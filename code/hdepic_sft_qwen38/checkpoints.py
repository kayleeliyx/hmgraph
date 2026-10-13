"""Small, collective FSDP2 LoRA checkpoints; frozen base weights are never saved.

Both ranks call save/load after an optimizer boundary, with FSDP parameters
resharded and no accumulated gradients. Resume requires the same two-rank mesh,
adapter architecture, named optimizer groups and exact caller-supplied identity.
The separately gathered PEFT adapter supports a fresh base-plus-adapter reload.
"""
from __future__ import annotations

import copy
import json
import random
import shutil
from pathlib import Path

from .common import canonical, read_json, sha, write_json

FORMAT = "hdepic_fsdp2_lora_checkpoint_v1"
RANK_FILES = ("rank-00000.pt", "rank-00001.pt")
ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json")
PROGRESS_FIELDS = ("step", "epoch", "microbatch_consumed", "sampler_state")


def _identity(value):
    if not isinstance(value, dict) or not value:
        raise ValueError("Checkpoint identity must be a nonempty JSON-safe dict")
    return json.loads(canonical(value))


def _progress(value):
    value = json.loads(canonical(value))
    if not isinstance(value, dict) or any(k not in value for k in PROGRESS_FIELDS):
        raise ValueError(f"Progress must contain {PROGRESS_FIELDS}")
    for key in ("step", "microbatch_consumed"):
        if type(value[key]) is not int or value[key] < 0:
            raise ValueError(f"Invalid progress {key}")
    if isinstance(value["epoch"], bool) or not isinstance(value["epoch"], (int, float)) or value["epoch"] < 0:
        raise ValueError("Invalid progress epoch")
    if not isinstance(value["sampler_state"], dict):
        raise ValueError("sampler_state must be a dict")
    return value


def _distributed():
    import torch.distributed as dist
    if not dist.is_initialized() or dist.get_world_size() != 2:
        raise RuntimeError("This checkpoint protocol requires both ranks of one two-rank process group")
    return dist, dist.get_rank()


def _phase(fn, label):
    """Propagate a local phase failure before the next collective/mutation."""
    dist, _ = _distributed()
    value, error = None, None
    try:
        value = fn()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = [None, None]
    dist.all_gather_object(errors, error)
    if any(errors):
        raise RuntimeError(f"Checkpoint {label} failed on ranks: {errors}")
    return value


def _mesh(mesh):
    return {"device_type": mesh.device_type, "ranks": mesh.mesh.cpu().tolist(),
            "dim_names": list(mesh.mesh_dim_names) if mesh.mesh_dim_names is not None else None}


def _placements(placements):
    from torch.distributed.tensor import Partial, Replicate, Shard
    result = []
    for placement in placements:
        if isinstance(placement, Shard):
            result.append({"type": "shard", "dim": placement.dim})
        elif isinstance(placement, Replicate):
            result.append({"type": "replicate"})
        elif isinstance(placement, Partial):
            result.append({"type": "partial", "reduce_op": placement.reduce_op})
        else:
            raise ValueError(f"Unsupported DTensor placement {placement}")
    return result


def _restore_placements(values):
    from torch.distributed.tensor import Partial, Replicate, Shard
    result = []
    for value in values:
        if value["type"] == "shard":
            result.append(Shard(value["dim"]))
        elif value["type"] == "replicate":
            result.append(Replicate())
        elif value["type"] == "partial":
            result.append(Partial(value["reduce_op"]))
        else:
            raise ValueError("Invalid saved DTensor placement")
    return result


def _tensor_record(tensor):
    import torch
    from torch.distributed.tensor import DTensor
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Expected tensor")
    is_dtensor = isinstance(tensor, DTensor)
    local = tensor.to_local() if is_dtensor else tensor
    result = {"kind": "dtensor" if is_dtensor else "tensor", "dtype": str(tensor.dtype),
              "shape": list(tensor.shape), "stride": list(tensor.stride()),
              "local_shape": list(local.shape), "device_type": local.device.type,
              "value": local.detach().cpu().clone(memory_format=torch.preserve_format)}
    if is_dtensor:
        result.update(mesh=_mesh(tensor.device_mesh), placements=_placements(tensor.placements))
    return result


def _restore_tensor(record, parameter):
    import torch
    from torch.distributed.tensor import DTensor
    local_parameter = parameter.to_local() if isinstance(parameter, DTensor) else parameter
    value = record["value"]
    if str(value.dtype) != record["dtype"] or list(value.shape) != record["local_shape"]:
        raise ValueError("Saved tensor payload disagrees with its dtype/shape metadata")
    device = "cpu" if record["device_type"] == "cpu" else local_parameter.device
    value = value.to(device=device).clone(memory_format=torch.preserve_format)
    if record["kind"] == "dtensor":
        if not isinstance(parameter, DTensor) or _mesh(parameter.device_mesh) != record["mesh"]:
            raise ValueError("Optimizer DTensor mesh changed")
        return DTensor.from_local(value, device_mesh=parameter.device_mesh,
                                  placements=_restore_placements(record["placements"]),
                                  run_check=False, shape=torch.Size(record["shape"]),
                                  stride=tuple(record["stride"]))
    if record["kind"] != "tensor":
        raise ValueError("Invalid tensor record kind")
    return value


def _pack(value):
    import torch
    if isinstance(value, torch.Tensor):
        return {"kind": "tensor_record", "record": _tensor_record(value)}
    if isinstance(value, dict):
        return {"kind": "dict", "items": [(k, _pack(v)) for k, v in value.items()]}
    if isinstance(value, (list, tuple)):
        return {"kind": "tuple" if isinstance(value, tuple) else "list", "items": [_pack(v) for v in value]}
    if value is None or isinstance(value, (str, bool, int, float)):
        return {"kind": "scalar", "value": value}
    raise TypeError(f"Unsupported optimizer/scheduler state value {type(value)}")


def _unpack(value, parameter):
    kind = value["kind"]
    if kind == "tensor_record":
        return _restore_tensor(value["record"], parameter)
    if kind == "dict":
        return {k: _unpack(v, parameter) for k, v in value["items"]}
    if kind in ("list", "tuple"):
        values = [_unpack(v, parameter) for v in value["items"]]
        return tuple(values) if kind == "tuple" else values
    if kind == "scalar":
        return value["value"]
    raise ValueError("Invalid saved nested state")


def _trainable(model):
    from torch.distributed.tensor import DTensor
    values = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if not values or any("lora_" not in name for name in values):
        raise ValueError("Only LoRA parameters may be trainable in this checkpoint protocol")
    if any(not isinstance(parameter, DTensor) for parameter in values.values()):
        raise ValueError("All trainable parameters must be resharded FSDP2 DTensors")
    if any(parameter.grad is not None for parameter in values.values()):
        raise ValueError("Save/load requires zero_grad(set_to_none=True) at an optimizer boundary")
    return dict(sorted(values.items()))


def _adapter(model):
    configs = getattr(model, "peft_config", {})
    if list(configs) != ["default"]:
        raise ValueError("This pilot requires one PEFT adapter named default")
    config = configs["default"]
    if str(config.peft_type) not in ("LORA", "PeftType.LORA") or config.bias != "none" or config.modules_to_save or config.use_dora:
        raise ValueError("Only pure LoRA with bias=none, no modules_to_save/DoRA is supported")
    return config


def _adapter_dict(config):
    def plain(value):
        if isinstance(value, dict):
            return {k: plain(v) for k, v in value.items()}
        if isinstance(value, set):
            return sorted(plain(v) for v in value)
        if isinstance(value, (tuple, list)):
            return [plain(v) for v in value]
        return value
    return json.loads(canonical(plain(config.to_dict())))


def _schema(parameters):
    result = {}
    for name, parameter in parameters.items():
        result[name] = {"dtype": str(parameter.dtype), "shape": list(parameter.shape),
                        "stride": list(parameter.stride()), "mesh": _mesh(parameter.device_mesh),
                        "placements": _placements(parameter.placements)}
    return result


def _optimizer_groups(optimizer, parameters):
    by_object = {id(value): name for name, value in parameters.items()}
    groups, seen = [], []
    for group in optimizer.param_groups:
        names = []
        for parameter in group["params"]:
            if id(parameter) not in by_object:
                raise ValueError("Optimizer contains a frozen or foreign parameter")
            names.append(by_object[id(parameter)])
        seen.extend(names)
        groups.append({"names": names, "options": _pack({k: v for k, v in group.items() if k != "params"})})
    if len(seen) != len(set(seen)) or set(seen) != set(parameters):
        raise ValueError("Optimizer must cover each trainable parameter exactly once")
    if any(id(parameter) not in by_object for parameter in optimizer.state):
        raise ValueError("Optimizer state contains a frozen or foreign parameter")
    return groups


def _rng():
    import numpy as np
    import torch
    result = {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state()}
    # Inspect only this process's current CUDA device; never initialize others.
    if torch.cuda.is_initialized():
        result["torch_cuda"] = torch.cuda.get_rng_state(torch.cuda.current_device()).cpu()
    return result


def _restore_rng(value):
    import numpy as np
    import torch
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"])
    if "torch_cuda" in value:
        if not torch.cuda.is_initialized():
            raise ValueError("CUDA must already be initialized to restore saved CUDA RNG")
        torch.cuda.set_rng_state(value["torch_cuda"], device=torch.cuda.current_device())


def _class(value):
    return f"{type(value).__module__}.{type(value).__qualname__}" if value is not None else None


def _same(left, right):
    import numpy as np
    import torch
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def _file(path):
    return {"sha256": sha(path), "bytes": path.stat().st_size}


def validate_checkpoint(path, identity):
    """Read-only validation, including hashes of every compact checkpoint file."""
    path = Path(path)
    marker = read_json(path / "completed.json")
    identity = _identity(identity)
    if marker.get("format") != FORMAT or marker.get("passed") is not True or marker.get("world_size") != 2:
        raise ValueError("Missing or incompatible checkpoint completion marker")
    if marker.get("identity") != identity or marker.get("identity_sha256") != __import__("hashlib").sha256(canonical(identity)).hexdigest():
        raise ValueError("Checkpoint identity differs from config/model/data identity")
    _progress(marker["state"])
    expected = set(RANK_FILES + ADAPTER_FILES)
    if set(marker.get("files", {})) != expected:
        raise ValueError("Incomplete checkpoint file inventory")
    if {v.name for v in path.iterdir()} != expected | {"completed.json"}:
        raise ValueError("Unexpected checkpoint file; frozen base weights must never be saved")
    if not marker.get("trainable_schema") or any("lora_" not in n for n in marker["trainable_schema"]):
        raise ValueError("Missing pure-LoRA trainable inventory")
    for name, record in marker["files"].items():
        source = path / name
        if source.is_symlink() or not source.is_file() or _file(source) != record:
            raise ValueError(f"Checkpoint file missing or changed: {name}")
    return marker


def save_checkpoint(model, optimizer, scheduler, path, state, identity):
    """Collectively save a new checkpoint without serializing frozen parameters.

    `state` contains step, epoch, microbatch_consumed and sampler_state. This
    function restores its captured RNG before returning, so saving cannot alter
    the subsequent training RNG stream. An existing valid identical checkpoint
    is immutable and reusable; an incompatible existing destination is rejected.
    """
    import hashlib
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file

    dist, rank = _distributed()
    path = Path(path)
    pending = path.with_name(path.name + ".incomplete")

    def prepare_local():
        exact_identity, exact_state = _identity(identity), _progress(state)
        parameters = _trainable(model)
        config = _adapter(model)
        groups = _optimizer_groups(optimizer, parameters)
        return exact_identity, exact_state, parameters, config, groups, _rng()

    identity, state, parameters, adapter_config, groups, rng = _phase(prepare_local, "local preparation")
    signature = {"path": str(path.resolve()), "identity": identity, "state": state, "schema": _schema(parameters),
                 "adapter_config": _adapter_dict(adapter_config),
                 "optimizer_class": _class(optimizer), "scheduler_class": _class(scheduler),
                 "groups": [g["names"] for g in groups]}
    signatures = [None, None]
    dist.all_gather_object(signatures, signature)
    if signatures[0] != signatures[1]:
        raise ValueError("Ranks disagree on checkpoint identity/progress/architecture")

    def destination():
        if path.exists():
            marker = validate_checkpoint(path, identity)
            if (marker["state"] != state or marker["trainable_schema"] != signature["schema"]
                    or marker["adapter_config"] != signature["adapter_config"]):
                raise ValueError("Existing checkpoint has different progress or architecture")
            old = torch.load(path / RANK_FILES[rank], map_location="cpu", weights_only=False)
            current = {"trainable": {n: _tensor_record(p) for n, p in parameters.items()},
                       "optimizer": {"class": _class(optimizer), "groups": groups,
                                     "state": {n: _pack(optimizer.state[p]) for n, p in parameters.items() if p in optimizer.state}},
                       "scheduler": {"class": _class(scheduler), "state": _pack(scheduler.state_dict()) if scheduler is not None else None},
                       "rng": rng}
            if any(not _same(old[k], v) for k, v in current.items()):
                raise ValueError("Existing checkpoint differs from current LoRA/optimizer/scheduler/RNG state")
            return marker
        return None

    existing = _phase(destination, "existing destination validation")
    if existing is not None:
        _restore_rng(rng)
        return existing

    def create_pending():
        if rank == 0:
            path.parent.mkdir(parents=True, exist_ok=True)
            if pending.exists():
                if pending.is_symlink() or not pending.is_dir():
                    raise ValueError("Unsafe incomplete checkpoint destination")
                shutil.rmtree(pending)
            pending.mkdir()
    _phase(create_pending, "temporary directory creation")

    def save_rank():
        payload = {"format": FORMAT, "rank": rank, "world_size": 2, "identity": identity,
                   "state": state, "trainable": {n: _tensor_record(p) for n, p in parameters.items()},
                   "optimizer": {"class": _class(optimizer), "groups": groups,
                                 "state": {n: _pack(optimizer.state[p]) for n, p in parameters.items() if p in optimizer.state}},
                   "scheduler": {"class": _class(scheduler), "state": _pack(scheduler.state_dict()) if scheduler is not None else None},
                   "rng": rng}
        torch.save(payload, pending / RANK_FILES[rank])
    _phase(save_rank, "rank-local serialization")

    # Only one small LoRA tensor is gathered at a time; no base state_dict call.
    full = {}
    for name, parameter in parameters.items():
        gathered = parameter.detach().full_tensor()
        if rank == 0:
            full[name] = gathered.cpu().contiguous().clone()
        del gathered

    def save_adapter():
        if rank == 0:
            adapter = get_peft_model_state_dict(model, state_dict=full, adapter_name="default", save_embedding_layers=False)
            if not adapter or any("lora_" not in n for n in adapter) or len(adapter) != len(parameters):
                raise ValueError("PEFT export lost trainable tensors or included non-LoRA weights")
            save_file(adapter, str(pending / ADAPTER_FILES[0]), metadata={"format": "pt"})
            exported_config = copy.deepcopy(adapter_config)
            exported_config.inference_mode = True
            exported_config.save_pretrained(pending)
    _phase(save_adapter, "PEFT adapter export")
    del full

    def complete():
        if rank == 0:
            marker = {"format": FORMAT, "passed": True, "world_size": 2,
                      "identity": identity, "identity_sha256": hashlib.sha256(canonical(identity)).hexdigest(),
                      "state": state, "trainable_schema": signature["schema"],
                      "adapter_config": signature["adapter_config"],
                      "optimizer_class": signature["optimizer_class"], "scheduler_class": signature["scheduler_class"],
                      "optimizer_group_names": signature["groups"],
                      "files": {n: _file(pending / n) for n in RANK_FILES + ADAPTER_FILES}}
            write_json(pending / "completed.json", marker)
            pending.rename(path)
    _phase(complete, "atomic completion")
    marker = _phase(lambda: validate_checkpoint(path, identity), "completed inventory validation")
    _restore_rng(rng)
    return marker


def load_checkpoint(model, optimizer, scheduler, path, identity):
    """Collectively restore local LoRA/AdamW shards, scheduler and rank RNG.

    Reconstruct the exact frozen base, PEFT adapter and FSDP2 architecture first.
    Recreate the sampler/iterator and skip consumed batches without consuming
    restored training RNG (use a dedicated DataLoader generator). The returned
    state defines the sampler epoch and consumed microbatch count.
    """
    import torch
    from torch.distributed.tensor import DTensor
    dist, rank = _distributed()
    path = Path(path)

    def validate_local():
        marker = validate_checkpoint(path, identity)
        parameters = _trainable(model)
        adapter_config = _adapter(model)
        groups = _optimizer_groups(optimizer, parameters)
        if marker["trainable_schema"] != _schema(parameters):
            raise ValueError("Trainable architecture, dtype, mesh or placement changed")
        if marker["adapter_config"] != _adapter_dict(adapter_config):
            raise ValueError("LoRA adapter configuration changed")
        if marker["optimizer_class"] != _class(optimizer) or marker["scheduler_class"] != _class(scheduler):
            raise ValueError("Optimizer or scheduler class changed")
        if marker["optimizer_group_names"] != [g["names"] for g in groups]:
            raise ValueError("Named optimizer groups changed")
        payload = torch.load(path / RANK_FILES[rank], map_location="cpu", weights_only=False)
        if any(payload.get(k) != v for k, v in {"format": FORMAT, "rank": rank, "world_size": 2,
                                               "identity": _identity(identity), "state": marker["state"]}.items()):
            raise ValueError("Rank payload identity/progress is inconsistent")
        if set(payload["trainable"]) != set(parameters):
            raise ValueError("Rank trainable inventory changed")
        if payload["optimizer"]["class"] != _class(optimizer) or payload["scheduler"]["class"] != _class(scheduler):
            raise ValueError("Rank payload optimizer/scheduler class changed")
        if [g["names"] for g in payload["optimizer"]["groups"]] != marker["optimizer_group_names"]:
            raise ValueError("Rank payload named optimizer groups changed")
        if not set(payload["optimizer"]["state"]).issubset(parameters):
            raise ValueError("Rank optimizer inventory includes foreign parameters")
        for name, parameter in parameters.items():
            record = payload["trainable"][name]
            local = parameter.to_local()
            if (record["kind"] != "dtensor" or record["dtype"] != str(parameter.dtype)
                    or record["shape"] != list(parameter.shape) or record["stride"] != list(parameter.stride())
                    or record["local_shape"] != list(local.shape) or record["mesh"] != _mesh(parameter.device_mesh)
                    or record["placements"] != _placements(parameter.placements)
                    or str(record["value"].dtype) != str(local.dtype) or list(record["value"].shape) != list(local.shape)):
                raise ValueError(f"Rank parameter shard changed: {name}")
        # Materialize all compact state before mutating model/optimizer.
        states = {n: _unpack(v, parameters[n]) for n, v in payload["optimizer"]["state"].items()}
        options = [_unpack(g["options"], parameters[g["names"][0]]) for g in payload["optimizer"]["groups"]]
        scheduler_state = _unpack(payload["scheduler"]["state"], next(iter(parameters.values()))) if scheduler is not None else None
        if "torch_cuda" in payload["rng"] and not torch.cuda.is_initialized():
            raise ValueError("CUDA RNG cannot be restored before CUDA initialization")
        return marker, parameters, payload, states, options, scheduler_state

    marker, parameters, payload, states, options, scheduler_state = _phase(validate_local, "resume prevalidation")
    markers = [None, None]
    dist.all_gather_object(markers, {"path": str(path.resolve()), "marker": marker})
    if markers[0] != markers[1]:
        raise ValueError("Ranks disagree on the checkpoint being resumed")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.to_local().copy_(payload["trainable"][name]["value"].to(parameter.to_local().device))
    optimizer.state.clear()
    for name, value in states.items():
        optimizer.state[parameters[name]] = value
    for group, restored in zip(optimizer.param_groups, options):
        for key in list(group):
            if key != "params":
                del group[key]
        group.update(restored)
    if scheduler is not None:
        scheduler.load_state_dict(scheduler_state)
    dist.barrier()
    _restore_rng(payload["rng"])
    return marker["state"]


def choose_resume(root, identity):
    """Return the newest fully verified matching checkpoint, or None.

    Incomplete temporary directories, mismatched identities and corrupted
    checkpoints are never candidates. This function reads only small checkpoint
    metadata/files and does not initialize Torch, a model or a CUDA device.
    """
    root = Path(root)
    identity = _identity(identity)
    candidates = []
    if not root.exists():
        return None
    for path in root.iterdir():
        if not path.is_dir() or path.is_symlink() or path.name.endswith(".incomplete"):
            continue
        try:
            marker = validate_checkpoint(path, identity)
            state = marker["state"]
            candidates.append(((state["step"], state["epoch"], state["microbatch_consumed"], path.name), path))
        except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError):
            continue
    return max(candidates, key=lambda item: item[0])[1] if candidates else None
