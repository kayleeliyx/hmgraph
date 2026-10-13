"""Native, text-only PEFT LoRA and two-rank FSDP2 for pinned Qwen3.8.

No inference device-map dispatch is used. The full base first loads on CPU;
bottom-up FSDP2 groups then move/shard it without a complete GPU replica.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time

import torch
from torch import nn
from torch.nn import functional as F

from .common import BASELINE, MODEL, MODEL_REPO, MODEL_REVISION, RUN, sha

SEED = 20261003
TARGET_REGEX = (
    r'model\.language_model\.layers\.\d+\.'
    r'(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|'
    r'linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)|'
    r'mlp\.(?:gate_proj|up_proj|down_proj))'
)
EXPECTED_BASE_PARAMETERS = 27_356_728_560
EXPECTED_LORA_PARAMETERS = 116_727_808
MODEL_INPUT_KEYS = frozenset({
    'input_ids', 'attention_mask', 'position_ids', 'pixel_values',
    'image_grid_thw', 'mm_token_type_ids', 'pixel_values_videos', 'video_grid_thw',
})


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def json_safe(value):
    """Normalize native loading diagnostics without dropping nested evidence."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        normalized = [json_safe(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def _parameter_audit(model):
    names = []
    frozen = Counter()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            _require('.lora_' in name and '.model.language_model.layers.' in name,
                     f'Unexpected trainable parameter: {name}')
            _require(not any(x in name for x in ('visual', 'merger', 'lm_head', 'embed_tokens')),
                     f'Forbidden trainable module: {name}')
            names.append({'name': name, 'shape': list(parameter.shape),
                          'numel': parameter.numel(), 'dtype': str(parameter.dtype)})
        else:
            frozen[str(parameter.dtype)] += parameter.numel()
    return {
        'trainable_parameters': names,
        'trainable_parameter_count': sum(x['numel'] for x in names),
        'trainable_tensor_count': len(names),
        'frozen_parameter_count_by_dtype': dict(frozen),
        'frozen_parameter_count': sum(frozen.values()),
        'all_non_lora_frozen': all(not p.requires_grad for n, p in model.named_parameters()
                                   if '.lora_' not in n),
    }


def inject_text_lora(native, *, enforce_full_architecture=True):
    """Inject exact text Linear targets; also usable by CPU/meta regression tests."""
    from peft import LoraConfig, TaskType, get_peft_model
    targets = [name for name, module in native.named_modules()
               if isinstance(module, nn.Linear) and re.fullmatch(TARGET_REGEX, name)]
    all_text_linears = [name for name, module in native.named_modules()
                        if isinstance(module, nn.Linear)
                        and name.startswith('model.language_model.layers.')]
    _require(targets and sorted(targets) == sorted(all_text_linears),
             'The fixed target regex no longer covers exactly all native text Linear modules')
    if enforce_full_architecture:
        _require(len(targets) == 496, f'Unexpected text target count: {len(targets)}')
    native.config.use_cache = False
    native.config.text_config.use_cache = False
    lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias='none',
                             task_type=TaskType.CAUSAL_LM, target_modules=TARGET_REGEX)
    model = get_peft_model(native, lora_config, autocast_adapter_dtype=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    # HF enables all GradientCheckpointingLayer objects, including frozen visual
    # blocks. Keep text recomputation and explicitly disable every visual flag.
    vision_disabled = []
    for name, module in native.model.visual.named_modules():
        if hasattr(module, 'gradient_checkpointing'):
            module.gradient_checkpointing = False
            vision_disabled.append('model.visual' + ('.' + name if name else ''))
    _require(all(not p.requires_grad for p in native.model.visual.parameters()),
             'Vision encoder/aligner must be frozen')
    _require(all(getattr(layer, 'gradient_checkpointing', False)
                 for layer in native.model.language_model.layers),
             'Text decoder checkpointing must remain enabled')
    audit = _parameter_audit(model)
    _require(all(x['dtype'] == 'torch.float32' for x in audit['trainable_parameters']),
             'LoRA master parameters must be FP32')
    if enforce_full_architecture:
        _require(audit['trainable_parameter_count'] == EXPECTED_LORA_PARAMETERS,
                 'Unexpected LoRA parameter count')
        _require(audit['frozen_parameter_count'] == EXPECTED_BASE_PARAMETERS,
                 'Unexpected frozen base parameter count')
    audit.update({'target_regex': TARGET_REGEX, 'target_modules': targets,
                  'rank': 16, 'alpha': 32, 'dropout': 0.0, 'bias': 'none',
                  'vision_checkpointing_disabled_modules': vision_disabled,
                  'text_checkpointing': {'enabled': True, 'use_reentrant': False},
                  'vision_and_aligner_frozen': True,
                  'use_cache': False})
    return model, audit


def _initial_lora_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        _require(parameter.device.type == 'cpu', 'Hash initial LoRA on CPU before sharding')
        digest.update(name.encode() + b'\0')
        digest.update(str(tuple(parameter.shape)).encode() + b'\0')
        digest.update(parameter.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _fsdp2_shard(model, rank):
    """Construct homogeneous groups bottom-up, retaining native parameter dtype."""
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor
    from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

    _require(dist.is_initialized() and dist.get_world_size() == 2,
             'Caller must initialize a two-rank process group before load_model(shard=True)')
    _require(rank == dist.get_rank() and rank in (0, 1), 'Expected one local rank per authorized GPU')
    torch.cuda.set_device(rank)
    mesh = init_device_mesh('cuda', (2,))
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                  cast_forward_inputs=True)
    groups = []

    def remaining(module):
        return [(name, parameter) for name, parameter in module.named_parameters()
                if not isinstance(parameter, DTensor)]

    def wrap(module, name, role):
        pending = remaining(module)
        if not pending:
            return
        dtypes = {parameter.dtype for _, parameter in pending}
        if len(dtypes) > 1:
            # A native FP32 frozen descendant is kept FP32 in a separate group.
            # We never cast native weights or replace the computation to satisfy
            # FSDP's homogeneous-original-dtype constraint.
            for child_name, child in module.named_children():
                wrap(child, name + '.' + child_name, role + '_dtype_subgroup')
            pending = remaining(module)
            if not pending:
                return
            dtypes = {parameter.dtype for _, parameter in pending}
        _require(len(dtypes) == 1,
                 f'Cannot group mixed native direct-parameter dtypes without altering math: {name}: {dtypes}')
        record = {
            'module': name, 'role': role, 'original_dtype': str(next(iter(dtypes))),
            'global_parameter_count': sum(p.numel() for _, p in pending),
            'global_trainable_parameter_count': sum(p.numel() for _, p in pending if p.requires_grad),
            'parameter_names': [name + '.' + n for n, _ in pending],
            'reshard_after_forward': True, 'compute_dtype': 'torch.bfloat16',
            'reduce_dtype': 'torch.float32',
        }
        fully_shard(module, mesh=mesh, reshard_after_forward=True, mp_policy=policy)
        groups.append(record)

    # PEFT's FP32 master A/B leaves must be managed before the BF16 decoder.
    lora_leaves = [(name, module) for name, module in model.named_modules()
                   if isinstance(module, nn.Linear)
                   and ('.lora_A.' in name or '.lora_B.' in name)]
    _require(len(lora_leaves) == 992, 'Unexpected LoRA leaf groups')
    for name, module in lora_leaves:
        wrap(module, name, 'text_lora_fp32')
    native = model.base_model.model
    prefix = 'base_model.model.'
    for index, layer in enumerate(native.model.language_model.layers):
        wrap(layer, prefix + f'model.language_model.layers.{index}', 'frozen_text_decoder')
    _require(native.get_input_embeddings().weight is not native.get_output_embeddings().weight,
             'Unexpected tied input/output embeddings require a shared-weight grouping audit')
    wrap(native.get_input_embeddings(), prefix + 'model.language_model.embed_tokens', 'frozen_embedding')
    wrap(native.get_output_embeddings(), prefix + 'lm_head', 'frozen_output_head')
    wrap(native.model.visual, prefix + 'model.visual', 'frozen_vision_and_aligner')
    wrap(model, 'root', 'remaining_frozen_native')
    # Root must be an FSDP module even if all its parameters were assigned to
    # subgroups, so that pre/post-forward communication state is coordinated.
    from torch.distributed._composable.fsdp import FSDPModule
    if not isinstance(model, FSDPModule):
        fully_shard(model, mesh=mesh, reshard_after_forward=True, mp_policy=policy)
    _require(all(isinstance(p, DTensor) for p in model.parameters()), 'Unsharded parameter remains')
    _require(all(p.to_local().device == torch.device('cuda', rank) for p in model.parameters()),
             'Parameter shard on unexpected device')
    _require(sum(g['global_parameter_count'] for g in groups)
             == EXPECTED_BASE_PARAMETERS + EXPECTED_LORA_PARAMETERS,
             'Sharding groups must cover each native/LoRA parameter exactly once')
    _require(sum(g['global_trainable_parameter_count'] for g in groups) == EXPECTED_LORA_PARAMETERS,
             'Trainable sharding coverage mismatch')
    return groups


def load_model(rank, run=RUN, shard=True):
    """Return (PEFT model, audit); caller initializes distributed/GPU allocation.

    ``shard=False`` leaves the complete model on CPU; it is not an inference
    multi-device fallback. This function never loads any previous adapter.
    """
    import accelerate
    import peft
    import transformers
    import torch.distributed as dist
    from transformers import Qwen3_5ForConditionalGeneration

    started = time.monotonic()
    _require(torch.__version__ == '2.6.0+cu124' and transformers.__version__ == '5.3.0'
             and peft.__version__ == '0.18.1' and accelerate.__version__ == '1.12.0',
             'Use the frozen Qwen3.8 SFT environment')
    if shard:
        _require(dist.is_initialized() and dist.get_world_size() == 2,
                 'Two-rank process group must be initialized before loading CPU weights')
        _require(rank == dist.get_rank() and rank in (0, 1), 'Unexpected distributed rank')
        torch.cuda.set_device(rank)
    _require(MODEL.name == MODEL_REVISION and MODEL.is_dir(), 'Pinned local model snapshot unavailable')
    for filename in ('config.json', 'model.safetensors.index.json'):
        _require(sha(MODEL / filename) == sha(BASELINE / 'model_metadata' / filename),
                 f'Pinned model metadata changed: {filename}')
    torch.manual_seed(SEED)
    native, loading_info = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(MODEL), dtype=torch.bfloat16, attn_implementation='sdpa',
        device_map=None, local_files_only=True, trust_remote_code=False,
        output_loading_info=True,
    )
    _require(all(p.device.type == 'cpu' for p in native.parameters()), 'Base must load entirely on CPU')
    _require(not getattr(native, 'hf_device_map', None), 'Inference device-map dispatch is forbidden')
    for key in ('missing_keys', 'mismatched_keys', 'error_msgs'):
        _require(not loading_info.get(key), f'Native weight loading failed: {key}={loading_info.get(key)}')
    _require(all(key.startswith('mtp.') for key in loading_info.get('unexpected_keys', [])),
             f'Unexpected checkpoint keys: {loading_info.get("unexpected_keys")}')
    index = json.loads((MODEL / 'model.safetensors.index.json').read_text())['weight_map']
    native_keys = set(native.state_dict())
    _require(not native_keys - set(index), 'Native model has keys absent from pinned checkpoint')
    extra_keys = sorted(set(index) - native_keys)
    _require(all(key.startswith('mtp.') for key in extra_keys), 'Checkpoint/native key mismatch beyond MTP')
    # Reset immediately before adapter initialization, independent of loader RNG
    # consumption; both ranks must start from identical LoRA A and zero B.
    torch.manual_seed(SEED)
    model, parameter_audit = inject_text_lora(native)
    initial_digest = _initial_lora_digest(model)
    gathered = [initial_digest]
    if shard:
        gathered = [None] * 2
        dist.all_gather_object(gathered, initial_digest)
        _require(gathered == [initial_digest, initial_digest], 'Ranks initialized different LoRA parameters')
    groups = _fsdp2_shard(model, rank) if shard else []
    model.train()
    audit = {
        'model': {'repo': MODEL_REPO, 'revision': MODEL_REVISION, 'path': str(MODEL),
                  'config_sha256': sha(MODEL / 'config.json'), 'adapter_loaded': False},
        'rank': rank, 'run': str(Path(run)), 'seed': SEED,
        'base_loading_device': 'cpu', 'base_full_gpu_replica_created': False,
        'native_loading_info': json_safe(loading_info),
        'checkpoint_extra_mtp_keys': extra_keys,
        'initial_lora_sha256': initial_digest, 'initial_lora_rank_hashes': gathered,
        'sharding': 'FSDP2 FULL_SHARD' if shard else 'none; CPU only',
        'sharding_groups': groups, 'group_count': len(groups),
        'load_and_shard_seconds': time.monotonic() - started,
        **parameter_audit,
    }
    return model, audit


def _loss_inputs(batch):
    _require('labels' in batch and 'input_ids' in batch, 'Batch needs input_ids and labels')
    _require(not set(batch) - MODEL_INPUT_KEYS - {'labels'},
             f'Unexpected training/scoring fields: {set(batch) - MODEL_INPUT_KEYS - {"labels"}}')
    labels = batch['labels']
    _require(labels.ndim == 2 and labels.shape[0] == 1, 'Per-device batch size must remain one')
    _require(labels.shape == batch['input_ids'].shape, 'Input/label shape mismatch')
    _require(labels.dtype == torch.long, 'Labels must be int64')
    _require(bool(((labels == -100) | (labels >= 0)).all()), 'Unexpected ignore label')
    supervised = torch.nonzero(labels[0] >= 0, as_tuple=False).flatten()
    _require(supervised.numel() > 0 and int(supervised[0]) > 0,
             'Every supervised token must have a causal predecessor')
    _require(torch.equal(labels[0, supervised], batch['input_ids'][0, supervised]),
             'Supervised tokens must be the unchanged assistant tokens in input_ids')
    return {key: value for key, value in batch.items() if key != 'labels'}, supervised


def training_loss(model, batch):
    """Mean NLL of every supervised assistant/termination token for one record.

    Native logits_to_keep indexes *predecessor* hidden states, preserving exact
    next-token supervision while avoiding prompt-position vocabulary logits.
    """
    inputs, supervised = _loss_inputs(batch)
    output = model(**inputs, logits_to_keep=supervised - 1, use_cache=False, return_dict=True)
    logits = output.logits
    _require(logits.shape[:2] == (1, supervised.numel()), 'Selected target logits were lost or reshaped')
    targets = batch['labels'][0, supervised]
    return F.cross_entropy(logits[0].float(), targets, reduction='mean')


@torch.no_grad()
def full_native_loss_parity(model, batch, *, atol=2e-5, rtol=2e-5):
    """Optional smoke check, computing full logits once; never used in the loop."""
    inputs, supervised = _loss_inputs(batch)
    was_training = model.training
    model.eval()
    try:
        selected = training_loss(model, batch)
        native = model(**inputs, labels=batch['labels'], logits_to_keep=0,
                       use_cache=False, return_dict=True).loss
        _require(bool(torch.isfinite(selected)) and bool(torch.isfinite(native)), 'Nonfinite loss parity check')
        _require(torch.allclose(selected.float(), native.float(), atol=atol, rtol=rtol),
                 f'Selected target loss differs from native loss: {selected.item()} vs {native.item()}')
        return {'selected_target_loss': selected.item(), 'native_full_loss': native.item(),
                'absolute_difference': abs(selected.item() - native.item()),
                'supervised_token_count': supervised.numel(), 'passed': True,
                'atol': atol, 'rtol': rtol}
    finally:
        model.train(was_training)
