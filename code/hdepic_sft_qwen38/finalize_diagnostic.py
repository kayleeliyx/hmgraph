"""Read-only two-rank diagnosis of final FSDP2 frozen-sample comparison.

Reconstruct original base/LoRA, restore checkpoint-1818, run exactly one
initial-SFT teacher-forced no-grad forward, and explicitly reshard every FSDP
module. No backward, generation, optimizer/scheduler step or completion-marker
write occurs here. Diagnostic evidence is separate from training artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import traceback

from .common import RUN, MODEL, GPU_UUIDS, configure, read_json, write_json, sha
from .train import identity, local_tensor, freeze_samples, memory


def differences(before, after):
    return sorted(n for n in before.keys() | after.keys() if before.get(n) != after.get(n))


def tensor_hash(tensor):
    import torch
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def descriptors(model):
    from torch.distributed.tensor import DTensor
    return {n: {'is_dtensor': isinstance(p, DTensor), 'shape': list(p.shape),
                'local_shape': list(local_tensor(p).shape), 'dtype': str(p.dtype),
                'local_dtype': str(local_tensor(p).dtype), 'requires_grad': p.requires_grad,
                'placements': [str(v) for v in p.placements] if isinstance(p, DTensor) else None}
            for n, p in model.named_parameters()}


def one_row(path, index):
    """Parse only the selected initial-SFT record, never a validation/test row."""
    with Path(path).open() as handle:
        for row_index, line in enumerate(handle):
            if row_index == index:
                return json.loads(line)
    raise ValueError(f'Missing row {index}: {path}')


def run_diagnostic(args):
    configure()
    run = Path(args.run).resolve()
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else run / 'training/main/checkpoints/checkpoint-1818'
    out = Path(args.output).resolve() if args.output else run / 'recovery/finalize_diagnostic'
    assert out.is_relative_to(run / 'recovery'), 'Diagnostic writes must stay under run/recovery'
    ident = identity(run)
    provenance = {'identity': ident, 'diagnostic_source_sha256': sha(__file__),
                  'checkpoint': str(checkpoint), 'checkpoint_completed_sha256': sha(checkpoint / 'completed.json'),
                  'no_optimizer_updates': True, 'no_backward': True, 'no_generation': True,
                  'training_completion_written': False}
    if (out / 'summary.json').exists():
        old = read_json(out / 'summary.json')
        assert all(old['provenance'].get(k) == v for k, v in provenance.items()) and old['passed'], 'Existing diagnosis is incompatible or failed; select a new --output'
        from .checkpoints import validate_checkpoint
        assert validate_checkpoint(checkpoint, ident)['files'] == old['provenance']['checkpoint_files']
        for key in ('torch_fsdp_module_source', 'torch_fsdp_state_source'):
            assert sha(old['provenance'][key]['path']) == old['provenance'][key]['sha256']
        print('Existing matching successful diagnosis: ' + str(out), flush=True)
        return
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ','.join(GPU_UUIDS)
    rank = int(os.environ['RANK'])
    assert rank == int(os.environ['LOCAL_RANK']) and rank in (0, 1) and os.environ['WORLD_SIZE'] == '2'
    import torch
    import torch.distributed as dist
    from torch.distributed._composable.fsdp import FSDPModule
    from torch.distributed.tensor import DTensor
    from transformers import get_cosine_schedule_with_warmup
    from safetensors import safe_open
    from .model import load_model, training_loss, EXPECTED_BASE_PARAMETERS, EXPECTED_LORA_PARAMETERS
    from .checkpoints import (load_checkpoint, validate_checkpoint, _optimizer_groups,
                              _pack, _rng, _same, _phase)
    from .data import build_training_inputs
    from code.hdepic_qwen38_baseline.processor import load_processor
    torch.set_num_threads(2)
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', device_id=torch.device('cuda', rank))
    out.mkdir(parents=True, exist_ok=True)
    stage = 'start'
    try:
        marker = _phase(lambda: validate_checkpoint(checkpoint, ident), 'diagnostic checkpoint validation')
        expected_state = {'step': 1818, 'epoch': 1, 'microbatch_consumed': 7272,
                          'sampler_state': {'epoch': 1, 'seed': 20261003}}
        assert marker['state'] == expected_state
        provenance['checkpoint_files'] = marker['files']
        provenance['torch_fsdp_module_source'] = {'path': inspect.getfile(FSDPModule), 'sha256': sha(inspect.getfile(FSDPModule))}
        from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState
        provenance['torch_fsdp_state_source'] = {'path': inspect.getfile(FSDPState), 'sha256': sha(inspect.getfile(FSDPState))}
        write_json(out / f'started_rank{rank}.json', provenance)
        stage = 'original_model'
        model, audit = load_model(rank, run, shard=True)
        write_json(out / f'model_audit_rank{rank}.json', audit)
        root_groups = [g for g in audit['sharding_groups'] if g['module'] == 'root']
        root_names = sorted(n.removeprefix('root.') for g in root_groups for n in g['parameter_names'])
        assert root_names == ['base_model.model.model.language_model.norm.weight'], root_names
        assert sum(g['global_parameter_count'] for g in root_groups) == 5120
        trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
        assert len(trainable) == 992 and sum(p.numel() for p in trainable.values()) == EXPECTED_LORA_PARAMETERS
        logical_base = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        assert logical_base == EXPECTED_BASE_PARAMETERS
        initial_lora = {n: local_tensor(p.detach()).cpu().clone() for n, p in trainable.items()}

        def snapshot(label):
            named = dict(model.named_parameters())
            root_evidence = {}
            for name in root_names:
                p = named[name].detach()
                full = p.full_tensor() if isinstance(p, DTensor) else p
                flat = full.reshape(-1)
                index = torch.arange(min(flat.numel(), 64), device=flat.device) * (flat.numel() - 1) // 63
                root_evidence[name] = {'full_tensor_sha256': tensor_hash(full),
                                       'full_64_sample_sha256': tensor_hash(flat[index]),
                                       'local_tensor_sha256': tensor_hash(local_tensor(p)),
                                       'full_numel': full.numel()}
            value = {'stage': label, 'frozen_samples': freeze_samples(model),
                     'parameter_descriptors': descriptors(model), 'root_norm': root_evidence,
                     'logical_frozen_base_parameter_count': logical_base,
                     'gradient_tensors': sum(p.grad is not None for p in model.parameters()), **memory()}
            write_json(out / f'{label}_rank{rank}.json', value)
            return value

        initial = snapshot('initial')
        optimizer = torch.optim.AdamW(list(trainable.values()), lr=1e-4, betas=(.9, .95), eps=1e-8,
                                     weight_decay=.1, foreach=False)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 55, 1818)
        stage = 'checkpoint_restore'
        state = load_checkpoint(model, optimizer, scheduler, checkpoint, ident)
        payload = torch.load(checkpoint / f'rank-{rank:05}.pt', map_location='cpu', weights_only=False)
        restored_rng_exact = _same(_rng(), payload['rng'])

        def restore_proof():
            exact = {n: torch.equal(local_tensor(p.detach()).cpu(), payload['trainable'][n]['value'])
                     for n, p in trainable.items()}
            optimizer_exact = {n: _same(_pack(optimizer.state[p]), payload['optimizer']['state'][n])
                               for n, p in trainable.items()}
            groups_exact = _same(_optimizer_groups(optimizer, trainable), payload['optimizer']['groups'])
            scheduler_exact = _same(_pack(scheduler.state_dict()), payload['scheduler']['state'])
            return {'adapter_local_tensor_exact': exact, 'optimizer_named_state_exact': optimizer_exact,
                    'optimizer_groups_exact': groups_exact, 'scheduler_exact': scheduler_exact,
                    'scheduler_last_epoch': scheduler.last_epoch,
                    'adapter_tensor_count': len(exact), 'optimizer_state_count': len(optimizer.state),
                    'all_exact': all(exact.values()) and all(optimizer_exact.values()) and groups_exact and scheduler_exact}

        restored = _phase(restore_proof, 'diagnostic exact restoration proof')
        restored['rng_exact_immediately_after_load'] = restored_rng_exact
        write_json(out / f'restored_rank{rank}.json', restored)
        assert restored['all_exact'] and restored_rng_exact and state == expected_state and scheduler.last_epoch == 1818
        # Compare every gathered full LoRA tensor to the ordinary PEFT export,
        # one tensor at a time, without ever materializing a base state_dict.
        adapter_exact = {}
        with safe_open(checkpoint / 'adapter_model.safetensors', framework='pt', device='cpu') as handle:
            assert set(handle.keys()) == {n.replace('.default', '') for n in trainable}
            for n, p in sorted(trainable.items()):
                full = p.detach().full_tensor().cpu()
                adapter_exact[n] = torch.equal(full, handle.get_tensor(n.replace('.default', '')))
        write_json(out / f'peft_adapter_exact_rank{rank}.json', {'full_tensor_exact': adapter_exact,
                   'count': len(adapter_exact), 'all_exact': all(adapter_exact.values()), 'no_base_state_dict': True})
        assert len(adapter_exact) == 992 and all(adapter_exact.values())
        loaded = snapshot('loaded_before_eval')
        stage = 'one_initial_sft_no_grad_forward'
        smoke_path = run / 'inputs/smoke_selection.json'
        selected = read_json(smoke_path)['records'][0]
        assert selected['partition'] == 'initial_sft'
        spec = read_json(run / 'inputs/manifest.json')['partitions']['initial_sft']
        for kind in ('export', 'scoring', 'prompts'):
            assert sha(spec[kind]) == spec[kind + '_sha256']
        export, scoring = (one_row(spec[k], selected['row_index']) for k in ('export', 'scoring'))
        assert scoring['id'] == selected['id'] and scoring['partition'] == 'initial_sft'
        processor = load_processor(MODEL)
        batch, input_audit = build_training_inputs(processor, export, selected['id'], selected['row_index'])
        assert input_audit['prompt_sha256'] == selected['prompt_sha256']
        batch = {k: v.cuda() for k, v in batch.items()}
        model.eval()
        with torch.no_grad():
            loss = training_loss(model, batch)
        model.train()
        torch.cuda.synchronize()
        write_json(out / f'forward_rank{rank}.json', {'id': selected['id'], 'partition': 'initial_sft',
                   'row_index': selected['row_index'], 'loss': loss.item(), 'finite_loss': bool(torch.isfinite(loss)),
                   'input_audit': input_audit, 'smoke_selection_sha256': sha(smoke_path),
                   'source_files': {k: {'path': spec[k], 'sha256': spec[k + '_sha256']} for k in ('export', 'scoring', 'prompts')},
                   'eval_no_grad_then_train': True, 'no_generation': True, 'no_optimizer_updates': True})
        after_eval = snapshot('after_eval')
        stage = 'explicit_recursive_reshard'
        resharded = []
        for n, module in reversed(list(model.named_modules())):
            if isinstance(module, FSDPModule):
                module.reshard()
                resharded.append(n or '<root>')
        torch.cuda.synchronize()
        dist.barrier()
        after_reshard = snapshot('after_reshard')
        final_restore = _phase(restore_proof, 'diagnostic no-update proof')
        deltas = {}
        for n, p in trainable.items():
            final = local_tensor(p.detach()).cpu()
            delta = final - initial_lora[n]
            deltas[n] = {'initial_local_sha256': tensor_hash(initial_lora[n]), 'final_local_sha256': tensor_hash(final),
                         'local_shape': list(final.shape), 'max_abs_delta': delta.abs().max().item(),
                         'changed': not torch.equal(final, initial_lora[n]), 'finite': bool(torch.isfinite(delta).all())}
        write_json(out / f'lora_deltas_rank{rank}.json', {'count': len(deltas), 'parameters': deltas})
        frozen_names = initial['frozen_samples'].keys()
        frozen_desc = lambda value: {n: value['parameter_descriptors'][n] for n in frozen_names}
        checks = {
            'loaded_frozen_hashes_match_initial': initial['frozen_samples'] == loaded['frozen_samples'],
            'loaded_frozen_descriptors_match_initial': frozen_desc(initial) == frozen_desc(loaded),
            'after_eval_hash_mismatch_set_equals_root_norm': differences(initial['frozen_samples'], after_eval['frozen_samples']) == root_names,
            'after_eval_descriptor_mismatch_set_equals_root_norm': differences(frozen_desc(initial), frozen_desc(after_eval)) == root_names,
            'after_reshard_all_frozen_hashes_match_initial': initial['frozen_samples'] == after_reshard['frozen_samples'],
            'after_reshard_all_frozen_descriptors_match_initial': frozen_desc(initial) == frozen_desc(after_reshard),
            'full_root_norm_hash_unchanged_all_stages': all(initial['root_norm'][n]['full_tensor_sha256'] == v['root_norm'][n]['full_tensor_sha256'] for n in root_names for v in (loaded, after_eval, after_reshard)),
            'all_992_lora_changed_finite': len(deltas) == 992 and all(d['changed'] and d['finite'] for d in deltas.values()),
            'exact_adapter_optimizer_scheduler_unchanged_after_eval_reshard': final_restore['all_exact'],
            'all_992_saved_full_peft_adapter_tensors_exact': len(adapter_exact) == 992 and all(adapter_exact.values()),
            'no_gradient_tensors': after_reshard['gradient_tensors'] == 0,
            'finite_teacher_forced_loss': bool(torch.isfinite(loss)),
            'scheduler_still_at_1818': scheduler.last_epoch == 1818,
        }
        result = {'provenance': provenance, 'rank': rank, 'passed': all(checks.values()), 'checks': checks,
                  'classification': 'root_fsdp_unsharded_after_eval_confirmed' if all(checks.values()) else 'diagnostic_failed',
                  'root_norm_names_from_model_audit': root_names, 'restored_state': state,
                  'logical_frozen_base_parameter_count': logical_base, 'global_lora_parameter_count': EXPECTED_LORA_PARAMETERS,
                  'after_eval_hash_mismatches': differences(initial['frozen_samples'], after_eval['frozen_samples']),
                  'after_eval_descriptor_diffs': {n: {'before': initial['parameter_descriptors'].get(n), 'after': after_eval['parameter_descriptors'].get(n)} for n in differences(frozen_desc(initial), frozen_desc(after_eval))},
                  'after_reshard_hash_mismatches': differences(initial['frozen_samples'], after_reshard['frozen_samples']),
                  'resharded_fsdp_modules': resharded, 'final_restoration_proof': final_restore,
                  'frozen_evidence_limit': '64 sampled local values per frozen parameter; entire root norm and every LoRA/optimizer value compared.',
                  **memory()}
        write_json(out / f'result_rank{rank}.json', result)
        gathered = [None, None]
        dist.all_gather_object(gathered, result)
        if rank == 0:
            write_json(out / 'summary.json', {'provenance': provenance, 'passed': all(r['passed'] for r in gathered),
                       'ranks': gathered, 'training_completion_written': False, 'no_optimizer_updates': True})
        dist.barrier()
        assert all(r['passed'] for r in gathered), 'Diagnostic failed; raw stages and failed checks preserved'
        print(json.dumps({'rank': rank, 'diagnostic_passed': True, 'no_optimizer_updates': True}), flush=True)
    except Exception as exc:
        write_json(out / f'failure_rank{rank}.json', {'provenance': provenance, 'stage': stage,
                   'error': repr(exc), 'traceback': traceback.format_exc(), 'passed': False,
                   'training_completion_written': False, 'no_optimizer_updates': True})
        raise
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default=str(RUN))
    parser.add_argument('--checkpoint')
    parser.add_argument('--output')
    run_diagnostic(parser.parse_args())


if __name__ == '__main__':
    main()
