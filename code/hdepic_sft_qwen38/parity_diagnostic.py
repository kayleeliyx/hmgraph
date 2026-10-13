"""Diagnose selected/full LM-head loss differences without any model updates.

Run under two-rank torchrun only after the caller's authorized idle-GPU gate.
The training/model/checkpoint implementation is imported unchanged. This module
does not alter their identity or relax their existing parity assertion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from .common import RUN, MODEL, GPU_UUIDS, configure, read_json, write_json, sha


def tensor_sha(tensor):
    import torch
    value = tensor.detach().cpu().contiguous()
    descriptor = str((tuple(value.shape), value.dtype)).encode() + b'\0'
    return hashlib.sha256(descriptor + value.view(torch.uint8).numpy().tobytes()).hexdigest()


def differences(first, second):
    import torch
    assert first.shape == second.shape
    delta = (first.float() - second.float()).abs()
    return {'exact_equal': torch.equal(first, second), 'shape': list(first.shape),
            'different_values': int((first != second).sum()), 'values': first.numel(),
            'max_absolute_difference': delta.max().item() if delta.numel() else 0.0,
            'mean_absolute_difference': delta.mean().item() if delta.numel() else 0.0,
            'first_sha256': tensor_sha(first), 'second_sha256': tensor_sha(second)}


def label_alignment(labels):
    import torch
    from torch.nn import functional as F
    assert labels.ndim == 2 and labels.shape[0] == 1
    positions = torch.nonzero(labels[0] >= 0, as_tuple=False).flatten()
    assert positions.numel() and int(positions[0]) > 0
    # Literal installed ForCausalLMLoss shifting operation, independently built.
    native_shifted = F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
    native_positions = torch.nonzero(native_shifted[0] >= 0, as_tuple=False).flatten()
    assert torch.equal(native_positions, positions - 1)
    assert torch.equal(native_shifted[0, native_positions], labels[0, positions])
    return positions, {
        'passed': True, 'selected_predecessor_positions': (positions - 1).cpu().tolist(),
        'native_nonignored_positions': native_positions.cpu().tolist(),
        'target_token_ids': labels[0, positions].cpu().tolist(),
        'supervised_tokens': positions.numel(), 'labels_sha256': tensor_sha(labels),
        'native_shifted_labels_sha256': tensor_sha(native_shifted),
        'mask_definition': 'labels>=0, including every assistant target token and im_end',
    }


def analyze(captures, targets, native_loss, alignment):
    """Pure-CPU semantic-free numerical comparison of saved forward evidence."""
    import torch
    from torch.nn import functional as F
    selected, full, repeat = (captures[name] for name in ('selected', 'full', 'repeat'))
    hidden_comparison = differences(selected['selected_hidden'], full['selected_hidden'])
    repeat_hidden = differences(selected['selected_hidden'], repeat['selected_hidden'])
    selected_full = differences(selected['selected_logits'], full['selected_logits'])
    selected_repeat = differences(selected['selected_logits'], repeat['selected_logits'])
    short_reference = differences(selected['selected_logits'], full['short_bf16_logits'])
    short_from_full = differences(full['selected_logits'], full['short_bf16_logits'])
    logits = {
        'selected': selected['selected_logits'], 'full_selected_positions': full['selected_logits'],
        'selected_repeat': repeat['selected_logits'], 'short_bf16_same_full_hidden': full['short_bf16_logits'],
        'fp32_same_full_hidden': full['fp32_logits'],
    }
    token_nll = {name: F.cross_entropy(value[0].float(), targets, reduction='none')
                 for name, value in logits.items()}
    losses = {name: value.mean().item() for name, value in token_nll.items()}
    native_masked_difference = abs(native_loss - full['masked_ce_gpu'])
    exact_inputs = hidden_comparison['exact_equal'] and repeat_hidden['exact_equal']
    same_weights = len({captures[k]['weight_sample_sha256'] for k in captures}) == 1
    native_masked_agrees = native_masked_difference <= 2e-5
    if (exact_inputs and same_weights and alignment['passed'] and native_masked_agrees
            and selected_repeat['exact_equal'] and short_reference['exact_equal']
            and not selected_full['exact_equal']):
        diagnosis = 'Same hidden states/weights and exact label mapping; BF16 LM-head projection differs with full versus selected sequence-row count.'
        classification = 'bf16_head_projection_shape_dependence_supported'
    elif not exact_inputs:
        diagnosis = 'Hidden states changed before the head; a head-only numerical explanation is not established.'
        classification = 'upstream_hidden_difference_requires_investigation'
    elif not native_masked_agrees:
        diagnosis = 'Masked CE from native full logits disagrees with native loss; label/reduction investigation remains necessary.'
        classification = 'native_loss_reduction_disagreement'
    elif not selected_repeat['exact_equal']:
        diagnosis = 'Repeated selected head logits changed; determinism/state investigation remains necessary.'
        classification = 'selected_forward_not_repeatable'
    elif selected_full['exact_equal']:
        diagnosis = 'Selected/full logits agree in this execution; the earlier discrepancy was not reproduced.'
        classification = 'earlier_discrepancy_not_reproduced'
    else:
        diagnosis = 'Evidence does not isolate head shape dependence; retain the failed parity gate pending further investigation.'
        classification = 'unresolved'
    return {
        'classification': classification, 'diagnosis': diagnosis,
        'label_alignment': alignment, 'same_weight_samples': same_weights,
        'selected_vs_full_hidden': hidden_comparison, 'selected_vs_repeat_hidden': repeat_hidden,
        'selected_vs_full_logits': selected_full, 'selected_vs_repeat_logits': selected_repeat,
        'selected_vs_short_bf16_on_full_hidden': short_reference,
        'full_vs_short_bf16_on_same_full_hidden': short_from_full,
        'selected_vs_fp32_reference': differences(selected['selected_logits'], full['fp32_logits']),
        'full_vs_fp32_reference': differences(full['selected_logits'], full['fp32_logits']),
        'mean_token_nll': losses, 'native_full_loss': native_loss,
        'masked_ce_from_native_full_logits_gpu': full['masked_ce_gpu'],
        'native_full_vs_same_logits_masked_ce_absolute_difference': native_masked_difference,
        'native_full_vs_same_logits_masked_ce_agrees_2e_minus5': native_masked_agrees,
        'token_evidence': [
            {'position': alignment['selected_predecessor_positions'][i], 'target_id': int(targets[i]),
             'nll': {name: value[i].item() for name, value in token_nll.items()},
             'target_logit': {name: value[0, i, targets[i]].float().item() for name, value in logits.items()},
             'argmax': {name: int(value[0, i].argmax()) for name, value in logits.items()}}
            for i in range(targets.numel())],
        'original_parity_tolerance_changed': False,
        'semantic_or_quality_evaluation': False,
    }


def inspect_forwards(model, batch, vocab_chunk=4096):
    import torch
    from torch.nn import functional as F
    from .model import training_loss
    positions, alignment = label_alignment(batch['labels'])
    predecessors = positions - 1
    targets = batch['labels'][0, positions]
    inputs = {key: value for key, value in batch.items() if key != 'labels'}
    captures = {}
    phase = {'name': None}
    head = model.base_model.model.lm_head

    def before(module, args):
        name = phase['name']
        hidden = args[0]
        selected_hidden = hidden[:, predecessors, :] if name == 'full' else hidden
        assert selected_hidden.shape[1] == positions.numel()
        # FSDP's prepend=True pre-hook has already made this frozen BF16 head
        # available. Nothing is modified; only small diagnostic projections run.
        weight = module.weight
        assert not hasattr(weight, 'to_local'), 'Head should be unsharded inside its forward pre-hook'
        captures[name] = {
            'selected_hidden': selected_hidden.detach().cpu().clone(),
            'head_input_shape': list(hidden.shape), 'head_input_stride': list(hidden.stride()),
            'weight_shape': list(weight.shape), 'weight_dtype': str(weight.dtype),
            'weight_sample_sha256': tensor_sha(torch.cat([weight[:2].flatten(), weight[-2:].flatten()])),
        }
        if name == 'full':
            captures[name]['short_bf16_logits'] = F.linear(selected_hidden, weight).detach().cpu()
            previous_tf32 = torch.backends.cuda.matmul.allow_tf32
            previous_autocast = torch.is_autocast_enabled('cuda')
            pieces = []
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
                with torch.autocast('cuda', enabled=False):
                    fp32_hidden = selected_hidden.float()
                    for start in range(0, weight.shape[0], vocab_chunk):
                        # Limit scratch to a vocabulary slice; never allocate a
                        # full FP32 head or full-prompt FP32 logits for this probe.
                        chunk = F.linear(fp32_hidden, weight[start:start + vocab_chunk].float())
                        pieces.append(chunk.detach().cpu())
                        del chunk
            finally:
                torch.backends.cuda.matmul.allow_tf32 = previous_tf32
            captures[name]['fp32_logits'] = torch.cat(pieces, dim=-1)
            captures[name]['fp32_reference_policy'] = {
                'tf32': False, 'global_autocast': False, 'vocab_chunk': vocab_chunk,
                'prior_tf32_restored': torch.backends.cuda.matmul.allow_tf32 == previous_tf32,
                'prior_autocast': previous_autocast,
            }

    def after(_module, _args, output):
        name = phase['name']
        chosen = output[:, predecessors, :] if name == 'full' else output
        captures[name]['selected_logits'] = chosen.detach().cpu().clone()
        captures[name]['masked_ce_gpu'] = F.cross_entropy(chosen[0].float(), targets).item()

    hooks = [head.register_forward_pre_hook(before), head.register_forward_hook(after)]
    model.eval()
    try:
        with torch.no_grad():
            phase['name'] = 'selected'
            selected_loss = training_loss(model, batch).item()
            phase['name'] = 'full'
            output = model(**inputs, labels=batch['labels'], logits_to_keep=0,
                           use_cache=False, return_dict=True)
            native_loss = output.loss.item()
            del output
            phase['name'] = 'repeat'
            repeat_loss = training_loss(model, batch).item()
    finally:
        for hook in hooks:
            hook.remove()
    report = analyze(captures, targets.cpu(), native_loss, alignment)
    report.update({'selected_training_loss': selected_loss, 'repeat_selected_training_loss': repeat_loss,
                   'head_calls': {name: {key: value for key, value in cap.items()
                                        if key not in ('selected_hidden', 'selected_logits', 'short_bf16_logits', 'fp32_logits')}
                                  for name, cap in captures.items()}})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--attempt', default='attempt-002')
    parser.add_argument('--record-index', type=int, default=0)
    parser.add_argument('--vocab-chunk', type=int, default=4096)
    args = parser.parse_args()
    assert Path(args.attempt).name == args.attempt and 0 <= args.record_index < 4 and args.vocab_chunk > 0
    configure()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ','.join(GPU_UUIDS)
    import torch, torch.distributed as dist
    from transformers import get_cosine_schedule_with_warmup
    from .train import identity, memory, local_tensor
    from .model import load_model
    from .checkpoints import load_checkpoint
    from .data import load_partition, build_training_inputs
    from code.hdepic_qwen38_baseline.processor import load_processor
    rank = int(os.environ['RANK'])
    assert rank == int(os.environ['LOCAL_RANK']) and os.environ['WORLD_SIZE'] == '2'
    torch.cuda.set_device(rank); torch.set_num_threads(2)
    dist.init_process_group('nccl', device_id=torch.device('cuda', rank))
    folder = args.run / 'smoke' / args.attempt
    destination = folder / 'parity_diagnostic'
    ident = identity(args.run)
    completed = read_json(folder / 'completed.json')
    assert completed['identity'] == ident and completed['optimizer_steps'] == 2
    model, audit = load_model(rank, args.run, shard=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=1e-4, betas=(.9, .95), eps=1e-8, weight_decay=.1, foreach=False)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 2)
    checkpoint = Path(completed['checkpoint']).resolve()
    state = load_checkpoint(model, optimizer, scheduler, checkpoint, ident)
    assert state['step'] == 2 and state['microbatch_consumed'] == 16
    # Record exact local adapter hashes before/after; no backward/step is called.
    before = {n: tensor_sha(local_tensor(p)) for n, p in model.named_parameters() if p.requires_grad}
    assert len(before) == 992
    del optimizer, scheduler
    processor = load_processor(MODEL)
    exports, scoring, _ = load_partition('initial_sft', args.run)
    selected = read_json(args.run / 'inputs/smoke_selection.json')['records'][args.record_index]
    assert selected['partition'] == 'initial_sft' and scoring[selected['row_index']]['id'] == selected['id']
    batch, input_audit = build_training_inputs(processor, exports[selected['row_index']], selected['id'], selected['row_index'])
    batch = {key: value.cuda() for key, value in batch.items()}
    torch.cuda.synchronize(); started = time.perf_counter()
    result = inspect_forwards(model, batch, args.vocab_chunk)
    torch.cuda.synchronize()
    after = {n: tensor_sha(local_tensor(p)) for n, p in model.named_parameters() if p.requires_grad}
    assert before == after, 'Diagnostic changed adapter parameters'
    result.update({'identity': ident, 'rank': rank, 'id': selected['id'], 'category': selected['category'],
                   'checkpoint': str(checkpoint), 'checkpoint_completed_sha256': sha(checkpoint / 'completed.json'),
                   'diagnostic_code_sha256': sha(Path(__file__)), 'restored_state': state,
                   'input_audit': input_audit, 'diagnostic_seconds': time.perf_counter() - started,
                   'no_optimizer_updates': True, 'all_992_local_adapter_hashes_unchanged': True,
                   'global_matmul_precision': torch.get_float32_matmul_precision(),
                   'bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                   **memory()})
    write_json(destination / f'record{args.record_index}_rank{rank}.json', result)
    dist.barrier()
    if rank == 0:
        combined = {'diagnostic_code_sha256': sha(Path(__file__)), 'identity': ident,
                    'ranks': [read_json(destination / f'record{args.record_index}_rank{r}.json') for r in (0, 1)],
                    'training_settings_changed': False, 'parity_tolerance_changed': False}
        write_json(destination / f'record{args.record_index}.json', combined)
    dist.barrier(); dist.destroy_process_group()
    print(json.dumps({'rank': rank, 'classification': result['classification'],
                      'selected_loss': result['selected_training_loss'], 'native_loss': result['native_full_loss']}), flush=True)


if __name__ == '__main__':
    main()
