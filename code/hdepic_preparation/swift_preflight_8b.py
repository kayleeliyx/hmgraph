"""CPU-only audit of the actual ms-swift Qwen3-VL-8B SFT data pipeline.

This intentionally does not load checkpoint tensors or perform a forward pass.
Swift's collator uses a meta-device architecture to calculate multimodal RoPE
positions; the real processor, dataset loader, lazy encoder, and collator run.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import time
from unittest.mock import patch


def configure_cpu_environment(project: Path, image_max_tokens: int, image_min_tokens: int) -> None:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['USE_HF'] = '1'
    os.environ['HF_HOME'] = str(project / '.cache/huggingface')
    os.environ['HF_DATASETS_CACHE'] = str(project / '.cache/hdepic_swift_dataset')
    os.environ['MODELSCOPE_CACHE'] = str(project / '.cache/modelscope')
    os.environ['SWIFT_CACHE_DIR'] = str(project / '.cache/swift')
    os.environ['IMAGE_MAX_TOKEN_NUM'] = str(image_max_tokens)
    os.environ['IMAGE_MIN_TOKEN_NUM'] = str(image_min_tokens)
    os.environ['OMP_NUM_THREADS'] = '4'


def read_jsonl(path: Path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def audit_batch(template, batch, row, scoring):
    """Check complete assistant labels and each individual image after collation."""
    import torch
    from .swift_smoke_audit import audit_collated_record

    shared_audit = audit_collated_record(batch, row, scoring, template)
    tokenizer = template.tokenizer
    input_ids = batch['input_ids'][0].tolist()
    labels = batch['labels'][0].tolist()
    answer = row['messages'][-1]['content']
    assert row['messages'][-1]['role'] == 'assistant'
    expected_answer = tokenizer.encode(answer, add_special_tokens=False)
    suffix = template.template_meta.suffix
    expected_suffix = []
    for part in suffix:
        expected_suffix += tokenizer.encode(part, add_special_tokens=False) if isinstance(part, str) else part
    supervised = [index for index, label in enumerate(labels) if label != -100]
    assert supervised, f"{scoring['id']}: no answer labels"
    start, stop = supervised[0], supervised[-1] + 1
    expected_labels = expected_answer + expected_suffix
    assert supervised == list(range(start, stop)), f"{scoring['id']}: discontinuous answer supervision"
    assert labels[start:stop] == expected_labels, f"{scoring['id']}: answer or suffix tokens changed/truncated"
    assert input_ids[start:stop] == expected_labels
    assert all(value == -100 for value in labels[:start] + labels[stop:])
    assert tokenizer.decode(input_ids[start:start + len(expected_answer)], skip_special_tokens=False) == answer

    grid = batch['image_grid_thw']
    assert tuple(grid.shape) == (16, 3), f"{scoring['id']}: image grids {tuple(grid.shape)}"
    patch_counts = grid.prod(dim=-1)
    merge_size = template.processor.image_processor.merge_size
    image_token = tokenizer.convert_tokens_to_ids('<|image_pad|>')
    start_token = tokenizer.convert_tokens_to_ids('<|vision_start|>')
    end_token = tokenizer.convert_tokens_to_ids('<|vision_end|>')
    image_counts = (patch_counts // merge_size ** 2).tolist()
    assert input_ids.count(image_token) == sum(image_counts)
    assert input_ids.count(start_token) == input_ids.count(end_token) == 16
    image_positions = [index for index, token in enumerate(input_ids) if token == image_token]
    assert all(labels[index] == -100 for index in image_positions)
    assert int(batch['pixel_values'].shape[0]) == int(patch_counts.sum())
    assert len(row['images']) == len(scoring['frames']) == 16
    assert len(set(row['images'])) == 16
    assert row['images'] == [frame['path'] for frame in scoring['frames']]
    assert all(Path(path).is_file() for path in row['images'])
    assert all(not isinstance(value, torch.Tensor) or value.device.type == 'cpu'
               for value in batch.values())
    assert 'position_ids' in batch
    goal_available = scoring['goal_status'] == 'available'
    assert ('<goal>' in answer) == goal_available
    if scoring['task'] == 'explain':
        assert '<why>' in answer and '<action>' not in answer
    else:
        assert '<action>' in answer and '<why>' not in answer
    assert '<next>' in answer
    return {
        **shared_audit,
        'id': scoring['id'], 'partition': scoring['partition'], 'task': scoring['task'],
        'goal_status': scoring['goal_status'], 'sequence_tokens': len(input_ids),
        'assistant_answer_tokens': len(expected_answer), 'assistant_suffix_tokens': len(expected_suffix),
        'supervised_tokens': len(supervised), 'masked_prompt_tokens': start,
        'image_count': 16, 'image_grid_thw': grid.tolist(),
        'image_tokens_per_image': image_counts,
        'pixel_values_shape': list(batch['pixel_values'].shape),
        'position_ids_shape': list(batch['position_ids'].shape),
        'complete_target': True, 'assistant_only_loss_mask': True,
        'all_images_preserved': True, 'cpu_tensors_only': True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path('/data/yaxuanli/ActGround'))
    parser.add_argument('--dataset', type=Path,
                        default=Path('/data/yaxuanli/ActGround/datasets/hdepic_sft_v2_repaired/sample/training20.jsonl'))
    parser.add_argument('--scoring', type=Path,
                        default=Path('/data/yaxuanli/ActGround/datasets/hdepic_sft_v2_repaired/sample/training20.scoring.jsonl'))
    parser.add_argument('--model', type=Path, default=Path('/data/yaxuanli/ActGround/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'))
    parser.add_argument('--output', type=Path,
                        default=Path('/data/yaxuanli/ActGround/runs/hdepic_sft_smoke_8b/cpu_preflight'))
    parser.add_argument('--max-length', type=int, default=8192)
    parser.add_argument('--image-max-token-num', type=int, default=144)
    parser.add_argument('--image-min-token-num', type=int, default=64)
    args = parser.parse_args(argv)
    assert 0 < args.image_min_token_num <= args.image_max_token_num
    configure_cpu_environment(args.project, args.image_max_token_num, args.image_min_token_num)
    args.output.mkdir(parents=True, exist_ok=True)

    import torch
    import safetensors
    from swift.llm import TrainArguments
    from swift.llm.dataset import LazyLLMDataset
    from swift.llm.train import SwiftSft

    torch.set_num_threads(4)
    rows = read_jsonl(args.dataset)
    scoring_rows = read_jsonl(args.scoring)
    assert len(rows) == len(scoring_rows) == 20
    expected_categories = {(task, available) for task in ['explain', 'track'] for available in [True, False]}
    assert {(row['task'], row['goal_status'] == 'available') for row in scoring_rows} == expected_categories

    class ProcessorOnlySft(SwiftSft):
        def _prepare_model_tokenizer(self, **kwargs):
            super()._prepare_model_tokenizer(load_model=False, **kwargs)
            assert self.model is None, 'Preflight must not load model weights'

    train_args = TrainArguments(
        model=str(args.model), model_type='qwen3_vl', torch_dtype='bfloat16',
        dataset=[str(args.dataset)], split_dataset_ratio=0.0,
        dataset_shuffle=False, lazy_tokenize=True, strict=True,
        max_length=args.max_length, truncation_strategy='delete', loss_scale='default',
        max_pixels=args.image_max_token_num * 32 ** 2,
        train_type='lora', lora_rank=16, lora_alpha=32,
        freeze_vit=True, freeze_aligner=True,
        per_device_train_batch_size=1, gradient_accumulation_steps=1,
        gradient_checkpointing=True, packing=False, padding_free=False,
        dataset_num_proc=1, dataloader_num_workers=0,
        output_dir=str(args.output / 'unused_training_output'),
        report_to=[], use_cpu=True, bf16=False, fp16=False,
        use_hf=True, attn_impl='sdpa', add_version=False,
    )

    def refuse_weights(*unused_args, **unused_kwargs):
        raise RuntimeError('Checkpoint tensor loading is forbidden in this CPU-only preprocessing audit')

    started = time.perf_counter()
    with ExitStack() as stack:
        stack.enter_context(patch.object(torch, 'load', side_effect=refuse_weights))
        stack.enter_context(patch.object(safetensors, 'safe_open', side_effect=refuse_weights))
        import safetensors.torch
        import transformers.modeling_utils
        stack.enter_context(patch.object(safetensors.torch, 'load_file', side_effect=refuse_weights))
        stack.enter_context(patch.object(safetensors.torch, 'safe_open', side_effect=refuse_weights))
        stack.enter_context(patch.object(transformers.modeling_utils, 'safe_open', side_effect=refuse_weights))
        pipeline = ProcessorOnlySft(train_args)
        train_dataset, val_dataset = pipeline._prepare_dataset()
        assert val_dataset is None
        assert isinstance(train_dataset, LazyLLMDataset)
        assert len(train_dataset) == len(rows)
        collator = pipeline._get_data_collator()
        results = []
        for index, (row, scoring) in enumerate(zip(rows, scoring_rows)):
            batch = collator([train_dataset[index]])
            result = audit_batch(pipeline.template, batch, row, scoring)
            result['row_index'] = index
            results.append(result)
            print(json.dumps({'row_index': index, 'id': result['id'],
                              'tokens': result['sequence_tokens'], 'status': 'passed'}), flush=True)
        dummy = pipeline.template.dummy_model
        assert dummy is not None, 'Expected Swift meta dummy for multimodal RoPE positions'
        assert all(parameter.device.type == 'meta' for parameter in dummy.parameters())
        assert not torch.cuda.is_initialized()

    versions = {name: importlib.metadata.version(name) for name in
                ['ms-swift', 'transformers', 'torch', 'torchvision', 'peft', 'accelerate', 'qwen-vl-utils']}
    report = {
        'status': 'passed', 'model': str(args.model), 'checkpoint_revision': args.model.name,
        'model_config_sha256': sha256(args.model / 'config.json'), 'versions': versions,
        'dataset': str(args.dataset), 'dataset_sha256': sha256(args.dataset),
        'scoring': str(args.scoring), 'scoring_sha256': sha256(args.scoring),
        'records': len(results), 'images': sum(row['image_count'] for row in results),
        'category_counts': dict(Counter(f"{row['task']}:{row['goal_status']}" for row in results)),
        'partition_counts': dict(Counter(row['partition'] for row in results)),
        'max_length': args.max_length, 'image_max_token_num': args.image_max_token_num,
        'image_min_token_num': args.image_min_token_num,
        'image_max_pixels': args.image_max_token_num * 32 ** 2,
        'image_min_pixels': args.image_min_token_num * 32 ** 2,
        'image_environment': {'IMAGE_MAX_TOKEN_NUM': os.environ['IMAGE_MAX_TOKEN_NUM'],
                              'IMAGE_MIN_TOKEN_NUM': os.environ['IMAGE_MIN_TOKEN_NUM']},
        'processor_global_vars': pipeline.processor.global_vars,
        'sequence_token_range': [min(row['sequence_tokens'] for row in results),
                                 max(row['sequence_tokens'] for row in results)],
        'loss_scale': 'default', 'target_loss': 'assistant answer and stop suffix only',
        'full_prompt_and_target_reconstruction': True,
        'truncation_strategy': pipeline.template.truncation_strategy,
        'no_target_truncation': True, 'all_16_individual_images_retained': True,
        'weights_loaded': False, 'cuda_initialized': False,
        'position_model': {'class': type(dummy).__name__, 'device': 'meta',
                           'purpose': 'architecture/config-only multimodal RoPE index calculation'},
        'pipeline': {'class': 'SwiftSft', 'model_processor_load_model': False,
                     'dataset_class': type(train_dataset).__name__,
                     'template_class': type(pipeline.template).__name__,
                     'template_mode': pipeline.template.mode,
                     'collator': 'SwiftSft._get_data_collator -> template.data_collator'},
        'source_sha256': {
            str(path): sha256(path) for path in [
                Path(__file__).resolve(),
                Path(__file__).with_name('swift_smoke_audit.py').resolve(),
                Path(inspect.getfile(SwiftSft)),
                Path(inspect.getfile(type(pipeline.template))),
            ]
        },
        'rerun_command': 'PYTHONPATH=/data/yaxuanli/ActGround PYTHONDONTWRITEBYTECODE=1 '
                         '/data/yaxuanli/ActGround/.venvs/hdepic-swift-8b/bin/python '
                         '-m code.hdepic_preparation.swift_preflight_8b',
        'elapsed_sec': time.perf_counter() - started, 'rows': results,
    }
    (args.output / 'preprocessing_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'rows'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
