"""Native Qwen3.8 processor and strict, prompt-only CPU preflight.

This module never imports a model class, loads weights, reads scoring files, or
runs generation. Model/tokenizer IDs are specific to the pinned Qwen3.8 model.
The frozen system/user text is retained exactly, with each literal <image>
replaced in place by one local-image content item for the native chat template.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import time

from .data import OUT, canonical_bytes, immutable_write, json_bytes, jsonl_bytes, prompt_digest, require, rows, sha256

MIN_PIXELS = 65536
MAX_PIXELS = 147456
MAX_LENGTH = 8192
GENERATION_BUDGET = 256
FRAME_COUNT = 16
IMAGE_ID = 248056
VISION_START_ID = 248053
VISION_END_ID = 248054
VIDEO_ID = 248057
EOS_IDS = [248046, 248044]
PAD_ID = 248044
NONTHINKING_PREFIX = '<|im_start|>assistant\n<think>\n\n</think>\n\n'
VISION_PATTERN = re.compile(r'<\|vision_start\|>(?:<\|image_pad\|>)+<\|vision_end\|>')
VERSION = 'hdepic_qwen38_native_processor_v1'
RECORD_KEYS = {'id', 'row_index', 'messages', 'images', 'prompt_sha256'}


def build_messages(record):
    """Return only native system/user content, preserving text and image order."""
    require(set(record) == RECORD_KEYS, 'Unapproved answering-input fields')
    require(isinstance(record['id'], str) and isinstance(record['row_index'], int), 'Invalid input identity')
    messages = record['messages']
    require([m.get('role') for m in messages] == ['system', 'user'], 'Only system and user input roles are permitted')
    require(all(set(m) == {'role', 'content'} and isinstance(m['content'], str) for m in messages),
            'Unexpected message fields or content')
    images = record['images']
    require(isinstance(images, list) and len(images) == FRAME_COUNT and all(isinstance(p, str) for p in images),
            'Expected 16 separate local-image paths')
    require('<image>' not in messages[0]['content'], 'System message cannot contain images')
    require(record['prompt_sha256'] == prompt_digest(messages, images), 'Frozen prompt fingerprint changed')
    require(all(Path(p).is_absolute() and Path(p).is_file() for p in images), 'Missing/nonabsolute cached image path')
    pieces = messages[1]['content'].split('<image>')
    require(len(pieces) == FRAME_COUNT + 1, 'Expected exactly 16 in-place image markers')
    content = []
    for index, piece in enumerate(pieces):
        if piece:
            content.append({'type': 'text', 'text': piece})
        if index < FRAME_COUNT:
            content.append({'type': 'image', 'image': images[index]})
    reconstructed = ''.join(item['text'] if item['type'] == 'text' else '<image>' for item in content)
    require(reconstructed == messages[1]['content'], 'Content conversion changed original user text')
    require([item['image'] for item in content if item['type'] == 'image'] == images, 'Content conversion changed image order')
    # Transformers 5.3's multimodal loading path iterates content items in
    # every role, including system. A single text item preserves the exact
    # string while avoiding that path treating a plain string as items.
    return [{'role': 'system', 'content': [{'type': 'text', 'text': messages[0]['content']}]},
            {'role': 'user', 'content': content}]


def load_processor(modeldir):
    """Load only the local native configuration/processor, with image limits."""
    from transformers import AutoConfig, AutoProcessor
    modeldir = Path(modeldir).resolve()
    require(modeldir.is_dir(), 'Local model metadata directory is missing')
    config = AutoConfig.from_pretrained(modeldir, local_files_only=True, trust_remote_code=False)
    require(config.model_type == 'qwen3_5' and config.architectures == ['Qwen3_5ForConditionalGeneration'],
            'Processor metadata is not the official Qwen3.8-27B architecture')
    require(config.image_token_id == IMAGE_ID and config.vision_start_token_id == VISION_START_ID
            and config.vision_end_token_id == VISION_END_ID and config.video_token_id == VIDEO_ID,
            'Unexpected Qwen3.8 multimodal token IDs')
    processor = AutoProcessor.from_pretrained(modeldir, local_files_only=True, trust_remote_code=False,
                                             use_fast=True, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    require(type(processor).__name__ == 'Qwen3VLProcessor', 'Unexpected native processor class')
    require(type(processor.image_processor).__name__ == 'Qwen2VLImageProcessorFast', 'Unexpected image processor class')
    image = processor.image_processor
    require(image.size == {'shortest_edge': MIN_PIXELS, 'longest_edge': MAX_PIXELS}, 'Image limits did not resolve exactly')
    require(image.patch_size == 16 and image.temporal_patch_size == 2 and image.merge_size == 2,
            'Unexpected visual patch geometry')
    require(image.image_mean == [0.5] * 3 and image.image_std == [0.5] * 3, 'Qwen3.8 image normalization was changed')
    tokenizer = processor.tokenizer
    for token, expected in (('<|image_pad|>', IMAGE_ID), ('<|vision_start|>', VISION_START_ID),
                            ('<|vision_end|>', VISION_END_ID), ('<|video_pad|>', VIDEO_ID),
                            ('<|im_end|>', EOS_IDS[0]), ('<|endoftext|>', PAD_ID)):
        require(tokenizer.convert_tokens_to_ids(token) == expected, f'Unexpected tokenizer ID: {token}')
    require(tokenizer.eos_token_id == EOS_IDS[0] and tokenizer.pad_token_id == PAD_ID, 'Unexpected tokenizer EOS/PAD')
    generation = json.loads((modeldir / 'generation_config.json').read_text())
    require(generation['eos_token_id'] == EOS_IDS and generation['pad_token_id'] == PAD_ID,
            'Original Qwen3.8 generation stopping configuration changed')
    return processor


def expected_serialization(record):
    return ''.join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in record['messages']) + NONTHINKING_PREFIX


def tensor_sha256(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def preprocess(processor, record):
    """Return actual native processor tensors and exact-input audit, on CPU."""
    import torch
    messages = build_messages(record)
    template_kwargs = {'add_generation_prompt': True, 'enable_thinking': False, 'add_vision_id': False}
    native = processor.apply_chat_template(messages, tokenize=False, **template_kwargs)
    expected = expected_serialization(record)
    require(VISION_PATTERN.sub('<image>', native) == expected,
            'Native template changed frozen prompt text (including edge whitespace)')
    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors='pt',
        padding=False, truncation=False,
        images_kwargs={'device': 'cpu', 'size': {'shortest_edge': MIN_PIXELS, 'longest_edge': MAX_PIXELS}},
        **template_kwargs,
    )
    require(set(inputs) == {'input_ids', 'attention_mask', 'mm_token_type_ids', 'pixel_values', 'image_grid_thw'},
            f'Unexpected processor output keys: {set(inputs)}')
    require(all(isinstance(value, torch.Tensor) and value.device.type == 'cpu' for value in inputs.values()),
            'Processor inputs must remain CPU tensors')
    ids = inputs['input_ids']
    require(ids.ndim == 2 and ids.shape[0] == 1, 'Expected one complete prompt per batch')
    ids = ids[0]
    require(ids.numel() + GENERATION_BUDGET <= MAX_LENGTH, 'Complete prompt plus generation budget exceeds context')
    require(inputs['attention_mask'].shape == (1, ids.numel()) and bool(inputs['attention_mask'].all()),
            'Prompt contains padding/masked/truncated tokens')
    grid = inputs['image_grid_thw']
    require(tuple(grid.shape) == (FRAME_COUNT, 3) and bool((grid[:, 0] == 1).all()),
            'Expected 16 individual-image grids, not a video/collage')
    require(bool((grid[:, 1:] == 24).all()), 'Frozen 384-pixel images did not retain their full configured resolution')
    tokens = grid.prod(-1) // processor.image_processor.merge_size ** 2
    require(bool(((tokens >= 64) & (tokens <= 144)).all()), 'Image token budget changed')
    require(bool((tokens == 144).all()), 'A frozen 384-pixel frame lost visual tokens')
    require(tuple(inputs['pixel_values'].shape) == (int(grid.prod(-1).sum()), 1536), 'Image patch tensor shape changed')
    require(bool(torch.isfinite(inputs['pixel_values']).all()), 'Nonfinite processed image pixels')
    starts = (ids == VISION_START_ID).nonzero().reshape(-1).tolist()
    ends = (ids == VISION_END_ID).nonzero().reshape(-1).tolist()
    require(len(starts) == len(ends) == FRAME_COUNT, 'Missing/extra image blocks')
    require(int((ids == IMAGE_ID).sum()) == int(tokens.sum()) and not bool((ids == VIDEO_ID).any()),
            'Incorrect image placeholders or video tokens')
    previous_end = -1
    for start, end, count in zip(starts, ends, tokens.tolist()):
        require(previous_end < start < end and end - start - 1 == count
                and bool((ids[start + 1:end] == IMAGE_ID).all()), 'Image tokens out of order or incomplete')
        previous_end = end
    mm = inputs['mm_token_type_ids']
    require(mm.shape == inputs['input_ids'].shape and torch.equal(mm[0], (ids == IMAGE_ID).to(mm.dtype)),
            'Multimodal token-type positions differ from image positions')
    decoded = processor.tokenizer.decode(ids.tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=False)
    require(VISION_PATTERN.sub('<image>', decoded) == expected, 'Tokenization changed or truncated the frozen prompt')
    require(decoded.endswith(NONTHINKING_PREFIX), 'Native non-thinking assistant prefix is incomplete')
    pixel_parts = inputs['pixel_values'].split(grid.prod(-1).tolist())
    audit = {
        'id': record['id'], 'row_index': record['row_index'], 'prompt_sha256': record['prompt_sha256'],
        'input_tokens': ids.numel(), 'generation_budget_tokens': GENERATION_BUDGET, 'context_limit': MAX_LENGTH,
        'individual_images': FRAME_COUNT, 'image_grid_thw': grid.tolist(),
        'image_tokens_per_image': tokens.tolist(), 'image_tokens': int(tokens.sum()),
        'processed_image_size_hw': [[int(row[1]) * 16, int(row[2]) * 16] for row in grid.tolist()],
        'pixel_values_shape': list(inputs['pixel_values'].shape), 'pixel_values_dtype': str(inputs['pixel_values'].dtype),
        'pixel_values_per_image_sha256': [tensor_sha256(part) for part in pixel_parts],
        'image_paths_sha256': hashlib.sha256(canonical_bytes(record['images'])).hexdigest(),
        'native_rendered_prompt_sha256': hashlib.sha256(native.encode()).hexdigest(),
        'decoded_expanded_prompt_sha256': hashlib.sha256(decoded.encode()).hexdigest(),
        'input_ids_sha256': tensor_sha256(inputs['input_ids']),
        'mm_token_type_ids_sha256': tensor_sha256(mm),
        'complete_prompt': True, 'original_text_exact': True, 'image_positions_preserved': True,
        'no_reference_fields_or_assistant_target': True, 'model_input_roles': ['system', 'user'],
        'enable_thinking': False, 'assistant_prefix': NONTHINKING_PREFIX,
        'native_edge_whitespace_trim_changed_input': False, 'truncation': False,
        'mm_token_type_ids_complete': True, 'all_tensors_cpu': True,
    }
    return inputs, audit


def verify_individual_image_order(processor, record, inputs):
    """Cross-check native batching against each source image separately, on CPU."""
    import torch
    from PIL import Image
    parts = inputs['pixel_values'].split(inputs['image_grid_thw'].prod(-1).tolist())
    for index, path in enumerate(record['images']):
        with Image.open(path) as image:
            require(image.size == (384, 384), 'Unexpected cached input resolution')
            single = processor.image_processor(images=[image.convert('RGB')], return_tensors='pt', device='cpu',
                                               size={'shortest_edge': MIN_PIXELS, 'longest_edge': MAX_PIXELS})
        require(torch.equal(single['image_grid_thw'][0], inputs['image_grid_thw'][index])
                and torch.equal(single['pixel_values'], parts[index]), 'Native batching changed image order/pixels')
    return {'passed': True, 'images': FRAME_COUNT, 'comparison': 'Exact torch.equal against separately processed local image in each original position'}


def run_cpu_preflight(modeldir, run=OUT):
    """Audit all frozen prompt-only rows; save outputs outside original artifacts."""
    run = Path(run).resolve(); modeldir = Path(modeldir).resolve()
    os.environ.update({'CUDA_VISIBLE_DEVICES': '', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                       'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
    import torch
    torch.set_num_threads(1)
    manifest_path = run / 'inputs/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    input_path = run / 'inputs/validation_explain.prompts.jsonl'
    require(manifest['partition'] == 'validation' and manifest['task'] == 'explain' and manifest['count'] == 228,
            'Preflight is restricted to the fixed 228 validation Explain inputs')
    require(sha256(input_path) == manifest['files_sha256']['inputs/validation_explain.prompts.jsonl'],
            'Prepared prompt file changed')
    records = rows(input_path)
    require(len(records) == 228 and [row['id'] for row in records] == manifest['ordered_ids'], 'Validation IDs/order changed')
    smoke = json.loads((run / 'inputs/smoke_selection.json').read_text())
    selected = set(smoke['ids'])
    require(selected <= {row['id'] for row in records}, 'Processor order-check selection changed')
    processor = load_processor(modeldir)
    versions = {name: importlib.metadata.version(name) for name in
                ('transformers', 'torch', 'torchvision', 'accelerate', 'huggingface-hub', 'tokenizers', 'safetensors', 'Pillow')}
    require(versions['transformers'] == '5.3.0', 'CPU preflight uses pinned Transformers 5.3.0')
    started = time.perf_counter(); audits = []
    for index, record in enumerate(records):
        inputs, audit = preprocess(processor, record)
        if record['id'] in selected:
            audit['separate_image_order_check'] = verify_individual_image_order(processor, record, inputs)
        audits.append(audit)
        del inputs
        if (index + 1) % 25 == 0 or index + 1 == len(records):
            print(json.dumps({'cpu_preflight_records': index + 1, 'total': len(records)}), flush=True)
    metadata_files = sorted(path for path in modeldir.iterdir() if path.is_file()
                            and path.suffix in ('.json', '.txt', '.jinja') and not path.name.endswith('.safetensors.index.json'))
    summary = {
        'version': VERSION, 'passed': True, 'partition': 'validation', 'task': 'explain', 'records': len(audits),
        'individual_frame_inputs': len(audits) * FRAME_COUNT, 'all_image_counts': dict(Counter(a['individual_images'] for a in audits)),
        'input_tokens_min': min(a['input_tokens'] for a in audits), 'input_tokens_max': max(a['input_tokens'] for a in audits),
        'all_visual_tokens_per_image': dict(Counter(t for a in audits for t in a['image_tokens_per_image'])),
        'all_16_frames_complete': True, 'all_original_text_exact': True, 'all_mm_token_type_ids_complete': True,
        'all_input_roles_system_user_only': True, 'all_targets_references_absent': True, 'truncation': False,
        'native_template_edge_whitespace_policy': 'Require exact frozen text. Native Jinja trim is permitted only when it changes no input bytes; stop on a mismatch.',
        'nonthinking_assistant_prefix': NONTHINKING_PREFIX, 'enable_thinking': False,
        'individual_order_checked_records': len(selected), 'individual_order_checked_images': len(selected) * FRAME_COUNT,
        'individual_order_check_ids': sorted(selected),
        'processor_class': type(processor).__name__, 'image_processor_class': type(processor.image_processor).__name__,
        'image_processor_config': processor.image_processor.to_dict(), 'max_length': MAX_LENGTH,
        'original_eos_token_ids': EOS_IDS, 'original_pad_token_id': PAD_ID, 'versions': versions,
        'prompt_inputs_sha256': sha256(input_path), 'input_manifest_sha256': sha256(manifest_path),
        'processor_source_sha256': sha256(Path(__file__)),
        'model_metadata_path': str(modeldir), 'model_metadata_sha256': {path.name: sha256(path) for path in metadata_files},
        'elapsed_sec': time.perf_counter() - started, 'model_weights_loaded': False, 'gpu_accessed': False,
        'inference_run': False, 'training_run': False, 'scoring_files_read': False, 'test_accessed': False,
    }
    output = run / 'processor_preflight'
    immutable_write(output / 'records.jsonl', jsonl_bytes(audits))
    # Runtime is intentionally not part of the deterministic cache identity.
    deterministic = dict(summary); deterministic.pop('elapsed_sec')
    immutable_write(output / 'audit.json', json_bytes(deterministic))
    print(json.dumps({'passed': True, 'records': len(audits), 'input_tokens_min': summary['input_tokens_min'],
                      'input_tokens_max': summary['input_tokens_max'], 'elapsed_sec': summary['elapsed_sec'],
                      'output': str(output)}), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True, help='Local pinned model metadata; weights are unnecessary')
    parser.add_argument('--run', type=Path, default=OUT)
    args = parser.parse_args(argv)
    run_cpu_preflight(args.model, args.run)


if __name__ == '__main__':
    main()
