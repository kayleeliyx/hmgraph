"""Package completed baseline evidence and blinded review, without private keys.

No weights, frame cache, original training artifacts, inference or GPU access.
Incomplete generation/review/report stages fail before any archive is written.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

from .data import OUT, immutable_write, json_bytes, require, rows, sha256, unique_by_id
from .report import load_active_configuration
from .review import HUMAN_FIELDS, completed_generation

VERSION = 'hdepic_qwen35_review_package_v1'
MODEL_METADATA = ('config.json', 'generation_config.json', 'preprocessor_config.json',
                  'tokenizer_config.json', 'chat_template.jinja', 'model.safetensors.index.json',
                  'README.md', 'official_model_api.json')
ROOT_FILES = ('README.md', 'frozen_config.json', 'frozen_config_v2.json',
              'active_configuration.json', 'frozen_parser.json', 'exact_commands.json',
              'model_identity.json', 'metrics.json', 'execution_report.md')
RUN_FOLDERS = ('inputs', 'scoring', 'audit', 'environment', 'processor_preflight',
               'resources', 'smoke', 'failures', 'attempts')
SUFFIXES = {'.json', '.jsonl', '.csv', '.txt', '.log', '.md', '.html', '.py', '.sh', '.jinja'}
WEIGHT_SUFFIXES = {'.safetensors', '.bin', '.pt', '.pth', '.ckpt', '.gguf', '.onnx', '.h5', '.pkl', '.npz', '.npy'}
REVIEW_INSTRUCTIONS = '''# Instructions for the blinded four-answer review

Use this blinded archive alone for reviewing. The separate full provenance
archive contains a named model's raw generation and can reveal answer identity
through text matching. Do not consult that archive, model predictions, previous
assistant ratings, or private identity keys before freezing your judgments.
Answer style and previous context can still limit blinding.

Open review/blinded/index.html. Every case has all16 ordered causal cached frames,
the exact action question, completed history, original narration, frozen reason
span/offsets and four saved answers in randomized A/B/C/D order. The model received
16 individual images. Contact sheets are review aids; their printed source-case
numbers differ from the newly randomized Q-case order and are explicitly labeled.

Read rubric.md and review_addendum.md before judging. Assess source attachment
separately. Judge specific stated purpose, accepting paraphrases and direct
entailments rather than keyword overlap. Mark source ambiguity/conflict
unjudgeable for all four answers. Unsupported additional claims and visual
support are separate decisions; visible action alone does not establish motive
or a future outcome. Do not inspect future frames or substitute proposed reasons.
Malformed NEXT is not a gate for uniquely extractable WHY. Missing/unextractable
WHY and length-limited answers remain in the228-case denominator.

Human fields are in review/blinded/human_review.csv, with one row per answer.
Preparation starts all human fields blank and preserves entered fields on rerun.
There are228 cases/912 answer slots and the original fixed50 flags are retained;
the subset is not an independent replicate. No new correctness judgment is
created by the archive builder. Human/assistant semantic review remains pending.
The blinded_package_manifest.json provides file hashes and byte counts.
'''


class PendingPackage(RuntimeError):
    pass


def allowed_member(name):
    """Apply a second archive-wide exclusion check, independent of collection."""
    path = PurePosixPath(name)
    lower_parts = [part.lower() for part in path.parts]
    if path.is_absolute() or '..' in path.parts:
        return False
    if any(part in ('private', '.cache', '__pycache__', 'frames', 'training', 'test') for part in lower_parts):
        return False
    if 'identity_key' in str(path).lower() or 'identitykey' in str(path).lower():
        return False
    if path.suffix.lower() in WEIGHT_SUFFIXES:
        return False
    if path.name.lower().startswith('test.') and path.suffix.lower() in {'.json', '.jsonl', '.csv'}:
        return False
    if path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}:
        return 'review/blinded/contact_sheets/' in str(path)
    return path.suffix.lower() in SUFFIXES


def contains_private_mapping(value):
    if isinstance(value, dict):
        if any(key in ('private_seed', 'case_slots') for key in value):
            return True
        return any(contains_private_mapping(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_private_mapping(item) for item in value)
    return False


def verify_safe_content(name, content):
    require(allowed_member(name), f'Forbidden archive member: {name}')
    if name.endswith('.json'):
        require(not contains_private_mapping(json.loads(content)), f'Private identity/seed content: {name}')
    elif name.endswith('.jsonl'):
        require(all(not contains_private_mapping(json.loads(line)) for line in content.splitlines() if line.strip()),
                f'Private identity/seed content: {name}')


def validate_ready(run):
    run = Path(run).resolve()
    required = [run / name for name in ('inputs/manifest.json', 'inputs/validation_explain.prompts.jsonl',
                'predictions/generation.jsonl', 'predictions/completed.json', 'review/materials_manifest.json',
                'metrics.json', 'execution_report.md')]
    if any(not path.is_file() for path in required):
        raise PendingPackage('Generation, four-answer review preparation and execution report must all complete before packaging')
    manifest = json.loads((run / 'inputs/manifest.json').read_text())
    require(manifest['count'] == 228 and len(manifest['ordered_ids']) == len(set(manifest['ordered_ids'])) == 228,
            'Package cohort is not the fixed228 Explain IDs')
    prompts = rows(run / 'inputs/validation_explain.prompts.jsonl')
    require([row['id'] for row in prompts] == manifest['ordered_ids'], 'Prepared input ID order differs')
    generated = completed_generation(run / 'predictions/generation.jsonl', run / 'predictions/completed.json',
                                     run / 'inputs/manifest.json', prompts)
    config, active = load_active_configuration(run)
    require(config['repo_id'] == 'Qwen/Qwen3.5-35B-A3B', 'Package uses a different model')
    marker = json.loads((run / 'predictions/completed.json').read_text())
    require(marker['configuration_sha256'] == config['configuration_sha256'], 'Generation used a different active profile')
    require(all(row['configuration_sha256'] == config['configuration_sha256'] and row['model_revision'] == config['revision']
                for row in generated.values()), 'Generated rows differ from active model/profile')
    identity = json.loads((run / 'model_identity.json').read_text())
    require(identity['repo_id'] == config['repo_id'] and identity['revision'] == config['revision'], 'Checkpoint identity differs')
    require(sha256(run / 'model_identity.json') == config['model_identity_sha256'], 'Frozen model identity changed')
    for name in MODEL_METADATA:
        path = run / 'model_metadata' / name
        require(path.is_file(), f'Missing model metadata: {name}')
        if name in identity['metadata_sha256']:
            require(sha256(path) == identity['metadata_sha256'][name], f'Model metadata changed: {name}')
    metrics = json.loads((run / 'metrics.json').read_text())
    require(metrics['generation_completed'] is True and metrics['generation']['denominator'] == metrics['generation']['completed'] == 228,
            'Execution report does not establish complete228 generation')
    require(metrics['configuration_sha256'] == config['configuration_sha256'], 'Execution report uses a different active profile')
    require(metrics['semantic_review']['materials_prepared'] is True, 'Report predates four-answer review preparation')
    require(metrics['preservation']['passed'] is True, 'Final original-artifact preservation audit failed')
    require(metrics['no_new_training'] is True and metrics['no_test_access'] is True
            and metrics['references_or_narration_supplied_to_answering_model'] is False, 'Answering boundary audit failed')
    review = json.loads((run / 'review/materials_manifest.json').read_text())
    require(review['cases'] == 228 and review['answer_slots'] == 912 and review['fixed50'] == 50,
            'Wrong four-answer review cohort')
    require(review['generation_sha256'] == sha256(run / 'predictions/generation.jsonl')
            and review['inputs_manifest_sha256'] == sha256(run / 'inputs/manifest.json'), 'Review predates current generation/inputs')
    for name, digest in review['files_sha256'].items():
        path = (run / 'review' / name).resolve()
        require(path.is_relative_to(run / 'review') and path.is_file(), f'Missing/escaped review artifact: {name}')
        require(sha256(path) == digest, f'Prepared review artifact changed: {name}')
    cases = rows(run / 'review/blinded/cases.jsonl')
    case_map = unique_by_id(cases, 'review case')
    require(set(case_map) == set(manifest['ordered_ids']) and sum(case['existing_fixed50'] for case in cases) == 50,
            'Review IDs/fixed50 differ from original cohort')
    require(all(set(case['answers']) == set('ABCD') and len(case['frames']) == 16 for case in cases),
            'Four answer slots/16 individual frame provenance missing')
    require(len({case['contact_sheet'] for case in cases}) == 228, 'Expected228 individual review sheets')
    require(all((run / 'review/blinded' / case['contact_sheet']).is_file() for case in cases),
            'A referenced review contact sheet is missing')
    # Do not read or reconstruct model/slot identity. Check row identity and
    # blank/preserved human fields without opening the separate private key.
    expected_human = [(case['case'], case['id'], slot) for case in cases for slot in 'ABCD']
    with (run / 'review/blinded/human_review.csv').open(newline='') as handle:
        human = list(csv.DictReader(handle))
    require([(row['case'], row['id'], row['slot']) for row in human] == expected_human,
            'Human CSV row identities/order differ from blinded cases')
    require(all(all(field in row for field in HUMAN_FIELDS) for row in human), 'Missing human review columns')
    return {'manifest': manifest, 'config': config, 'active_configuration': active, 'review': review,
            'metrics': metrics, 'human_fields_all_blank': all(row[field] == '' for row in human for field in HUMAN_FIELDS)}


def collect_files(run, project):
    run, project = Path(run).resolve(), Path(project).resolve()
    result = {}

    def add(path, name, root):
        path = Path(path)
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root),
                f'Missing/linked/escaped package source: {path}')
        require(allowed_member(name), f'Forbidden package source: {name}')
        require(name not in result, f'Duplicate package member: {name}')
        result[name] = path

    for name in ROOT_FILES:
        add(run / name, name, run)
    for folder in RUN_FOLDERS:
        for path in sorted((run / folder).rglob('*')):
            if path.is_file() and path.suffix.lower() in SUFFIXES:
                add(path, str(path.relative_to(run)), run)
    for path in sorted(run.glob('*.log')):
        add(path, path.name, run)
    for name in ('generation.jsonl', 'completed.json'):
        add(run / 'predictions' / name, 'predictions/' + name, run)
    for name in MODEL_METADATA:
        add(run / 'model_metadata' / name, 'model_metadata/' + name, run)
    for name in ('rubric.md', 'review_addendum.md', 'materials_manifest.json'):
        add(run / 'review' / name, 'review/' + name, run)
    instructions = run / 'review/REVIEW_INSTRUCTIONS.md'
    if instructions.is_file():
        add(instructions, 'review/REVIEW_INSTRUCTIONS.md', run)
    for path in sorted((run / 'review/blinded').rglob('*')):
        if path.is_file():
            add(path, str(path.relative_to(run)), run)
    for path in sorted((project / 'code/hdepic_qwen35_baseline').glob('*.py')):
        add(path, 'source/' + str(path.relative_to(project)), project)
    for path in sorted((project / 'code/tests').glob('test_hdepic_qwen35_baseline_*.py')):
        add(path, 'source/' + str(path.relative_to(project)), project)
    for path in sorted((project / 'scripts').glob('*hdepic_qwen35*.sh')):
        add(path, 'source/' + str(path.relative_to(project)), project)
    for name in ('code/__init__.py', 'code/tests/__init__.py', 'code/hdepic_experiment/__init__.py',
                 'code/hdepic_experiment/metrics.py', 'code/hdepic_review_followup/__init__.py',
                 'code/hdepic_review_followup/fields.py'):
        add(project / name, 'source/' + name, project)
    return result


def deterministic_zip(destination, top, payloads):
    """Stable ZIP bytes and complete CRC/content-hash readback before publish."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix='.review-package-', suffix='.zip', dir=destination.parent) as temporary:
        with zipfile.ZipFile(temporary.name, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name, content in sorted(payloads.items()):
                verify_safe_content(name, content)
                info = zipfile.ZipInfo(top + '/' + name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
        with zipfile.ZipFile(temporary.name) as archive:
            require(archive.testzip() is None, 'ZIP CRC validation failed')
            require(len(archive.namelist()) == len(set(archive.namelist())) == len(payloads), 'ZIP member count/uniqueness differs')
            for name, content in payloads.items():
                member = top + '/' + name
                require(allowed_member(member), f'Forbidden final ZIP member: {member}')
                require(hashlib.sha256(archive.read(member)).digest() == hashlib.sha256(content).digest(),
                        f'ZIP bytes differ from source: {member}')
        actual = Path(temporary.name).read_bytes()
        immutable_write(destination, actual)
    return hashlib.sha256(actual).hexdigest()


def build_package(run=OUT, project=None, archive=None, blinded_archive=None):
    run = Path(run).resolve(); ready = validate_ready(run)
    project = Path(project or ready['manifest']['project']).resolve()
    require(run.is_relative_to(project / 'runs'), 'Package run must be inside project/runs')
    archive = Path(archive or run / (run.name + '.review.zip')).resolve()
    require(archive.is_relative_to(run) and archive.suffix == '.zip', 'Archive must be a new ZIP inside this baseline run')
    blinded_archive = Path(blinded_archive or run / (run.name + '.review.blinded.zip')).resolve()
    require(blinded_archive.is_relative_to(run) and blinded_archive.suffix == '.zip' and blinded_archive != archive,
            'Blinded archive must be a separate new ZIP inside this baseline run')
    files = collect_files(run, project)
    payloads = {name: path.read_bytes() for name, path in files.items()}
    for name in ('data.py', 'model.py', 'processor.py', 'resources.py', 'runner.py',
                 'review.py', 'report.py', 'package.py'):
        require('source/code/hdepic_qwen35_baseline/' + name in payloads, f'Missing new baseline source: {name}')
    for name in ('run_hdepic_qwen35_baseline.sh', 'prepare_hdepic_qwen35_environment.sh'):
        require('source/scripts/' + name in payloads, f'Missing baseline script: {name}')
    for name, digest in ready['config'].get('answering_code_sha256', {}).items():
        member = 'source/code/hdepic_qwen35_baseline/' + name
        require(member in payloads and hashlib.sha256(payloads[member]).hexdigest() == digest,
                f'Packaged answering source differs from active configuration: {name}')
    parser = json.loads((run / 'frozen_parser.json').read_text())
    for kind, member in (('strict', 'source/code/hdepic_experiment/metrics.py'),
                         ('supplementary', 'source/code/hdepic_review_followup/fields.py')):
        require(hashlib.sha256(payloads[member]).hexdigest() == parser[kind + '_parser_sha256'],
                f'Packaged {kind} parser differs from frozen source')
    instructions = REVIEW_INSTRUCTIONS.encode()
    if 'review/REVIEW_INSTRUCTIONS.md' in payloads:
        require(payloads['review/REVIEW_INSTRUCTIONS.md'] == instructions, 'Review instructions changed')
    payloads['review/REVIEW_INSTRUCTIONS.md'] = instructions
    profiles = {name: {'sha256': sha256(run / name), 'bytes': (run / name).stat().st_size}
                for name in ('frozen_config.json', 'frozen_config_v2.json', 'active_configuration.json')}
    profile_manifest = {'active_configuration_sha256': ready['config']['configuration_sha256'],
                        'active_file': str(Path(ready['active_configuration']['path']).relative_to(run)),
                        'active_file_sha256': ready['active_configuration']['file_sha256'],
                        'previous_experimental_settings_unchanged': ready['active_configuration']['experimental_settings_unchanged'],
                        'profiles': profiles}
    profile_bytes = json_bytes(profile_manifest)
    payloads['profile_manifest.json'] = profile_bytes
    blind_payloads = {name: content for name, content in payloads.items()
                      if name.startswith('review/blinded/') or name in
                      ('review/rubric.md', 'review/review_addendum.md', 'review/REVIEW_INSTRUCTIONS.md')}
    blind_manifest = {'version': VERSION + '_blinded', 'cases': 228, 'four_answer_slots': 912, 'fixed50': 50,
                      'human_fields_all_blank': ready['human_fields_all_blank'],
                      'human_or_assistant_review_claimed_complete': False,
                      'identity_keys_private_seed_named_prediction_files_model_identity_code_weights_absent': True,
                      'scope': 'Only blinded review/source evidence/contact sheets, frozen rubric, operational addendum and review instructions.',
                      'files': {name: {'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}
                                for name, content in sorted(blind_payloads.items())}}
    blind_manifest_bytes = json_bytes(blind_manifest)
    blind_payloads['blinded_package_manifest.json'] = blind_manifest_bytes
    manifest = {'version': VERSION, 'cases': 228, 'four_answer_slots': 912, 'fixed50': 50,
                'profile_manifest_sha256': hashlib.sha256(profile_bytes).hexdigest(),
                'blinded_package_manifest_sha256': hashlib.sha256(blind_manifest_bytes).hexdigest(),
                'blinded_archive_name': blinded_archive.name, 'blinded_archive_members': len(blind_payloads),
                'generation_sha256': sha256(run / 'predictions/generation.jsonl'),
                'inputs_manifest_sha256': sha256(run / 'inputs/manifest.json'),
                'rubric_sha256': ready['review']['rubric_sha256'],
                'human_fields_all_blank': ready['human_fields_all_blank'],
                'semantic_review_status': ready['review']['status'], 'semantic_accuracy_claimed': False,
                'identity_keys_included': False, 'private_seed_included': False,
                'weights_included': False, 'full_frame_cache_included': False,
                'old_training_artifacts_copied': False, 'test_data_accessed': False,
                'old_full_prediction_files_copied': False,
                'original_three_model_answers': 'All raw answers retained in the four-slot blinded cases; original2209-row files are not duplicated.',
                'omitted_model_asset': 'tokenizer.json vocabulary; pinned identity SHA retained, available from original official checkpoint.',
                'blinding_limitations': ['Full evidence archive includes named new-model raw generation, permitting answer-text matching. For blind review use only review/blinded with rubric/addendum.',
                                        'Prior context and recognizable answer styles limit blinding. Private slot map/seed remain outside archive.'],
                'files': {name: {'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}
                          for name, content in sorted(payloads.items())}}
    payloads['package_manifest.json'] = json_bytes(manifest)
    # The gate and collection may be separated by file writes in another stage.
    # Require stable complete evidence before saving a review artifact.
    second = validate_ready(run)
    require(ready['config']['configuration_sha256'] == second['config']['configuration_sha256']
            and ready['review'] == second['review'] and ready['metrics'] == second['metrics'], 'Evidence changed during packaging')
    require(all(sha256(path) == hashlib.sha256(payloads[name]).hexdigest() for name, path in files.items()),
            'Package source bytes changed during collection')
    require(all(name.startswith('review/blinded/') or name in
                ('review/rubric.md', 'review/review_addendum.md', 'review/REVIEW_INSTRUCTIONS.md', 'blinded_package_manifest.json')
                for name in blind_payloads), 'Unblinded evidence entered blinded-only archive')
    immutable_write(run / 'review/REVIEW_INSTRUCTIONS.md', instructions)
    digest = deterministic_zip(archive, run.name, payloads)
    blind_digest = deterministic_zip(blinded_archive, run.name, blind_payloads)
    immutable_write(run / 'packaging/profile_manifest.json', profile_bytes)
    immutable_write(run / 'packaging/package_manifest.json', json_bytes(manifest))
    immutable_write(run / 'packaging/blinded_package_manifest.json', blind_manifest_bytes)
    result = {'archive': str(archive), 'sha256': digest, 'bytes': archive.stat().st_size, 'members': len(payloads),
              'blinded_archive': str(blinded_archive), 'blinded_sha256': blind_digest,
              'blinded_bytes': blinded_archive.stat().st_size, 'blinded_members': len(blind_payloads),
              'blinded_package_manifest_sha256': hashlib.sha256(blind_manifest_bytes).hexdigest(),
              'profile_manifest_sha256': manifest['profile_manifest_sha256'], 'all_member_bytes_verified': True,
              'identity_keys_included': False, 'private_seed_included': False, 'weights_included': False,
              'full_frame_cache_included': False, 'test_data_accessed': False, 'human_review_complete': False,
              'semantic_review_status': ready['review']['status']}
    immutable_write(run / 'packaging/package.json', json_bytes(result))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=OUT)
    parser.add_argument('--project', type=Path)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--blinded-archive', type=Path)
    args = parser.parse_args(argv)
    try:
        result = build_package(args.run, args.project, args.archive, args.blinded_archive)
    except PendingPackage as exc:
        print(json.dumps({'status': 'pending_completion', 'archive_created': False, 'reason': str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
