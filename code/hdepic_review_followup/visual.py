"""Portable visual review of the existing fixed fifty blinded WHY pairs.

Uses only preserved validation inputs, saved answers and cached JPEG bytes.
Contact sheets are review aids; the model consumed sixteen individual images.
No video decoding, model inference, annotation editing or identity reassignment.
"""
from __future__ import annotations

from collections import Counter
import csv
from fractions import Fraction
import hashlib
import html
from html.parser import HTMLParser
import io
import json
import math
from pathlib import Path
import re
import shutil
from urllib.parse import unquote, urlparse

from PIL import Image, ImageDraw, ImageFont

HUMAN_FIELDS = (
    'reference reason valid?', 'reference attached to correct action?',
    'answer A specific purpose', 'answer A observational support',
    'answer B specific purpose', 'answer B observational support', 'reviewer notes',
)
PURPOSE_OPTIONS = ('clear', 'partial', 'different or contradictory', 'cannot judge')
SUPPORT_OPTIONS = ('supported', 'unsupported', 'insufficient evidence')
REFERENCE_OPTIONS = ('yes', 'no', 'cannot judge')
FRAME_PATTERN = re.compile(r'^Frame (\d{2}), source time ([0-9.]+) s: <image>$', re.MULTILINE)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def font(size, bold=False):
    paths = (
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/clear-sans/ClearSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/clear-sans/ClearSans-Regular.ttf',
    )
    for path in paths:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def make_contact_sheet(paths, frames, destination, number, query_time):
    """Paste each cached 384px image in order, with its source timestamp."""
    cell, margin, gap, header, label = 384, 16, 12, 54, 40
    width = margin * 2 + cell * 4 + gap * 3
    height = header + margin * 2 + (cell + label) * 4 + gap * 3
    sheet = Image.new('RGB', (width, height), '#f3f5f7')
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 12), f'Review {number:02d}  |  query {query_time:.6f} s  |  16 cached input frames',
              font=font(22, True), fill='#172437')
    for index, (path, frame) in enumerate(zip(paths, frames)):
        x = margin + (index % 4) * (cell + gap)
        y = header + margin + (index // 4) * (cell + label + gap)
        with Image.open(path) as original:
            assert original.size == (frame['width'], frame['height'])
            image = original.convert('RGB')
            # Preserved input frames are square384. Retain aspect if a future
            # compatible cache is smaller; the exact JPEG is linked separately.
            image.thumbnail((cell, cell), Image.Resampling.LANCZOS)
            draw.rectangle((x, y, x + cell, y + cell), fill='#ffffff')
            sheet.paste(image, (x + (cell - image.width) // 2, y + (cell - image.height) // 2))
        draw.text((x + 8, y + cell + 8), f'{index + 1:02d}  source {frame["timestamp_sec"]:.6f} s',
                  font=font(20), fill='#172437')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name('.' + destination.name + '.tmp')
    sheet.save(temporary, format='JPEG', quality=94, subsampling=0)
    temporary.replace(destination)
    return {'width': width, 'height': height, 'sha256': sha256(destination), 'bytes': destination.stat().st_size}


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images, self.links = [], []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == 'img':
            self.images.append(attributes['src'])
        if tag == 'a' and 'href' in attributes:
            self.links.append(attributes['href'])


def audit_html_links(html_path):
    """Can also run against the HTML after extracting the portable ZIP."""
    html_path = Path(html_path).resolve()
    parser = LinkParser()
    parser.feed(html_path.read_text())
    checked = set()
    for link in parser.images + parser.links:
        parsed = urlparse(link)
        assert not parsed.scheme and not parsed.netloc, f'External/absolute HTML link: {link}'
        if not parsed.path:
            continue
        assert not Path(unquote(parsed.path)).is_absolute()
        target = (html_path.parent / unquote(parsed.path)).resolve()
        assert target.is_relative_to(html_path.parent), f'HTML asset escapes visual directory: {link}'
        assert target.is_file(), f'Broken HTML asset: {link}'
        checked.add(parsed.path)
    return {'passed': True, 'image_links': len(parser.images), 'unique_file_links': len(checked),
            'all_links_relative_and_local': True}


def reviewer_control(number, field, options=None):
    attributes = f'data-row="{number - 1}" data-field="{html.escape(field, quote=True)}"'
    label = f'<label>{html.escape(field)}'
    if options is None:
        control = f'<textarea {attributes} rows="3"></textarea>'
    else:
        control = f'<select {attributes}><option value="">— leave blank until reviewed —</option>'
        control += ''.join(f'<option value="{html.escape(option, quote=True)}">{html.escape(option)}</option>' for option in options)
        control += '</select>'
    return label + control + '</label>'


def render_html(records, csv_rows):
    parts = ['''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>HD-EPIC blinded WHY visual review</title>
<style>body{margin:0;background:#f2f5f8;color:#172437;font:16px/1.55 system-ui,sans-serif}main{max-width:1250px;margin:auto;padding:26px}h1,h2,h3{line-height:1.2}header,.record{background:white;border:1px solid #dbe3ea;border-radius:10px;padding:24px;margin-bottom:22px}.meta{color:#4d6076;font-size:14px}.sheet{display:block;width:100%;height:auto;border:1px solid #dbe3ea}.answers{display:grid;grid-template-columns:1fr 1fr;gap:18px}.answer,.reference{padding:16px;border:1px solid #dbe3ea;border-radius:8px}.reference{background:#fbfaf4;margin-top:18px}.answer p{white-space:pre-wrap}.history{font-size:15px}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f7;padding:14px;font:14px/1.5 ui-monospace,monospace}a{color:#185c9e}.frames{display:flex;flex-wrap:wrap;gap:10px;font-size:13px;margin:12px 0}.review{display:grid;grid-template-columns:1fr 1fr;gap:12px}label{display:block;font-size:14px;font-weight:600}select,textarea{display:block;box-sizing:border-box;width:100%;padding:9px;margin-top:5px;font:14px system-ui;border:1px solid #a6b5c5;border-radius:5px;background:white}button{padding:11px 16px;background:#185c9e;color:white;border:0;border-radius:5px;font:600 15px system-ui;cursor:pointer}.jump{max-height:120px;overflow:auto}.note{font-size:14px;color:#4d6076}@media(max-width:760px){main{padding:12px}.record,header{padding:16px}.answers,.review{grid-template-columns:1fr}}</style></head><body><main><header>
<h1>Blinded WHY visual review</h1><p>The same fixed 50 validation examples and A/B assignments from the existing review. Human review is pending.</p>
<p>Compare each answer with the <strong>specific stated purpose</strong>, allowing paraphrases. Assess support from frames and completed history separately. A plausible alternative explanation does not verify the actor's intent. The source reason is provisional and may require attachment or span adjudication.</p>
<p class="note">Contact sheets are presentation aids. The model received 16 individual cached JPEGs; numbered links open their exact bytes. All labels show actual source timestamps, in original input order. Source narration and reference reason below are review material, not model observations.</p>
<p><a href="review.csv">Blank review CSV</a> · <a href="records.jsonl">Portable record evidence</a> · <a href="timestamp_audit.json">Timestamp/image audit</a></p>
<button type="button" id="download">Download CSV with current review entries</button><p class="note">Entries remain in this page until downloaded. No server requests or automatic human judgments occur.</p><nav class="jump">''']
    parts.append(' · '.join(f'<a href="#record-{record["blind_row"]:02d}">{record["blind_row"]:02d}</a>' for record in records))
    parts.append('</nav></header>')
    for record in records:
        number = record['blind_row']
        parts.append(f'<article class="record" id="record-{number:02d}"><h2>{number:02d}. {html.escape(record["action_question_text"])}</h2>')
        parts.append(f'<p class="meta">{html.escape(record["id"])}<br>Query time {record["query_time_sec"]:.6f} s · action interval [{record["action_start_sec"]:.6f}, {record["action_end_sec"]:.6f}] s</p>')
        parts.append('<h3>Completed actions supplied to the model</h3><ol class="history">')
        for action in record['history']:
            parts.append(f'<li>[{action["start_sec"]:.3f}, {action["end_sec"]:.3f}] {html.escape(action["action"])}</li>')
        parts.append('</ol>' if record['history'] else '</ol><p>No previous completed actions were supplied.</p>')
        parts.append(f'<a href="{record["contact_sheet"]}"><img class="sheet" src="{record["contact_sheet"]}" alt="Review {number:02d}, sixteen input frames with actual source times" loading="lazy"></a>')
        parts.append('<div class="frames">' + ''.join(
            f'<a href="{frame["path"]}">{index + 1:02d} · {frame["timestamp_sec"]:.6f} s</a>'
            for index, frame in enumerate(record['frames'])) + '</div>')
        parts.append('<div class="answers">')
        for letter in ('A', 'B'):
            parts.append(f'<section class="answer"><h3>Answer {letter}</h3><p>{html.escape(record[f"answer_{letter}_why"])}</p><details><summary>Inspect raw response {letter}</summary><pre>{html.escape(record[f"answer_{letter}_raw"])}</pre></details></section>')
        parts.append('</div><section class="reference"><h3>Source reference</h3>')
        parts.append(f'<p><strong>Extracted reason:</strong> {html.escape(record["source_stated_reason"])}</p><p><strong>Original narration:</strong> {html.escape(record["original_narration"])}</p>')
        parts.append(f'<p class="meta">Narration ID: {html.escape(record["source_narration_id"])} · exact reason characters [{record["reason_char_start"]}, {record["reason_char_end"]}) · extraction rule: {html.escape(record["extraction_rule"])}</p></section>')
        parts.append(f'<details><summary>Inspect exact user prompt supplied to the model</summary><pre>{html.escape(record["input_prompt"])}</pre></details>')
        parts.append('<h3>Human review — initially blank</h3><div class="review">')
        parts.append(reviewer_control(number, HUMAN_FIELDS[0], REFERENCE_OPTIONS))
        parts.append(reviewer_control(number, HUMAN_FIELDS[1], REFERENCE_OPTIONS))
        for letter in ('A', 'B'):
            parts.append(reviewer_control(number, f'answer {letter} specific purpose', PURPOSE_OPTIONS))
            parts.append(reviewer_control(number, f'answer {letter} observational support', SUPPORT_OPTIONS))
        parts.append(reviewer_control(number, 'reviewer notes'))
        parts.append('</div><p><a href="#">Back to review index</a></p></article>')
    # JSON is escaped for a script context while preserving decoded field text.
    payload = json.dumps(csv_rows, ensure_ascii=True).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    fields = json.dumps(list(csv_rows[0]), ensure_ascii=True)
    parts.append(f'''<script type="application/json" id="review-data">{payload}</script><script>
const rows=JSON.parse(document.getElementById('review-data').textContent);const fields={fields};
document.getElementById('download').addEventListener('click',()=>{{
document.querySelectorAll('[data-row][data-field]').forEach(el=>{{rows[Number(el.dataset.row)][el.dataset.field]=el.value;}});
const quote=value=>'"'+String(value??'').replaceAll('"','""')+'"';
const content=[fields.map(quote).join(','),...rows.map(row=>fields.map(field=>quote(row[field])).join(','))].join('\\r\\n')+'\\r\\n';
const url=URL.createObjectURL(new Blob([content],{{type:'text/csv;charset=utf-8'}}));const link=document.createElement('a');link.href=url;link.download='visual_why_review.csv';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}});</script></main></body></html>''')
    return '\n'.join(parts) + '\n'


def build_visual(project, run, dataset, out):
    project, run, dataset, out = [Path(path).resolve() for path in (project, run, dataset, out)]
    assert out.is_relative_to(run) and out != run
    source_csv = run / 'review/blind_why_review.csv'
    source_key = run / 'review/blind_why_review_identity_key.json'
    with source_csv.open(newline='') as handle:
        source_rows = list(csv.DictReader(handle))
    assert len(source_rows) == 50 and len({row['id'] for row in source_rows}) == 50
    assert [int(row['blind_row']) for row in source_rows] == list(range(1, 51))
    manifest = read_json(run / 'review/blind_why_review_manifest.json')
    assert [row['id'] for row in source_rows] == manifest['frozen_ids']
    frozen = read_json(run / 'frozen_evaluation.json')
    assert [row['id'] for row in source_rows] == frozen['blind_review_ids']
    config = read_json(run / 'frozen_config.json')
    assert sha256(run / 'frozen_config.json') == frozen['frozen_config_sha256']
    source_paths = {key: dataset / 'swift' / f'validation{suffix}.jsonl'
                    for key, suffix in [('export', ''), ('prompts', '.prompts'), ('scoring', '.scoring')]}
    for key, path in source_paths.items():
        assert str(path) == config['data']['validation'][key]
        assert sha256(path) == config['data']['validation'][key + '_sha256']
    text_path = dataset / 'text/validation.jsonl'
    assert sha256(text_path) == config['data']['validation']['text_sha256']
    sources = {key: read_jsonl(path) for key, path in source_paths.items()}
    text = read_jsonl(text_path)
    assert all(len(items) == 2209 for items in [*sources.values(), text])
    selected = read_json(run / 'experiment_completed.json')['selected_stage']
    assert selected == 'epoch1', 'Preserve the completed experiment selection'
    predictions = {}
    for stage in ('base', selected):
        path = run / 'predictions' / stage / 'generation.jsonl'
        completed = read_json(path.parent / 'completed.json')
        assert completed['passed'] and sha256(path) == completed['generation_sha256']
        predictions[stage] = {row['id']: row for row in read_jsonl(path)}
        assert len(predictions[stage]) == 2209

    visual = out / 'visual'
    csv_path = visual / 'review.csv'
    if csv_path.exists():
        with csv_path.open(newline='') as handle:
            previous = list(csv.DictReader(handle))
        assert not any(row.get(field, '') for row in previous for field in HUMAN_FIELDS), 'Refusing to overwrite human review entries'
    visual.mkdir(parents=True, exist_ok=True)
    for name in ('images', 'contact_sheets'):
        (visual / name).mkdir(exist_ok=True)
    records, review_rows, image_assets, contact_assets = [], [], {}, {}
    max_delta, outside_nominal_window = -math.inf, []
    from code.hdepic_experiment.report import _display_why
    for number, original in enumerate(source_rows, 1):
        index = int(original['validation_row_index'])
        exported, prompt, score, record = [sources[key][index] for key in ('export', 'prompts', 'scoring')] + [text[index]]
        identity = original['id']
        assert score['id'] == record['id'] == identity and score['row_index'] == index
        assert score['partition'] == record['partition'] == 'validation' and score['participant'] == 'P07'
        assert score['task'] == record['task'] == 'explain'
        assert prompt == {'messages': exported['messages'][:2], 'images': exported['images']}
        assert [message['role'] for message in prompt['messages']] == ['system', 'user']
        assert original['input_prompt'] == prompt['messages'][1]['content']
        assert json.loads(original['individual_images_json']) == exported['images'] == [frame['path'] for frame in score['frames']]
        assert len(exported['images']) == len(set(exported['images'])) == len(score['frames']) == 16
        query = float(original['query_time_sec'])
        assert query == score['query_time_sec'] == record['query_time_sec'] == score['current_action']['end_sec']
        history = json.loads(original['history_json'])
        assert history == record['input']['history']
        assert history == [{key: action[key] for key in ('action', 'start_sec', 'end_sec')} for action in score['history_actions']]
        assert len(history) <= 8 and all(action['end_sec'] <= query for action in history)
        assert score['current_action']['narration_id'] not in {action['narration_id'] for action in score['history_actions']}
        reason = score['reason_annotation']
        assert original['source_narration_id'] == reason['narration_id']
        assert original['original_narration'] == reason['original_narration']
        assert original['source_stated_reason'] == reason['reason_text'] == record['target']['reason']
        start, end = int(original['reason_char_start']), int(original['reason_char_end'])
        assert (start, end) == (reason['char_start'], reason['char_end'])
        assert original['original_narration'][start:end] == original['source_stated_reason']
        assert original['action_question_text'] == record['input']['action_question_text']
        assert Counter([original['answer_A_raw'], original['answer_B_raw']]) == Counter(
            predictions[stage][identity]['prediction'] for stage in ('base', selected))
        # A/B order comes from the existing CSV. Do not consult the identity key
        # or choose an answer based on the target/reference.
        for letter in ('A', 'B'):
            matches = [predictions[stage][identity] for stage in ('base', selected)
                       if predictions[stage][identity]['prediction'] == original[f'answer_{letter}_raw']]
            assert any(_display_why(row) == original[f'answer_{letter}_why'] for row in matches)
        prompt_frames = FRAME_PATTERN.findall(original['input_prompt'])
        assert len(prompt_frames) == 16
        frames = []
        for frame_index, (source_image, saved_frame, (label_index, prompt_time)) in enumerate(
                zip(exported['images'], score['frames'], prompt_frames)):
            path = Path(source_image)
            assert path.is_file() and path.is_relative_to(project / 'datasets/hdepic_sft_v1/frames')
            assert int(label_index) == frame_index + 1
            assert saved_frame['timestamp_sec'] <= query
            assert f'{saved_frame["timestamp_sec"]:.6f}' == prompt_time
            pts_time = float(Fraction(saved_frame['source_pts']) * Fraction(saved_frame['source_time_base']))
            assert math.isclose(pts_time, saved_frame['timestamp_sec'], rel_tol=0, abs_tol=1e-9)
            assert saved_frame['bytes'] == path.stat().st_size
            actual_hash = sha256(path)
            assert actual_hash == saved_frame['sha256']
            relative = f'images/{actual_hash}.jpg'
            destination = visual / relative
            if destination.exists():
                assert sha256(destination) == actual_hash
            else:
                shutil.copyfile(path, destination)
            with Image.open(destination) as image:
                assert image.size == (saved_frame['width'], saved_frame['height'])
            image_assets[relative] = {'sha256': actual_hash, 'bytes': destination.stat().st_size,
                                      'width': saved_frame['width'], 'height': saved_frame['height']}
            frames.append({**saved_frame, 'path': relative, 'original_cached_path': source_image,
                           'input_order': frame_index + 1, 'prompt_timestamp_text': prompt_time})
            max_delta = max(max_delta, saved_frame['timestamp_sec'] - query)
            if saved_frame['timestamp_sec'] < max(0, query - 5):
                outside_nominal_window.append({'id': identity, 'frame': frame_index + 1,
                                               'timestamp_sec': saved_frame['timestamp_sec'], 'query_time_sec': query})
        assert [frame['timestamp_sec'] for frame in frames] == sorted(frame['timestamp_sec'] for frame in frames)
        sheet_path = f'contact_sheets/{number:02d}.jpg'
        contact_assets[sheet_path] = make_contact_sheet([visual / frame['path'] for frame in frames], frames,
                                                       visual / sheet_path, number, query)
        portable = {'blind_row': number, 'id': identity, 'participant': original['participant'],
                    'source_video_id': original['source_video_id'], 'validation_row_index': index,
                    'query_time_sec': query, 'action_start_sec': score['current_action']['start_sec'],
                    'action_end_sec': score['current_action']['end_sec'],
                    'action_question_text': original['action_question_text'], 'history': history,
                    'input_prompt': original['input_prompt'], 'frames': frames, 'contact_sheet': sheet_path,
                    'source_narration_id': original['source_narration_id'], 'original_narration': original['original_narration'],
                    'source_stated_reason': original['source_stated_reason'], 'reason_char_start': start,
                    'reason_char_end': end, 'extraction_rule': reason['extraction_rule'],
                    **{key: original[key] for key in ('answer_A_why', 'answer_B_why', 'answer_A_raw', 'answer_B_raw',
                                                     'answer_A_saved_status', 'answer_B_saved_status')},
                    'human_review_status': 'pending', **{field: '' for field in HUMAN_FIELDS}}
        records.append(portable)
        review_rows.append({key: value for key, value in portable.items() if key not in ('frames', 'history')} | {
            'history_json': json.dumps(history, ensure_ascii=False),
            'individual_images_json': json.dumps([frame['path'] for frame in frames]),
            'actual_frame_timestamps_json': json.dumps([frame['timestamp_sec'] for frame in frames]),
        })
    assert [record['id'] for record in records] == [row['id'] for row in source_rows]
    assert all(not row[field] for row in review_rows for field in HUMAN_FIELDS)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(review_rows[0]))
    writer.writeheader()
    writer.writerows(review_rows)
    if csv_path.exists():
        assert csv_path.read_bytes() == buffer.getvalue().encode('utf-8'), 'Refusing to overwrite changed review CSV content or custom columns'
    atomic_text(csv_path, buffer.getvalue())
    atomic_text(visual / 'records.jsonl', ''.join(json.dumps(record, ensure_ascii=False) + '\n' for record in records))
    identity_key_destination = run / (out.name + '.identity_key.json')
    assert not identity_key_destination.is_relative_to(out)
    if identity_key_destination.exists():
        assert sha256(identity_key_destination) == sha256(source_key), 'Refusing to replace a different model identity key'
    else:
        shutil.copyfile(source_key, identity_key_destination)
    timestamp_audit = {'passed': True, 'examples': 50, 'frame_references': 800,
                       'unique_exact_cached_jpegs': len(image_assets), 'frames_after_query': 0,
                       'max_frame_time_minus_query_sec': max_delta,
                       'frames_before_nominal_5_second_window': outside_nominal_window,
                       'source_pts_times_match_saved_timestamps': True,
                       'actual_model_prompt_frame_times_match': True,
                       'input_path_order_and_cached_sha256_match': True,
                       'history_actions_finish_by_query': True,
                       'source_reason_exact_offsets_verified': True,
                       'same_50_ids_and_original_AB_assignments': True,
                       'saved_raw_answer_pairs_verified_without_identity_key_mapping': True,
                       'video_decodes': 0, 'new_inference_or_training': False,
                       'human_review_status': 'pending',
                       'review_rubric': {'purpose': PURPOSE_OPTIONS, 'support': SUPPORT_OPTIONS,
                                         'reference_validity_and_attachment': REFERENCE_OPTIONS},
                       'source_review_csv_sha256': sha256(source_csv),
                       'frozen_evaluation_sha256': sha256(run / 'frozen_evaluation.json'),
                       'source_input_file_sha256': {key: sha256(path) for key, path in source_paths.items()},
                       'image_assets': image_assets, 'contact_sheet_assets': contact_assets}
    atomic_json(visual / 'timestamp_audit.json', timestamp_audit)
    atomic_text(visual / 'index.html', render_html(records, review_rows))
    links = audit_html_links(visual / 'index.html')
    summary = {'passed': True, 'examples': 50, 'contact_sheets': 50, 'frame_references': 800,
               'unique_individual_images': len(image_assets), 'human_columns_blank': True,
               'original_AB_assignments_preserved': True, 'timestamp_audit_passed': True,
               'frames_after_query': 0, 'html': str(visual / 'index.html'), 'review_csv': str(csv_path),
               'record_evidence': str(visual / 'records.jsonl'), 'html_link_audit': links,
               'identity_key_outside_package': str(identity_key_destination),
               'identity_key_sha256': sha256(identity_key_destination),
               'assets_bytes': sum(asset['bytes'] for asset in image_assets.values())
                              + sum(asset['bytes'] for asset in contact_assets.values()),
               'asset_hashes': {**{name: asset['sha256'] for name, asset in image_assets.items()},
                                **{name: asset['sha256'] for name, asset in contact_assets.items()}},
               'source_module_sha256': sha256(Path(__file__))}
    atomic_json(visual / 'manifest.json', summary)
    return summary
