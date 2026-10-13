"""Complete the saved-prediction follow-up without model or video operations."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import shutil
import tempfile
from urllib.parse import unquote, urlsplit
import zipfile

PROJECT = Path(__file__).resolve().parents[2]
RUN = PROJECT / "runs/hdepic_sft_8b_v1"
DATASET = PROJECT / "datasets/hdepic_sft_v2_repaired"
OUTPUT = RUN / "review_followup_v1"
ZIP_PATH = RUN / "review_followup_v1.zip"
PYTHON = PROJECT / ".venvs/hdepic-swift-8b/bin/python"
COMMAND = f"PYTHONDONTWRITEBYTECODE=1 PYTHONPATH={PROJECT} {PYTHON} -m code.hdepic_review_followup.workflow"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(data)
    temporary.replace(path)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def protected_paths():
    paths = []
    for path in RUN.rglob("*"):
        if not path.is_file() or path.name == "workflow.lock":
            continue
        relative = path.relative_to(RUN)
        if relative.parts[0].startswith("review_followup_v1"):
            continue
        paths.append(path)
    # The cached inputs reside outside the repaired dataset and are independently
    # audited by visual.py. Protect every existing repaired-dataset artifact.
    paths.extend(path for path in DATASET.rglob("*") if path.is_file())
    for directory in (PROJECT / "code/hdepic_experiment", PROJECT / "code/hdepic_preparation"):
        paths.extend(directory.glob("*.py"))
    return sorted(set(paths))


def snapshot_sources():
    result = {}
    for path in protected_paths():
        stat = path.stat()
        item = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        # No model weights are loaded or copied. Checkpoint tensor files are
        # protected by size/mtime; their original completion manifests remain.
        if path.suffix not in {".bin", ".pt", ".safetensors"}:
            item["sha256"] = sha256(path)
        result[str(path)] = item
    return result


def verify_sources(saved):
    current_paths = {str(path) for path in protected_paths()}
    assert current_paths == set(saved), "Original artifact membership changed"
    for raw, expected in saved.items():
        path = Path(raw)
        stat = path.stat()
        assert stat.st_size == expected["bytes"] and stat.st_mtime_ns == expected["mtime_ns"], raw
        if "sha256" in expected:
            assert sha256(path) == expected["sha256"], raw


class AssetLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "img":
            self.images.append(values.get("src", ""))
        if tag == "a" and "href" in values:
            self.links.append(values["href"])


def verify_portable_tree(root):
    pages = sorted(Path(root).rglob("*.html"))
    assert pages, "No HTML review found"
    image_links = 0
    local_links = 0
    for page in pages:
        parser = AssetLinks()
        parser.feed(page.read_text())
        assert parser.images, f"No contact sheets in {page}"
        for url in parser.images + parser.links:
            parsed = urlsplit(url)
            if not parsed.path:
                continue
            assert not parsed.scheme and not parsed.netloc and not parsed.path.startswith("/"), url
            target = (page.parent / unquote(parsed.path)).resolve()
            assert target.is_relative_to(Path(root).resolve()) and target.is_file(), (page, url)
            local_links += 1
        image_links += len(parser.images)
    return {"passed": True, "html_pages": len(pages), "image_links_verified": image_links,
            "local_links_verified": local_links, "no_absolute_or_remote_asset_links": True}


def copy_code(out):
    destinations = []
    for path in sorted((PROJECT / "code/hdepic_review_followup").glob("*.py")):
        target = out / "source" / path.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        destinations.append(target)
    originals = [PROJECT / "code/__init__.py", PROJECT / "code/hdepic_experiment/__init__.py",
                 PROJECT / "code/hdepic_experiment/metrics.py"]
    tests = list((PROJECT / "code/tests").glob("test_hdepic_review_followup*.py"))
    for path in originals + tests:
        target = out / "source" / path.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        destinations.append(target)
    return {str(path.relative_to(out)): sha256(path) for path in destinations}


def pack(out):
    files = sorted(path for path in out.rglob("*") if path.is_file() and path.name != "workflow.lock")
    assert all("identity_key" not in path.name for path in files), "Identity key must remain outside the archive"
    assert all(path.suffix.lower() not in {".pt", ".bin", ".safetensors", ".mp4", ".avi", ".mov"} for path in files)
    # Asset selection is restricted to the 50-row visual audit, never a cache tree.
    assert not any("frame_cache" in path.relative_to(out).parts for path in files)
    fd, raw_temp = tempfile.mkstemp(prefix="hdepic-followup-", suffix=".zip", dir="/tmp")
    os.close(fd)
    temporary = Path(raw_temp)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in files:
                archive.write(path, str(path.relative_to(out)))
        with zipfile.ZipFile(temporary) as archive:
            assert archive.testzip() is None
            assert len(archive.namelist()) == len(files)
            for path in files:
                assert hashlib.sha256(archive.read(str(path.relative_to(out)))).hexdigest() == sha256(path)
            with tempfile.TemporaryDirectory(prefix="hdepic-followup-html-") as extract_dir:
                archive.extractall(extract_dir)
                portability = verify_portable_tree(extract_dir)
        if ZIP_PATH.exists():
            # Compare member content, since ZIP container timestamps may differ.
            with zipfile.ZipFile(ZIP_PATH) as old, zipfile.ZipFile(temporary) as new:
                assert set(old.namelist()) == set(new.namelist()), "Existing package differs; preserve it and use a new version"
                assert all(old.read(name) == new.read(name) for name in new.namelist()), "Existing package differs; preserve it and use a new version"
        else:
            shutil.copyfile(temporary, ZIP_PATH)
    finally:
        temporary.unlink(missing_ok=True)
    return {"zip": str(ZIP_PATH), "zip_bytes": ZIP_PATH.stat().st_size, "zip_sha256": sha256(ZIP_PATH),
            "members": len(files), "extracted_html_audit": portability, "identity_key_excluded": True,
            "weights_videos_full_cache_excluded": True}


def build_report(out, fields, forecast, visual, issues):
    lines = ["# HD-EPIC offline evaluation follow-up v1", "",
             "Supplementary diagnostics and a human-review preparation bundle, using only saved validation predictions. Human reason-label review is pending. No training, inference, external API, video decoding, or test evaluation was performed. Original predictions, dataset, strict parser/report, NLL and checkpoints are preserved. Epoch 1 remains selected under the original Explain NLL criterion.", "",
             "## Independent-field lexical diagnostics", "",
             "Each requested field must have one complete, nonempty literal opening/closing pair, with no duplicate occurrences or ambiguous boundary. WHY, action and supervised goal are extracted independently: malformed NEXT and prose outside a field do not invalidate its content. Missing, empty, duplicate and ambiguous flags are explicit and can overlap; unextractable predictions remain in the denominator. Generated content is HTML-decoded once, then the original normalization applies NFKC, case folding and whitespace collapse, retaining punctuation and wording. References use the original normalization. Exact match is a lexical diagnostic, not semantic accuracy or visual support.", "",
             "| Saved model | WHY lexical exact match | Current-action lexical exact match | Supervised-goal lexical exact match | Original strict format |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for stage in ("base", "epoch1", "epoch2"):
        summary = fields["stages"][stage]
        scope = summary["scopes"]["all"]
        cells = []
        for field in ("why", "action", "goal"):
            metric = scope[field]
            cells.append(f"{metric['exact_matches']}/{metric['denominator']} ({100*metric['rate']:.2f}%)")
        strict = summary["original_strict_metric"]["scopes"]["all"]
        lines.append(f"| {stage} | " + " | ".join(cells) + f" | {strict['strict_format_valid']}/{strict['expected']} |")
    lines.extend(["", "| Saved model | Extractable WHY | Extractable current action | Extractable supervised goal |",
                  "| --- | ---: | ---: | ---: |"])
    for stage in ("base", "epoch1", "epoch2"):
        scope = fields["stages"][stage]["scopes"]["all"]
        cells = [f"{scope[field]['extractable']}/{scope[field]['denominator']}" for field in ("why", "action", "goal")]
        lines.append(f"| {stage} | " + " | ".join(cells) + " |")
    lines.extend(["", "All 228 base WHY fields are extractable despite zero strict whole-response validity; none exactly match its reference after the fixed lexical normalization. This does not establish that its reasons are semantically wrong. Epoch 2 gains one current-action lexical match whose NEXT field is malformed. Detailed statuses, raw predictions, field offsets, extracted/reference fields and per-example diagnostic scores are in scoring/{base,epoch1,epoch2}.fields.jsonl; the original strict metrics are reproduced separately and checked against the frozen report.", "",
             "The frozen strict parser requires semicolon-separated NEXT actions. The saved validation prompts request three actions but do not explicitly specify that delimiter. This mismatch is documented separately; prompts, generations and original strict results were not changed.", "",
             f"Saved-prompt audit: {fields['semicolon_prompt_audit']['prompts_explicitly_requesting_semicolon']}/2,209 explicitly request a semicolon. Evidence is in scoring/semicolon_prompts.jsonl and scoring/semicolon_mismatch_audit.json.", "",
             "## Forecasting repetition", "",
             "One independently extractable NEXT block must contain three nonempty semicolon-separated strings. Strings are normalized with the same rules. Count when all three are identical; do not use overall strict response validity as a gate. All 2,209 IDs remain in the denominator. Malformed triples are separate and are not silently repaired. Repetition can be appropriate and is not automatically an error.", "",
             "| Source | All three strings identical | Malformed triples | Valid triples |",
             "| --- | ---: | ---: | ---: |"])
    for stage in ("epoch1", "epoch2", "references"):
        metric = forecast["stages"][stage]
        lines.append(f"| {stage} | {metric['all_three_identical']}/{metric['denominator']} ({100*metric['rate']:.2f}%) | {metric['malformed_triples']}/{metric['denominator']} | {metric['valid_triples']} |")
    lines.extend(["", "Counts are calculated from saved files, without inserting the previously reported totals. Every predicted/reference triple and affected ID is in forecast/; malformed rows are retained separately.", "",
             "## Fixed human-review bundle", "",
             "Open visual/index.html for the exact fixed 50 WHY examples and original A/B assignments. The portable sheets and individual-frame links use only the corresponding cached input JPEGs, ordered by audited source timestamps. Review reference validity/attachment separately from answer purpose and visual support. Plausible alternative purposes do not establish the actor's stated intent; exact wording is not a semantic correctness criterion. All human review fields start blank.", "",
             "The model input history, action question, query time, original narration/reference reason and raw A/B responses are shown for every example. Timestamp/path/SHA evidence is in visual/; HTML image links are checked both in place and after extracting the ZIP. Contact sheets are presentation aids, not substituted model inputs.", "",
             "## Source-label and question-scope advisories", "",
             "The advisory issues/ list retains original narration/spans/offsets/question/target and keeps any proposed correction separate, pending adjudication. Neither annotations nor original NLL targets were revised. Scope omissions are advisory, with no broad modifier enrichment.", "",
             "Required concerns are P07-20240529-191007-94 (subsequent knife action in the reason), P07-20240529-131737-170 (subsequent spice-jar placement), and P07-20240529-102652-196 (unclear 'it fits' referent). The fixed sample also contains spoon-to-spoon cookie-dough transfers whose broad question may omit decisive context; their evidence and pending proposals are recorded in the advisory list.", "",
             "Model identities remain only in the separate server key file. For blinded human review distribute visual/ alone: the complete technical bundle's stage-labeled scoring records could be manually matched to A/B answers.", "",
             "## Rerun", "", "```bash", COMMAND, "```", "",
             "A completed rerun verifies source/artifact hashes and reuses the finished outputs. It does not regenerate annotations or predictions. Source and focused extraction tests are included under source/.", "",
             "Machine-readable counts and definitions are in component_summaries.json, scoring/field_scoring_summary.json and forecast/repetition_summary.json. Preservation and portable-link audits accompany the bundle. No semantic correctness or human-review result has been assigned.", ""])
    (out / "supplementary_report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "workflow.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protected_path = OUTPUT / "protected_originals.json"
        if protected_path.exists():
            protected = read_json(protected_path)
            verify_sources(protected)
        else:
            protected = snapshot_sources()
            write_json(protected_path, protected)
        selected = read_json(RUN / "experiment_completed.json")
        assert selected["selected_stage"] == "epoch1" and selected["optimizer_steps"] == 1818
        marker_path = RUN / "review_followup_v1.completed.json"
        if marker_path.exists():
            marker = read_json(marker_path)
            for relative, expected in marker["output_files_sha256"].items():
                assert sha256(OUTPUT / relative) == expected, relative
            for raw, expected in marker["code_sha256"].items():
                assert sha256(raw) == expected, raw
            package = pack(OUTPUT)
            assert package["zip_sha256"] == marker["package"]["zip_sha256"]
            print(json.dumps({"already_complete": True, "passed": True, **package}, indent=2))
            return
        from .fields import build_fields
        from .forecast import build_forecast
        from .visual import build_visual
        from .issues import build_issues
        fields = build_fields(PROJECT, RUN, DATASET, OUTPUT)
        forecast = build_forecast(PROJECT, RUN, DATASET, OUTPUT)
        visual = build_visual(PROJECT, RUN, DATASET, OUTPUT)
        issues = build_issues(PROJECT, RUN, DATASET, OUTPUT)
        write_json(OUTPUT / "component_summaries.json", {"fields": fields, "forecast": forecast, "visual": visual, "issues": issues})
        build_report(OUTPUT, fields, forecast, visual, issues)
        code_outputs = copy_code(OUTPUT)
        html_audit = verify_portable_tree(OUTPUT)
        write_json(OUTPUT / "portability_audit.json", html_audit)
        verify_sources(protected)
        write_json(OUTPUT / "preservation_audit.json", {"passed": True, "original_files_verified_unchanged": len(protected),
                   "metadata_and_prediction_hashes_unchanged": True, "checkpoint_tensor_size_mtime_unchanged": True,
                   "selected_checkpoint_unchanged": selected["selected_checkpoint"], "no_training_or_inference": True})
        # Completion metadata is kept beside the ZIP rather than modifying an
        # archive that includes its own hash. All actual review assets are inside.
        files = {str(path.relative_to(OUTPUT)): sha256(path) for path in OUTPUT.rglob("*") if path.is_file() and path.name != "workflow.lock"}
        package = pack(OUTPUT)
        code = {str(PROJECT / Path(path).relative_to("source")): value for path, value in code_outputs.items()}
        marker = {"passed": True, "version": "hdepic_review_followup_v1", "offline_only": True,
                  "finished_at_utc": datetime.now(timezone.utc).isoformat(), "command": COMMAND,
                  "package": package, "output_files_sha256": files, "code_sha256": code,
                  "epoch1_remains_selected": True, "human_reason_label_review": "pending"}
        # Keep marker outside the content tree so completed reruns have exactly
        # the same archive members as the original package.
        write_json(marker_path, marker)
        ZIP_PATH.with_suffix(".zip.sha256").write_text(f"{package['zip_sha256']}  {ZIP_PATH.name}\n")
        print(json.dumps({"passed": True, **package, "human_reason_label_review": "pending"}, indent=2))


if __name__ == "__main__":
    main()
