#!/usr/bin/env python3
"""Render only a complete, fully judged 228-example three-model comparison.

Run after sourcing scripts/hdepic_data_storage.sh. This is a report renderer;
it never changes evaluator metrics, semantic judgments, or model outputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
from statistics import median
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODELS = ("base", "epoch1", "epoch2")
NAMES = {"base": "Base", "epoch1": "Epoch 1", "epoch2": "Epoch 2"}
LABELS = ("correct", "partial", "incorrect")
ESC = html.escape


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def pct(rate):
    return "not measured" if rate is None else f"{100 * rate:.1f}%"


def fraction(count, total):
    return f"{count}/{total} ({pct(count / total)})" if total else "not measured"


def action_metric(value):
    return fraction(value["correct"], value["denominator"])


def difference(rate, base):
    return f"{100 * (rate - base):+.1f} pp"


def md_table(headers, values):
    clean = lambda x: str(x).replace("|", "\\|").replace("\n", "<br>")
    return "\n".join(["| " + " | ".join(map(clean, headers)) + " |", "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(map(clean, value)) + " |" for value in values])


def html_table(headers, values):
    return '<div class="table-wrap"><table><thead><tr>' + ''.join('<th scope="col">' + ESC(str(x)) + '</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + ESC(str(x)) + '</td>' for x in value) + '</tr>' for value in values) + '</tbody></table></div>'


def decision(record, field):
    entry = record["semantic"][field]
    if not entry["eligible"]:
        return None
    if entry["judgment"]:
        return entry["judgment"]
    if not entry["prediction"]:
        return {"label": "incorrect", "justification": "No extractable answer in this completed generation.",
                "flags": {}, "provenance": {"method": "missing_prediction_deterministic_failure"}}
    raise ValueError(f"Unjudged {field} answer for {record['id']}")


def next_correct(record, reference, position, component=None):
    predictions = record["predicted_next"]
    prediction = predictions[position] if len(predictions) > position else [None, None]
    target = reference["next_pairs"][position]
    return prediction == target if component is None else prediction[component] == target[component]


def transitions(before, after):
    assert len(before) == len(after)
    return {"both_correct": sum(a and b for a, b in zip(before, after)),
            "gained_correctness": sum(not a and b for a, b in zip(before, after)),
            "lost_correctness": sum(a and not b for a, b in zip(before, after)),
            "neither_correct": sum(not a and not b for a, b in zip(before, after)),
            "denominator": len(before)}


def change_name(before, after):
    return "gain" if not before and after else "loss" if before and not after else "same"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=HERE / "full_comparison")
    args = parser.parse_args()
    snapshot = args.snapshot.resolve()
    if not snapshot.is_relative_to(HERE):
        raise SystemExit("Report snapshot must be inside full_validation")
    cohort = read_json(snapshot / "cohort.json")
    summary = read_json(snapshot / "scores/summary.json")
    manifest = read_json(snapshot / "scores/manifest.json")
    if not cohort["full_generation_complete_for_requested_models"] or cohort["evaluated_examples"] != 228:
        raise SystemExit("No final report: generation is not complete for the full 228-example cohort")
    if set(summary["models"]) != set(MODELS):
        raise SystemExit("No final report: base, epoch1 and epoch2 are all required")
    if not summary["completed_semantic_evaluation"] or summary["pending_packet_count"]:
        raise SystemExit("No final report: semantic review is still pending")
    for path, expected in manifest["input_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise SystemExit(f"Scoring input changed after evaluation: {path}")

    refs = {r["id"]: r for r in rows(snapshot / "references.jsonl")}
    ids = cohort["selected_query_ids"]
    records = {m: {r["id"]: r for r in rows(snapshot / "scores/per_example" / f"{m}.jsonl")} for m in MODELS}
    raw = {m: {r["id"]: r for r in rows(snapshot / f"{m}.jsonl")} for m in MODELS}
    assert len(ids) == len(set(ids)) == 228 and set(refs) == set(ids)
    assert all(set(values) == set(ids) for values in records.values())
    assert all(rec["generation_completed"] for values in records.values() for rec in values.values())
    assert all(r["task"] == "explain" for r in refs.values())
    scores = {m: summary["models"][m]["all"] for m in MODELS}
    for model in MODELS:
        assert scores[model]["examples"] == 228
        for field, expected in (("why", 226), ("goal", 228)):
            metric = scores[model][field]
            assert metric["eligible"] == expected and not metric["pending"]
            assert sum(metric["counts"].values()) == expected

    audited = read_json(HERE.parent / "analysis.json")
    validation, training = audited["validation"], audited["training"]
    cleanup_path = HERE.parent / "checkpoint_cleanup_20261011/completed.json"
    cleanup = read_json(cleanup_path) if cleanup_path.exists() else None
    source = {r["id"]: r for r in rows(ROOT / "datasets/hdepic_sft_v2_repaired/text/validation.jsonl") if r["id"] in refs}
    examples = []
    for identity in ids:
        ref = refs[identity]
        example = {"id": identity, "video_id": ref["video_id"], "query_time_sec": ref["query_time_sec"],
                   "question": ref["question"], "reference_why": ref["why"]["reference"],
                   "original_narration": ref["why"]["original_narration"],
                   "why_source_status": ref["why"]["source_status"], "why_eligible": ref["why"]["eligible"],
                   "reference_goal": ref["goal"]["reference"],
                   "goal_scope_review_pending": ref["goal"]["profile"]["manual_scope_review_pending"],
                   "reference_next": source[identity]["target"]["next_actions"],
                   "reference_next_pairs": ref["next_pairs"]}
        for model in MODELS:
            record = records[model][identity]
            example[model] = {"why": record["fields"]["why"]["text"], "goal": record["fields"]["goal"]["text"],
                              "next": record["next_extraction"]["actions"], "next_pairs": record["predicted_next"],
                              "why_judgment": decision(record, "why"), "goal_judgment": decision(record, "goal"),
                              "next_correct_positions": [next_correct(record, ref, pos) for pos in range(3)]}
        examples.append(example)

    paired = {}
    for model in MODELS[1:]:
        fields = {}
        for field in ("why", "goal"):
            eligible = [i for i in ids if refs[i][field]["eligible"]]
            before = [decision(records["base"][i], field)["label"] for i in eligible]
            after = [decision(records[model][i], field)["label"] for i in eligible]
            fields[field] = {**transitions([v == "correct" for v in before], [v == "correct" for v in after]),
                             "label_transitions": dict(Counter(a + " → " + b for a, b in zip(before, after)))}
        for position in range(3):
            for component, index in (("verb", 0), ("noun", 1), ("action", None)):
                fields[f"next_{position + 1}_{component}"] = transitions(
                    [next_correct(records["base"][i], refs[i], position, index) for i in ids],
                    [next_correct(records[model][i], refs[i], position, index) for i in ids])
        fields["next_complete_sequence"] = transitions(
            [all(next_correct(records["base"][i], refs[i], pos) for pos in range(3)) for i in ids],
            [all(next_correct(records[model][i], refs[i], pos) for pos in range(3)) for i in ids])
        paired[model] = fields

    primary_rows = []
    for field, label in (("why", "WHY: specific narrated purpose"), ("goal", "GOAL: complete annotated activity")):
        primary_rows.append([label, *[fraction(scores[m][field]["counts"]["correct"], scores[m][field]["eligible"]) for m in MODELS],
                             *[difference(scores[m][field]["correct_rate"], scores["base"][field]["correct_rate"]) for m in MODELS[1:]]])
    for pos in range(1, 4):
        values = {m: scores[m]["next"][f"position_{pos}"]["action"] for m in MODELS}
        primary_rows.append([f"NEXT {pos}: verb + noun", *[action_metric(values[m]) for m in MODELS],
                             *[difference(values[m]["rate"], values["base"]["rate"]) for m in MODELS[1:]]])
    totals = {m: sum(scores[m]["next"][f"position_{pos}"]["action"]["correct"] for pos in range(1, 4)) for m in MODELS}
    primary_rows.append(["NEXT: all positions pooled", *[fraction(totals[m], 684) for m in MODELS],
                         *[difference(totals[m] / 684, totals["base"] / 684) for m in MODELS[1:]]])
    primary_rows.append(["NEXT: all three actions correct", *[action_metric(scores[m]["next"]["exact_ordered_sequence"]["action"]) for m in MODELS],
                         *[difference(scores[m]["next"]["exact_ordered_sequence"]["action"]["rate"], scores["base"]["next"]["exact_ordered_sequence"]["action"]["rate"]) for m in MODELS[1:]]])
    primary_headers = ["Metric", "Base", "Epoch 1", "Epoch 2", "Epoch 1 − base", "Epoch 2 − base"]

    semantic_rows = []
    flags_rows = []
    for field in ("why", "goal"):
        for model in MODELS:
            metric = scores[model][field]
            semantic_rows.append([field.upper(), NAMES[model], metric["eligible"],
                                  *[metric["counts"][label] for label in LABELS], pct(metric["ordinal_score"])])
            flags_rows.append([field.upper(), NAMES[model],
                               *[f"{metric['flags'][flag]}/{metric['flag_judgment_counts'][flag]}" for flag in ("unsupported_addition", "contradiction", "too_broad", "too_narrow")]])

    goal_scope_rows, goal_scope_diagnostic = [], {}
    for flagged, label in ((True, "Scope review flagged"), (False, "Scope review not flagged")):
        selected_ids = [i for i in ids if refs[i]["goal"]["profile"]["manual_scope_review_pending"] == flagged]
        rates, ordinal, groups = [], [], {}
        for model in MODELS:
            counts = Counter(decision(records[model][i], "goal")["label"] for i in selected_ids)
            score = (2 * counts["correct"] + counts["partial"]) / (2 * len(selected_ids)) if selected_ids else None
            rates.append(fraction(counts["correct"], len(selected_ids)))
            ordinal.append(pct(score))
            groups[model] = {"examples": len(selected_ids), "counts": {key: counts[key] for key in LABELS},
                             "correct_rate": counts["correct"] / len(selected_ids) if selected_ids else None,
                             "ordinal_score": score}
        goal_scope_rows.append([label, len(selected_ids), *rates, *ordinal])
        goal_scope_diagnostic[str(flagged).lower()] = {"manual_scope_review_pending": flagged,
                                                      "ids": selected_ids, "models": groups}
    goal_scope_headers = ["Source profile", "N", "Base correct", "Epoch 1 correct", "Epoch 2 correct", "Base partial credit", "Epoch 1 partial credit", "Epoch 2 partial credit"]

    component_rows = []
    for position in range(1, 4):
        for component in ("verb", "noun", "action"):
            component_rows.append([f"Position {position}", "verb + noun" if component == "action" else component,
                                   *[action_metric(scores[m]["next"][f"position_{position}"][component]) for m in MODELS]])
    for component in ("verb", "noun", "action"):
        correct = {m: sum(scores[m]["next"][f"position_{pos}"][component]["correct"] for pos in range(1, 4)) for m in MODELS}
        component_rows.append(["Pooled positions", "verb + noun" if component == "action" else component,
                               *[fraction(correct[m], 684) for m in MODELS]])
    sequence_rows = []
    for component in ("verb", "noun", "action"):
        sequence_rows.append(["Entire ordered sequence correct ↑", component, *[action_metric(scores[m]["next"]["exact_ordered_sequence"][component]) for m in MODELS]])
    for metric, label in (("normalized_edit_distance", "Normalized Levenshtein ↓"), ("normalized_damerau_osa", "Normalized Damerau-OSA ↓")):
        for component in ("verb", "noun", "action"):
            sequence_rows.append([label, component, *[f"{scores[m]['next'][metric][component]['mean']:.4f}" for m in MODELS]])

    paired_rows = []
    for model in MODELS[1:]:
        for key, label in (("why", "WHY"), ("goal", "GOAL"), ("next_1_action", "NEXT 1"), ("next_2_action", "NEXT 2"), ("next_3_action", "NEXT 3"), ("next_complete_sequence", "All NEXT")):
            changes = paired[model][key]
            paired_rows.append([NAMES[model], label, changes["denominator"], *[changes[k] for k in ("both_correct", "gained_correctness", "lost_correctness", "neither_correct")]])

    video_rows = []
    for video, metrics in summary["models"]["base"]["by_video"].items():
        for field, label in (("why", "WHY"), ("goal", "GOAL"), ("next", "NEXT 1")):
            result = []
            for model in MODELS:
                metric = summary["models"][model]["by_video"][video][field]
                result.append(fraction(metric["counts"]["correct"], metric["eligible"]) if field != "next" else action_metric(metric["position_1"]["action"]))
            video_rows.append([video, label, *result])
    nll_rows = []
    for task in ("explain", "track"):
        value = validation[task]
        nll_rows.append([f"{task.title()} · {value['count']:,}", *[f"{value['mean_nll'][m]:.4f}" for m in MODELS],
                         *[f"{value['comparisons'][m]['relative_change_percent']:+.2f}%" for m in MODELS[1:]]])
    loss_video_rows = [[r["video_id"], r["task"], r["n"], *[f"{r[m]:.4f}" for m in MODELS]] for r in audited["by_video_validation"]]

    diagnostic_rows = []
    generation_diagnostics = {}
    for model in MODELS:
        values = list(raw[model].values())
        generation_diagnostics[model] = {"examples": len(values), "finish_reasons": dict(Counter(r["finish_reason"] for r in values)),
                                         "tokens_median": median(r["generated_tokens"] for r in values),
                                         "tokens_min": min(r["generated_tokens"] for r in values),
                                         "tokens_max": max(r["generated_tokens"] for r in values),
                                         "missing_why": scores[model]["why"]["missing_prediction"],
                                         "missing_goal": scores[model]["goal"]["missing_prediction"],
                                         "next": scores[model]["next"]["forecast_diagnostics"]}
        d = generation_diagnostics[model]
        diagnostic_rows.append([NAMES[model], d["examples"], json.dumps(d["finish_reasons"], sort_keys=True),
                                d["missing_why"], d["missing_goal"], d["next"]["missing_positions"],
                                f"{d['tokens_median']:g} ({d['tokens_min']}–{d['tokens_max']})"])

    goal_pending = sum(ref["goal"]["profile"]["manual_scope_review_pending"] for ref in refs.values())
    lead = (
        f"For base → epoch 1, first-next-action joint accuracy improves "
        f"{pct(scores['base']['next']['position_1']['action']['rate'])} → {pct(scores['epoch1']['next']['position_1']['action']['rate'])}. "
        f"Specific WHY agreement is {pct(scores['base']['why']['correct_rate'])} → {pct(scores['epoch1']['why']['correct_rate'])}; "
        f"complete GOAL agreement is {pct(scores['base']['goal']['correct_rate'])} → {pct(scores['epoch1']['goal']['correct_rate'])}. "
        f"Epoch 1 remains selected by the lowest validation loss. All three models are evaluated on the same 228 completed Explain queries. "
        "WHY and GOAL are provisional assistant judgments of reference agreement."
    )
    scope = "Generated-answer evaluation: 228 Explain queries from nine validation videos, all participant P07. WHY has 226 eligible sources, with two source-quality exclusions fixed before model comparison. GOAL and NEXT use all 228. The input prompts, image preprocessing, greedy generation settings, and native model identity match across all models. No unfinished generation is counted as an error."
    supersedes = "This full-cohort report supersedes the earlier 47-example preliminary comparison. The original report and its judgments are preserved; identical packet IDs and prompt hashes reuse the same decisions. The cohort has changed, so its percentages should not be compared as though they measured different models on identical samples."
    loss_note = "NLL averages each example's loss over its complete reference answer, with correct preceding tokens supplied. Explain includes WHY, GOAL, NEXT and termination, so this is not WHY accuracy. Epoch 1 lowers full Explain loss by 15.55% versus base. Epoch 2 has higher validation loss than epoch 1 for both tasks while training loss keeps falling, consistent with overfitting. Compared with base, epoch 2 Explain loss is still 7.09% lower, while Track loss is 8.38% higher. Checkpoint selection was fixed to Explain validation NLL and was not changed after viewing generated accuracy."
    semantic_note = "Correct means agreement with the specific narrated purpose (WHY) or all essential units of the annotated active activity (GOAL). Partial means a real component is present but a material detail or scope is missing. Incorrect includes another purpose/activity, central contradiction, or an unextractable answer from a completed generation. Partial-credit score = (2 × correct + partial) / (2 × eligible); strict correctness is the primary score. Assistant review uses identical frozen rubrics and judge packets that omit model names, with fixed source eligibility and hash-bound judgments. Orchestration and file paths can reveal model context, so this is not a fully blinded study. The judgments do not establish visual inferability or human-validated accuracy."
    goal_note = f"{goal_pending}/228 GOAL profiles remain marked for manual scope review. Activity labels vary in granularity, and some combine tasks. Strict GOAL correctness requires their essential activity units; naming only a relevant micro-action is not automatically correct. Keep complete and partial counts separate and calibrate compound scope with human review before treating this as a definitive intention metric."
    goal_scope_note = "Diagnostic only: these subsets use the original reference-side automatic scope flag, unchanged across all models. An unflagged profile is not human-validated and need not be a better reference. The main GOAL denominator remains all 228 examples. Differences between these subsets can reflect activity and video composition as well as annotation scope."
    action_note = "NEXT predicts the next three fine-grained annotated actions. Each position is mapped to official verb and noun classes by the frozen train-only heuristic mapper. Joint accuracy requires both classes at the same position. Missing slots are errors; repeated reference actions remain separate positions. The three-position pooled score has 684 decisions. The mapper still needs human checks on generated paraphrases, so class accuracy is an operational benchmark rather than a perfect measure of natural-language action correctness."
    sequence_note = "Exact ordered sequence requires all three predictions to match. Levenshtein allows insertions, deletions and substitutions; Damerau-OSA additionally allows an adjacent transposition. Distances are normalized by the three-action horizon; lower is better. There is one greedy sequence per example, not best-of-K sampling. These definitions do not make scores directly comparable to papers using other horizons, inputs, datasets or sampling protocols."
    limitations = "The benchmark covers one validation participant and uses completed annotated action history as input. Test performance, current-action recognition and Track GOAL/NEXT generation accuracy remain unmeasured. Explain already names the queried action, so it cannot measure action recognition. Semantic judgments and generated-phrase mapping are provisional. Per-video breakdowns describe this cohort; no statistical significance or population generalization claim is made."
    training_note = (
        "Training completed all 1,818 optimizer steps over two epochs: 14,543 initial-SFT examples "
        "(1,403 Explain and 13,140 Track), text LoRA rank 16, 116,727,808 trainable parameters, frozen vision/aligner, "
        "effective batch size 16, and 16 image frames per example. Epoch checkpoints 909 and 1818 were retained. "
        "The earlier audit verified both rank logs, finite losses/gradients, epoch checkpoint hashes and unchanged frozen code. "
        "An end-of-run audit failure was recovered without additional optimizer updates. The original run took about 95.16 hours, "
        "including validation/checkpointing; the short recovery marker does not represent training duration."
    )
    if cleanup:
        training_note += (
            f" The user-authorized cleanup removed {len(cleanup['deleted_steps'])} intermediate checkpoints "
            f"and recovered {cleanup['allocated_gib']:.2f} GiB. Both epoch checkpoint hashes remained unchanged."
        )
    flags_note = "Flags describe explicitly reviewed semantic answers, not a separate overall accuracy. Each cell is flagged decisions / decisions with that flag recorded; missing answers and automatic identical-reference decisions may lack flag reviews. Unsupported ancillary text can coexist with a correct core. A central contradiction requires an incorrect label."
    qa_paths = sorted((HERE / "review_audit").glob("*comparison.json"))
    qa_checks = [{"path": str(path.relative_to(HERE)), **read_json(path)} for path in qa_paths]
    semantic_qa = None
    qa_note = ""
    qa_rows = []
    if qa_checks:
        disagreements = [row for check in qa_checks for row in check["label_disagreements"]]
        alternatives = {row["packet_id"]: row["independent"] for row in disagreements}
        assert len(alternatives) == len(disagreements), "Overlapping independent QA disagreements need explicit reconciliation"
        semantic_qa = {"checks": qa_checks,
                       "sample_size": sum(check["sample_size"] for check in qa_checks),
                       "label_agreements": sum(check["label_agreements"] for check in qa_checks),
                       "label_disagreements": disagreements, "human_validation": False,
                       "primary_reviews_corrected_for_consistency": any((HERE / "review_audit").glob("*_consistency_corrections.json")),
                       "independent_qa_imported": False}
        sensitivity = {}
        for model in MODELS:
            sensitivity[model] = {}
            for field in ("why", "goal"):
                eligible = [record for record in records[model].values() if record["semantic"][field]["eligible"]]
                primary_count = sum(decision(record, field)["label"] == "correct" for record in eligible)
                alternative_count = sum(
                    alternatives.get(record["semantic"][field].get("packet_id"), decision(record, field))["label"] == "correct"
                    for record in eligible)
                sensitivity[model][field] = {"primary_correct": primary_count, "qa_alternative_correct": alternative_count,
                                             "eligible": len(eligible)}
                qa_rows.append([NAMES[model], field.upper(), fraction(primary_count, len(eligible)), fraction(alternative_count, len(eligible))])
        semantic_qa["disagreement_sensitivity"] = sensitivity
        qa_note = (
            f"Additional independent assistant review covered {semantic_qa['sample_size']} shuffled, selected "
            f"packets without the original labels or model context; {semantic_qa['label_agreements']}/{semantic_qa['sample_size']} "
            f"labels agreed. {len(disagreements)} disagreements remain visible in review_audit/ and concern rubric boundaries. "
            "This is a consistency check on selected samples, not human validation, an estimate of population accuracy, "
            "or a confidence interval. Independent QA ratings were not imported into the benchmark scores. "
            "The sensitivity table replaces only the disputed packet labels with the independent alternatives wherever those exact packets occur; "
            "all other ratings remain fixed. It is not an uncertainty interval for unreviewed rubric disagreements."
        )
        consistency_paths = sorted((HERE / "review_audit").glob("*_consistency_corrections.json"))
        if consistency_paths:
            semantic_qa["consistency_audits"] = [str(path.relative_to(HERE)) for path in consistency_paths]
            qa_note += " Separately, initial ratings were corrected for consistent application of the frozen rubric: identical GOAL reference/answer text and material WHY details such as complete ingredient removal or improved fit. These corrections apply across reviewed outputs; full before/after records are retained in the consistency audits. Original matched-47 judgment files remain unchanged."
    now = datetime.now(timezone.utc)
    created = now.isoformat()
    display_created = now.astimezone(ZoneInfo("America/Detroit")).strftime("%Y-%m-%d %H:%M %Z")
    details = {"created_utc": created, "scope": cohort, "validation": validation, "training": training,
               "paired_transitions": paired, "generation_diagnostics": generation_diagnostics,
               "checkpoint_cleanup": cleanup,
               "goal_profiles_requiring_scope_review": goal_pending,
               "goal_scope_diagnostic": goal_scope_diagnostic, "independent_semantic_spot_check": semantic_qa,
               "primary_table": {"headers": primary_headers, "rows": primary_rows},
               "provisional_semantic_assessment": True, "human_validation_performed": False}

    # Example selection is reproducible: first chronological gains/losses for
    # WHY, GOAL and NEXT-1 against the selected epoch, at most two per category.
    chosen = []
    for field in ("why", "goal", "next"):
        for desired in ("gain", "loss"):
            selected = 0
            for example in examples:
                if field == "next":
                    before, after = (example[m]["next_correct_positions"][0] for m in ("base", "epoch1"))
                else:
                    if field == "why" and not example["why_eligible"]:
                        continue
                    before, after = (example[m][f"{field}_judgment"]["label"] == "correct" for m in ("base", "epoch1"))
                if change_name(before, after) == desired:
                    if example not in chosen:
                        chosen.append(example)
                    selected += 1
                    if selected == 2:
                        break

    markdown = ["# Qwen3.8-27B: complete validation evaluation", "", f"**Generated {display_created} · 228/228 answers per model · Semantic review provisional**", "", lead, "", supersedes,
                "", "## Evaluation coverage", "", scope, "", "## Generated-answer accuracy", "", md_table(primary_headers, primary_rows),
                "", "Changes are percentage points. All models use exactly the same examples and source-eligibility masks.",
                "", "## Semantic decisions", "", md_table(["Field", "Model", "Eligible", "Correct", "Partial", "Incorrect", "Partial credit"], semantic_rows),
                "", semantic_note, "", goal_note, "", "### GOAL scope diagnostic", "", md_table(goal_scope_headers, goal_scope_rows), "", goal_scope_note,
                "", "## NEXT components", "", md_table(["Position", "Component", "Base", "Epoch 1", "Epoch 2"], component_rows),
                "", action_note, "", "## Complete sequences and edit distances", "", md_table(["Metric", "Component", "Base", "Epoch 1", "Epoch 2"], sequence_rows),
                "", sequence_note, "", "## Paired gains and losses", "", md_table(["Compared with base", "Field", "N", "Both correct", "Gained", "Lost", "Neither correct"], paired_rows),
                "", "## Results by video", "", md_table(["Video", "Metric", "Base", "Epoch 1", "Epoch 2"], video_rows),
                "", "## Full teacher-forced validation loss", "", md_table(["Task · N", "Base", "Epoch 1", "Epoch 2", "Epoch 1 relative change", "Epoch 2 relative change"], nll_rows),
                "", loss_note, "", "## Concrete gains and regressions", "", "The examples below are the first chronological gains/losses for WHY, GOAL and NEXT-1 against epoch 1, up to two per category. The HTML report and examples.json contain all 228 examples.", ""]
    for example in chosen:
        markdown += [f"### {example['video_id']} · {example['query_time_sec']:.2f} s · {example['question']}", "", md_table(["Field", "Reference", "Base", "Epoch 1", "Epoch 2"], example_rows(example)), ""]
    markdown += ["## Generation diagnostics", "", md_table(["Model", "Completed", "Finish reasons", "Missing WHY", "Missing GOAL", "Missing NEXT slots", "Tokens median (range)"], diagnostic_rows),
                 "", "Missing WHY counts only source-eligible examples. A missing extracted answer within a completed generation is distinct from an unfinished generation.",
                 "", "## Semantic flags", "", md_table(["Field", "Model", "Unsupported addition", "Contradiction", "Too broad", "Too narrow"], flags_rows), "", flags_note,
                 "", "## Training and limitations", "", training_note, "", limitations,
                 "", "## Independent semantic review", "", qa_note, "", md_table(["Model", "Field", "Primary correct", "Disputed labels replaced"], qa_rows),
                 "", "## Validation loss by video", "", md_table(["Video", "Task", "N", "Base", "Epoch 1", "Epoch 2"], loss_video_rows),
                 "", "## Reproducibility", "", f"Snapshot: `{snapshot.relative_to(HERE)}`. The canonical evaluator's `scores/manifest.json` records references, mapper, config, judgments and evaluator hashes. `scores/metrics.csv` and `scores/per_example/` preserve every metric and evaluated answer. `analysis.json` stores paired changes and diagnostics; `examples.json` and `report.html` expose every comparison. No evaluator or training implementation was changed for this report.", ""]

    def paragraph(value):
        return '<p>' + ESC(value) + '</p>'

    def section(identity, title, value):
        return '<section id="' + identity + '"><h2>' + ESC(title) + '</h2>' + value + '</section>'

    content = '<header><div class="eyebrow">ACTGROUND · COMPLETE VALIDATION</div><h1>Qwen3.8-27B after fine-tuning</h1><p class="subtitle">Base, epoch 1 and epoch 2 · Same 228 Explain queries · Nine P07 videos</p><div class="badges"><span>Generation complete</span><span>All semantic packets reviewed</span><span class="provisional">Assistant judgments · provisional</span></div></header>'
    content += '<div class="lead">' + paragraph(lead) + '</div><nav aria-label="Report sections"><a href="#accuracy">Accuracy</a><a href="#semantic">WHY &amp; GOAL</a><a href="#next">NEXT</a><a href="#gallery">All answers</a><a href="#loss">Validation loss</a><a href="#method">Method</a></nav>'
    content += section('coverage', 'What is evaluated', paragraph(scope) + '<div class="callout">' + paragraph(supersedes) + '</div>')
    content += section('accuracy', 'Generated-answer accuracy', html_table(primary_headers, primary_rows) + paragraph('Changes are percentage points. Strict semantic correctness requires the full specific purpose or annotated activity. Pooled NEXT is joint verb+noun accuracy across all 684 positions.'))
    content += section('semantic', 'WHY and GOAL: complete, partial or incorrect', html_table(['Field', 'Model', 'Eligible', 'Correct', 'Partial', 'Incorrect', 'Partial credit'], semantic_rows) + paragraph(semantic_note) + '<div class="callout">' + paragraph(goal_note) + '</div>')
    content += section('goal-scope', 'GOAL scope diagnostic', html_table(goal_scope_headers, goal_scope_rows) + paragraph(goal_scope_note))
    content += section('next', 'NEXT: verb, noun and ordered action', html_table(['Position', 'Component', 'Base', 'Epoch 1', 'Epoch 2'], component_rows) + paragraph(action_note))
    content += section('sequences', 'Three-action sequences', html_table(['Metric', 'Component', 'Base', 'Epoch 1', 'Epoch 2'], sequence_rows) + paragraph(sequence_note))
    content += section('paired', 'Which examples gained or lost correctness', html_table(['Compared with base', 'Field', 'N', 'Both correct', 'Gained', 'Lost', 'Neither correct'], paired_rows))
    content += section('videos', 'Results by video', html_table(['Video', 'Metric', 'Base', 'Epoch 1', 'Epoch 2'], video_rows))

    gallery = '<p>Expand any example to inspect the reference and all three answers. Filter gains or regressions by field; labels show strict correctness against the frozen reference.</p><div class="gallery-controls"><label>Search<input id="search" type="search" placeholder="Action, purpose, video or answer"></label><label>Video<select id="video"><option value="">All videos</option>' + ''.join('<option>' + ESC(v) + '</option>' for v in sorted({x['video_id'] for x in examples})) + '</select></label><label>Compare<select id="model"><option value="epoch1">Epoch 1 vs base</option><option value="epoch2">Epoch 2 vs base</option></select></label><label>Field<select id="field"><option value="why">WHY</option><option value="goal">GOAL</option><option value="next">First NEXT action</option></select></label><label>Change<select id="change"><option value="">All changes</option><option value="gain">Became correct</option><option value="loss">Lost correctness</option><option value="same">Correctness unchanged</option></select></label></div><div class="gallery-actions"><output id="shown" aria-live="polite">228 of 228 examples</output><button id="expand" type="button">Expand visible</button><button id="collapse" type="button">Collapse all</button></div><div id="cases">'
    for example in examples:
        attributes = {'video': example['video_id']}
        for model in MODELS[1:]:
            for field in ('why', 'goal', 'next'):
                if field == 'next':
                    a, b = (example[m]['next_correct_positions'][0] for m in ('base', model))
                elif field == 'why' and not example['why_eligible']:
                    attributes[model + '-' + field] = 'excluded'
                    continue
                else:
                    a, b = (example[m][field + '_judgment']['label'] == 'correct' for m in ('base', model))
                attributes[model + '-' + field] = change_name(a, b)
        title = f"{example['video_id']} · {example['query_time_sec']:.2f} s · {example['question']}"
        gallery += '<details class="case" ' + ' '.join('data-' + key + '="' + ESC(value, quote=True) + '"' for key, value in attributes.items()) + '><summary>' + ESC(title) + '</summary>'
        gallery += html_table(['Field', 'Reference', 'Base', 'Epoch 1', 'Epoch 2'], example_rows(example))
        notes = f"Source WHY status: {example['why_source_status']}. " + ('GOAL scope review pending. ' if example['goal_scope_review_pending'] else '')
        gallery += '<p class="case-note">' + ESC(notes) + '</p><details class="technical"><summary>Original narration and mapped NEXT class pairs</summary>' + paragraph(example['original_narration'] or '[none]')
        gallery += html_table(['Class pairs', 'Reference', 'Base', 'Epoch 1', 'Epoch 2'], [['(verb ID, noun ID)', str(example['reference_next_pairs']), *[str(example[m]['next_pairs']) for m in MODELS]]]) + '</details></details>'
    content += section('gallery', 'Inspect all 228 answers', gallery + '</div>')
    content += section('loss', 'Full validation loss: 2,209 examples', html_table(['Task · N', 'Base', 'Epoch 1', 'Epoch 2', 'Epoch 1 relative change', 'Epoch 2 relative change'], nll_rows) + paragraph(loss_note))
    content += section('diagnostics', 'Generation completeness', html_table(['Model', 'Completed', 'Finish reasons', 'Missing WHY', 'Missing GOAL', 'Missing NEXT slots', 'Tokens median (range)'], diagnostic_rows) + paragraph('Missing extracted fields are within completed generations. WHY missingness uses only eligible sources. No whole-response formatting metric is used.'))
    content += section('flags', 'Semantic error flags', html_table(['Field', 'Model', 'Unsupported addition', 'Contradiction', 'Too broad', 'Too narrow'], flags_rows) + paragraph(flags_note))
    content += section('method', 'Training audit and limits', paragraph(training_note) + paragraph(limitations))
    if semantic_qa:
        content += section('semantic-qa', 'Independent semantic review', paragraph(qa_note) + html_table(['Model', 'Field', 'Primary correct', 'Disputed labels replaced'], qa_rows))
    content += section('loss-videos', 'Validation loss by video', html_table(['Video', 'Task', 'N', 'Base', 'Epoch 1', 'Epoch 2'], loss_video_rows))
    content += '<footer>Generated ' + ESC(display_created) + ' · <a href="report.md">Markdown report</a> · <a href="examples.json">All example data</a> · <a href="' + ESC(str(snapshot.relative_to(HERE))) + '/scores/metrics.csv">Canonical metrics CSV</a> · <a href="README.md">Reproduce this evaluation</a></footer>'

    css = '''
    :root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f3f6f9;color:#1d3044;font:15px/1.65 system-ui,-apple-system,sans-serif}main{max-width:1400px;margin:auto;padding:44px 28px 72px}header{padding:0 4px 26px}.eyebrow{color:#14786f;font-size:11px;font-weight:750;letter-spacing:1.7px}h1{font-size:36px;line-height:1.2;letter-spacing:-1px;margin:12px 0}.subtitle{color:#667b8d}.badges{display:flex;gap:8px;flex-wrap:wrap}.badges span{background:#def1e9;color:#226851;font-size:12px;padding:4px 10px;border-radius:24px}.badges .provisional{background:#fff0d7;color:#866017}.lead,section{background:white;border:1px solid #dce5ed;border-radius:12px;padding:24px 28px;margin-bottom:18px}.lead{border-left:5px solid #238077}.lead p{font-size:17px;margin:0}nav{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:18px}nav a{display:inline-block;padding:6px 14px;background:#e7eef3;border-radius:20px;color:#304d65;text-decoration:none;font-size:13px}a{color:#116b86}h2{font-size:22px;letter-spacing:-.35px;margin:0 0 15px}p{color:#435a6d;margin:13px 0}.callout{border-left:3px solid #dbb979;background:#fffbf3;padding:1px 16px;margin:14px 0}.table-wrap{overflow:auto;margin:12px 0}table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.5}th,td{text-align:left;vertical-align:top;padding:10px 12px;border-bottom:1px solid #e4ebf1;white-space:pre-line;min-width:90px}th{background:#f0f5f8;color:#486176;font-size:12px;font-weight:700}td:first-child{font-weight:550}tbody tr:hover{background:#f8fbfc}.gallery-controls{display:grid;grid-template-columns:2fr 1.6fr 1fr 1fr 1fr;gap:10px;background:#f2f6f9;padding:14px;border-radius:8px}label{font-size:12px;font-weight:650;color:#526a7e}input,select{display:block;width:100%;border:1px solid #c8d7e3;border-radius:6px;padding:9px;margin-top:4px;background:#fff;color:#243e54;font:13px system-ui,sans-serif}.gallery-actions{display:flex;align-items:center;gap:8px;padding:12px 0}.gallery-actions output{margin-right:auto;font-size:13px;color:#5b7286}button{border:1px solid #cbd8e2;border-radius:6px;background:#fff;padding:6px 11px;color:#34546c;font:12px system-ui,sans-serif;cursor:pointer}button:hover{background:#e9f3f3}summary{cursor:pointer;font-weight:650}.case{border:1px solid #dae5ed;border-radius:8px;margin:8px 0;background:#fff}.case>summary{padding:13px 15px;font-size:13px}.case[open]>summary{background:#eff6f6;border-bottom:1px solid #d8e7e5}.case .table-wrap{margin:0;padding:8px}.case table{table-layout:fixed;min-width:800px}.case table th:first-child{width:65px}.case table td{font-size:12px;font-weight:400}.case-note{font-size:12px;padding:0 16px}.technical{margin:10px 16px;font-size:12px}.technical summary{font-weight:500;color:#647b8f}.technical table{min-width:600px}.technical p{font-size:12px}footer{font-size:12px;color:#6a7d8e;padding:12px 4px}.case[hidden]{display:none}@media(max-width:900px){main{padding:24px 14px}h1{font-size:29px}.lead,section{padding:20px 16px}.gallery-controls{grid-template-columns:1fr 1fr}.gallery-controls label:first-child{grid-column:1/-1}}@media print{body{background:white}main{max-width:none;padding:0}nav,.gallery-controls,.gallery-actions{display:none}section,.lead{break-inside:avoid;border:0;border-bottom:1px solid #ddd}#gallery{break-inside:auto}.case{break-inside:avoid}}
    '''
    script = '''
    const cases=[...document.querySelectorAll('.case')];
    const controls=['search','video','model','field','change'].map(id=>document.getElementById(id));
    function filter(){
      const [search,video,model,field,change]=controls.map(e=>e.value);
      const query=search.trim().toLowerCase(); let count=0;
      for(const row of cases){
        const visible=(!video||row.dataset.video===video)&&(!change||row.getAttribute('data-'+model+'-'+field)===change)&&(!query||row.textContent.toLowerCase().includes(query));
        row.hidden=!visible; if(visible)count++;
      }
      document.getElementById('shown').textContent=count+' of '+cases.length+' examples';
    }
    controls.forEach(e=>e.addEventListener(e.id==='search'?'input':'change',filter));
    document.getElementById('expand').addEventListener('click',()=>cases.filter(e=>!e.hidden).forEach(e=>e.open=true));
    document.getElementById('collapse').addEventListener('click',()=>cases.forEach(e=>e.open=false));
    '''
    document = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen3.8-27B complete validation evaluation</title><style>' + css + '</style></head><body><main>' + content + '</main><script>' + script + '</script></body></html>\n'
    write_json(HERE / "analysis.json", details)
    write_json(HERE / "examples.json", examples)
    (HERE / "report.md").write_text("\n".join(markdown))
    (HERE / "report.html").write_text(document)
    print(json.dumps({"report": str(HERE / "report.html"), "examples": len(examples), "models": list(MODELS),
                      "pending_semantic_packets": 0, "primary_metrics": primary_rows}, indent=2))


def example_rows(example):
    values = []
    for field in ("why", "goal", "next"):
        reference = example["reference_" + field]
        reference = " → ".join(str(x) for x in reference) if isinstance(reference, list) else reference or "[source excluded]"
        cells = []
        for model in MODELS:
            value = example[model][field]
            if field == "next":
                text = " → ".join(str(x) if x is not None else "[missing]" for x in value)
                positions = example[model]["next_correct_positions"]
                cells.append(text + f"\n\n{sum(positions)}/3 class pairs correct · " + " / ".join("✓" if x else "×" for x in positions))
            else:
                judgment = example[model][field + "_judgment"]
                label = judgment["label"] if judgment else "source excluded"
                cells.append(f"[{label}] {value or '[missing]'}\n\n" + (judgment["justification"] if judgment else "Not scored under the frozen source-quality mask."))
        values.append([field.upper(), reference, *cells])
    return values


if __name__ == "__main__":
    main()
