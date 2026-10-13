# E0: HD-EPIC Why Recognition Pilot

E0 asks whether general background facts change multiple-choice why-recognition accuracy when the official HD-EPIC action clip is available. It uses one saved, participant-balanced set of 100 development questions and four fixed conditions:

- C0: question and displayed choices;
- C1: C0 plus 16 timestamp-bounded action-clip frames;
- C2: C1 plus answer-blind, self-generated general kitchen facts;
- C3: C1 plus answer-blind facts selected from public Wikipedia reference pages.

The same deterministic option permutation is used for every condition. Scoring labels are stored separately from inference-facing records. C1–C3 use the same cached ordered image payload, and C0 contains no image payload. C2 generation sees only the bracketed target action and a fixed general kitchen context. C3 queries and ranking use those same fields and retain URL/revision/span provenance.

The answering and C2 generation client uses the OpenAI-compatible Chat Completions interface at `http://messi.eecs.umich.edu:8080/v1`, with the exact model identifier `azure/gpt-5.6-sol`. Both operations use medium reasoning effort. Supply the endpoint key through `ACTGROUND_E0_API_KEY`. The credential value is never copied into E0 source, YAML, logs, or experiment artifacts.

Run from any directory:

```bash
bash /data/yaxuanli/ActGround/scripts/run_e0.sh --stage all
bash /data/yaxuanli/ActGround/scripts/run_e0.sh --stage prepare
bash /data/yaxuanli/ActGround/scripts/run_e0.sh --stage all --limit 2
bash /data/yaxuanli/ActGround/scripts/run_e0.sh --stage run --conditions C0 C1 C2 C3
bash /data/yaxuanli/ActGround/scripts/run_e0.sh --stage report
bash /data/yaxuanli/ActGround/scripts/run_e0.sh --stage all --dry-run
```

`--limit 2` always means the first two records in the persisted `dev100.json`; it never resamples or changes the experiment denominator. Completed frame, knowledge, and answer operations are content-addressed and reused. The command refuses a materially incompatible configuration once model artifacts exist. A failed provider setup with zero predictions and zero C2 records may be replaced in place, with its resolved metadata retained in `protocol_history.json`.

The experiment does not run the W/L1/graph pipeline, extract full videos, use memory/hierarchy/recipe inputs, or test future context or reveal time. Reports are exploratory and do not claim statistical significance or a completed human audit.

## C3 external-retrieval development revision

`configs/e0_c3_v2.yaml` defines the isolated retrieval repair at
`runs/e0/hd_epic_why_dev100_c3_v2`. It loads the original `dev100.json`,
reserved-ID file, 1,572 JPEGs, frame records, and C0-C2/C3-v1 predictions as
read-only reference artifacts. It never resamples, decodes video, copies the
frames, or reruns C0-C2.

The revised Wikipedia retriever parses action/object/container structure, runs
the action-and-kitchen query before the broad concept query, resolves generic
kitchen word senses, and uses bounded MediaWiki generator searches with
`exintro` and `exchars=1200`. Raw errors, warnings, continuation, and explicit
page acquisition outcomes are cached. Disambiguation pages cannot supply facts.
The deterministic selector retains one to three exact captured source spans;
zero useful facts creates an explicit retrieval abstention. Reports show both
the pure C3-v2 eligible subset and a separately labeled C1-fallback policy.

```bash
bash /data/yaxuanli/ActGround/scripts/run_e0.sh --config /data/yaxuanli/ActGround/configs/e0_c3_v2.yaml --stage all --dry-run
bash /data/yaxuanli/ActGround/scripts/run_e0.sh --config /data/yaxuanli/ActGround/configs/e0_c3_v2.yaml --stage retrieve
bash /data/yaxuanli/ActGround/scripts/run_e0.sh --config /data/yaxuanli/ActGround/configs/e0_c3_v2.yaml --stage run --conditions C3
bash /data/yaxuanli/ActGround/scripts/run_e0.sh --config /data/yaxuanli/ActGround/configs/e0_c3_v2.yaml --stage report
bash /data/yaxuanli/ActGround/scripts/run_e0.sh --config /data/yaxuanli/ActGround/configs/e0_c3_v2.yaml --stage all
```

The endpoint key is still supplied only through `ACTGROUND_E0_API_KEY`. A
cache-only resume and report generation do not require the key.
