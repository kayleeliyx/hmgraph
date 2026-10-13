# E1: online why-recognition at annotated action end

E1 evaluates why-recognition at the annotation's action-end time while allowing
the model to inspect any past interval in `[0, t]`. It begins with 16 frames from
the fixed five seconds ending at `t`, then permits at most three model-selected
16-frame reads. Each view is transported as four chronological 2×2 contact sheets
whose tiles carry source timestamps, keeping the maximum request below the model
endpoint's 50-image limit while retaining all earlier frames. The annotation's
start boundary, duration, gold answer,
narrations, and purpose annotations never enter a model request.

The experiment reads the immutable E0 dev100 split and E0 C1 predictions. It
does not alter E0 and does not invoke the W/L1/graph pipeline.

```bash
bash scripts/run_e1_online_why.sh --stage prepare
bash scripts/run_e1_online_why.sh --stage all --limit 2
bash scripts/run_e1_online_why.sh --stage all
bash scripts/run_e1_online_why.sh --stage report
```

The exact E0 model endpoint is used. Supply its key only through
`ACTGROUND_E0_API_KEY`.
