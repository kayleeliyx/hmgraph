# HD-EPIC field evaluation v1

Saved outputs only. Action scores use a train-only heuristic mapper. Semantic scores are provisional; pending judgments produce no completed rate. No human review or visualization was performed.

| Model / task | N | WHY correct | GOAL correct | Current pair | NEXT 1 pair | NEXT 2 pair | NEXT 3 pair | NEXT mean pair | NEXT OSA distance ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base / explain | 228 | 43.36% | 0.88% | — | 13.60% | 5.70% | 5.70% | 8.33% | 0.9167 |
| epoch1 / explain | 228 | 33.19% | 3.51% | — | 22.37% | 7.46% | 4.39% | 11.40% | 0.8787 |
| epoch2 / explain | 228 | 35.84% | 4.82% | — | 21.49% | 5.26% | 3.95% | 10.23% | 0.8874 |

WHY retains the Codex narrated-purpose rubric; Claude contributed action-mapping heuristics. GOAL measures active annotated activity at its original scope. Identical references can be resolved directly; nonidentical wording is never automatically marked wrong.

NEXT positions 1–3 are primary. Full ordered sequence, verb/noun scores, classic Levenshtein and restricted Damerau/OSA are in summary.json. One forecast of three class actions is not INSIGHT's best-of-five twenty-action benchmark.

Every model's intended cohort retains absent/failed predictions. Missing gold components affect only their own denominators. WHY source exclusions are shared. GOAL annotation absence is not a negative label.

Semantic decisions still needed: 0 unique packets. Supply hash-bound automated judgments and rerun; visualization and human calibration remain deferred.
