# F-2: longer adapter training and the model's text form

Status: designed 9 October 2026, run and completed 12 October 2026 on the lab computer (stamp `1012-pending`), results read 14 October. Script: `scripts/asr/long_adapter.sh`, with the shared coordinator `long_runs.py`.

## Idea

The rebuilt adapter on stock scored 208 errors of 425 words (48.9%), and whole-block training was 68.5%. The adapter's usual schedule is only 15 epochs on 72–156 personal training clips, so maybe it stops too early. Also, asking a frozen prediction network to output lowercase text with no punctuation might make adaptation harder. The `stock` form is a formatting heuristic (initial capital and terminal punctuation), not a transcript inferred from the test audio. Historical results are from `logs/pool_table-20261007-231043-940765.log`.

## Setup

Seven recipes, all starting from stock Parakeet-TDT-0.6B-v2 with the adapter after block 8:

| Recipe | Epochs | Peak learning rate | Target form |
|---|---:|---:|---|
| control | 15 | 3e-4 | scoring |
| scoring-75 | 75 | 3e-4 | scoring |
| scoring-200 | 200 | 3e-4 | scoring |
| stock-75 | 75 | 3e-4 | stock |
| stock-200 | 200 | 3e-4 | stock |
| stock-200-low-lr | 200 | 1e-4 | stock |
| stock-200-high-lr | 200 | 1e-3 | stock |

Every full run finishes its fixed epoch budget and averages the final four epochs, with no dev checkpoint picking. Speed perturbation, SpecAugment, frozen batch-norm statistics, effective batch 16 and cosine decay stay the same. A longer budget also stretches the cosine schedule, so these are whole recipes, not extra epochs at the same rates.

## Plan

The search trains all three speakers at seed 17 and picks one recipe on pooled dev error counts, with the control winning ties. That recipe and the control are then repeated at seeds 18/19. Neither the confirmation nor the test scores can change the pick. Only these two finalists are decoded on the same historical M01/M02/M04 test split, at all three seeds. That's at most 33 full training runs plus three two-step smoke runs. I had no measured wall-time for this schedule on the lab GPU.

Pre-run hypotheses: keeping the model's output form is worth testing, and 75 epochs may help while 200 may overfit. I didn't assume more epochs alone would reach 35%. No prediction was recorded. This was a new plan, so no historical prediction was changed.

## Running it

From the lab checkout's `app/`, after copying the code over:

```bash
bash scripts/asr/long_adapter.sh 1009-adapter --dry-run
bash scripts/asr/long_adapter.sh 1009-adapter
```

By default it runs smoke, search, confirmation, test and a 140-clip synthetic non-speech battery. `--phase smoke` runs only the short checks, and `--phase search` stops after dev selection. `--resume` reuses finished work under the same stamp, but only if code, environments, manifests and source audio haven't changed. Incomplete training is never overwritten. `--batch 4 --grad-accum 4` keeps the 16-clip effective batch if the GPU has room. Don't run the scripts at the same time.

Results go in `reports/f2-long-adapter/<stamp>/`, models in `models/parakeet-ft-long-adapter-<stamp>/`, and the live child-process log in `logs/lab-long-adapter-<stamp>.txt`. `selection.json` records the dev decision, `confirmation-dev.json` has all three dev seeds, and `summary.json` has test means and ranges. Each test table includes prompt-clustered paired intervals, and the evaluation folders split words and sentences.

By default, intermediate checkpoints and losing search models are deleted after scoring. Curves, provenance, hypotheses, the current dev leader and the control are kept. `--keep-all-models` turns the cleanup off. `lab_space.py pack` includes the report and logs but not the final model weights.

A useful improvement should repeat in all three pooled seed comparisons and help at least two speakers. The lab's threshold for a practical change is 4.5 percentage points, and each paired interval gets checked too. 35% means at most 148 errors on 425 reference words. The test clips were used in earlier experiments, so this is exploratory. The battery measures output on generated non-speech, not live-room triggers, and Pi/int8 behaviour needs its own check.

## Results

Queue run `1012-pending` on 12 October 2026, on the lab RTX 4000 Ada, read 14 October. It took about 6.5 hours (14:21 to about 20:50 on 12 Oct): 3 smoke runs, 21 search runs and 12 confirmation runs, then test decodes and the battery.

Seed 17 search, wrong words out of the 112 dev words (M01 18, M02 69, M04 25):

| Recipe | Dev wrong (seed 17) | Recipe | Dev wrong (seed 17) |
|---|---:|---|---:|
| control | 46 | stock-75 | 44 |
| scoring-75 | 45 | stock-200 | 37 |
| scoring-200 | 44 | stock-200-low-lr | 40 |
| | | stock-200-high-lr | 57 |

The rule picked stock-200 (200 epochs, stock-form targets, 3e-4) by 9 dev words. Confirmation dev at seeds 18/19 didn't repeat that: control got 43 and 44 wrong, stock-200 got 44 and 45. With only 112 dev words and 7 recipes, the seed-17 lead was a lucky pick.

Test, three speakers, 425 reference words (stock 323 wrong, 76.0%):

| Seed | control (15 epochs) | stock-200 | stock-200 vs control (95% CI, prompts) |
|---|---:|---:|---|
| 17 | 208 (48.9%) | 221 (52.0%) | -3.1 (-9.2 to +3.1), unclear |
| 18 | 219 (51.5%) | 217 (51.1%) | +0.5 (-4.7 to +5.5), unclear |
| 19 | 212 (49.9%) | 230 (54.1%) | -4.2 (-10.9 to +2.4), unclear |
| Mean | 213 (50.1%) | 222.7 (52.4%) | |

By speaker, stock-200 is better on M04 in all three seeds (+3.1, +9.9, +3.7 points) and worse on M01 and M02 in all three (M01 -7.1, -2.4, -12.4; M02 -6.3, -10.5, -3.2). No seed shows a real difference, and the 4.5-point threshold isn't met either way. Longer training and the model's own text form didn't help overall. The control at seed 17 matches the 7 Oct rebuild exactly (208 wrong), and the three seeds span 11 words (2.6 points).

On the synthetic non-speech battery (seed 3000, 140 clips), stock writes words on 3 of 140. Control adapters write words on 1, 3 and 2 clips for M01 (seeds 17/18/19), 20, 5 and 13 for M02, and 0, 1 and 4 for M04. stock-200 does it on 4, 1 and 4 (M01), 8, 9 and 8 (M02), and 6, 12 and 14 (M04). So the adapters do hallucinate words on non-speech, much more on M02 and M04, and it varies by seed.

Nothing reaches the bar. The best single run is 208 wrong, 60 above 148. No prediction was recorded and that field stays blank.

Sentences only (secondary score, added 21 October 2026). Same decodes, scored on the sentence clips alone without the single-word clips. That's 303 of the 425 reference words, and 35% or lower means 106 wrong or fewer. Means over the listed seeds, lower is better. Read from each run's `results.json` with `scripts/asr/sentence_rescore.py`. The full ranking is in `docs/measurements.md` under "Secondary score: sentences only".

| Recipe (run name) | Seeds | Sentence words wrong (of 303) | Sentence WER | M01 / M02 / M04 | All-clip WER |
| --- | ---: | ---: | ---: | --- | ---: |
| `f2-long-adapter/1012-pending/control/` | 3 | 117.7 | 38.8% | 39.9 / 35.3 / 39.8 | 50.1% |
| `f2-long-adapter/1012-pending/stock-200/` | 3 | 121.0 | 39.9% | 47.0 / 39.7 / 32.4 | 52.4% |
| `stock 0.6B v2 (reference)` | 1 | 188.0 | 62.0% | 61.5 / 47.1 / 71.7 | 76.0% |

Sources: the long-adapter log (`logs/lab-long-adapter-1012-pending.txt`) (test tables near the end, battery at the very end). `summary.json`, `selection.json`, `battery.json` and `test-seed-<17|18|19>.json` are in this checkout under `reports/f2-long-adapter/1012-pending/`. They arrived with commit 538487e on 15 October, and their seed scores match the tables above.
