# F-2: personal adaptation with other speakers mixed into every epoch

Status: designed 9 October 2026, smoke checks only on 13 October 2026. Two seed-17 searches on 27 to 28 October 2026 (stamps `1027-1530` and `1028-rest`) didn't finish: the old clip-budget check failed trials, then a mid-study pull stopped the second. Completed under stamp `1029-night` (29 October 16:04 to 30 October 10:08, lab clock) with the fixed budget check, no failures, and no prediction written first. Dev chose `severe50`, which was worse than the control on test at every seed. Read 30 October from branch `lab-results-1030-100827-454341`. Script: `scripts/asr/speaker_replay.sh`, shared coordinator `long_runs.py`.

## Idea

Lab 11 (severe-speaker pretraining, then personal training, in consecutive phases) scored 58.4%, behind the adapters. Here other voices stay in the mix during personal adaptation, to see if that keeps useful acoustic distinctions. Replay is just a proposed mechanism; nothing earlier established it.

## Setup

All four recipes train the bottom eight encoder blocks for 100 epochs at 1e-5, freeze the prediction network and joint, use stock-form targets, keep batch-norm statistics, average the last four epochs, and share speed perturbation/SpecAugment and effective batch 16. N is the number of personal training clips no longer than 25 seconds:

| Recipe | Personal uses / epoch | Distinct other clips / epoch | Total uses / epoch |
|---|---:|---:|---:|
| control | 4N | 0 | 4N |
| personal75 | 3N | N | 4N |
| personal50 | 2N | 2N | 4N |
| severe50 | 2N | 2N, other severe speakers only | 4N |

Total clip uses, batches and optimizer updates are fixed. Audio minutes, phonetic coverage and personal exposure change with replay, on purpose. The other clips are a fixed, nested, speaker-balanced subset of `loso-<speaker>/train.jsonl` sampled without replacement (subset seed 17), the same across epochs and optimization seeds, so not every other-speaker recording is used.

The builder writes separate exports under `data/public/torgo/export/long-replay-<stamp>/` and leaves the original manifests and audio alone. It excludes the target speaker, the target's normalized dev/test prompts, and any audio identical to a target held-out clip. Only other speakers' training clips qualify. Each export has the target's original dev clips and no test manifest. Test metadata is only used for exclusion and integrity checks. Source and copied-file hashes are recorded, and if NeMo's loaded budget doesn't match the planned 4N uses the run stops.

## Plan

Train M01/M02/M04 at seed 17, pick one recipe by pooled personal dev errors, then repeat it and the control at seeds 18/19 and test only those finalists. That's at most 24 full runs plus four two-step smoke runs. The control is matched to replay on purpose and isn't the best historical adapter, so beating it doesn't prove a better deployment model; compare the WER with the adapter results.

My guesses before the run: 75% personal / 25% other has a better chance than 50/50 of keeping distinctions without diluting adaptation, and severe-only replay may transfer worse than mixed. Uncertain, no sub-35% result promised, and no prediction recorded.

```bash
bash scripts/asr/speaker_replay.sh 1009-replay --dry-run
bash scripts/asr/speaker_replay.sh 1009-replay
```

Run from the lab checkout's `app/`. Missing other-speaker exports get prepared with the existing corpus setup, keeping past manifests. If too few eligible severe clips are left the script fails clearly and doesn't duplicate clips.

Results go to `reports/f2-speaker-replay/<stamp>/`, models to `models/parakeet-ft-long-replay-<stamp>/`, live log to `logs/lab-long-replay-<stamp>.txt`. Selection, confirmation, cleanup, resume, paired intervals and the synthetic battery follow the [long-adapter protocol](../f2-long-adapter/notes.md).

A real gain should be over 4.5 percentage points, repeat across seeds and help at least two speakers. Replay is rejected if it hurts the target speaker or only helps single words while sentences get worse. Small dev sets, repeated looks at the test set and untested Pi/int8 parity remain limits.

## Results

### Smoke checks, 13 October 2026

Lab RTX 4000 Ada, read 14 October. The four two-step smoke runs on M02 passed (control, personal75, personal50, severe50: dev WER 59.4%, 59.4%, 59.4%, 56.5% after two optimizer steps, plumbing only). The main run stopped on 13 Oct at 08:46, before control seed 17, because free disk was under the 14 GiB guard (13 GB free of 150 GB; queue output at `9c2c21d:app/logs/lab-queue-1012-pending.out`). A retry at 14:37 with 30 GB free refused `--resume` since the saved fingerprint no longer matched the code, so it needed a new stamp. No recognition results. Sources: replay log (`logs/lab-long-replay-1012-pending.txt`).

### First seed-17 search, 27 to 28 October 2026

Stamp `1027-1530`, my lab run, dev only. Sources `logs/lab-long-replay-1027-1530.txt`, `logs/lab-1027-1530-speaker-replay.txt`.

All twelve trials (four recipes, three speakers, 100 epochs, 25 to 49 minutes each) trained fully. Four were then marked failed because NeMo loaded one or two fewer clip uses per epoch than 4N (`personal75` M02 623 of 624, `personal50` M01 287 of 288, M02 622 of 624, M04 327 of 328), probably clips under the loader's 0.1 s minimum. An incomplete search picks no winner, but the dev scores are complete. Wrong of 112 dev words (M01 18, M02 69, M04 25):

| Recipe | M01 | M02 | M04 | Total | Dev WER |
|---|---:|---:|---:|---:|---:|
| control (personal only) | 4 | 22 | 10 | 36 | 32.1% |
| `personal75` (75/25) | 5 | 19 | 12 | 36 | 32.1% |
| `personal50` (50/50) | 4 | 19 | 12 | 35 | 31.3% |
| `severe50` (severe speakers only, 50/50) | 5 | 18 | 9 | 32 | 28.6% |

`severe50`'s 4-word "current leader" margin is noise for 112 words at one seed. The confirmation phase (about 18 trainings, 10 hours) never started.

### Second seed-17 search, 28 October 2026

16:34 to 20:07, stamp `1028-rest`, my lab run, dev only, read 29 October. Sources `logs/lab-long-replay-1028-rest.txt`, `logs/lab-1028-rest-speaker-replay.txt`, `reports/f2-speaker-replay/1028-rest/failures.jsonl` on branch `lab-results-1028-200718-136772`.

PR 8 had loosened the budget check that day (skip clips under 0.1 s, allow up to max(2, 0.5%) dropped clip uses), but the lab checkout hadn't pulled it, so this ran with the old check.

| Trial | M01 | M02 | M04 |
|---|---:|---:|---:|
| control | 4 | 22 | 10 |
| `personal75` | 5 | failed (623 of 624 clip uses) | 12 |
| `personal50` | not started | not started | not started |
| `severe50` | not started | not started | not started |

Each scored trial matched its `1027-1530` count exactly (control 4, 22, 10; `personal75` M01 5, M04 12), so seed-17 GPU retraining is reproducible. `personal75` M02 trained 49 minutes and failed again. PR 8 got pulled at 19:46 while `personal75` M04 was training. It finished at 20:07 and the other six stopped with "Code changed during the experiment: scripts/asr/long_runs.py; use a new stamp", since the study checks the code fingerprint before each trial. Any pull touching `long_runs.py` kills a running long study. About 3.5 hours of GPU time for nothing new except reproducibility. A rerun needed a new stamp (or `--restart-changed`): about 7 hours of search, then about 10 of confirmation.

### Rerun plan, 29 October 2026

Stamp `1029-night`, queued after lab test 34 with `python lab.py queue positions --stamp 1029-night --detach`. Fresh study with the PR 8 check: the 12-trial seed-17 search (about 7 hours), then the dev pick and control at seeds 18 and 19 and all finalists on test (about 10 hours). No pulls touching `long_runs.py` on the lab computer until it's done.

Numbers recorded beforehand, for comparison:

- Dev at seed 17 (112 words): control 36, `personal75` 36, `personal50` 35, `severe50` 32. Lowest pooled dev wins, the control wins ties, so `severe50` gets picked if these repeat. All five trials scored twice had repeated exactly.
- The control (bottom 8 blocks, head frozen, 1e-5, 100 epochs at 4N clip uses) isn't any recorded test system. Closest is the acoustic-only study's `b8-frozen-high-lr` (same recipe at N uses per epoch), 50 to 51 dev wrong. Its chosen `b24-frozen-high-lr` got 40 on dev and 191.7 of 425 on test.
- Test (wrong of 425): stage P 178.0, acoustic-only 191.7, adapter on stock 213.0, stock 323. Bar 148.

No prediction was written before the `1029-night` queue started, and none added after.

### Complete study, 29 to 30 October 2026

Stamp `1029-night`, my lab run. Sources `logs/lab-long-replay-1029-night.txt`, `logs/lab-1029-night-speaker-replay.txt`, `logs/pool_table-20261030-075331-943184.log`, `-113030-081075.log`, `-150807-761825.log` and `logs/battery-20261030-150826-501098.log` on branch `lab-results-1030-100827-454341`.

Mixing in other speakers made personal training slightly worse, and the dev pick reversed on test. All twelve search trials passed the budget check. Dev wrong at seed 17 (of 112; M01 18, M02 69, M04 25):

| Recipe | M01 | M02 | M04 | Total |
|---|---:|---:|---:|---:|
| control (personal only, 4N uses) | 4 | 22 | 10 | 36 |
| `personal75` (75/25) | 5 | 17 | 12 | 34 |
| `personal50` (50/50) | 5 | 17 | 12 | 34 |
| `severe50` (severe speakers only, 50/50) | 5 | 18 | 9 | 32 |

Control and `severe50` repeated the first search exactly, and `personal75` and `personal50` (never cleanly scored before) got 34. The rule picked `severe50`, so it and the control were trained at seeds 17, 18 and 19 for test. Test wrong of 425 (M01 of 169, M02 of 95, M04 of 161; intervals resample prompts):

| Recipe and seed | M01 | M02 | M04 | Pooled | WER | Sentences (of 303) |
|---|---:|---:|---:|---:|---:|---:|
| stock | 131 | 62 | 130 | 323 | 76.0% | 188 |
| control, 17 | 77 | 34 | 72 | 183 | 43.1% | 95 |
| control, 18 | 81 | 34 | 74 | 189 | 44.5% | 102 |
| control, 19 | 80 | 35 | 73 | 188 | 44.2% | 98 |
| control, mean | 79.3 | 34.3 | 73.0 | 186.7 | 43.9% | 98.3 (32.4%) |
| `severe50`, 17 | 83 | 41 | 78 | 202 | 47.5% | 107 |
| `severe50`, 18 | 83 | 40 | 76 | 199 | 46.8% | 104 |
| `severe50`, 19 | 87 | 40 | 78 | 205 | 48.2% | 111 |
| `severe50`, mean | 84.3 | 40.3 | 77.3 | 202.0 | 47.5% | 107.3 (35.4%) |

`severe50` is worse at every seed and on every speaker, 15.3 words (3.6 points) on the mean. That's under the 19-word rule, so not an established loss, but the pooled interval excludes zero at seed 17 (-8.7 to -0.2 points) and 19 (-7.0 to -0.9), not at 18 (-6.1 to +1.2). By the pre-run reading ("reject replay if it hurts the target speaker"), replay is rejected. As in the long-adapter study, a 4-of-112 dev lead didn't hold on test.

The control is the best recorded recipe with no pretraining on other speakers: 186.7, against acoustic-only 191.7 (5 words, noise), adapter on stock 213.0 and stage P 178.0 (8.7 lower, inside the rule, which doesn't make them equal). On sentences it's under the 35% line (106 of 303) at every seed. Its only difference from the acoustic-only `b8-frozen-high-lr` (50 dev wrong at one use per epoch) is the 4x repetition per epoch.

Battery (140 synthetic non-speech clips per model): control models wrote words on 2 to 6 clips, `severe50` on 3 to 7, stock on 3, totals over the nine models 36 and 45 of 1,260. Trials took about 25 to 50 minutes each, about 18 hours of GPU in all, leaving 23.9 GB free.

Interpretation (30 October 2026, after the results): replay is closed. Other speakers' clips, even just the severe ones, didn't help a personal model on these three speakers. The personal-only control at 4N uses per epoch is a decent cheap single-speaker recipe, but it doesn't beat stage P.
