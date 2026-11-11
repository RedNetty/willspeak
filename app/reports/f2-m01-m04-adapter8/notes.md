# F-2: M01 and M04, adapter after 8 blocks

Proposal: F-2 (single-speaker fine-tuning, §5.2 / H3) and the F-1 comparison. Same adapter as the [M02 runs](../f2-m02-adapter8/notes.md), on the other two severe speakers with two or more sessions.

Status: run 1 Oct 2026 on Yack, 12:13–12:20. Prediction entered before the runs, results taken from the saved reports. See also the [Report Context](../Report%20Context.md) and the [Measurement guide](../../../docs/measurements.md).

## Idea

Train the adapter after block 7 on M01's earlier sessions and, separately, on M04's, then score each on that speaker's last session. With only about 6 minutes of speech per speaker, does a 1-million-number adapter beat retraining 8 whole blocks, and does it get close to the F-1 model that never heard the speaker?

## Setup

F-2 trains on one speaker's earlier sessions and tests on his last session, with no prompt shared between them. F-1 is the shared model: blocks 0-7 retrained on about 110 minutes of other speakers, never hearing the test speaker. The adapter is the layer after block 7 from the M02 runs, `h + W h + b`. It starts at zero, so the model starts as stock, and only its 1,049,600 numbers train. A fixed schedule (`--select last`) trains a set number of epochs and averages the last 4 checkpoints, so dev picks nothing. The adapter has no int8 export, so everything here is a NeMo score, decoded in NeMo before any export.

The test clips are each speaker's last session, the same ones the F-2 runs used:

| Speaker | Train clips (minutes) | Dev clips | Test clips | Test reference words | One word is |
| --- | ---: | ---: | ---: | ---: | ---: |
| M01 | 72 (5.7) | 4 | 65 | 169 | 0.59 points |
| M04 | 82 (6.1) | 9 | 64 | 161 | 0.62 points |
| M02 (for contrast) | 156 (9.0) | 24 | 37 | 95 | 1.05 points |

M01 and M04 have fewer training clips than M02 but bigger test sets, so their intervals are narrower. The models, all on the same test clips:

| | M01 | M04 |
| --- | ---: | ---: |
| Stock, NeMo | 77.51% (131 word errors) | 80.75% (130) |
| F-2 bottom-8 model, fixed 5 epochs, NeMo | 74.56% (126) | 82.61% (133) |
| F-2 bottom-8 model, fixed 5 epochs, int8 | 72.78% | 83.23% |
| F-1 shared model, int8 | 58.58% (99) | 65.22% (105) |
| F-1 shared model, NeMo | 57.40% (97 word errors) | 65.22% (105) |
| Adapter, NeMo | 49.70% (84 word errors) | 49.07% (79) |

F-1 had the best recorded score on these clips, but only in int8, and int8 and NeMo don't line up here. Stock on M01 gets 70.41% in int8 and 77.51% in NeMo, and on M04 88.20% and 80.75%. So I decoded the F-1 models in NeMo too.

Going in, the F-2 bottom-8 models were 14.20 (M01) and 18.01 (M04) points behind F-1 in int8 after only 25 and 30 optimizer steps, and the [F-2 note](../f2-severe-bottom8-5epochs/notes.md) couldn't tell too little data from too little training. On M02 the adapter scored 48.42% on all three seeds (17, 18 and 19), 16.84 points better than stock and 4 words behind bottom-8's 44.21%. Its dev WER (30 to 32%) looked much better than its test score, maybe because M02's dev clips share sessions with its training clips, but that's a guess. M01's and M04's dev sets (4 and 9 clips) are too small to pick a checkpoint on, so the per-epoch dev WER is just for the record.

Why 15 epochs: an update uses 16 clips (batch 2, 8 accumulated), and Lightning also updates on the last partial group each epoch. That gives M01 `ceil(36 / 8) = 5` updates per epoch and M04 `ceil(41 / 8) = 6`, so 75 and 90 in total. My first estimate of about 67 and 77 forgot the partial group. On M02 the best dev epochs came at 60 and 80 updates (seed 17 at zero-based epoch 5, seed 19 at epoch 7, 10 updates per epoch), so this is a similar update budget. F-2 used 5 epochs. Both recipes spend their first 10 updates on warm-up.

## Plan

1. Check that the earlier outputs exist in the WSL tree (first command block).
2. For M01 and M04, train the adapter on the earlier sessions: `--adapter-after-blocks 8`, learning rate 3e-4, batch 2 with 8 accumulated, 10 warmup steps, frozen batch norm, normalised transcripts, seed 17, `--epochs 15 --select last --no-keep-stock` (averages epochs 11 to 14).
3. Decode the test clips in NeMo with the adapter, and with that speaker's F-1 model.
4. Score with `evaluate.py` and compare with `ci.py`: adapter against stock, bottom-8 and F-1, plus bottom-8 against F-1 in NeMo.

Two differences from the bottom-8 runs can't be avoided. Only the adapter trains, where F-2 trained blocks 0-7, the prediction network and the joint, and the learning rate is 3e-4 instead of 2e-5. I also chose 15 epochs instead of 5. So a win over bottom-8 can't be pinned on the adapter alone.

Rules set before the run: a difference counts only if the `ci.py` interval is wholly on one side of zero. Four points (about 7 words on M01, 6 on M04) is my run-to-run noise guide, since two runs of one recipe landed 3 to 4 points apart on single speakers on 30 Sep. Against bottom-8, 4 or more points better means the adapter uses a few minutes of data better, within 4 is a tie, and 4 or more worse means it didn't help. Against F-1, within 4 points or better means a personal adapter on about 6 minutes reaches what other speakers' speech gives. More than 4 behind on both would point to an adapter on top of F-1 (shared plus personal, closest to the device design), which isn't set up.

## Why

At 5.7 and 6.1 minutes the personal bottom-8 models lost to F-1 by 14 to 18 points, and a client will also start with only a few minutes of speech. The adapter trains about 1/201 of the numbers, so it might cope better with that little data.

## Limits

This shows, for M01 and M04 in NeMo, how the adapter compares with stock, bottom-8 and F-1 by the margins above. It's one run per speaker, and M02's zero seed spread in word errors may not carry over. It can't separate the adapter from the extra updates (15 epochs against 5). The 65 and 64 clips don't allow fine margins; the F-2 against F-1 intervals here were about 16 points wide. Nothing here covers the Pi or an adapter on top of F-1.

## Reading and Video

Found by search and not opened, so I'm judging them from the title only.

- [Parameter-efficient Dysarthric Speech Recognition Using Adapter Fusion and Householder Transformation](https://arxiv.org/abs/2306.07090): research paper on adapters for dysarthric speech.
- [Personalized Fine-Tuning with Controllable Synthetic Speech from LLM-Generated Transcripts for Dysarthric Speech Recognition](https://arxiv.org/abs/2505.12991): research paper on personal fine-tuning with little recorded speech.
- The [videos in the first adapter note](../f2-m02-adapter8/notes.md#videos) explain the `h + W h + b` step and why a layer that starts at zero is safe.

## Results

Run 1 Oct 2026, 12:13 to 12:20, on Yack, after `git pull` to commit 9aafe98. NeMo scores on the last-session test clips (M01: 65 clips, 169 words; M04: 64 clips, 161 words), word errors in brackets. Positive Delta means the adapter is better.

| | M01 | M04 |
| --- | ---: | ---: |
| Stock | 77.51% (131) | 80.75% (130) |
| F-2 bottom-8 model, fixed 5 epochs | 74.56% (126) | 82.61% (133) |
| F-1 shared model | 57.40% (97) | 65.22% (105) |
| Adapter after block 7 | 49.70% (84) | 49.07% (79) |
| Delta against stock (95% CI) | 27.81 (16.98 to 39.72) | 31.68 (16.96 to 48.22) |
| Delta against the F-2 bottom-8 model | 24.85 (13.77 to 37.32) | 33.54 (18.71 to 52.86) |
| Delta against the F-1 model | 7.69 (-1.16 to 16.77) | 16.15 (4.29 to 28.30) |

The adapter did much better than I predicted. I'd said 65 to 75% (M01) and 78 to 85% (M04) with no win over F-1, and the scores came in 15.3 and 28.9 points under those ranges. The mismatch I'd flagged before the run held too: F-1's NeMo scores (57.40% and 65.22%) are close to its int8 ones, so my ranges never fit a delta near 0. Against stock and bottom-8 it's far more than 4 points better, with intervals clearing zero. It's ahead of F-1 on both speakers too, but only M04's interval clears zero. On M01 it includes zero and the CER is identical to F-1's (35.24%).

Most of the gain over stock is fewer inserted and deleted words. On M04 the 51 fewer errors are 29 inserted, 16 deleted and 6 wrong. On M01 the 47 are 24 wrong, 20 inserted and 3 deleted. Single-word clips are still mostly wrong (87.2% and 75.0%), and the full breakdown is in Full Results. The adapter runs made 75 and 90 updates at a higher learning rate, so the gap to bottom-8 doesn't isolate the adapter. There was also one seed per speaker and no int8 export for the Pi.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

Written on 1 Oct 2026 at 12:05 (a first answer at 11:58 had no numbers) and entered here before any run of this test. Stock scores 77.51% on M01's 65 test clips and 80.75% on M04's 64 in NeMo. The F-2 bottom-8 model scores 74.56% and 82.61%. The F-1 model's int8 scores are 58.58% and 65.22%. Its NeMo scores aren't known yet.

1. The adapter's NeMo score on M01 and on M04 will be:

I believe M01 will likely be 65-75% WER and M02 78-85% based on the F-2 Bottom Eight,

[Note added before the run: I'm reading the second range as M04's, since M02 isn't in this test, and both ranges as NeMo WER. The 11:58 answer expected "similar results that we saw on M02 margin wise, and maybe only slightly better due to data", and the numbers above replaced it. In words, 65 to 75% is 110 to 127 of M01's 169 reference words wrong, and 78 to 85% is 126 to 137 of M04's 161. The M01 range runs from 2.5 to 12.5 points better than stock, with the F-2 bottom-8 model (74.56%) near its top. The M04 range contains both stock (80.75%) and the F-2 bottom-8 model (82.61%), so it predicts no measurable gain over either on M04.]

2. For each speaker, the adapter will beat the F-1 shared model, match it, or lose to it:

I do not believe it will beat it from the M02's test on the same adapter, It might have a delta near 0, but I likely don't see it beating it.

3. The reason I expect that is:

Due to our previous runs and tests the data doesn't point towards it being a likely success

[Note added before the run: answers 1 and 2 only agree if the F-1 model's NeMo scores are well above its int8 scores. Against the F-1 int8 scores (58.58% and 65.22%), the answer-1 ranges sit about 6 to 16 points behind F-1 on M01 and about 13 to 20 points behind on M04. So `ci.py` would print a Delta (positive means the adapter is better) of about -6 to -16 and -13 to -20, which isn't near 0. Whether "near 0" holds depends on the F-1 NeMo scores, which this test decodes. For a Delta near 0 the adapter would have to do about 14 to 18 points better against F-1 than the F-2 bottom-8 models did in int8, and answer 1 doesn't predict that.]

### Commands and Setup

Run from `app/` in the WSL tree on Yack after `git pull`, with the GPU free and Windows apps closed. Training and decoding run in `~/venvs/willspeak-train`, and `evaluate.py` and `ci.py` in `app/.venv`. The data is `data/public/torgo/export/M01` and `data/public/torgo/export/M04`, the same test clips as the F-2 runs. Each training run should take about 2 to 3 minutes plus model loading. The training and scoring blocks stop at their first error.

Check that the earlier outputs exist (every line should start with `ok`):

```bash
for S in M01 M04; do
  for p in models/parakeet-ft-f1-loso-$S-bottom8/parakeet-ft.nemo \
           reports/finetune-torgo/$S-test-stock-nemo/results.json \
           reports/finetune-torgo/f2-$S-bottom8-5epochs-test-nemo/results.json \
           data/public/torgo/export/$S/test.jsonl; do
    [ -e "$p" ] && echo "ok       $p" || echo "MISSING  $p"
  done
done
```

Train the adapters, then decode the adapters and the F-1 models in NeMo:

```bash
( set -e
source ~/venvs/willspeak-train/bin/activate
for S in M01 M04; do
  python scripts/asr/train.py --export data/public/torgo/export/$S --out models/parakeet-ft-f2-$S-adapter8 --adapter-after-blocks 8 --lr 3e-4 --batch 2 --grad-accum 8 --warmup-steps 10 --epochs 15 --select last --no-keep-stock --freeze-batchnorm --normalize-targets --dev-scoring evaluate --seed 17
  python scripts/asr/nemo_hyp.py --stock --adapter models/parakeet-ft-f2-$S-adapter8/adapter.pt --manifest data/public/torgo/export/$S/test.jsonl --audio-root data/public/torgo/export/$S --out reports/finetune-torgo/f2-$S-adapter8-test-nemo.jsonl
done
for S in M01 M04; do
  python scripts/asr/nemo_hyp.py --nemo models/parakeet-ft-f1-loso-$S-bottom8/parakeet-ft.nemo --manifest data/public/torgo/export/$S/test.jsonl --audio-root data/public/torgo/export/$S --out reports/finetune-torgo/f1-loso-$S-bottom8-on-F2-test-nemo.jsonl
done )
```

Score and compare:

```bash
( set -e
source .venv/bin/activate
for S in M01 M04; do
  M=data/public/torgo/export/$S
  python scripts/asr/evaluate.py --manifest $M/test.jsonl --audio-root $M --hyp reports/finetune-torgo/f2-$S-adapter8-test-nemo.jsonl --group-by kind --out reports/finetune-torgo/f2-$S-adapter8-test-nemo
  python scripts/asr/evaluate.py --manifest $M/test.jsonl --audio-root $M --hyp reports/finetune-torgo/f1-loso-$S-bottom8-on-F2-test-nemo.jsonl --group-by kind --out reports/finetune-torgo/f1-loso-$S-bottom8-on-F2-test-nemo
  echo "== $S adapter against stock"
  python scripts/asr/ci.py --manifest $M/test.jsonl --stock reports/finetune-torgo/$S-test-stock-nemo --finetuned reports/finetune-torgo/f2-$S-adapter8-test-nemo
  echo "== $S adapter against the F-2 bottom-8 model"
  python scripts/asr/ci.py --manifest $M/test.jsonl --stock reports/finetune-torgo/f2-$S-bottom8-5epochs-test-nemo --finetuned reports/finetune-torgo/f2-$S-adapter8-test-nemo
  echo "== $S adapter against the F-1 model"
  python scripts/asr/ci.py --manifest $M/test.jsonl --stock reports/finetune-torgo/f1-loso-$S-bottom8-on-F2-test-nemo --finetuned reports/finetune-torgo/f2-$S-adapter8-test-nemo
  echo "== $S F-2 bottom-8 model against the F-1 model"
  python scripts/asr/ci.py --manifest $M/test.jsonl --stock reports/finetune-torgo/f1-loso-$S-bottom8-on-F2-test-nemo --finetuned reports/finetune-torgo/f2-$S-bottom8-5epochs-test-nemo
done )
```

Before trusting the scores, check the training logs (two newest, M01 then M04):

```bash
for f in $(ls -t logs/train_parakeet-*.log | head -2 | sort); do echo "== $f"; grep -hE "train [0-9]+ clips|parameters exact|fixed schedule|kept epochs|averaged|saved models/parakeet-ft-f2-M0[14]-adapter8" "$f"; done
for S in M01 M04; do echo "== $S curve"; cat models/parakeet-ft-f2-$S-adapter8/curve.csv; done
```

Each log should show `1049600 trainable`, `fixed schedule (--select last): kept epochs 12, 13, 14, 15`, `averaged 4 checkpoints` and a saved `adapter.pt`. The console counts epochs from 1 now, but checkpoint filenames and `curve.csv` still count from 0, so the kept files are epochs 11 to 14. F-2's older logs used zero-based console epochs too. `curve.csv` should list epochs 0 to 14, one row each, with the dev WER for the record. There's no `stock dev WER` line, because `--no-keep-stock` skips it.

This is the first run of `--select last` together with an adapter, so if any of those lines is missing, stop and read the log before decoding anything. In every `ci.py` block a positive Delta means the "Fine-tuned" line is better than the "Stock" line. In the second block "Stock" is the F-2 bottom-8 model, and in the third and fourth it's the F-1 model.

### Full Results

Recorded 1 Oct from the saved NeMo reports and the training `run.json` files. Delta is comparison-model WER minus candidate WER, in percentage points, so positive means the candidate made fewer word errors.

| Speaker | Adapter WER | Word errors | F-1 model WER (NeMo) | Against stock: Delta (95% CI) | Against the F-2 bottom-8 model: Delta (95% CI) | Against the F-1 model: Delta (95% CI) | F-2 bottom-8 model against the F-1 model: Delta (95% CI) |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| M01 | 49.70% | 84 / 169 | 57.40% | 27.81 (16.98 to 39.72) | 24.85 (13.77 to 37.32) | 7.69 (−1.16 to 16.77) | −17.16 (−27.98 to −8.54) |
| M04 | 49.07% | 79 / 161 | 65.22% | 31.68 (16.96 to 48.22) | 33.54 (18.71 to 52.86) | 16.15 (4.29 to 28.30) | −17.39 (−38.71 to −0.66) |

| Speaker | Adapter CER | Adapter PER | Sentence WER | Single-word WER | Updates | Training minutes | Peak GPU reserved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M01 | 35.24% | 33.94% | 35.25% | 87.23% | 75 | 2.0 | 4.94 GB |
| M04 | 39.79% | 40.62% | 38.05% | 75.00% | 90 | 2.3 | 5.51 GB |

Both runs finished the fixed 15 epochs and averaged checkpoints 11 to 14 (zero-based), with no checkpoint chosen on dev. Both beat stock and the shorter F-2 bottom-8 recipe with intervals clearing zero. M04 also beats F-1 on these clips, while M01's 7.69-point lead over F-1 stays inconclusive. M01's CER equals F-1's 35.24% even with fewer word errors. The prediction ranges were above the observed WER on both speakers.

These are whole-recipe comparisons, since the adapter runs use a higher learning rate and three times the epochs of the F-2 bottom-8 runs. They don't isolate architecture, show equivalence when an interval includes zero, measure seed spread on M01/M04, or say anything about Pi performance.

Code: before these runs I updated the WSL tree with `git pull` from af07966 to 9aafe98 (the cleanup of `train.py`, `ci.py` and other files). Commits 167598f (12:21) and b5e0589 (12:26) came after the runs. 167598f changed the release step in `train.py` again, so the dev check of the averaged model now also runs for fixed-schedule runs, and that code hasn't been run yet. Each model folder's `run.json` holds a hash of the `train.py` that made it. The existence check printed `ok` for all eight paths.

Word errors by type, from `results.json` (wrong word / deleted / inserted, out of 169 reference words for M01 and 161 for M04):

| | M01 | M04 |
| --- | ---: | ---: |
| Stock | 93 / 7 / 31 (131) | 63 / 29 / 38 (130) |
| F-2 bottom-8 model | 93 / 6 / 27 (126) | 60 / 30 / 43 (133) |
| F-1 model | 79 / 5 / 13 (97) | 66 / 13 / 26 (105) |
| Adapter | 69 / 4 / 11 (84) | 57 / 13 / 9 (79) |

Error rate by recording kind (sentences / single words). A single-word rate above 100% means inserted words outnumber the reference words:

| | M01 | M04 |
| --- | ---: | ---: |
| Stock | 61.5% / 119.2% | 71.7% / 102.1% |
| F-2 bottom-8 model | 59.8% / 112.8% | 63.7% / 127.1% |
| F-1 model | 45.1% / 89.4% | 54.0% / 91.7% |
| Adapter | 35.2% / 87.2% | 38.0% / 75.0% |

Training, from the two logs (`train_parakeet-20261001-171322-658423.log` and `train_parakeet-20261001-171612-487643.log`): 72 and 82 training clips (5.72 and 6.07 minutes), and dev sets of 4 and 9 clips (18 and 25 reference words). Training loss fell from 5.15 to 1.82 (M01) and from 6.01 to 2.63 (M04) between epoch 1 and epoch 15. Dev WER went from 55.56% to 44.44% on M01, flat at 44.44% from epoch 10, and from 92.00% to 64.00% on M04, flat from epoch 11.

Each log shows `parameters exact: 618875526 total, 1049600 trainable`, `kept epochs 12, 13, 14, 15; dev chose nothing`, `averaged 4 checkpoints` and a saved `adapter.pt` of 4.2 MB. `nemo_hyp.py` reported `Added the adapter ... after encoder blocks 0-7`. No `stock dev WER` line appears, because `--no-keep-stock` skips it.

Scoring and comparisons, as printed (`ci.py` no longer prints a `Check:` line). For each speaker, the "Stock" line in the first three blocks is stock, the F-2 bottom-8 model and the F-1 model in turn, and in the fourth it's the F-1 model. The "Fine-tuned" line is the adapter in the first three and the F-2 bottom-8 model in the fourth:

```text
WER 49.70%  CER 35.24%  PER 33.94%        (M01 adapter)
WER 57.40%  CER 35.24%  PER 35.26%        (M01 F-1 model, NeMo)
== M01 adapter against stock
n = 65
Stock WER 77.51%  (95% CI 63.48% to 94.45%)
Fine-tuned WER 49.70%
Delta 27.81 points  (95% CI 16.98 to 39.72)
== M01 adapter against the F-2 bottom-8 model
n = 65
Stock WER 74.56%  (95% CI 60.25% to 91.14%)
Fine-tuned WER 49.70%
Delta 24.85 points  (95% CI 13.77 to 37.32)
== M01 adapter against the F-1 model
n = 65
Stock WER 57.40%  (95% CI 45.30% to 70.86%)
Fine-tuned WER 49.70%
Delta 7.69 points  (95% CI -1.16 to 16.77)
== M01 F-2 bottom-8 model against the F-1 model
n = 65
Stock WER 57.40%  (95% CI 45.30% to 70.86%)
Fine-tuned WER 74.56%
Delta -17.16 points  (95% CI -27.98 to -8.54)
WER 49.07%  CER 39.79%  PER 40.62%        (M04 adapter)
WER 65.22%  CER 51.04%  PER 50.00%        (M04 F-1 model, NeMo)
== M04 adapter against stock
n = 64
Stock WER 80.75%  (95% CI 65.45% to 97.22%)
Fine-tuned WER 49.07%
Delta 31.68 points  (95% CI 16.96 to 48.22)
== M04 adapter against the F-2 bottom-8 model
n = 64
Stock WER 82.61%  (95% CI 65.84% to 107.25%)
Fine-tuned WER 49.07%
Delta 33.54 points  (95% CI 18.71 to 52.86)
== M04 adapter against the F-1 model
n = 64
Stock WER 65.22%  (95% CI 52.83% to 80.92%)
Fine-tuned WER 49.07%
Delta 16.15 points  (95% CI 4.29 to 28.30)
== M04 F-2 bottom-8 model against the F-1 model
n = 64
Stock WER 65.22%  (95% CI 52.83% to 80.92%)
Fine-tuned WER 82.61%
Delta -17.39 points  (95% CI -38.71 to -0.66)
```

### Conclusions

On M01 and M04 the adapter after block 7, trained on about 6 minutes of each speaker's earlier sessions, scored 49.70% and 49.07% in NeMo on their last sessions. That's 27.81 and 31.68 points better than stock, with intervals that clear zero (16.98 to 39.72 and 16.96 to 48.22).

It was ahead of the F-1 shared model on both speakers. On M04 the lead (16.15 points, interval 4.29 to 28.30) clears zero. On M01 the lead is 7.69 with an interval of -1.16 to 16.77, which includes zero, and the character error rate is identical to F-1's (35.24%). So the M01 word-error comparison is inconclusive. It shows neither an improvement nor equivalence.

It was 24.85 and 33.54 points better than the F-2 bottom-8 models, with intervals clearing zero. Those models trained at a learning rate of 2e-5 for 25 and 30 updates, though, and the adapter trained at 3e-4 for 75 and 90. This compares two recipes, and nothing here says how much of the gap is the adapter and how much is the rate and the number of updates. The F-2 note's reading that F-1's lead came from too little data is weaker now, because the adapter on the same data got ahead of F-1. Whether the bottom-8 recipe was undertrained or overfits a few minutes of data is still open.

On the prediction: I said 65 to 75% (M01) and 78 to 85% (M04), and the scores were 15.3 and 28.9 points below the bottom of those ranges. I said the adapter would likely not beat F-1, and it was ahead on both. My reason, that earlier runs didn't point to success, didn't hold for the adapter at a higher learning rate with more updates.

Under the rules written before the run, the adapter gains over stock, is more than 4 points better than the F-2 bottom-8 models with intervals clearing zero, and is within 4 points of F-1 or better on both speakers. The case that would have sent the next test to an adapter on top of the F-1 model (more than 4 points behind F-1 on both) didn't happen. That test is still open as a way to combine shared and personal training.

Most of the gain over stock is fewer inserted and deleted words. On M04 the 51 fewer errors are 29 fewer inserted and 16 fewer deleted words, while wrong words only fell from 63 to 57. On M01 the 47 fewer are 24 wrong, 20 inserted and 3 deleted. Against F-1, M01's 13 fewer errors are mostly wrong words (10) and M04's 26 are mostly inserted words (17).

Single words are still poor, with adapter error rates of 87.2% (M01) and 75.0% (M04). These count word errors including insertions, so they aren't the share of clips decoded wrong. The lead over F-1 on M01 is mainly in sentences (35.2% against 45.1%). On M04 it's in both kinds (38.0% against 54.0%, and 75.0% against 91.7%).

The three personal adapters so far land close together: 48.42% (M02), 49.70% (M01) and 49.07% (M04), on different clips and against different stock and bottom-8 baselines.

What this doesn't show: seed spread on M01 and M04, since there's one run each (M02's three seeds gave identical word-error counts). It also doesn't show why inserted and deleted words fell. That could be fitting the speaker's voice, learning how many words a clip holds, or fitting the recording setup, and none of those was tested. Nothing here is about the Pi, since the adapter has no int8 export. Finally, these runs used 9aafe98, so the code in commits 167598f and b5e0589 is untested.

</details>

## Related

- [F-2: M02, adapter after 8 blocks](../f2-m02-adapter8/notes.md)
- [F-2: M02, adapter after 8 blocks, seeds 18 and 19](../f2-m02-adapter8-seeds/notes.md)
- [F-2: severe speakers, bottom 8 blocks, fixed 5 epochs](../f2-severe-bottom8-5epochs/notes.md)
- [F-1: severe speakers, leave-one-speaker-out, bottom 8 blocks](../f1-loso-bottom8/notes.md)
