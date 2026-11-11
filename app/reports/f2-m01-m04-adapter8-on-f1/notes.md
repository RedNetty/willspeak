# F-2: M01 and M04, adapter after 8 blocks, on the F-1 model

Proposal: F-2 (single-speaker fine-tuning, §5.2 / H3) and the F-1 comparison. This is the setup the device would use: a base trained on other dysarthric speakers plus a small personal layer.

Status: run 1 Oct 2026, 17:07 to about 17:14, one run per speaker (seed 17). Against the adapter on stock, the adapter on F-1 is clearly better on M04 and not shown to differ on M01. The prediction was given at 17:04 and revised at 17:08, both before the run. Both versions are scored in the Conclusions.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

Take each speaker's F-1 model, which never heard him, and train the same adapter after block 7 on his earlier sessions. Score on his last session. Does that beat the adapter on stock, and does it beat F-1 alone?

## Setup

F-1 is the shared model: blocks 0-7 retrained on about 110 minutes of other speakers, never a word from the test speaker. F-2 trains on one speaker's earlier sessions and tests on his last, with no shared prompts. The adapter is the layer from the [M02 runs](../f2-m02-adapter8/notes.md), `h + W h + b`. It starts at zero, so the model starts out as its base, and only its 1,049,600 numbers train. The adapter file records the base's identity and `nemo_hyp.py` won't attach it to a different one. A fixed schedule (`--select last`) trains a set number of epochs and averages the last 4 checkpoints, so dev picks nothing.

The only change from the [M01 and M04 adapter runs](../f2-m01-m04-adapter8/notes.md) is the base, now the speaker's F-1 model (`models/parakeet-ft-f1-loso-<SPK>-bottom8/parakeet-ft.nemo`, frozen batch norm). The rest is the same: 3e-4, batch 2 with 8 accumulated, 10 warmup updates, 15 epochs, seed 17, normalised transcripts, `--select last --no-keep-stock`. The device would really start from the prototype model, but that was trained on all eight speakers including M01's and M04's earlier sessions, so it can't be scored fairly here.

Earlier scores on the same clips (NeMo; M01 has 65 test clips and 169 words, M04 has 64 and 161):

| Model | M01 | M04 |
| --- | --- | --- |
| Stock | 77.51% (131 word errors) | 80.75% (130) |
| F-2 bottom-8 model | 74.56% (126) | 82.61% (133) |
| F-1 shared model | 57.40% (97) | 65.22% (105) |
| Adapter on stock, seed 17 | 49.70% (84) | 49.07% (79) |

In the [check of the cleaned-up `train.py`](../check-train-py-cleanup/notes.md), three seeds of the adapter on stock got 84, 87 and 86 errors on M01 (spread 1.78 points) and 79, 76 and 82 on M04 (3.73). Its gain over F-1 was 7.69 points on M01 (interval -1.16 to 16.77) and 16.15 on M04 (4.29 to 28.30). One word is 0.59 points on M01 and 0.62 on M04.

## Plan

1. Check that the earlier outputs exist in the WSL tree (first command block).
2. For M01 and M04, train the adapter with `--base` set to the speaker's F-1 model, then decode his test clips with both attached (`--nemo` with `--adapter`).
3. Score with `evaluate.py` and compare with `ci.py` against stock, the adapter on stock, and F-1 alone. In every `ci.py` block a positive Delta means "Fine-tuned" beats "Stock".
4. Check each training log names the F-1 file as base, not stock.

Four points is the rough run-to-run noise per speaker (about 7 words on M01, 6 on M04), and the adapter's seed spread was 1.8 and 3.7 points. A difference only counts when the `ci.py` interval is entirely on one side of zero. With 65 and 64 clips these intervals are about 9 to 13 points wide, so a 4-point gap just means more seeds are needed.

If it's clearly better than the adapter on stock on both speakers, the shared model gives the adapter something about 6 minutes of one speaker can't, and the device recipe is base plus adapter. About the same means F-1 adds nothing measurable at this size. Worse means retrained blocks 0-7 are a worse start. Beating F-1 alone means the personal layer helps on top. If the speakers disagree I report them separately.

## Why

The device is meant to be a shared model plus a personal layer, and I've only tested each half alone. F-1 beat the personal whole-block models on M01 and M04, and the adapter on stock matched or beat F-1 (shown on M04, inconclusive on M01). The answer also decides what F-3 varies: an adapter on a base, or whole blocks.

## Limits

One run each, so this combination's seed spread is unknown. It doesn't compare against retraining blocks on F-1, and 3e-4 and the block-7 position were tuned on stock. M02 isn't included, since its dev set is big enough for the default recipe. The prototype model can't be the base, and there's nothing about the Pi since the adapter has no int8 export. The test clips have been scored many times, so I only read the comparisons above and pick nothing by test score.

## Reading and Video

None added. See the [M01 and M04 note](../f2-m01-m04-adapter8/notes.md#reading-and-video) and the [first adapter note](../f2-m02-adapter8/notes.md#videos).

## Results

Run 1 Oct 2026, 17:07 to about 17:14. Both training logs name the speaker's F-1 file as base and every log line the plan listed is there. The copied output starts partway through M01 training, so I didn't see the pull, starting commit, `train.py` hash or input check. The final commit was 3c2585e, and `train.py`, `nemo_hyp.py`, `block_adapter.py` and `evaluate.py` are identical there and in 9991cd5, the commit the [cleanup check](../check-train-py-cleanup/notes.md) passed.

Test WER (word errors in brackets; M01 has 169 words, M04 161):

| Model | M01 | M04 |
| --- | --- | --- |
| Stock | 77.51% (131) | 80.75% (130) |
| F-1 shared model | 57.40% (97) | 65.22% (105) |
| Adapter on stock, seed 17 | 49.70% (84) | 49.07% (79) |
| Adapter on F-1, seed 17 | 53.25% (90) | 40.99% (66) |

`ci.py` comparisons (95% paired bootstrap over clips). Gain is the other model's WER minus the adapter on F-1's, so positive favours the adapter on F-1:

| The adapter on F-1 against | M01 gain | M01 interval | M04 gain | M04 interval |
| --- | ---: | --- | ---: | --- |
| Stock | 24.26 | 14.88 to 36.13 | 39.75 | 25.55 to 56.50 |
| Adapter on stock | -3.55 | -13.07 to 6.59 | 8.07 | 1.38 to 15.43 |
| F-1 model alone | 4.14 | -4.26 to 13.66 | 24.22 | 13.73 to 35.86 |

On M04 the adapter on F-1 got 66 of 161 wrong, 13 fewer than the adapter on stock and 39 fewer than F-1. The interval against the adapter on stock excludes zero, but its lower end is only 1.38 points (2 words). It's also 10 words (6.2 points) below the best of the adapter on stock's three seeds (76, 79 and 82 errors, 47.20% to 50.93%), though that isn't a paired test. CER fell from 39.79% to 32.54% and PER from 40.62% to 31.43% (no intervals).

On M01 it got 90 of 169 wrong, 6 more than the adapter on stock and 7 fewer than F-1, with both intervals including zero. That's 3 words above the highest stock-adapter seed, the size of their whole spread (1.78 points). CER barely moved (35.10% against 35.24%). On M01 neither adapter is shown to beat F-1 or each other.

With the F-1 base the training loss starts and ends much lower: 1.74 to 0.80 on M01 and 2.45 to 1.17 on M04, against 5.15 to 1.82 and 6.01 to 2.63 on stock. That's training loss, and it jumps between epochs at this size (M01: 0.78 at epoch 10, 1.04 at epoch 11). Released dev WER was 44.44% on M01, same as the adapter on stock, and 48.00% on M04 against 64.00%. The dev sets are 18 and 25 words (5.6 and 4.0 points per word), so the 4-word M04 gap agrees with test but isn't evidence. Training took 2m 03s and 2m 19s and each decode about 25 seconds. The whole thing took about 6 minutes against the 10 to 12 I'd planned.

`ci.py` printed "repeated transcripts are resampled separately; --cluster-by text keeps their takes together" on every comparison, meaning some prompts repeat in the test clips. Treating repeated takes as independent can make intervals too narrow, and the [cleanup check](../check-train-py-cleanup/notes.md) had the same issue. Nothing was rerun with `--cluster-by text` on 1 Oct. The [seed-repeat note](../f2-m01-m04-adapter8-on-f1-seeds/notes.md) did it later: every interval that excluded zero still does, and M04's lower end went from 1.38 to 0.45.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

Written on 1 Oct 2026 at 17:04 and revised at 17:08, both before any run of this test. At 17:06 I saw that answers 1 and 2 couldn't both hold and chose "close to F-1 alone" for answer 2. Revised answers 1 and 3 below are interpretations of that choice, so they aren't word-for-word predictions. Answer 2 is my original wording. My 17:04 answers come after, kept as given with spelling and punctuation fixed, so the record shows what changed. The scores to answer against are in the table in Setup: the adapter on stock scored 49.70% on M01 and 49.07% on M04, and the F-1 model alone 57.40% and 65.22%.

Revised answers (17:08):

1. The adapter on F-1, against the adapter on stock, on M01 and on M04: better, about the same, or worse. (Clarified wording.) The adapter on F-1 will not beat the adapter on stock. It will land close to the F-1 model alone (57.40% on M01, 65.22% on M04), which is behind the adapter on stock by about 8 points on M01 and about 16 on M04.
2. For each speaker, whether the adapter on F-1 beats the F-1 model alone. (My original words.) I think the margins will be fairly close when comparing adapter vs F-1. This is read as the adapter on F-1 against the F-1 model alone: close to it, not clearly ahead.
3. Which speaker gains more over the adapter on stock, and whether either goes below 45%. (Clarified wording.) Neither speaker gains over the adapter on stock, and neither goes below 45%.

[Note added before the run: "close" uses the 4-point guide, so about 53 to 61% on M01 (90 to 104 of 169 words wrong) and 61 to 69% on M04 (99 to 111 of 161). A difference only counts as shown when the `ci.py` interval sits entirely on one side of zero.]

Answers as first given (17:04):

1. The adapter on F-1, against the adapter on stock, on M01 and on M04: better, about the same, or worse, and by roughly how many points:

I believe the adapter on F-1 will 100% beat the adapter on stock, as we've seen gains from both F-1 and the Adapter separately, whereas stock is not great at dysarthric speech past mild. Maybe 10 points.

2. For each speaker, whether the adapter on F-1 beats the F-1 model alone:

I think the margins will be fairly close when comparing adapter vs F-1.

3. Which speaker gains more over the adapter on stock, whether either goes below 45%, and the reason for the answers above:

I believe M04 will gain the most as we've typically seen this.

[Notes added before the run on the 17:04 answers. Answer 1: 10 points better than the adapter on stock would be about 39.6% on M01 (67 word errors) and 39.1% on M04 (63). "100%" is stronger than the seed spread supports, since the adapter on stock moved 1.8 points on M01 and 3.7 on M04 across seeds. The reason given compares the F-1 base with stock, but this test compares with the adapter on stock. That adapter already gained 27.8 and 31.7 points over stock, more than the F-1 model's own head start over stock (20.1 points on M01, 15.5 on M04).

Answers 1 and 2: if answer 2 means the adapter on F-1 against the F-1 model alone, "fairly close" puts the adapter on F-1 about 4 to 12 points (M01) and 12 to 20 points (M04) behind the adapter on stock. Answer 1's 10 points ahead puts it 17.8 points (M01) and 26.1 points (M04) ahead of the F-1 model alone. Both can't hold, which is why they were revised.

Answer 3: whether either speaker goes below 45% wasn't answered. The recorded gains back up "typically seen" for the adapter on stock. Over the F-1 model it gained 16.15 points on M04 against 7.69 on M01, and over stock 31.68 against 27.81. M04's seed spread (3.73 points) is also bigger than M01's (1.78).]

### Commands and Setup

Run from `app/` in the WSL tree on Yack after `git pull`, with the GPU free and Windows apps closed. Training and decoding run in `~/venvs/willspeak-train`, and `evaluate.py` and `ci.py` in `app/.venv`. The data is `data/public/torgo/export/M01` and `data/public/torgo/export/M04`, the same test clips as the earlier runs. Each training run should take about 2 to 3 minutes, plus loading the 2.4 GB F-1 file, which gets read once more to record its identity. That's about 10 to 12 minutes in all. The training and scoring blocks stop at their first error.

Check that the earlier outputs exist (every line should start with `ok`):

```bash
for S in M01 M04; do
  for p in models/parakeet-ft-f1-loso-$S-bottom8/parakeet-ft.nemo \
           reports/finetune-torgo/$S-test-stock-nemo/results.json \
           reports/finetune-torgo/f1-loso-$S-bottom8-on-F2-test-nemo/results.json \
           reports/finetune-torgo/f2-$S-adapter8-test-nemo/results.json \
           data/public/torgo/export/$S/test.jsonl; do
    [ -e "$p" ] && echo "ok       $p" || echo "MISSING  $p"
  done
done
```

Train the adapters on the F-1 models, then decode with the F-1 model and the adapter attached:

```bash
( set -e
source ~/venvs/willspeak-train/bin/activate
for S in M01 M04; do
  F1=models/parakeet-ft-f1-loso-$S-bottom8/parakeet-ft.nemo
  python scripts/asr/train.py --export data/public/torgo/export/$S --base $F1 --out models/parakeet-ft-f2-$S-adapter8-on-f1 --adapter-after-blocks 8 --lr 3e-4 --batch 2 --grad-accum 8 --warmup-steps 10 --epochs 15 --select last --no-keep-stock --freeze-batchnorm --normalize-targets --dev-scoring evaluate --seed 17
  python scripts/asr/nemo_hyp.py --nemo $F1 --adapter models/parakeet-ft-f2-$S-adapter8-on-f1/adapter.pt --manifest data/public/torgo/export/$S/test.jsonl --audio-root data/public/torgo/export/$S --out reports/finetune-torgo/f2-$S-adapter8-on-f1-test-nemo.jsonl
done )
```

Score and compare:

```bash
( set -e
source .venv/bin/activate
for S in M01 M04; do
  M=data/public/torgo/export/$S; R=f2-$S-adapter8-on-f1-test-nemo
  python scripts/asr/evaluate.py --manifest $M/test.jsonl --audio-root $M --hyp reports/finetune-torgo/$R.jsonl --group-by kind --out reports/finetune-torgo/$R
  echo "== $S adapter on F-1 against stock"
  python scripts/asr/ci.py --manifest $M/test.jsonl --stock reports/finetune-torgo/$S-test-stock-nemo --finetuned reports/finetune-torgo/$R
  echo "== $S adapter on F-1 against the adapter on stock"
  python scripts/asr/ci.py --manifest $M/test.jsonl --stock reports/finetune-torgo/f2-$S-adapter8-test-nemo --finetuned reports/finetune-torgo/$R
  echo "== $S adapter on F-1 against the F-1 model"
  python scripts/asr/ci.py --manifest $M/test.jsonl --stock reports/finetune-torgo/f1-loso-$S-bottom8-on-F2-test-nemo --finetuned reports/finetune-torgo/$R
done )
```

Check in the training logs (two newest, M01 then M04) before trusting the scores:

```bash
for f in $(ls -t logs/train_parakeet-*.log | head -2 | sort); do echo "== $f"; grep -hE "Base model|loading|parameters exact|fixed schedule|kept epochs|Checking the averaged|averaged|Released|saved .*adapter8-on-f1" "$f"; done
for S in M01 M04; do echo "== $S curve"; cat models/parakeet-ft-f2-$S-adapter8-on-f1/curve.csv; done
```

Each log should have a `Base model` and a `loading` line naming `models/parakeet-ft-f1-loso-<SPK>-bottom8/parakeet-ft.nemo`. If either says `nvidia/parakeet-tdt-0.6b-v2`, the run used the stock model and isn't this test. The log should also show `1049600 trainable`, `fixed schedule (--select last): kept epochs 12, 13, 14, 15`, `Checking the averaged model on dev before release...`, `averaged 4 checkpoints`, a `Released model dev WER` that's a number, and a saved `adapter.pt`. There's no `stock dev WER` line because `--no-keep-stock` skips it. `curve.csv` should list epochs 0 to 14, one row each.

As run: the four blocks ran as one script, `logs/adapter-on-f1.sh`, with output also saved to `logs/adapter-on-f1-full-<date and time>.txt`. It printed the commit and the `train.py` hash at the start (expected c696b4873186) and stopped if an input was missing. It printed the log lines after each training run and the two `curve.csv` files at the end. Both runs, the decodes, the scoring and the three comparisons per speaker finished with no errors.

### Full Results

Log lines from the two training logs and the decode and scoring output. The logs are `train_parakeet-20261001-220706-883834.log` for M01 and `train_parakeet-20261001-221006-803080.log` for M04, and their file names carry the time in UTC.

| Line | M01 | M04 |
| --- | --- | --- |
| Base model, and the `loading` line | `models/parakeet-ft-f1-loso-M01-bottom8/parakeet-ft.nemo` | `models/parakeet-ft-f1-loso-M04-bottom8/parakeet-ft.nemo` |
| `parameters exact` | 618875526 total, 1049600 trainable | 618875526 total, 1049600 trainable |
| Training data | 36 batches and 5 updates per epoch (the data line was cut off) | 82 clips, 6.07 min; 9 dev clips, 0.70 min; 41 batches, 6 updates |
| Schedule line | kept epochs 12, 13, 14, 15; dev chose nothing | kept epochs 12, 13, 14, 15; dev chose nothing |
| Averaging | `averaged 4 checkpoints`, then `Checking the averaged model on dev before release...` | the same |
| `Released model dev WER` | 44.44% | 48.00% |
| Elapsed | 2m 03s | 2m 19s |
| Saved | `models/parakeet-ft-f2-M01-adapter8-on-f1/adapter.pt` (4.2 MB) | `models/parakeet-ft-f2-M04-adapter8-on-f1/adapter.pt` (4.2 MB) |
| `nemo_hyp.py` | `Added the adapter ... after encoder blocks 0-7`; 65 clips | the same; 64 clips |
| Test WER / CER / PER | 53.25% / 35.10% / 33.77% | 40.99% / 32.54% / 31.43% |

The console counts epochs from 1, so the kept epochs 12 to 15 are the checkpoint files numbered 11 to 14.

Summary table, as printed:

```text
report                                           WER errors
f2-M01-adapter8-test-nemo                     49.70%     84
f2-M01-adapter8-on-f1-test-nemo               53.25%     90
f1-loso-M01-bottom8-on-F2-test-nemo           57.40%     97
f2-M04-adapter8-test-nemo                     49.07%     79
f2-M04-adapter8-on-f1-test-nemo               40.99%     66
f1-loso-M04-bottom8-on-F2-test-nemo           65.22%    105
```

Comparisons, as printed. `ci.py` now labels its lines Baseline, Candidate and Gain (baseline - candidate), with the same sign as before, so a positive gain means the candidate is better. Every block began with `Paired percentile bootstrap: 65 clip units` (64 for M04), `2000 resamples, seed 0` and the repeated-transcripts warning described in Results:

```text
== M01 adapter on F-1 against stock
Baseline WER 77.51%  (95% CI 63.48% to 94.45%)
Candidate WER 53.25%
Gain (baseline - candidate) 24.26 points  (95% CI 14.88 to 36.13)
== M01 adapter on F-1 against the adapter on stock
Baseline WER 49.70%  (95% CI 39.19% to 62.11%)
Candidate WER 53.25%
Gain (baseline - candidate) -3.55 points  (95% CI -13.07 to 6.59)
== M01 adapter on F-1 against the F-1 model
Baseline WER 57.40%  (95% CI 45.30% to 70.86%)
Candidate WER 53.25%
Gain (baseline - candidate) 4.14 points  (95% CI -4.26 to 13.66)
== M04 adapter on F-1 against stock
Baseline WER 80.75%  (95% CI 65.45% to 97.22%)
Candidate WER 40.99%
Gain (baseline - candidate) 39.75 points  (95% CI 25.55 to 56.50)
== M04 adapter on F-1 against the adapter on stock
Baseline WER 49.07%  (95% CI 39.52% to 62.12%)
Candidate WER 40.99%
Gain (baseline - candidate) 8.07 points  (95% CI 1.38 to 15.43)
== M04 adapter on F-1 against the F-1 model
Baseline WER 65.22%  (95% CI 52.83% to 80.92%)
Candidate WER 40.99%
Gain (baseline - candidate) 24.22 points  (95% CI 13.73 to 35.86)
```

Training curves from `curve.csv`, which counts epochs from 0. The table counts from 1 like the console does. Loss is the logged training loss, and dev WER is on 18 words (M01) and 25 words (M04):

| Epoch | M01 dev WER | M01 loss | M04 dev WER | M04 loss |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 44.44% | 1.7375 | 72.00% | 2.4493 |
| 2 | 33.33% | 1.6165 | 72.00% | 2.1016 |
| 3 | 44.44% | 1.5667 | 56.00% | 1.9299 |
| 4 | 33.33% | 1.3459 | 64.00% | 1.6899 |
| 5 | 33.33% | 1.1789 | 60.00% | 1.4947 |
| 6 | 50.00% | 1.0752 | 56.00% | 1.5128 |
| 7 | 44.44% | 0.9116 | 52.00% | 1.5180 |
| 8 | 50.00% | 1.0519 | 60.00% | 1.4890 |
| 9 | 38.89% | 1.1436 | 60.00% | 1.2424 |
| 10 | 44.44% | 0.7797 | 56.00% | 1.2439 |
| 11 | 44.44% | 1.0415 | 52.00% | 1.2376 |
| 12 | 44.44% | 0.8222 | 48.00% | 1.1341 |
| 13 | 44.44% | 0.8219 | 48.00% | 1.2762 |
| 14 | 44.44% | 0.7946 | 52.00% | 1.0921 |
| 15 | 44.44% | 0.7985 | 56.00% | 1.1671 |

### Conclusions

1. Against the adapter on stock, the answer depends on the speaker. On M04 the adapter on F-1 is clearly better, with 13 fewer words wrong (66 against 79). That's a gain of 8.07 points with an interval of 1.38 to 15.43, which excludes zero but only just. On M01 it isn't shown to differ: 6 more words wrong (90 against 84), a gain of -3.55 points with an interval of -13.07 to 6.59. In the plan's terms, on M04 the shared model gave the adapter something that about 6 minutes of M04 alone didn't, and on M01 it added nothing measurable.

2. Against the F-1 model alone, the adapter on F-1 is ahead on both speakers but only shown to be ahead on M04 (24.22 points, interval 13.73 to 35.86). On M01 it's 4.14 points with an interval of -4.26 to 13.66. So the personal layer helps on top of the shared model on M04 and isn't shown to on M01. The adapter on stock had the same status against F-1 in the [M01 and M04 adapter note](../f2-m01-m04-adapter8/notes.md).

3. M04 at 40.99% is the lowest test WER of any F-2 run so far (next is the M02 whole-block model at 44.21%). It's still about 13 points above my goal in the high 20s, and M01 at 53.25% is about 25 points above it. No run has reached it yet.

4. A base plus an adapter isn't established as the device recipe. One speaker shows the benefit and the other doesn't, and each figure is a single run. The adapter on stock moved 1.78 points on M01 and 3.73 on M04 across three seeds. This test's gains over it (8.07 on M04, -3.55 on M01) are bigger than those spreads, but a spread from a different recipe is a weak guide, and this combination's own spread hasn't been measured.

5. Scoring the predictions. The Prediction text is the record, and neither version has been rewritten.

   Revised answers (17:08). Answer 1 (clarified wording: no win over the adapter on stock, landing near the F-1 model alone) fits M01. There the adapter on F-1 didn't beat the adapter on stock (6 words worse) and landed 7 words ahead of F-1 (90 errors, the lower edge of the 90 to 104 range set for "close"). It doesn't fit M04, where the adapter on F-1 beat the adapter on stock (shown) and beat F-1 by 24.22 points. Answer 2 (my original words, read as the adapter on F-1 against F-1 alone) fits M01 (4.14 points, interval includes zero) and not M04 (24.22 points, shown). Answer 3 (clarified wording: neither speaker gains over the adapter on stock, neither goes below 45%) fits M01 (no gain, 53.25%) and not M04 (a gain of 8.07 points, 40.99%). So the revised set matched M01 and missed M04 on all three answers.

   Answers as first given (17:04). Answer 1 ("100%" better than the adapter on stock, maybe 10 points) fits M04, where it was better and the difference was shown. Ten points would have been about 39.1% (63 errors) and the result was 40.99% (66 errors). It doesn't fit M01, where the adapter on F-1 was 6 words worse and 10 points would have meant 39.6% against the actual 53.25%. The "100%" didn't hold, since one of the two speakers failed it. Answer 2 ("fairly close" margins, adapter against F-1) fits M01 only, same as in the revised set. Answer 3 (M04 will gain the most) fits on every comparison. M04 gained more than M01 against stock (39.75 against 24.26 points), against the adapter on stock (8.07 against -3.55) and against F-1 (24.22 against 4.14). Whether either speaker would go below 45% wasn't answered. M04 did and M01 didn't.

   The 17:04 answers fit the results better than the 17:08 revision. The revision came from noticing that answers 1 and 2 couldn't both hold, and it kept the reading that fits M01. The two revised lines that missed M04 (answers 1 and 3) are interpretations of the "close to F-1 alone" choice and not word-for-word predictions. The first-given answers 1 and 2 couldn't both hold, as the note on them said, and the results split the same way: answer 1 held on M04 and answer 2 on M01.

6. What this doesn't show. Nothing here separates the speaker, the quality of the F-1 base and chance, so it can't say why M04 benefits and M01 doesn't. There's one run per speaker, so the adapter on F-1's seed spread is unknown. I haven't checked whether the intervals survive resampling by prompt (`--cluster-by text`), which matters most for M04's lower end of 1.38. It doesn't cover whole blocks retrained on the F-1 base, an adapter at another position or learning rate on that base, M02, the prototype model as the base, or the Pi. The dev WERs rest on 18 and 25 words, so I don't read anything from them. These test clips have now been scored many times, so I only read the three comparisons per speaker named in the plan and don't choose any variant by its test score.

7. Possible next tests, not chosen yet. One is repeating this with seeds 18 and 19 on both speakers (about 2 to 3 minutes of training each), to see whether M04's +8.07 and M01's -3.55 come from the base or the seed. Another is re-scoring these six comparisons with `ci.py --cluster-by text`, which only needs the saved outputs. Later options are the same stack on M02 (first checking that its bottom-8 F-1 model has a score on M02's F-2 test clips) and F-3. The seed repeat and the prompt-grouped re-score were run afterwards (see the [seed-repeat note](../f2-m01-m04-adapter8-on-f1-seeds/notes.md)). M04's lead repeated on seeds 18 and 19, and M01's -3.55 didn't.

</details>

## Related

[F-2: M01 and M04, adapter after 8 blocks](../f2-m01-m04-adapter8/notes.md), [F-2: M02, adapter after 8 blocks](../f2-m02-adapter8/notes.md), [F-1: leave-one-speaker-out, bottom 8 blocks](../f1-loso-bottom8/notes.md), [Check: cleaned-up train.py](../check-train-py-cleanup/notes.md)
