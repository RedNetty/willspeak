# Prototype model: all eight dysarthric speakers, bottom 8 blocks

Proposal: §2 deliverable ("a tuned transcription model out, deployable on the Pi 5"). This is the model F-4 and the P-tests use.

Status: Run 30 Sep 2026 and results recorded. Conclusions written 21 October 2026 from the recorded results.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

Build the one model the prototype will actually run: the bottom-eight recipe trained on every dysarthric speaker in TORGO, with nobody held out.

## Setup

F-1 and F-2 hold something back so a hypothesis can be tested. The device just needs the strongest single model. The F-1 results point to the recipe: multi-speaker training beat single-speaker training for M01 and M04 (H3 test), and every F-1 model got better at typical speech too (typical-speech check, +2.3 to +2.7 points). So this run uses the F-1 bottom-eight recipe unchanged, with all eight dysarthric speakers' training prompts instead of seven.

## Plan

1. Rebuild the exports. This adds `export/all-dysarthric`, with every dysarthric speaker's train and dev prompts as train and dev, and every dysarthric speaker's test prompts as test.
2. Train with the F-1 bottom-eight recipe: blocks 0-7 whole, fixed dev score, stock kept if no checkpoint beats it.
3. Export to int8, the Pi's format.
4. Score it on the control speakers (typical speech) and on each of the eight dysarthric speakers' test clips.

## Why

The live demo runs one model on the Pi. It should be the best one I can make from public data, and its numbers should sit next to stock and F-1 on the same clips.

## Limits

This shows how much the prototype model beats stock and F-1 on speakers it trained on, using prompts it never saw, and whether it keeps typical speech. Every test speaker was in its training, so it says nothing about a new speaker like the intended user. For "a speaker it has never heard", F-1 is the number to quote. It doesn't test any hypothesis.

## Results

The prototype model beats stock for every speaker, and beats each severe speaker's F-1 model (the one that never heard him): M01 48.31% against 59.42%, M02 45.24% against 52.86%, M04 59.11% against 70.44% and F01 58.70% against 60.87%. Typical speech is 9.86% against stock's 12.40%. Full table in the Run Record.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

Prediction before the run:

1. On the typical-speech recordings (the control speakers), I expect the prototype model
   to get about 10-12% of words wrong. For comparison, the stock model gets 12.40% wrong and
   the F-1 models get about 9.8% wrong.

2. On the four severe speakers' test recordings, I expect the prototype model to do
   better, than each speaker's F-1 model (the one that never
   heard him), because It has more overall training data, but still doesn't tell us, if we had enough of one speakers data, does that beat out training on a bunch.

3. Of the four severe speakers, I expect M04 to improve the most over his F-1 model,
   because his improvement margins were large previously on multi-speaker training.
   (F-1 scores on these recordings: F01 60.87%, M01 59.42%, M02 52.86%, M04 70.44%.)

4. I would be surprised if M04 gets a score below 50%.

### Commands

Run from app/ in WSL with the GPU free and the PC awake (Instant Replay off). Training takes about 40-50 minutes and scoring about 45 more. Each block stops at its first error. The first line rewrites every export folder's manifests, so don't run it while another training run is going.

```bash
( set -e
source ~/venvs/willspeak-train/bin/activate
python scripts/corpus/torgo_manifest.py --export
python scripts/asr/train.py --export data/public/torgo/export/all-dysarthric --check
python scripts/asr/train.py --export data/public/torgo/export/all-dysarthric --out models/parakeet-ft-prototype-all-dysarthric-bottom8 --train-parts all --batch 2 --grad-accum 8 --freeze-batchnorm --normalize-targets --dev-scoring evaluate --keep-stock
python scripts/asr/export.py --nemo models/parakeet-ft-prototype-all-dysarthric-bottom8/parakeet-ft.nemo --out models/parakeet-ft-prototype-all-dysarthric-bottom8/onnx-int8 )

( set -e
source .venv/bin/activate
M=prototype-all-dysarthric-bottom8
python scripts/asr/evaluate.py --manifest data/public/torgo/export/controls/test.jsonl --audio-root data/public/torgo/export/controls --group-by kind --model-dir models/parakeet-ft-$M/onnx-int8 --out reports/finetune-torgo/controls-$M
python scripts/asr/ci.py --manifest data/public/torgo/export/controls/test.jsonl --stock reports/finetune-torgo/controls-stock --finetuned reports/finetune-torgo/controls-$M
for S in F01 M01 M02 M04 M05 F03 F04 M03; do
  if [ ! -f reports/finetune-torgo/loso-$S-stock/results.json ]; then
    python scripts/asr/evaluate.py --manifest data/public/torgo/export/loso-$S/test.jsonl --audio-root data/public/torgo/export/loso-$S --group-by kind --out reports/finetune-torgo/loso-$S-stock
  fi
  python scripts/asr/evaluate.py --manifest data/public/torgo/export/loso-$S/test.jsonl --audio-root data/public/torgo/export/loso-$S --group-by kind --model-dir models/parakeet-ft-$M/onnx-int8 --out reports/finetune-torgo/$M-on-$S
  echo "== $S: stock against the prototype model"
  python scripts/asr/ci.py --manifest data/public/torgo/export/loso-$S/test.jsonl --stock reports/finetune-torgo/loso-$S-stock --finetuned reports/finetune-torgo/$M-on-$S
done
for S in F01 M01 M02 M04; do
  echo "== $S: F-1 bottom 8 (never heard $S) against the prototype model (heard $S)"
  python scripts/asr/ci.py --manifest data/public/torgo/export/loso-$S/test.jsonl --stock reports/finetune-torgo/f1-loso-$S-bottom8 --finetuned reports/finetune-torgo/$M-on-$S
done )
```

Before trusting the training log, look for "train ... clips" (should be larger than any F-1 run's 1,836-2,012), "210461318 trainable", "stock dev WER" (a value), and either "averaged ... checkpoints" or "releasing the stock model". If it released stock, no checkpoint beat stock on dev and the prototype model is just stock.

Each speaker's test clips are that speaker's test prompts, the same clips as the F-1 test sets. No model trained on those prompts, but the prototype model did train on the speaker.

### Results

  Training: train 2095 clips (119.2 min), dev 358; stock dev WER 49.90%; selected fine-tuned, best epoch 7
  (the eighth pass, counting from 0), dev WER 29.6%, 4 checkpoints averaged
  Training curve: not recorded. This run started before the train.py that records the curve reached WSL
  (no "curve:" lines in its log). The kept checkpoints' file names give four points of it.

  The four kept checkpoints (the four best dev scores; epochs 0-5 were all worse than these):

  | epoch (from 0) | dev WER |
  | ---: | ---: |
  | 6 | 29.9% |
  | 7 | 29.6% (best) |
  | 8 | 29.7% |
  | 9 | 30.1% (the last epoch) |

  From epoch 6 on, dev WER stayed within half a point (29.6-30.1%), so the run had levelled off by
  its last epochs.

| Test set | n clips | Stock | F-1 bottom 8 (never heard him) | Prototype model | Prototype against stock: Delta (95% CI) | Prototype against F-1: Delta (95% CI) |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Controls (typical speech) | 1179 | 12.40% | 9.70-10.07% | 9.86% | 2.53 (1.62 to 3.54) | - |
| F01 (severe) | 23 | 73.91% | 60.87% | 58.70% | 15.22 (1.96 to 42.86) | 2.17 (0.00 to 9.09) |
| M01 (severe) | 80 | 70.05% | 59.42% | 48.31% | 21.74 (12.96 to 32.19) | 11.11 (4.61 to 17.62) |
| M02 (severe) | 84 | 59.52% | 52.86% | 45.24% | 14.29 (3.83 to 26.47) | 7.62 (3.14 to 13.33) |
| M04 (severe) | 87 | 91.63% | 70.44% | 59.11% | 32.51 (21.31 to 44.08) | 11.33 (4.44 to 18.23) |
| M05 (moderate-severe) | 111 | 44.60% | - | 30.66% | 13.94 (8.47 to 20.37) | - |
| F03 (moderate) | 116 | 30.54% | - | 21.81% | 8.72 (4.50 to 13.95) | - |
| F04 (mild) | 103 | 15.89% | - | 14.34% | 1.55 (-2.32 to 5.62) | - |
| M03 (mild) | 92 | 8.71% | - | 7.05% | 1.66 (-1.29 to 5.37) | - |

A positive Delta means the prototype model is better. Run 30 Sep 2026 on Yack, all scores Int8.

What the numbers show, before interpreting them. Against stock, the prototype model is better for every speaker. The interval clears zero for all four severe speakers, M05, F03 and the controls. The two mild speakers (F04, M03) point better, but their intervals include zero, since stock already gets 84-91% of their words right.

Against each severe speaker's F-1 model (never heard him), the prototype model (heard his other recordings) is better for all four. M04 is +11.33, M01 +11.11 and M02 +7.62, all with intervals clear of zero. F01 is +2.17, and its interval touches zero on 23 clips. On typical speech it lands where the F-1 models did, 9.86% against their 9.70-10.07%.

Prediction against actual:

1. Controls: predicted 10-12%, measured 9.86%, just under the range.
2. Better than F-1 on all four severe speakers: predicted yes, measured yes (three clear zero, F01 touches zero).
3. Most improvement over F-1: predicted M04, measured M04 (+11.33), with M01 (+11.11) close behind.
4. Surprise if M04 below 50%: not reached, M04 measured 59.11%.

### Conclusions

*Written 21 October 2026 from the recorded results above.*

Trained on all eight dysarthric speakers, the prototype model is better than stock for every speaker (Int8, LOSO test prompts, lower WER is better). The mild speakers F04 and M03 only move by 1.55 and 1.66 points, with intervals that include zero, so there's no sign of a loss for them. Against F-1 it's 7.62 to 11.33 points better for M01, M02 and M04. The prototype heard each speaker's other recordings and F-1 didn't, so this measures what hearing the speaker adds. It isn't a test of single-speaker against multi-speaker training (H3).

Three of my four predictions held. The miss was the controls, at 9.86% against the predicted 10 to 12%.

These figures are on the LOSO test sets (prompt-disjoint, all sessions). Every later recipe was ranked on the pooled lab set instead (M01, M02 and M04, 166 clips, 425 words), so these can't be put next to the later 178 to 213 wrong-word results. The prototype model was never scored on that pooled set.

For the expo model this stays a fallback candidate. Its best severe-speaker figure (M01 48.31%) is above the 35% bar. Later recipes on the pooled set start from a similar idea, other speakers' data first (severe pretraining then the block-8 adapter, 178.0 of 425, 41.9%, `reports/f2-adapter-levers`). Scoring this model on the pooled lab set would show where it stands.

</details>

## Related

[F-1: severe speakers, leave-one-speaker-out, bottom 8 blocks](../f1-loso-bottom8/notes.md), [Typical-speech check](../typical-speech-check/notes.md), [F-2: severe speakers, bottom 8 blocks, fixed 5 epochs](../f2-severe-bottom8-5epochs/notes.md)
