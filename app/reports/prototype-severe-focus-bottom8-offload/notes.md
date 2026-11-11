# Prototype model, severe focus, repeated with the optimizer in RAM: all eight dysarthric speakers, bottom 8 blocks

Proposal: §2 deliverable (the tuned model for the Pi), aimed at the severe speakers, who are the project's goal.

Status: run 30 Sep 2026 (22:40–23:41), results recorded. No prediction was written before the run.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

Train the severe-focus model again with the same recipe, data and seed, changing only one thing: `--offload-optimizer`. That keeps AdamW's running averages and its weight update in system RAM instead of on the GPU.

## Setup

The severe-focus run (30 Sep 2026) is the best model so far for the severe speakers (37.64% on their test sentences), but training took about 2 hours 40 minutes. From epoch 5 on it ran at about 1.1 batches per second instead of about 3, because the GPU's 8 GB was full and Windows moved GPU memory into system RAM.

A 30-update check on 30 Sep 2026 (`logs/offload-check-off.log`, `logs/offload-check-on.log`) compared the two settings on the same data and seed:

| | Offload off | Offload on |
| --- | ---: | ---: |
| Peak GPU memory | 8.11 GB | 6.45 GB |
| 240 training batches | 1:16 | 1:18 |
| Dev WER after 30 updates | 48.03% | 47.93% |
| Mean training loss | 3.9547 | 3.9542 |

Training was the same (loss equal to the third decimal) with 1.66 GB less GPU memory, and about 2% slower per batch. The check was too short to show the slowdown, which only starts in later epochs.

## Plan

1. Train the severe-focus recipe with `--offload-optimizer` added and nothing else changed.
2. Export to int8.
3. Score it on the same recordings as the first severe-focus run.
4. Compare training time per epoch and peak GPU memory (speed), and WER against the first severe-focus run (how much two runs of one recipe differ).

## Why

One run answers two things. First, speed: if every epoch stays near 3 batches per second, the next experiments take about half the time, and there's GPU memory left to train more of the encoder.

Second, run-to-run noise. The method is the same, so any WER difference from the first severe-focus run is just what changes between two runs of one recipe on this GPU (rounding and non-repeatable GPU arithmetic). That tells me how big a gap between two different recipes has to be before it means anything.

## Limits

This shows whether the epochs stay fast with the optimizer in RAM, and how far apart two runs of the same recipe land on each test set. It can't say the recipe itself is better or worse, since nothing in it changed. One repeat is also just one example of run-to-run noise, not its full range.

## Videos

- [ADAM Optimization Algorithm Explained Visually](https://www.youtube.com/watch?v=MWZakqZDgfQ): the two running averages per weight, which is the memory this run moves to RAM.
- [AdamW explained (short)](https://www.youtube.com/shorts/iyoIx7IallQ): the AdamW variant the runs use (weight decay kept separate from the update).
- Reading, since I found no good short video: [DeepSpeed ZeRO-Offload tutorial](https://www.deepspeed.ai/tutorials/zero-offload/), the same idea (optimizer state and update on the CPU) used for much larger models.

## Results

Speed is fixed. Every epoch's training batches took 5.6–6.1 minutes (about 4 batches per second) and peak GPU memory stayed at 7.20–7.76 GB, so the GPU never spilled into system RAM. Training took 60 minutes in all, against about 2 hours 40 minutes for the first severe-focus run with the same recipe.

The two runs land within a few points of each other and no difference clears zero. On the goal set (the severe speakers' test sentences) this run gets 38.51% against the first run's 37.64% (Delta −0.88, CI −5.40 to 3.24). Per speaker the gaps reach 4.35 points (F01, 23 clips) and 3.33 points (M02), in both directions. Both runs beat the prototype (41.79%) on the goal set, by 4.16 and 3.28 points.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

Prediction before the run: not recorded.

### Commands

Run from app/ in WSL after `git pull`, with Windows apps closed. Training should take about 1.5 hours if the epochs stay fast (the first severe-focus run took 2 hours 40 minutes), and scoring about 15 minutes. Each block stops at its first error.

```bash
( set -e
source ~/venvs/willspeak-train/bin/activate
python scripts/asr/train.py --export data/public/torgo/export/all-dysarthric --out models/parakeet-ft-prototype-severe-focus-bottom8-offload --dev-speakers F01,M01,M02,M04 --repeat-speakers F01,M01,M02,M04 --repeat 2 --lr 5e-5 --train-parts all --batch 2 --grad-accum 8 --freeze-batchnorm --normalize-targets --dev-scoring evaluate --keep-stock --offload-optimizer
python scripts/asr/export.py --nemo models/parakeet-ft-prototype-severe-focus-bottom8-offload/parakeet-ft.nemo --out models/parakeet-ft-prototype-severe-focus-bottom8-offload/onnx-int8 )

( set -e
source .venv/bin/activate
E=data/public/torgo/export/all-dysarthric
ALL=prototype-all-dysarthric-bottom8
PREV=prototype-severe-focus-bottom8
NEW=prototype-severe-focus-bottom8-offload
echo "== SPEED: one line per epoch (minutes of training batches, peak GPU memory)"
cat models/parakeet-ft-$NEW/curve.csv
python scripts/asr/evaluate.py --manifest $E/test-severe-sentences.jsonl --audio-root $E --group-by speaker --model-dir models/parakeet-ft-$NEW/onnx-int8 --out reports/finetune-torgo/$NEW-on-all-dysarthric-test-severe-sentences
echo "== GOAL, severe test sentences: first severe-focus run against this run"
python scripts/asr/ci.py --manifest $E/test-severe-sentences.jsonl --stock reports/finetune-torgo/$PREV-on-all-dysarthric-test-severe-sentences --finetuned reports/finetune-torgo/$NEW-on-all-dysarthric-test-severe-sentences
echo "== GOAL, severe test sentences: prototype (all) against this run"
python scripts/asr/ci.py --manifest $E/test-severe-sentences.jsonl --stock reports/finetune-torgo/$ALL-on-all-dysarthric-test-severe-sentences --finetuned reports/finetune-torgo/$NEW-on-all-dysarthric-test-severe-sentences
for S in F01 M01 M02 M04 F04 M03; do
  python scripts/asr/evaluate.py --manifest data/public/torgo/export/loso-$S/test.jsonl --audio-root data/public/torgo/export/loso-$S --group-by kind --model-dir models/parakeet-ft-$NEW/onnx-int8 --out reports/finetune-torgo/$NEW-on-$S
  echo "== $S full test: first severe-focus run against this run"
  python scripts/asr/ci.py --manifest data/public/torgo/export/loso-$S/test.jsonl --stock reports/finetune-torgo/$PREV-on-$S --finetuned reports/finetune-torgo/$NEW-on-$S
done
python scripts/asr/evaluate.py --manifest data/public/torgo/export/controls/test.jsonl --audio-root data/public/torgo/export/controls --group-by kind --model-dir models/parakeet-ft-$NEW/onnx-int8 --out reports/finetune-torgo/controls-$NEW
echo "== GUARD, controls: first severe-focus run against this run"
python scripts/asr/ci.py --manifest data/public/torgo/export/controls/test.jsonl --stock reports/finetune-torgo/controls-$PREV --finetuned reports/finetune-torgo/controls-$NEW )
```

In the training log, check for "--offload-optimizer: AdamW's running averages and its update live in system RAM", the `--dev-speakers` and `--repeat-speakers` lines (dev 358 -> 144, train 2095 -> 2945 clip uses, as in the first run), and one `curve:` line per epoch ending in minutes and peak GPU memory.

In the "first severe-focus run against this run" lines, ci.py prints the first run as "Stock WER". A positive Delta means this run scored better and a negative one worse. With the same recipe, either sign is noise.

### Results

  Training: selected fine-tuned, best epoch 6, dev WER 43.86%, 4 checkpoints averaged; ran all 10 epochs; minutes of training batches per epoch 5.6–6.1; peak GPU memory 7.20–7.76 GB; 60 minutes in all.
  First run for comparison: best epoch 7, dev WER 41.51%; epochs 4–7 took 16–21 minutes each; about 2 hours 40 minutes in all.

  | Epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | Dev WER (severe dev) | 59.79% | 51.70% | 49.35% | 46.48% | 47.52% | 44.13% | 43.86% | 44.91% | 44.91% | 45.17% |
  | Training loss | 2.738 | 1.296 | 0.963 | 0.801 | 0.707 | 0.618 | 0.595 | 0.565 | 0.565 | 0.538 |
  | Minutes | 5.8 | 5.7 | 6.1 | 6.1 | 5.8 | 5.6 | 5.6 | 5.6 | 5.6 | 5.6 |
  | Peak GPU GB | 7.32 | 7.20 | 7.25 | 7.51 | 7.34 | 7.57 | 7.75 | 7.75 | 7.55 | 7.76 |

  Curve: models/parakeet-ft-prototype-severe-focus-bottom8-offload/curve.csv

| Test recordings | n clips | Prototype (all) | Severe focus, first run | Severe focus, this run | First run against this run: Delta (95% CI) |
| --- | ---: | ---: | ---: | ---: | --- |
| Goal: severe speakers' test sentences | 65 | 41.79% | 37.64% | 38.51% | −0.88 (−5.40 to 3.24) |
| F01 full test | 23 | 58.70% | 50.00% | 54.35% | −4.35 (−13.90 to 3.58) |
| M01 full test | 80 | 48.31% | 40.10% | 40.10% | 0.00 (−2.80 to 2.74) |
| M02 full test | 84 | 45.24% | 36.19% | 39.52% | −3.33 (−8.02 to 0.83) |
| M04 full test | 87 | 59.11% | 57.64% | 56.65% | 0.99 (−7.74 to 9.00) |
| Guard: F04 (mild) full test | 103 | 14.34%* | 13.57%† | 13.18% | 0.39 (−2.56 to 3.44) |
| Guard: M03 (mild) full test | 92 | 7.05%* | 5.81%† | 6.64% | −0.83 (−2.34 to 0.00) |
| Guard: controls (typical speech) | 1179 | 9.86%* | 10.20%† | 10.14% | 0.07 (−0.38 to 0.49) |

\* Scored before the British-spelling fix to the scorer (30 Sep 2026); not re-scored yet.
† Re-scored with the fix by this run's ci.py lines; the first run's notes give 14.73%, 6.64% and 10.51% under the old scorer. The goal set and the severe speakers did not change under the fix. This run was scored with the fix throughout.

### Conclusions

`--offload-optimizer` removes the slowdown. Every epoch ran at full speed (5.6–6.1 minutes) and the run took 60 minutes instead of about 2 hours 40 minutes, with peak GPU memory 7.2–7.8 GB. It should be the default for bottom-8 runs on this GPU.

Two runs of one recipe differ by up to about 1 point on the goal set and up to 3–4 points on single speakers' test sets, and dev picked a different best epoch (6 against 7). A gap between two recipes smaller than that, from one run each, can't be told apart from run-to-run noise.

Both severe-focus runs beat the prototype on the goal set (by 4.16 and 3.28 points), so the severe-focus gain holds across runs, though neither interval clears zero on its own. Rounding from the CPU optimizer and the GPU's non-repeatable arithmetic both contribute, and this run can't separate them (the 30-update check showed the two optimizers start out matching).

</details>

## Related

[Prototype model, severe focus](../prototype-severe-focus-bottom8/notes.md), [Prototype model: all eight dysarthric speakers, bottom 8 blocks](../prototype-all-dysarthric-bottom8/notes.md)
