# Prototype model, severe focus: all eight dysarthric speakers, bottom 8 blocks

Proposal: §2 deliverable (the tuned model for the Pi), aimed at the severe speakers, who are the project's goal.

Status: run 30 Sep 2026, results recorded. No prediction was written before the run.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

Rebuild the prototype model with training aimed at the four severe speakers (F01, M01, M02, M04), still keeping every other speaker's recordings in the mix.

## Setup

The prototype model (all recordings) gets 41.79% wrong on the severe speakers' test sentences (stock 58.42%). It picks checkpoints on dev recordings from all eight speakers, and the easy mild speakers pull that choice toward what already works. Its dev WER also flattened after about six epochs. Three changes, all aimed at the severe speakers:

1. `--dev-speakers F01,M01,M02,M04` picks checkpoints on the severe speakers' dev recordings only.
2. `--repeat-speakers F01,M01,M02,M04 --repeat 2` uses severe training recordings twice per epoch. Mild and moderate ones stay in once, since removing data hurt the sentences-only run.
3. `--lr 5e-5` instead of 2e-5, since the last run stopped improving.

The rest is the prototype recipe: bottom 8 blocks, frozen batch norm, normalised transcripts, fixed dev score, stock kept if nothing beats it, up to 10 epochs. The goal here is the best severe-speaker model. With three changes at once, a win can't be traced to any one of them.

## Plan

1. Train with the three changes on the same `export/all-dysarthric` data.
2. Export to int8.
3. Score on the prototype's recordings: the severe test sentences (the goal), each severe speaker's full test set, mild speakers F04 and M03, and the controls (typical speech).
4. Compare with the prototype on each.

## Why

The device is for severe speakers, so I want lower WER on their sentences without losing typical speech.

## Limits

It can't say which change is responsible, and it trained on all eight speakers, so nothing about an unseen one.

## Results

On the goal set (65 clips) the severe-focus model gets 37.64% wrong, against 41.79% for the prototype. The 4.16-point gain doesn't clear zero, so it's likely but unproven. It beats the prototype on all four severe speakers' full tests. That clears zero for M01 and M02, touches zero for F01, and is small and uncertain for M04. The mild speakers don't move. Controls are 0.64 points worse than the prototype but still better than stock. Full numbers are in the table in the Run Record.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

Not recorded.

### Commands

Run from app/ in WSL after `git pull` (train.py needs `--dev-speakers` and `--repeat-speakers`). GPU free, PC awake, Instant Replay off. Training takes about 50-60 minutes and scoring 15 minutes. Each block stops at its first error.

```bash
( set -e
source ~/venvs/willspeak-train/bin/activate
python scripts/asr/train.py --export data/public/torgo/export/all-dysarthric --check --dev-speakers F01,M01,M02,M04 --repeat-speakers F01,M01,M02,M04 --repeat 2
python scripts/asr/train.py --export data/public/torgo/export/all-dysarthric --out models/parakeet-ft-prototype-severe-focus-bottom8 --dev-speakers F01,M01,M02,M04 --repeat-speakers F01,M01,M02,M04 --repeat 2 --lr 5e-5 --train-parts all --batch 2 --grad-accum 8 --freeze-batchnorm --normalize-targets --dev-scoring evaluate --keep-stock
python scripts/asr/export.py --nemo models/parakeet-ft-prototype-severe-focus-bottom8/parakeet-ft.nemo --out models/parakeet-ft-prototype-severe-focus-bottom8/onnx-int8 )

( set -e
source .venv/bin/activate
E=data/public/torgo/export/all-dysarthric
ALL=prototype-all-dysarthric-bottom8
NEW=prototype-severe-focus-bottom8
python scripts/asr/evaluate.py --manifest $E/test-severe-sentences.jsonl --audio-root $E --group-by speaker --model-dir models/parakeet-ft-$NEW/onnx-int8 --out reports/finetune-torgo/$NEW-on-all-dysarthric-test-severe-sentences
echo "== GOAL, severe test sentences: stock against the new model"
python scripts/asr/ci.py --manifest $E/test-severe-sentences.jsonl --stock reports/finetune-torgo/all-dysarthric-test-severe-sentences-stock --finetuned reports/finetune-torgo/$NEW-on-all-dysarthric-test-severe-sentences
echo "== GOAL, severe test sentences: prototype (all) against the new model"
python scripts/asr/ci.py --manifest $E/test-severe-sentences.jsonl --stock reports/finetune-torgo/$ALL-on-all-dysarthric-test-severe-sentences --finetuned reports/finetune-torgo/$NEW-on-all-dysarthric-test-severe-sentences
for S in F01 M01 M02 M04 F04 M03; do
  python scripts/asr/evaluate.py --manifest data/public/torgo/export/loso-$S/test.jsonl --audio-root data/public/torgo/export/loso-$S --group-by kind --model-dir models/parakeet-ft-$NEW/onnx-int8 --out reports/finetune-torgo/$NEW-on-$S
  echo "== $S full test: prototype (all) against the new model"
  python scripts/asr/ci.py --manifest data/public/torgo/export/loso-$S/test.jsonl --stock reports/finetune-torgo/$ALL-on-$S --finetuned reports/finetune-torgo/$NEW-on-$S
done
python scripts/asr/evaluate.py --manifest data/public/torgo/export/controls/test.jsonl --audio-root data/public/torgo/export/controls --group-by kind --model-dir models/parakeet-ft-$NEW/onnx-int8 --out reports/finetune-torgo/controls-$NEW
echo "== GUARD, controls: prototype (all) against the new model"
python scripts/asr/ci.py --manifest data/public/torgo/export/controls/test.jsonl --stock reports/finetune-torgo/controls-$ALL --finetuned reports/finetune-torgo/controls-$NEW )
```

What to check in the training log: the `--dev-speakers` line (dev 358 -> about 150-200 clips), the `--repeat-speakers` line (train 2095 -> more clip uses), the Optimizer config `lr: 5e-05`, "stock dev WER" (severe dev only, so higher than before), the `curve:` lines, and "averaged ... checkpoints" or "releasing the stock model".

A positive Delta in the "prototype (all) against the new model" lines means the new model is better.

First attempt, 30 Sep 2026, I stopped by hand in epoch 4. The GPU's 8 GB had filled up (7,921 of 8,188 MiB, since the card also drives the Windows display and a browser was open). The driver pushed the overflow into system RAM and training slowed from about 2.0 to 0.7 batches per second, with the GPU at 100% but 38 C. I moved its folder aside to models/parakeet-ft-prototype-severe-focus-bottom8-stalled and used nothing from it. For the rerun I closed Windows apps and set the NVIDIA "CUDA - Sysmem Fallback Policy" to "Prefer No Sysmem Fallback", so a full GPU errors out instead of crawling.

Second attempt, 30 Sep 2026, started 19:07 and ran normally (1.76 batches per second, epoch 0 severe dev WER 61.1%). It ended at 19:24 in epoch 1 when the PC restarted. That folder was moved aside with an "-interrupted" name and nothing from it is used. The third attempt is the run reported below.

### Results

  Training: train 2945 clip uses per epoch (2095 recordings, 850 severe ones used twice), dev 144 (severe only, from 358); stock dev WER 86.42%; selected fine-tuned,
  best epoch 7, dev WER 41.51%, 4 checkpoints averaged; ran all 10 epochs (epoch 9 dev WER 43.86%, training loss 0.54).
  Training slowed to about 1.1 batches per second from epoch 5 on (GPU memory full, see docs/training.md Troubleshooting).
  Curve: models/parakeet-ft-prototype-severe-focus-bottom8/curve.csv

| Test recordings | n clips | Stock | Prototype (all) | Severe focus | Prototype against severe focus: Delta (95% CI) |
| --- | ---: | ---: | ---: | ---: | --- |
| Goal: severe speakers' test sentences | 65 | 58.42% | 41.79% | 37.64% | 4.16 (−1.38 to 9.40) |
| F01 full test | 23 | 73.91% | 58.70% | 50.00% | 8.70 (0.00 to 19.51) |
| M01 full test | 80 | 70.05% | 48.31% | 40.10% | 8.21 (0.45 to 16.89) |
| M02 full test | 84 | 59.52% | 45.24% | 36.19% | 9.05 (2.30 to 16.49) |
| M04 full test | 87 | 91.63% | 59.11% | 57.64% | 1.48 (−6.44 to 8.76) |
| Guard: F04 (mild) full test | 103 | 15.89% | 14.34% | 14.73% | −0.39 (−4.21 to 3.29) |
| Guard: M03 (mild) full test | 92 | 8.71% | 7.05% | 6.64% | 0.41 (−2.11 to 2.88) |
| Guard: controls (typical speech) | 1179 | 12.40% | 9.86% | 10.51% | −0.64 (−1.28 to −0.03) |

### Conclusions

The severe-focus model is the best so far for the severe speakers. It has a lower WER than the prototype on the goal set and on every severe speaker, and the gain is shown (CI clears zero) for M01 and M02. On the goal set the gain over the prototype isn't shown (CI −1.38 to 9.40), but the gain over stock is (20.79 points, CI 12.61 to 29.14).

The mild speakers are unchanged. Typical speech is 0.64 points worse than the prototype, and the interval only just clears zero, so it's a small real cost.

Three things changed at once (severe dev, severe recordings twice, lr 5e-5), so this run can't say which one helped. 37.64% is also still about 10 points above the high-20s goal.

This was scored before the British-spelling fix to the scorer (30 Sep 2026). With the fix, stock (58.42%) and the prototype (41.79%) on the goal set don't change. This model's guard numbers do drop with the fix (from the repeat run's ci.py lines): F04 14.73% → 13.57%, M03 6.64% → 5.81%, controls 10.51% → 10.20%. The guard comparisons above used the old scorer on both sides, so they still stand. The prototype's and stock's guard numbers haven't been re-scored.

</details>

## Related

[Prototype model: all eight dysarthric speakers, bottom 8 blocks](../prototype-all-dysarthric-bottom8/notes.md), [Prototype model, sentences only](../prototype-all-dysarthric-sentences-bottom8/notes.md)
