# F-1: severe speakers, leave-one-speaker-out, bottom 8 blocks

Proposal: §5.2 / H2, needed before H3.

Status: recorded 29 Sep 2026. Training metadata is incomplete.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

Repeat F-1 using the bottom-eight recipe from the later single-speaker runs.

## Setup

Same four held-out speakers and LOSO test sets as the top-eight F-1 run. Encoder blocks 0–7 train whole and fixed dev scoring picks the checkpoint. F-2 releases after five epochs instead.

## Plan

Train one model per held-out speaker, score its int8 export against stock and the first F-1 run, then run each model on the F-2 test set for H3.

## Why

H3 compares single-speaker and multi-speaker training, so both sides need the same encoder recipe.

## Limits

Shows whether this recipe helps held-out-speaker WER. It can't answer H3 alone: the LOSO clips differ from F-2's later-session clips, and so do the release rules and amount of training data.

## Results

| Held-out speaker | Stock WER | Top-8 F-1 | Bottom-8 F-1 |
|---|---:|---:|---:|
| F01 | 73.91% | 67.39% | 60.87% |
| M01 | 70.05% | 62.32% | 59.42% |
| M02 | 59.52% | 51.43% | 52.86% |
| M04 | 91.63% | 74.38% | 70.44% |

Against stock, the bottom-eight intervals clear zero for M01 and M04. Every top-eight vs bottom-eight interval includes or touches zero.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

Why: H3 is F-2 against F-1. The first F-1 run trained the top eight blocks and picked checkpoints with the old dev
score. The top-8 rerun then showed that on M02 the top eight gain about nothing while the bottom eight (the bottom-8 model)
gain about 16 points, and the F-2 speaker runs use the bottom eight. Against the first F-1 run, F-2 would mostly measure the
recipe. So this reruns F-1 with the bottom-8 model's recipe: encoder blocks 0-7 whole (--train-parts all),
the fixed dev score, and stock kept if no checkpoint beats it.

Checkpoints are still picked on dev, here the other speakers' dev clips, which is big enough to choose on (the
per-speaker dev sets in the F-2 runs aren't). Warm-up stays at the default 100 steps (an epoch is about 115-125
steps). Everything else matches the first F-1 run.

Prediction before the run: not recorded.

Run after the F-2 speaker runs finish. About 30 minutes per speaker, 2 hours in all. GPU free, PC awake.

```bash
( set -e
source ~/venvs/willspeak-train/bin/activate
for S in F01 M01 M02 M04; do
  python scripts/asr/train.py --export data/public/torgo/export/loso-$S --out models/parakeet-ft-f1-loso-$S-bottom8 --train-parts all --batch 2 --grad-accum 8 --freeze-batchnorm --normalize-targets --dev-scoring evaluate --keep-stock
  python scripts/asr/export.py --nemo models/parakeet-ft-f1-loso-$S-bottom8/parakeet-ft.nemo --out models/parakeet-ft-f1-loso-$S-bottom8/onnx-int8
done )

( set -e
source .venv/bin/activate
for S in F01 M01 M02 M04; do
  python scripts/asr/evaluate.py --manifest data/public/torgo/export/loso-$S/test.jsonl --audio-root data/public/torgo/export/loso-$S --group-by kind --model-dir models/parakeet-ft-f1-loso-$S-bottom8/onnx-int8 --out reports/finetune-torgo/f1-loso-$S-bottom8
  python scripts/asr/ci.py --manifest data/public/torgo/export/loso-$S/test.jsonl --stock reports/finetune-torgo/loso-$S-stock --finetuned reports/finetune-torgo/f1-loso-$S-bottom8
  python scripts/asr/ci.py --manifest data/public/torgo/export/loso-$S/test.jsonl --stock reports/finetune-torgo/loso-$S --finetuned reports/finetune-torgo/f1-loso-$S-bottom8
done )
```

The second ci.py line compares the first F-1 run (first folder) with the rerun. Positive Delta means the new recipe wins. Check each training log for "210461318 trainable", "only norm_feed_forward1, ...
in encoder blocks 0-7 trainable" and "stock dev WER" (it should print a value and not crash).

Result (Int8, each speaker's loso test clips), run 29 Sep 2026 on Yack.

I started this run before the folder renaming reached WSL, so its files are named
models/parakeet-ft-loso-<SPK>-b and reports/finetune-torgo/loso-<SPK>-b. The first lines of the H3 block in the
[F-2 comparison notes](../f2-severe-bottom8-5epochs/notes.md) rename them to the f1-loso-<SPK>-bottom8 names used above.

A positive Delta means the F-1 rerun (bottom 8) is better.

| speaker | stock | top 8 (first run) | bottom 8 (rerun) | bottom 8 against stock: Delta (95% CI) | bottom 8 against top 8: Delta (95% CI) |
| --- | --- | --- | --- | --- | --- |
| F01 | 73.91% | 67.39% | 60.87% | 13.04 (0.00 to 39.03) | 6.52 (0.00 to 14.63) |
| M01 | 70.05% | 62.32% | 59.42% | 10.63 (4.59 to 17.86) | 2.90 (-3.53 to 10.16) |
| M02 | 59.52% | 51.43% | 52.86% | 6.67 (-3.34 to 18.97) | -1.43 (-8.18 to 6.10) |
| M04 | 91.63% | 74.38% | 70.44% | 21.18 (11.89 to 31.96) | 3.94 (-3.14 to 11.35) |

Test clips: F01 23, M01 80, M02 84, M04 87.

Training, M04: picked the fine-tuned model, best epoch 5, dev WER 27.2%, 4 checkpoints averaged.

Training, F01 / M01 / M02: not recorded. The 29 September training logs (`logs/train_parakeet-20260929-*.log`) stayed in the WSL tree on Yack and aren't in this checkout. The command below reads them there.

What the numbers show, before interpreting them:

All four speakers beat stock. The interval clears zero for M01 and M04, touches zero for F01 (lower end exactly 0.00), and includes zero for M02.

Against the first F-1 run (top 8), no speaker moves by more than 23-87 clips can detect, since every interval includes or touches zero. F01, M01 and M04 lean slightly better and M02 slightly worse.

The first F-1 run cleared zero against stock for M01, M02 and M04. The rerun clears it for M01 and M04.

Training lines for the other three speakers (run from app/ in WSL):

```bash
for f in $(grep -l -- "loso-.*-b/" logs/train_parakeet-20260929-*.log); do echo "== $f"; grep -hE "train [0-9]+ clips|parameters exact|stock dev WER|best dev WER|averaged|releasing the stock|saved .*-b/" "$f"; done
```

</details>

## Related

[F-1: severe speakers, leave-one-speaker-out, top 8 blocks](../f1-loso-top8/notes.md), [F-2: severe speakers, bottom 8 blocks, fixed 5 epochs](../f2-severe-bottom8-5epochs/notes.md)
