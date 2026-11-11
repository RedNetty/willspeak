# F-2: M02, top 8 blocks, fixed dev score

Proposal: §5.2 / F-2 recipe comparison

Status: recorded 29 Sep 2026.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

Rerun the top-eight recipe with the corrected dev score, so the comparison with bottom eight uses the same scoring rule on both sides.

## Setup

Same 37 M02 test clips. The earlier top-eight model (normalized transcripts) scored 52.63% in int8, but its checkpoints were picked with the old score.

## Plan

Train blocks 16–23 with --dev-scoring evaluate --keep-stock, score in NeMo and int8, and compare with the fixed-score whole-block bottom-eight model.

## Why

Comparing two encoder regions only makes sense if both used the same checkpoint-selection rule.

## Limits

This compares the two recipes on M02 only. It can't say which encoder region is best for other speakers. GPU runs vary and each recipe ran once.

## Results

| Model | NeMo WER | Int8 WER |
|---|---:|---:|
| Stock | 65.26% | 60.00% |
| Top 8, fixed dev score | 64.21% | 62.11% |
| Bottom 8, fixed dev score | 44.21% | 44.21% |

Bottom eight beats top eight in int8 by 17.89 points, 95% CI 8.49 to 30.56.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

Before this, the bottom-8 model beat the top-8 model in int8 by 8.42 points (95% CI 1.05 to 17.05), but the top-8 checkpoints had been picked by the old val_wer. The rerun keeps the top-8 recipe and changes one thing: checkpoints are picked by the fixed score, and stock is kept if nothing beats it. That makes bottom eight against top eight a like-for-like comparison.

### Prediction (Before the Run)

I still predict the bottom-8 model will beat the top-8 rerun. Even before the val wer changes, training the layers chosen for the bottom-8 model looked better overall.

Run from app/ in WSL. Each block stops at its first error.

```bash
( set -e
source ~/venvs/willspeak-train/bin/activate
python scripts/asr/train.py --export data/public/torgo/export/M02 --out models/parakeet-ft-M02f --freeze-encoder-layers 16 --batch 2 --grad-accum 8 --warmup-steps 10 --freeze-batchnorm --normalize-targets --dev-scoring evaluate --keep-stock
python scripts/asr/nemo_hyp.py --nemo models/parakeet-ft-M02f/parakeet-ft.nemo --manifest data/public/torgo/export/M02/test.jsonl --audio-root data/public/torgo/export/M02 --out reports/finetune-torgo/M02f-test-nemo.jsonl
python scripts/asr/export.py --nemo models/parakeet-ft-M02f/parakeet-ft.nemo --out models/parakeet-ft-M02f/onnx-int8 )

( set -e
source .venv/bin/activate
python scripts/asr/evaluate.py --manifest data/public/torgo/export/M02/test.jsonl --audio-root data/public/torgo/export/M02 --hyp reports/finetune-torgo/M02f-test-nemo.jsonl --group-by kind --out reports/finetune-torgo/M02f-test-nemo
python scripts/asr/evaluate.py --manifest data/public/torgo/export/M02/test.jsonl --audio-root data/public/torgo/export/M02 --group-by kind --model-dir models/parakeet-ft-M02f/onnx-int8 --out reports/finetune-torgo/M02f-test-int8
python scripts/asr/ci.py --manifest data/public/torgo/export/M02/test.jsonl --stock reports/finetune-torgo/M02-test-stock-nemo --finetuned reports/finetune-torgo/M02f-test-nemo
python scripts/asr/ci.py --manifest data/public/torgo/export/M02/test.jsonl --stock reports/finetune-torgo/M02-stock --finetuned reports/finetune-torgo/M02f-test-int8
python scripts/asr/ci.py --manifest data/public/torgo/export/M02/test.jsonl --stock reports/finetune-torgo/M02f-test-int8 --finetuned reports/finetune-torgo/M02e-A3-test-int8 )
```

### Results

    Training: selected fine-tuned, best epoch 0, dev WER 55.1%, 4 checkpoints averaged (all under stock's 57.97%); stopped early
    NeMo score 64.21%   Int8 Score 62.11%
    ci.py against stock: NeMo Delta 1.05 (95% CI -5.68 to 8.43), Int8 Delta -2.11 (95% CI -5.62 to 0.00)
    ci.py, the top-8 rerun against the bottom-8 model (Int8): Delta 17.89 (95% CI 8.49 to 30.56)
    Words, Int8:  wrong 46, deleted 4, inserted 9 of 95
    Single words / sentences, Int8: 111.1% / 42.6%

</details>

## Related

[F-2: M02, bottom 8 blocks, which parts to train](../f2-m02-bottom8-parts/notes.md)
