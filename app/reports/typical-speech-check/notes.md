# Check: TORGO controls, typical-speech forgetting

Proposal: §6.2, the forgetting pitfall. Supports F-1 and F-2.

Status: run 30 Sep 2026. Interpretation added 21 October 2026.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

I wanted to check whether fine-tuning on dysarthric speech makes the model worse on typical speech.

## Setup

The TORGO controls' prompt-disjoint test set. The models are the ones in the F-1 and F-2 comparison columns, plus the dev-picked M02 bottom-eight model.

## Plan

Score stock and every listed int8 model on the same controls manifest, with WER and paired bootstrap intervals.

## Why

A gain on dysarthric speech might come with a loss somewhere else, and that cost should sit next to the gains.

## Limits

This measures forgetting on the TORGO controls only. It doesn't cover all typical English or the intended user.

## Results

No fine-tuned model got worse on typical speech. The eight F-1 models gained 2.3-2.7 points over stock's 12.40%, all intervals above zero. The F-2 models gained 0-1 point. The full table is in the Run Record.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

Every model in the F-1 or F-2 column is scored on the control speakers' test set
(export/controls/test.jsonl, typical speech), int8 only since that's what the Pi runs. Stock is the
reference, and each model gets ci.py's interval against it. If a model gains on its speaker but
loses more than a few points here, that loss is the main finding for it.

### Prediction (Before the Run)

I suspect that most models could potentially see a gain in points, mainly due to the model hopefully learning on Dysarthric speech changing how it identifies speech as a whole. In the sense that the model is trying to take this normalized speech and apply it to what it's learned about dysarthric, and that may cause a higher WER (I suspect by 1-5 points).

### Commands

Run from app/ in WSL once the F-2 speaker runs and the F-1 rerun are done and every model below has its onnx-int8 folder. It stops at the first error.

```bash
( set -e
source .venv/bin/activate
python scripts/asr/evaluate.py --manifest data/public/torgo/export/controls/test.jsonl --audio-root data/public/torgo/export/controls --group-by kind --out reports/finetune-torgo/controls-stock
for M in M02e-A3 f2-M02-bottom8-5epochs f2-M01-bottom8-5epochs f2-M04-bottom8-5epochs f2-F01-bottom8-5epochs loso-F01 loso-M01 loso-M02 loso-M04 f1-loso-F01-bottom8 f1-loso-M01-bottom8 f1-loso-M02-bottom8 f1-loso-M04-bottom8; do
  python scripts/asr/evaluate.py --manifest data/public/torgo/export/controls/test.jsonl --audio-root data/public/torgo/export/controls --group-by kind --model-dir models/parakeet-ft-$M/onnx-int8 --out reports/finetune-torgo/controls-$M
  python scripts/asr/ci.py --manifest data/public/torgo/export/controls/test.jsonl --stock reports/finetune-torgo/controls-stock --finetuned reports/finetune-torgo/controls-$M
done )
```

### Results

Run 30 Sep 2026 on Yack, 1179 test clips for every model. The reference word count wasn't recorded, because the controls' `results.json` stayed in the WSL tree and isn't in this checkout.

| model | controls WER (Int8) | Delta against stock (95% CI) |
| --- | --- | --- |
| stock | 12.40% | - |
| M02e-A3 (F-2, M02, bottom 8, dev-picked) | 11.39% | 1.01 (0.27 to 1.77) |
| f2-M02-bottom8-5epochs (F-2, fixed rule) | 11.79% | 0.61 (0.00 to 1.25) |
| f2-M01-bottom8-5epochs | 11.72% | 0.68 (0.16 to 1.25) |
| f2-M04-bottom8-5epochs | 12.26% | 0.14 (-0.41 to 0.71) |
| f2-F01-bottom8-5epochs (context only) | 12.36% | 0.03 (-0.44 to 0.57) |
| loso-F01 (F-1, top 8) | 10.03% | 2.36 (1.42 to 3.34) |
| loso-M01 (F-1, top 8) | 9.86% | 2.53 (1.59 to 3.52) |
| loso-M02 (F-1, top 8) | 9.83% | 2.57 (1.69 to 3.48) |
| loso-M04 (F-1, top 8) | 9.86% | 2.53 (1.63 to 3.44) |
| f1-loso-F01-bottom8 (F-1, bottom 8) | 9.83% | 2.57 (1.69 to 3.51) |
| f1-loso-M01-bottom8 (F-1, bottom 8) | 9.70% | 2.70 (1.78 to 3.76) |
| f1-loso-M02-bottom8 (F-1, bottom 8) | 10.07% | 2.33 (1.40 to 3.34) |
| f1-loso-M04-bottom8 (F-1, bottom 8) | 10.00% | 2.40 (1.46 to 3.41) |


A negative Delta means the fine-tuned model is worse than stock on typical speech.

What the numbers show, before any reading of them. No model is worse than stock: no interval sits below zero. All eight F-1 models are 2.3-2.7 points better, every interval clear of zero. The F-2 models change less. M02e-A3 (+1.01) and f2-M01 (+0.68) clear zero, f2-M02 (+0.61) touches zero, and f2-M04 and f2-F01 show no change. The top 8 and bottom 8 F-1 models land within 0.4 points of each other.

The control speakers were recorded in the same TORGO room, with the same microphones and read-prompt style as the training speakers. So part of any gain could come from the recording setup and not from typical English in general, and this check can't separate the two.

### Conclusions

*Interpretation added 21 October 2026 from the recorded results above.*

On the 1,179 control clips (lower is better), the exact ranges are 2.33 to 2.70 points better for the F-1 models and 0.03 to 1.01 for the F-2 models.

My prediction expected typical-speech WER to rise by 1 to 5 points. It didn't rise for any model, and the F-1 models improved instead. The recording-setup caveat above may explain part of that.

Later guard rows on the same 1,179 clips agree. The all-recordings prototype got 9.86%. The severe-focus prototype got 10.51% under the old scorer (0.64 points worse than the all-recordings prototype, interval just clear of zero) and 10.20% after the British-spelling fix, and its offload repeat 10.14%. The sentences-only prototype, at 13.07%, is the one model no better than stock.

The models now leading on severe speech have never been scored here: no block-8 adapter, no adapter on F-1, and no severe-pretrained adapter (stage P). Their forgetting is unmeasured. Before an adapter goes to the expo, it should be scored on this controls set with `ci.py` against stock, like the rows above.

</details>

## Related

[F-1: severe speakers, leave-one-speaker-out, bottom 8 blocks](../f1-loso-bottom8/notes.md), [F-2: severe speakers, bottom 8 blocks, fixed 5 epochs](../f2-severe-bottom8-5epochs/notes.md)
