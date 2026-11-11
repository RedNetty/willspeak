# Setup: F01, batch-norm diagnosis

Proposal: §4.2 / fine-tuning prerequisite

Status: recorded 24 Sep 2026.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

Repeat the F01 smoke test with batch-norm statistics frozen, to find out what damaged it.

## Setup

Same 85 training clips, six optimizer steps and 23 test clips as the smoke test. The only change is --freeze-batchnorm.

## Plan

Compare frozen batch norm against updating batch norm and stock, and repeat the silence check.

## Why

Freezing the encoder weights doesn't freeze the batch-norm statistics. If the baseline recipe itself damages the model, fine-tuning could look like it fails when it doesn't.

## Limits

This tests the batch-norm explanation for this one smoke run. It can't show that longer training improves recognition, or that F01 generalizes across sessions.

## Results

| Model | Test WER | Silence |
|---|---:|---|
| Stock | 73.91% | Empty output |
| Batch norm updating | 93.48% | "I don't know." |
| Batch norm frozen | 73.91% | Empty output |

Frozen batch norm against stock: Δ 0.00 points, 95% CI 0.00 to 0.00.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

The smoke test lost 19.57 points and started hallucinating text on silence. The batch-norm diagnosis changes one thing, --freeze-batchnorm, to test whether the batch-norm statistics caused that.

### What I expect before running the batch-norm diagnosis

WER on the 23 F01 clips: 60-70%, closer to stock than the smoke test, because I think batch norm caused the near 20 point drop.
On 1 s of silence: nothing. If batch norm wasn't the cause, it might still hallucinate text.
ci.py, stock against the diagnosis: probably a ~10 point gain or loss, and with 23 clips the interval will include zero. I don't fully understand the cause of the loss yet, but if batch norm was the problem I'd expect a ~10 point gain, somewhere in the range of 60-70%.
I would be surprised if the result is dramatically good, below 50%.

### Results

Machine: Yack, WSL2, 24 GB given to WSL

The batch-norm diagnosis, the smoke test again with frozen batch norm

```bash
python scripts/asr/train.py --export data/public/torgo/export/F01 --out models/b6-f01 --epochs 1 --freeze-encoder-layers 16 --batch 2 --grad-accum 8 --warmup-steps 10 --freeze-batchnorm
```

  85 train clips, 3.4 min, 10 dev; 6 optimizer steps; dev WER during training 87.5% (the smoke test: 95.8%)
  F01 test, 23 clips: WER 73.91%   CER 51.85%   PER 51.61%
  1 s of silence: ""
  ci.py, stock through export.py against the batch-norm diagnosis: Delta 0.00 points (95% CI 0.00 to 0.00)

The same 23 clips, all four models
  Prebuilt stock:           73.91%   silent
  Stock through export.py:  73.91%   silent
  Smoke test (batch norm updating):         93.48%   "I don't know."
  Batch-norm diagnosis (batch norm frozen): 73.91%   silent

Prediction against actual
  WER: predicted 60-70%, measured 73.91%
  Silence: predicted nothing, measured nothing
  ci.py: predicted the interval includes zero, measured 0.00 to 0.00
  Batch norm as the cause of the smoke test's loss: predicted yes, measured yes
  Surprise range, below 50%: not reached

### What I concluded

1. The smoke test and the diagnosis differ only in --freeze-batchnorm, and the diagnosis shows the 19.57 point loss and the hallucinations on silence may well have been caused by batch norm. This time the model matched stock at 73.91% and stayed silent on silence.
2. With only 6 small steps this wasn't really a test of getting better results. It was a check that training doesn't damage the model or make it worse than stock. That's still promising, though my prediction was a bit off: I assumed 60-70% and got the same as stock, 73.9%.
3. The first F-2 run would probably have been a complete failure without this diagnosis and the fixes that came after it. M02 has 156 training clips in batches of 2, so 78 batches per epoch, or about 780 over 10 epochs. The smoke test wiped out 99% of stock's batch-norm numbers in just 43 batches, so after 780 they'd have been gone entirely. Any gains would have been fighting about 20 points of loss, like the smoke test shows, and the first F-2 run could have wrongly concluded that fine-tuning doesn't work.

</details>

## Related

[Setup: F01, smoke test and stock export check](../setup-f01-smoke-test/notes.md), [F-2: M02, top 8 blocks, batch norm updating](../f2-m02-top8-bn-updating/notes.md)
