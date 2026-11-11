# F-2: M02, top 8 blocks, batch norm frozen

Proposal: §5.2 / F-2 development

Status: results recorded 24 Sep 2026. No prediction was written before the run. The conclusions were added on 21 October 2026, working from the recorded results.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

Repeat M02's first fine-tune, but with the batch-norm statistics frozen.

## Setup

Same 37 M02 test clips. Transcripts were still raw since the normalization change hadn't reached WSL. Early stop after epoch 7, four best checkpoints averaged.

## Plan

Decode stock and both fine-tuned checkpoints in NeMo, keeping ONNX export out of the comparison.

## Why

To check whether protecting the stock statistics undoes the first run's damage.

## Limits

Same decoder and test set, so the runs compare fairly, but the small gain over stock is inconclusive. Without F-1 on the same clips this can't credit transcript normalization or show H3.

## Results

| Model, NeMo | WER |
|---|---:|
| Stock | 65.26% |
| Batch norm updating | 103.16% |
| Batch norm frozen | 63.16% |

The gain over stock is 2.11 points, with a 95% CI of −3.57 to 8.96. The exported-model check is in the normalized-transcript note.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### What I expected before running this run

(none written before the run)

### Results

Machine: Yack, WSL2, 24 GB given to WSL


```bash
python scripts/asr/train.py --export data/public/torgo/export/M02 --out models/parakeet-ft-M02b --freeze-encoder-layers 16 --batch 2 --grad-accum 8 --warmup-steps 10
```

  Batch norm frozen (log line present); transcripts as written in the manifest (the normalised-transcript change was not yet on the training machine)
  Early stop after epoch 7; best checkpoint epoch 4; 4 best averaged

M02 test, 37 clips, every model decoded by NeMo (no export)
  Stock:              WER 65.26%   CER 50.71%   (ci.py 95% CI 46.40% to 88.41%)
  First F-2 run (batch norm updating):   WER 103.16%  CER 71.23%
  This run (batch norm frozen):          WER 63.16%   CER 45.28%
  ci.py, stock against the batch-norm-frozen run: Delta 2.11 points (95% CI -3.57 to 8.96), the interval includes zero

Decoder matters at this sample size: stock scored 60.00% exported and 65.26% in NeMo on the same 37 clips.

### What I concluded

*Added 21 October 2026, from the recorded results above.*

Freezing the batch-norm statistics fixed the damage from the first run. On M02's 37 test clips, all decoded in NeMo (lower is better), the frozen run got 63.16% and the run with updating statistics got 103.16%. That's a 40-point difference from a single change.

Against stock (65.26% in NeMo) the frozen run is 2.11 points better. The interval (-3.57 to 8.96) includes zero though, so there's no established gain over stock. It doesn't prove the run equals stock either.

Later runs also got little out of training the top eight blocks on M02. The top-8 fixed-dev run scored 62.11% Int8 against stock's 60.00%, while training the bottom eight blocks got to 44.21% ([top 8, fixed dev](../f2-m02-top8-fixed-dev/notes.md), [bottom 8, which parts](../f2-m02-bottom8-parts/notes.md)). So the useful change was moving training to the bottom of the encoder. The batch-norm fix alone wasn't it.

Stage R later found the same batch-norm effect on a different recipe. The block-8 adapter with trainable statistics got 324.7 of 425 pooled words wrong, as bad as stock (323). With the statistics frozen it got 212.0 (`reports/f2-adapter-levers`).

Every model that could go to the expo needs frozen batch-norm statistics.

</details>

## Related

[F-2: M02, top 8 blocks, normalized transcripts](../f2-m02-top8-normalized-transcripts/notes.md)
