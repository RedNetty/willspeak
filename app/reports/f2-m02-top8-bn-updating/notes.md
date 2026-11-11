# F-2: M02, top 8 blocks, batch norm updating

Proposal: §5.2 / F-2 development

Status: recorded 24 Sep 2026. Interpretation added 21 October 2026.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

Run the first full single-speaker fine-tune, and if it fails, work out whether the checkpoint or the export is to blame.

## Setup

M02 has 156 training clips (9 min), 24 dev clips from Session1 and 37 test clips from Session2, prompt-disjoint, head mic. Encoder blocks 16–23 train. The run started before the frozen batch-norm default reached the training machine.

## Plan

Compare stock with the fine-tuned int8 model on the same test clips, then check dev output in NeMo, fp32 ONNX and int8 ONNX.

## Why

Splitting training damage from export damage tells me what the next run has to change.

## Limits

This can document how this recipe and its export failed on M02. It can't rule out fine-tuning in general, and this batch-norm-updating run shouldn't be read as the later frozen recipe. Each decoder should be compared with its own stock baseline.

## Results

| Decoder / set | Stock WER | Fine-tuned WER |
|---|---:|---:|
| Int8 / test, 37 clips | 60.00% | 105.26% |
| NeMo / dev, 24 clips | 57.97% | 75.36% |
| Int8 / dev, 24 clips | 62.32% | 89.86% |

On int8 test the fine-tuned model is 45.26 points worse, 95% CI 22.92 to 66.33 points worse.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

first real fine-tune, M02's earlier sessions in, last session out,
batch norm updating, about 100 steps

### What I expect before running the first F-2 run

M02 stock WER on its 37 test clips: somewhere in the range of 60-68%. T-1 had M02 at 45.99% on sentences,
and single-word clips score worse, and from F01 testing on untrained model, it got 73.91% of words wrong on F01's 23 test clips, which likely include single words (the dev clips did)
Which is nearly a 16 point increase from F01's 57.61% on sentences.
M02 fine-tuned WER: 35-44%. T-1 showed that M02's sentences are transcribed with a 45.99% WER and expecting that this round will provide better results.
ci.py, stock against fine-tuned: While I believe we may see some possible, gain of about 16-33 points going by my stock and fine-tuned guesses, so the interval should clear zero, 37 clips is still small
Fine-tuned model on 1 s of silence: nothing. I expect nothing, as we found out that our batch-norm diagnosis run where we froze batch-norm, may have
resolved this issue
I would be surprised if: we see a higher than stock range for fine-tuned, and it hallucinates on silence

### Results

Machine: Yack, WSL2, 24 GB given to WSL (32 GB PC)

The first F-2 run, F-2 on M02, batch norm updating

```bash
python scripts/asr/train.py --export data/public/torgo/export/M02 --out models/parakeet-ft-M02 --freeze-encoder-layers 16 --batch 2 --grad-accum 8 --warmup-steps 10
```

  156 train clips (9.0 min), 24 dev; 10 epochs, 100 optimizer steps; 4 best checkpoints averaged
  Training-log dev WER best 84.3% at epoch 9 (counts capitals and punctuation, so not comparable with evaluate.py)
  Epochs 6 and 7 ran about 40 times slower while NVIDIA Instant Replay shared the GPU; speed does not change the numbers
  Batch norm was NOT frozen in this run: the frozen default was committed at 17:48, after
  The run started at 17:16, and the log has no "batch-norm statistics stay" line. It repeats the smoke test's setting.
  Splits: train and dev Session1, test Session2, all head mic
    train 123 words + 33 sentences, dev 17 + 7, test 27 + 10

M02 test, 37 clips (Session2)
  Stock:    WER 60.00%    CER 46.70%   PER 47.78%   (ci.py 95% CI 41.33% to 82.41%)
  Fine-tuned:  WER 105.26%   CER 87.03%   PER 86.11%
  ci.py: Delta -45.26 points (95% CI -66.33 to -22.92), fine-tuned is worse
  Fine-tuned on 1 s of silence: ""

M02 dev, 24 clips (Session1, the same session as training)
  Stock:    WER 62.32%   CER 43.51%
  Fine-tuned:  WER 89.86%   CER 78.60%

The NeMo decode check, M02 dev decoded by NeMo itself (no export, no int8)
  Stock:    WER 57.97%   CER 44.56%
  Fine-tuned:  WER 75.36%   CER 54.04%
  Word errors out of 69 dev words (substitutions / deletions / insertions)
    stock, NeMo 26 / 7 / 7;  stock, export 30 / 2 / 11
    fine-tuned, NeMo 34 / 12 / 6;  fine-tuned, export 31 / 30 / 1

The export-parts check, which part of the export (M02 dev, fine-tuned)
  NeMo, no export:           WER 75.36%   deletions 12
  fp32 export (no int8):     WER 81.16%   deletions 12
  int8 encoder only:         WER 89.86%   deletions 28
  int8 everything (the first F-2 run):      WER 89.86%   deletions 30

Prediction against actual
  Stock: predicted 60-68%, measured 60.00%
  Fine-tuned: predicted 35-44%, measured 105.26%
  ci.py: predicted a 16-33 point gain clearing zero, measured a 45.26 point loss clearing zero
  Silence: predicted nothing, measured nothing
  Surprise, fine-tuned worse than stock: reached

### What I concluded

Interpretation added 21 October 2026 from the recorded results above.

The checkpoint was already damaged before any export. On M02's 24 dev clips (69 words) decoded by NeMo itself, the fine-tuned model scored 75.36% against stock's 57.97%, 17.39 points worse (lower is better). On the 37 test clips in NeMo it scored 103.16% against 65.26% (recorded in the batch-norm-frozen note).

The export then made it worse, mostly in the int8 encoder. Along the export chain on the same dev clips it went NeMo 75.36%, fp32 export 81.16%, int8 encoder only 89.86%, all int8 89.86%. Deletions jumped from 12 to 28 at the int8 encoder step, so most of the export loss is dropped words, not substitutions.

So the answer is both, in that order. Training with batch-norm statistics updating damaged the model, and int8 quantisation of the encoder added more loss on top. The int8 test result (105.26% against 60.00%, 45.26 points worse, interval clear of zero) is the sum of the two.

Three of my five predictions didn't hold. The fine-tuned WER (predicted 35 to 44%, measured 105.26%) and the 16 to 33 point gain (a 45.26-point loss instead) went the other way, and the surprise (fine-tuned worse than stock) happened. The stock range (60 to 68%, measured 60.00%) and the silence check (nothing, measured nothing) held.

Later runs confirmed the batch-norm part. Frozen statistics on the same recipe gave 63.16% in NeMo (next note). With statistics trainable, the block-8 adapter in stage R lost its whole gain: 324.7 wrong of 425 pooled test words, against 323 for stock and 212.0 for the frozen default (`reports/f2-adapter-levers`). Batch-norm statistics stay frozen in every recipe now. The int8 part didn't repeat for the bottom-8 whole-block model (44.21% in NeMo and in Int8), and lab test 19 measured the default int8 recipe at 8 to 12 extra wrong words on the pooled set, with `--int8-ops matmul` costing nothing measurable.

</details>

## Related

[F-2: M02, top 8 blocks, batch norm frozen](../f2-m02-top8-bn-frozen/notes.md)
