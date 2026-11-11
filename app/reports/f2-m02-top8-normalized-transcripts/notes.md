# F-2: M02, top 8 blocks, normalized transcripts

Proposal: §5.2 / F-2 development

Status: recorded 24 Sep 2026.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

First check the int8 export of the healthy frozen-batch-norm model. Then test training on normalized transcripts.

## Setup

Every comparison uses the same 37 M02 test clips (95 reference words). Stock scores 65.26% in NeMo and 60.00% after int8 export.

## Plan

Part 1 compares the frozen-batch-norm export with stock int8. Part 2 changes the form of the training transcripts, scores the result in NeMo, then scores its int8 export.

## Why

Training should spend its effort on the speech and not on formatting, and the exported model still has to give usable output.

## Limits

This compares transcript forms and checks these exports on M02. The intervals against stock include zero, so the gains are inconclusive. With one run per recipe I can't separate the transcript effect from run-to-run variation, and a desktop export isn't F-4 on the Pi.

## Results

| Model | NeMo WER | Int8 WER |
|---|---:|---:|
| Stock | 65.26% | 60.00% |
| Frozen batch norm, raw transcripts | 63.16% | 56.84% |
| Frozen batch norm, normalized transcripts | 57.89% | 52.63% |

The normalized-transcript model's int8 gain over stock is 7.37 points, with a 95% CI of −1.12 to 17.65.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### What the numbers so far are (M02 test, the same 37 clips every time)

Scored by NeMo (the checkpoint itself, no export):
  Stock 65.26%   first F-2 run 103.16% (batch norm updating)   batch-norm-frozen run 63.16% (batch norm frozen)
Scored after export to int8 (what the Pi runs):
  Stock 60.00%, with 11 of its 95 reference words deleted (the model said nothing for them)
The same stock model scores about 5 points apart in the two decoders, so I only compare NeMo with NeMo and export with export.
On the broken first F-2 model, putting the encoder in int8 made it go silent on words it had said in NeMo (M02 dev: 12 deleted words in NeMo, 28 with the int8 encoder).

### Part 1: the batch-norm-frozen run through the int8 export

Now that the model isn't broken, does the int8 encoder still make it go silent? I compare the batch-norm-frozen run exported with stock exported (60.00%).

What I expect before running

Exported WER: about 58-60%, a 3-5 point gain over its 63.16% in NeMo. I expect the gain to be small but in the right direction, now that some of the fine-tuning problems are fixed. Stock gained about the same when exported (65.26% in NeMo, 60.00% exported), so I'm basing it on that.
Deleted words (stock exported had 11): more. The int8 encoder seems to delete more (on the broken first F-2 model, M02 dev went from 12 deleted words in NeMo to 28 with the int8 encoder).
ci.py, stock exported against this run exported: around a 0-2 point difference, which is what my WER guess gives against stock's 60.00%. So the interval will include zero (this run's interval in NeMo was about 12 points wide), consistent with stock.
I would be surprised by a big WER gain, below 50% exported. I do expect the type of error to change (more deleted words, fewer wrong words).

### Part 2: the normalised-transcripts run (batch-norm-frozen run plus normalised transcripts)

The one change is that training transcripts are lowercase with no punctuation and no ⁇, the same form evaluate.py scores. That could help in two ways. The model stops spending training on capitals and full stops, and the dev score that picks the best checkpoint now counts words instead of formatting. The batch-norm-frozen run stopped after epoch 7 with its best checkpoint at epoch 4. I compare this run in NeMo with the batch-norm-frozen run in NeMo (63.16%) and stock in NeMo (65.26%).

What I expect before running

WER in NeMo: around 53-56%. Stock gained about 5 points going from NeMo to the export (65.26% to 60.00%), and I expect taking formatting out of training to give a gain of about that size or a bit more.
Against the batch-norm-frozen run: maybe a 7-10 point improvement, which is what 53-56% works out to against its 63.16%. Again the stock model is what's giving me this idea.
ci.py, stock against this run: I think it will clear zero. My guess puts this run about 9-12 points better than stock's 65.26% in NeMo, and the batch-norm-frozen run's interval ran about 6 points either side of its difference (37 clips gave an interval about 12 points wide), so a gain that size should stay clear of zero.
I would be pleasantly surprised by sub 50% WER, and by fewer deleted words than stock in NeMo.

### Results

Machine: Yack, WSL2, 24 GB given to WSL

Part 1, the batch-norm-frozen run exported (M02 test, 37 clips)
  WER 56.84%   CER 40.57%   PER 43.06%   deleted words 0 of 95 (stock exported: 11); wrong words 43 (stock: 34)
  ci.py, stock exported against the batch-norm-frozen run exported: Delta 3.16 points (95% CI -4.26 to 11.39), the interval includes zero
  Export against NeMo for the same checkpoint: the batch-norm-frozen run 63.16% in NeMo, 56.84% exported; stock 65.26% and 60.00%

Part 2, the normalised-transcripts run
  Log lines present: batch-norm statistics frozen, transcripts in scoring form
  Training: best checkpoint epoch 7 (dev WER 60.9%, now counting words only), ran all 10 epochs, 4 best averaged
  WER in NeMo 57.89%   CER 40.80%   PER 42.50%
  ci.py, stock in NeMo against the normalised-transcripts run: Delta 7.37 points (95% CI -1.11 to 17.50), the interval just includes zero
  The normalised-transcripts run against the batch-norm-frozen run, both in NeMo: 57.89% against 63.16%, 5.27 points better

The normalised-transcripts run through the int8 export (no prediction written)
  WER 52.63%   CER 38.68%   PER 41.39%   deleted words 8 of 95, wrong words 35
  ci.py, stock exported against the normalised-transcripts run exported: Delta 7.37 points (95% CI -1.12 to 17.65), the interval just includes zero
  Single words 88.9% (stock exported 107.4%); sentences 38.2% (stock exported 41.2%)

Prediction against actual
  Part 1, the batch-norm-frozen run exported WER: predicted 58-60%, measured 56.84%
  Part 1, deleted words against stock's 11: predicted more, measured 0
  Part 1, ci.py: predicted a 0-2 point difference with the interval including zero, measured 3.16 points, interval includes zero
  Part 1, surprise below 50%: not reached
  Part 2, the normalised-transcripts run WER in NeMo: predicted 53-56%, measured 57.89%
  Part 2, the normalised-transcripts run against the batch-norm-frozen run: predicted 7-10 points better, measured 5.27 points better
  Part 2, ci.py: predicted the interval clears zero, measured -1.11 to 17.50 (just includes zero)
  Part 2, surprise below 50%: not reached

### What I concluded

1. The int8 export doesn't silence a healthy fine-tuned model. The batch-norm-frozen run scored 63.16% in NeMo and 56.84% after export, the same direction and about the same size as stock (65.26% in NeMo, 60.00% exported). The words the first F-2 run dropped came from its batch-norm damage, not from the int8 encoder. For the Pi, a fine-tuned model can go through the same int8 export as stock with no extra fix. Against stock exported it's 3.16 points better, but the interval (-4.26 to 11.39) includes zero, so on 37 clips there's no measurable difference.
2. Normalised transcripts helped. The normalised-transcripts run scored 57.89% in NeMo against the batch-norm-frozen run's 63.16%, 5.27 points better, and the transcripts were the only change. That's one run each, so some of those 5 points could be run-to-run noise. Against stock in NeMo (65.26%) it's 7.37 points better. The interval (-1.11 to 17.50) only just includes zero, so it's close to a measurable gain but not there yet on 37 clips. The first F-1 run should use frozen batch norm and normalised transcripts, which are now the defaults.
3. I was close on Part 1. I said 58-60% and it came in at 56.84%, because exporting helped the batch-norm-frozen run a bit more than it helped stock (6.3 points against 5.3). I was too optimistic on Part 2. I expected 53-56% and 7-10 points over the batch-norm-frozen run, and got 57.89% and 5.27 points. I based that on stock's export gain, but that gain comes from how the model is exported and decoded, while normalizing changes what the model learns, so it was a bad yardstick. I expected the interval to clear zero and it missed by about a point. With 37 clips, a gain of about 7 points is right at the edge of what this test set can show. I also expected more deleted words in Part 1 and there were none. The exported run deleted 0 of 95 words against stock's 11 and made more wrong words instead (43 against 34). So the error type did change, just the opposite way to what I said.

</details>

## Related

[F-2: M02, top 8 blocks, batch norm frozen](../f2-m02-top8-bn-frozen/notes.md), [F-2: M02, bottom 8 blocks, which parts to train](../f2-m02-bottom8-parts/notes.md)
