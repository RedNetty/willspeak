# F-2: M02, adapter after 8 blocks

Proposal F-2 (single-speaker fine-tuning, §5.2 / H3), using a way to fine-tune that isn't in the proposal.

Status: run 1 Oct 2026 (seed 17). Repeated with two more seeds in [the seeds note](../f2-m02-adapter8-seeds/notes.md).

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

Instead of training 8 encoder blocks (about 210 million numbers), I add one small layer after block 7 and train only that (1,049,600 numbers). Does it fix M02's recognition as well as the bottom-8 model?

## Setup

In F-2 the model learns from one speaker's earlier sessions and is tested on their last session, with no shared prompts. M02 is the severe TORGO speaker with the most data (9.0 minutes of training audio, 37 test clips). The encoder has 24 blocks, so blocks 0-7 are the bottom 8. Frozen weights don't change in training, and a NeMo score is decoded directly in NeMo before any export.

All three models are scored on the same 37 last-session test clips. Stock (Parakeet-TDT-0.6B-v2, nothing trained) gets 65.26% of words wrong in NeMo (60.00% as int8). The bottom-8 model (`models/parakeet-ft-M02e-A3`), with blocks 0-7, the prediction network and the joint trained on M02's earlier sessions, gets 44.21% in NeMo and as int8. The adapter model is stock plus one trainable layer after block 7. For every 80 ms of audio:

```
audio -> front end -> blocks 0-7 -> [ ADAPTER: h + W h + b ] -> blocks 8-23 -> decoder + joint -> text
          frozen        frozen       the only part that trains     frozen          frozen
```

h is the row of 1024 numbers that block 7 passes on. W is a 1024 x 1024 grid of weights and b is 1024 numbers, so 1024 x 1024 + 1024 = 1,049,600. A small example with 3 numbers instead of 1024:

```
h = (2, 1, 3)
W = row 1: (0.5, 0, 0)    row 2: (0, 0, 1)    row 3: (1, 1, 0)
b = (0.1, 0, -0.1)

W h, each output a weighted sum of all of h:
  row 1: 0.5x2 + 0x1 + 0x3 = 1
  row 2: 0x2   + 0x1 + 1x3 = 3
  row 3: 1x2   + 1x1 + 0x3 = 3
W h + b = (1.1, 3, 2.9)
h + W h + b = (3.1, 4, 5.9)        what block 8 receives
```

W and b start at zero, so at first the model is just stock. That's why the log line `stock dev WER 0.5797` matters: with the zero adapter in place it has to equal stock's own dev score (57.97%), or the run is invalid.

The idea comes from round 7 of the earlier CPU-only research experiments (30 Sep 2026, background only). There an adapter like this matched a bottom-8 fine-tune within noise on three of four comparisons. But its splits kept prompts apart and not sessions, so its gains are probably bigger than F-2 will show. And an adapter fitted on other speakers didn't help a new one, so this one is personal to M02.

## Plan

1. Train the adapter on M02's earlier sessions with the bottom-8 recipe: same data, seed 17, frozen batch norm, normalised transcripts, fixed dev score, stock kept if nothing beats it on dev, usual speed perturbation and SpecAugment. Two things have to differ. Only the adapter trains (the bottom-8 run also trained the prediction network and joint), and the learning rate is 3e-4 instead of 2e-5, since a zero-start adapter barely moves at the lower rate (3e-4 is what the research used).
2. Decode the 37 test clips in NeMo with the adapter and score with `evaluate.py`.
3. Compare with stock and the bottom-8 model, already scored on the same clips, using `ci.py`.

Decision rule, written before the run: the adapter counts as equal to the bottom-8 model if the interval on their difference includes zero, and is better or worse only if the whole interval is on one side. Under about 8 points can't be told from noise on 37 clips (the bottom-8 model's 8.42-point lead over the top-8 model only just cleared zero).

Correction (1 Oct): the rule stays as written, but an interval including zero only means inconclusive, not equal. And the smallest detectable gain isn't a fixed 8 points; it depends on the paired errors and reference lengths.

## Why

If a 1-million-number adapter reaches the bottom-8 score, a personal model is a 4 MB file on top of one shared base model, which is the form the client phase needs. If it doesn't, the 210 million trained numbers in blocks 0-7 are doing something an adapter there can't, and that's worth knowing too.

## Limits

This shows whether an adapter after block 7 beats stock on M02's last session and roughly how far it is from the bottom-8 model. It's one speaker and one run, so nothing about M01 or M04 and no way to separate model differences from training noise. With 37 clips and 95 reference words, uncertainty comes from each paired interval. It's NeMo only (no int8 export yet), so nothing about the Pi, and it doesn't address reaching 38% or lower on the severe speakers' pooled test sentences.

## Videos

Found by search and not opened, so I haven't checked length or content.

- [Why ResNet (Residual Connections) Works](https://www.youtube.com/watch?v=7crFKmTKH3A): why a layer that passes on `h + something` can start as a do-nothing layer.
- [Matrix & Vector Multiplication explained](https://www.youtube.com/watch?v=JNlWbaGLmi4): the W h step, each output a weighted sum of all the inputs.
- [Matrix-Vector Multiplication Example](https://www.youtube.com/watch?v=ebdfJwHM5vo): a worked example of the same step.
- [5 Reasons Why Adapters are the Future of Fine-tuning LLMs](https://www.youtube.com/watch?v=s2BF_gC0X1o): why add a small layer and freeze the rest.

## Results

The adapter (seed 17) scored 48.42% WER on M02's 37 test clips in NeMo, 46 word errors out of 95 (CER 34.91%, PER 35.00%). On the same clips stock scores 65.26% (62 errors) and the bottom-8 model 44.21% (42 errors).

That's 16.84 points better than stock (95% CI 5.10 to 30.44), which clears zero. It got about 80% of the bottom-8 model's gain (16.84 of 21.05 points) with about 1/201 of the trained numbers. Against the bottom-8 model it's 4.21 points worse, 4 more word errors (95% CI -15.79 to 6.94). That includes zero, so it's inconclusive; the pre-run rule called it "equal", but equivalence wasn't tested.

Training touched 1,049,600 of 618,875,526 numbers, and the zero start held (stock dev WER 0.5797). It ran epochs 0 to 8 and stopped after three epochs without a better dev WER. The best was 31.88% at epoch 5, with 4 checkpoints averaged. About 13 seconds an epoch, 4.73 GB peak GPU memory, about 3 minutes in all, so no slowdown and no need for `--offload-optimizer`.

One odd thing: the adapter's best dev WER (31.88%) is far below the bottom-8 model's (46.4%), yet it's 4 words worse on test. Dev comes from the same earlier sessions as training, and test is the last session, so maybe the adapter fits what those earlier sessions share and less of it carries over. I didn't test this.

No prediction was written before this run. Two more seeds are in [the seeds note](../f2-m02-adapter8-seeds/notes.md), and both also scored 48.42%.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

No prediction was written before this run (training started at 11:28 on 1 Oct 2026), so the four lines below are empty. The template asked for full-sentence answers written before the run, given that stock scores 65.26% in NeMo and the bottom-8 model 44.21%.

1. The adapter model's NeMo score on M02's 37 test clips will be:

2. Compared with stock, I expect the interval from `ci.py` to clear zero (the adapter is measurably better) or not, because:

3. Compared with the bottom-8 model, I expect the adapter to be better, about the same, or worse, by roughly this many points:

4. The reason I expect that is:

### Commands and Setup

Run from `app/` in the WSL tree on Yack after `git pull`, with the GPU free and Windows apps closed. The adapter has 1 M trainable numbers, so it should be much quicker than the bottom-8 run. Training and decoding use `~/venvs/willspeak-train`, and `evaluate.py` and `ci.py` use `app/.venv`. Data is `data/public/torgo/export/M02` (earlier sessions in, last session out, prompt-disjoint, 37 test clips).

```bash
source ~/venvs/willspeak-train/bin/activate
python scripts/asr/train.py --export data/public/torgo/export/M02 --out models/parakeet-ft-f2-M02-adapter8 --adapter-after-blocks 8 --lr 3e-4 --batch 2 --grad-accum 8 --warmup-steps 10 --freeze-batchnorm --normalize-targets --dev-scoring evaluate --keep-stock
python scripts/asr/nemo_hyp.py --stock --adapter models/parakeet-ft-f2-M02-adapter8/adapter.pt --manifest data/public/torgo/export/M02/test.jsonl --audio-root data/public/torgo/export/M02 --out reports/finetune-torgo/f2-M02-adapter8-test-nemo.jsonl
deactivate
source .venv/bin/activate
python scripts/asr/evaluate.py --manifest data/public/torgo/export/M02/test.jsonl --audio-root data/public/torgo/export/M02 --hyp reports/finetune-torgo/f2-M02-adapter8-test-nemo.jsonl --group-by kind --out reports/finetune-torgo/f2-M02-adapter8-test-nemo
echo "== adapter against stock"
python scripts/asr/ci.py --manifest data/public/torgo/export/M02/test.jsonl --stock reports/finetune-torgo/M02-test-stock-nemo --finetuned reports/finetune-torgo/f2-M02-adapter8-test-nemo
echo "== adapter against the bottom-8 model"
python scripts/asr/ci.py --manifest data/public/torgo/export/M02/test.jsonl --stock reports/finetune-torgo/M02e-A3-test-nemo --finetuned reports/finetune-torgo/f2-M02-adapter8-test-nemo
```

In the first minute of the training log, check two lines and stop the run if either is wrong: `parameters exact: ... 1049600 trainable` (only the adapter trains) and `stock dev WER 0.5797` (the zero adapter changed nothing). After that there's one epoch summary per epoch (`curve:` in older logs), then `saved .../adapter.pt`. In the `ci.py` output, a positive Delta means the adapter beat the model it's compared with. `adapter.pt` records its base model, and `nemo_hyp.py` stops if it's attached to a different one.

### Full Results

Training, from `logs/train_parakeet-20261001-162804-150693.log` in the WSL tree (started 11:28 on 1 Oct 2026):

```text
parameters exact: 618875526 total, 1049600 trainable
stock dev WER 0.5797 (a checkpoint must beat this to be released)
epoch 8, the last: 90 optimizer steps, dev WER 0.3478, mean training loss 1.4593, 0.2 min of training batches, peak GPU memory 4.73 GB
best dev WER 0.3188 at epoch 5 (best-epoch=5-val_wer=0.319.ckpt)
averaged 4 checkpoints
saved models/parakeet-ft-f2-M02-adapter8/adapter.pt
```

The other epochs are in `models/parakeet-ft-f2-M02-adapter8/curve.csv` in the WSL tree, not copied here.

Decode, scoring and the two comparisons:

```text
Added the adapter from models/parakeet-ft-f2-M02-adapter8/adapter.pt: after encoder blocks 0-7
Transcribing 37 clips with NeMo on the GPU...
WER 48.42%  CER 34.91%  PER 35.00%
== adapter against stock
Check: yours 0.6526, results.json 0.6526
n = 37
Stock WER 65.26%  (95% CI 46.40% to 88.41%)
Fine-tuned WER 48.42%
Delta 16.84 points  (95% CI 5.10 to 30.44)
== adapter against the bottom-8 model
Check: yours 0.4421, results.json 0.4421
n = 37
Stock WER 44.21%  (95% CI 30.52% to 60.83%)
Fine-tuned WER 48.42%
Delta -4.21 points  (95% CI -15.79 to 6.94)
```

In the second block the "Stock" line is really the bottom-8 model, because `ci.py` calls the first folder it gets the reference.

### Conclusions

The 1,049,600-number adapter after block 7 took M02's test WER from 65.26% to 48.42% in NeMo, and the gain over stock (16.84 points, CI 5.10 to 30.44) clears zero. It finished 4 word errors (4.21 points) behind the bottom-8 model's 44.21%, with an interval of -15.79 to 6.94 that includes zero. That's inconclusive. It doesn't show equivalence, and it doesn't set a universal minimum detectable gain.

The code works end to end on the real model: training through the wrapper, saving `adapter.pt`, reattaching it to stock and decoding. The zero start held (stock dev WER 0.5797).

Not shown: that the adapter equals the bottom-8 model, that it works for M01 or M04, or anything about the Pi (no int8 export yet). Still open is why the adapter's dev score is much better than the bottom-8 model's while its test score is slightly worse; fitting the earlier sessions is one guess, untested. The seeds 18 and 19 repeat is in [the seeds note](../f2-m02-adapter8-seeds/notes.md).

</details>

## Related

[F-2: M02, bottom 8 blocks, which parts to train](../f2-m02-bottom8-parts/notes.md), [F-2: severe speakers, bottom 8 blocks, fixed 5 epochs](../f2-severe-bottom8-5epochs/notes.md), [Prototype model, severe focus](../prototype-severe-focus-bottom8/notes.md)
