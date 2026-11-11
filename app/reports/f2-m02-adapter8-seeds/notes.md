# F-2: M02, adapter after 8 blocks, seeds 18 and 19

Proposal: F-2 (single-speaker fine-tuning, §5.2 / H3). This repeats the [first adapter run](../f2-m02-adapter8/notes.md) to measure run-to-run spread. It isn't a new method.

Status: run 1 Oct 2026 (seeds 18 and 19). Prediction entered before the runs.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

Train the adapter twice more with only the seed changed and see how much its score moves on M02's 37 test clips. The first run (seed 17) scored 48.42% in NeMo, 4.21 points behind the bottom-8 model's 44.21%, with an interval that includes zero. Is that a real difference between methods, or just the noise between two runs of one method?

## Setup

A seed fixes a training run's random choices: clip order, how much each clip is sped up or slowed (0.85 to 1.15 times), and which spectrogram patches SpecAugment blanks. It doesn't change the adapter's start, which is zero. Two GPU runs on Yack aren't bit-for-bit identical even with one seed, so rerunning seed 17 wouldn't reproduce the first run anyway, and I used new seeds on purpose. Run-to-run noise is how much the score moves when only that randomness changes.

M02's 37 test clips (his last session) hold 95 reference words, so one word is 1.05 points of WER and the 4.21-point gap is 4 words. Stock (Parakeet-TDT-0.6B-v2, untrained) gets 65.26%, 62 word errors. The bottom-8 model (`models/parakeet-ft-M02e-A3`: blocks 0-7, the prediction network and the joint, trained on his earlier sessions) gets 44.21%, 42 errors, from one run only. The seed 17 adapter gets 48.42%, 46 errors. Seeds 18 and 19 use the same recipe.

For scale, in the severe-focus repeat (30 Sep) two runs of one recipe differed by up to about 1 point on the 65-clip goal set and 3 to 4 points on single speakers, but that was a different recipe and speakers.

## Plan

1. Rerun the first adapter command with only `--seed` changed (18, then 19) and a new output folder. Everything else stays: learning rate 3e-4, batch 2 with 8 accumulated, 10 warmup steps, frozen batch norm, normalised transcripts, fixed dev score, stock kept if nothing beats it on dev, at most 10 epochs, stop after 3 without improvement.
2. Decode the 37 test clips in NeMo, score with `evaluate.py`, and compare with stock and the bottom-8 model using `ci.py`.
3. Compare the three adapter scores: lowest, highest, gap in points and words, mean.

Written before the run: if the three scores are 4 or more points apart, the gap to bottom-8 is within the adapter's own noise and the methods can't be separated here. If they're within 2 points and the mean stays at least 3 points above 44.21%, the adapter is consistently a little worse, with the bottom-8 spread still unknown. Anything between is unclear and needs more seeds.

## Why

One run of each can't say if a 4-word gap means anything, and choosing between a personal adapter and retraining blocks shouldn't rest on it. Two more runs cost about 6 minutes of GPU time.

## Limits

This only measures training randomness. Repeating on the same 37 clips can't shrink the uncertainty from having so few clips (the intervals `ci.py` prints), and it doesn't measure the bottom-8 model's spread. Nothing on M01, M04 or the severe speakers' pooled sentences. NeMo only, since the adapter has no int8 export.

## Reading and Video

Found by search and not opened, so each one is judged by its title.

- [Why Do I Get Different Results Each Time in Machine Learning?](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/): why two runs of one method differ.
- [Single training runs and estimates of variance](https://bricksdont.github.io/posts/2020/12/single-training-runs/): why one run of each method is a weak comparison.
- [Calibrated Chaos: Variance Between Runs of Neural Network Training is Harmless and Inevitable](https://ar5iv.labs.arxiv.org/html/2304.01910): a research paper, harder going than the other two.
- [Reproducible results with Keras](https://www.youtube.com/watch?v=HcW0DeWRggs): what a seed controls (uses Keras, not NeMo).

## Results

All three runs (seeds 17, 18, 19) scored 48.42% WER, 46 word errors of 95 each, so the WER spread is 0 points. CER ranged from 33.96% to 35.14% (1.18 points), PER from 35.00% to 36.39% (1.39 points), and the intervals differ. They're three different models that happened to make the same number of word errors.

That matches the Plan's "consistently a little worse" case: about 4 words behind the bottom-8 model's single run. The intervals against it all include zero (-15.79 to 6.94, -18.19 to 7.97, -15.67 to 8.33), so the gap isn't shown to be real.

Three earlier whole-block bottom-8 runs on these clips scored 45.26% in NeMo (first run, old dev score, 43 errors), 44.21% (fixed dev score, the one compared against here, 42 errors) and 51.58% (fixed five epochs, 49 errors). The first two are in [F-2: M02, bottom 8 blocks, which parts to train](../f2-m02-bottom8-parts/notes.md), the third in [F-2: severe speakers, bottom 8 blocks, fixed 5 epochs](../f2-severe-bottom8-5epochs/notes.md). They span 7 words with the adapter's 46 inside, but they differ in checkpoint choice, so it isn't a pure seed comparison.

Seed 19 ran all 10 epochs, best dev WER 30.43% at epoch 7 (seed 17: 31.88% at epoch 5, stopped early), about 13 seconds per epoch, peak GPU memory 4.76 GB. Seed 18's training lines aren't recorded here.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

Written 1 Oct 2026, 11:38, before any seed 18 or 19 run. The first adapter run (seed 17) scored 48.42% and the bottom-8 model 44.21%.

1. The three adapter scores (seed 17 and the two new seeds) will differ between the best and the worst by:

I'm assuming we will see a max gap of around 8 points either way.

2. The average of the two new seeds will land better than, about the same as, or worse than the bottom-8 model's 44.21%:

I think the average will likely land around the same as the bottom-8 model.

3. The reason I expect that is:

I expect that because I'm estimating based on our first run, that with noise we will probably see the average land within 2-3 points of the bottom-8 model.

### Commands and Setup

Run from `app/` in the WSL tree on Yack after `git pull`, with the GPU free and Windows apps closed. Each training run took about 3 minutes the first time. Training and decoding run in `~/venvs/willspeak-train`, and `evaluate.py` and `ci.py` in `app/.venv`. Data is `data/public/torgo/export/M02`, the same 37 test clips as every earlier M02 run.

```bash
source ~/venvs/willspeak-train/bin/activate
for SEED in 18 19; do
  python scripts/asr/train.py --export data/public/torgo/export/M02 --out models/parakeet-ft-f2-M02-adapter8-seed$SEED --adapter-after-blocks 8 --lr 3e-4 --batch 2 --grad-accum 8 --warmup-steps 10 --freeze-batchnorm --normalize-targets --dev-scoring evaluate --keep-stock --seed $SEED && \
  python scripts/asr/nemo_hyp.py --stock --adapter models/parakeet-ft-f2-M02-adapter8-seed$SEED/adapter.pt --manifest data/public/torgo/export/M02/test.jsonl --audio-root data/public/torgo/export/M02 --out reports/finetune-torgo/f2-M02-adapter8-seed$SEED-test-nemo.jsonl
done
deactivate
source .venv/bin/activate
for SEED in 18 19; do
  python scripts/asr/evaluate.py --manifest data/public/torgo/export/M02/test.jsonl --audio-root data/public/torgo/export/M02 --hyp reports/finetune-torgo/f2-M02-adapter8-seed$SEED-test-nemo.jsonl --group-by kind --out reports/finetune-torgo/f2-M02-adapter8-seed$SEED-test-nemo
  echo "== seed $SEED against stock"
  python scripts/asr/ci.py --manifest data/public/torgo/export/M02/test.jsonl --stock reports/finetune-torgo/M02-test-stock-nemo --finetuned reports/finetune-torgo/f2-M02-adapter8-seed$SEED-test-nemo
  echo "== seed $SEED against the bottom-8 model"
  python scripts/asr/ci.py --manifest data/public/torgo/export/M02/test.jsonl --stock reports/finetune-torgo/M02e-A3-test-nemo --finetuned reports/finetune-torgo/f2-M02-adapter8-seed$SEED-test-nemo
done
```

As with the first run, check each training log for `parameters exact: ... 1049600 trainable` and `stock dev WER 0.5797`. In the `ci.py` lines a positive Delta means the adapter beat the model it's compared with. In the second block the line labelled "Stock" is the bottom-8 model.

### Full Results

| Adapter run | WER | Word errors | CER | PER | Against stock: Delta (95% CI) | Against the bottom-8 model: Delta (95% CI) |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| seed 17 (first run) | 48.42% | 46 | 34.91% | 35.00% | 16.84 (5.10 to 30.44) | -4.21 (-15.79 to 6.94) |
| seed 18 | 48.42% | 46 | 35.14% | 36.39% | 16.84 (4.39 to 31.65) | -4.21 (-18.19 to 7.97) |
| seed 19 | 48.42% | 46 | 33.96% | 36.11% | 16.84 (7.22 to 28.99) | -4.21 (-15.67 to 8.33) |

Stock is 65.26% (62 errors) and the bottom-8 model 44.21% (42 errors) on the same 37 clips.

Seed 19 training, from `logs/train_parakeet-20261001-164439-868957.log` in the WSL tree (started 11:44 on 1 Oct 2026):

```text
epoch 9, the last: 100 optimizer steps, dev WER 0.3043, mean training loss 1.2377, 0.2 min of training batches, peak GPU memory 4.76 GB
`Trainer.fit` stopped: `max_epochs=10` reached.
best dev WER 0.30434781312942505 at epoch 7 (best-epoch=7-val_wer=0.304.ckpt)
averaged 4 checkpoints
saved models/parakeet-ft-f2-M02-adapter8-seed19/adapter.pt
```

Scoring and comparisons for seeds 18 and 19:

```text
seed 18: WER 48.42%  CER 35.14%  PER 36.39%
== seed 18 against stock
Check: yours 0.6526, results.json 0.6526
n = 37
Stock WER 65.26%  (95% CI 46.40% to 88.41%)
Fine-tuned WER 48.42%
Delta 16.84 points  (95% CI 4.39 to 31.65)
== seed 18 against the bottom-8 model
Check: yours 0.4421, results.json 0.4421
n = 37
Stock WER 44.21%  (95% CI 30.52% to 60.83%)
Fine-tuned WER 48.42%
Delta -4.21 points  (95% CI -18.19 to 7.97)
seed 19: WER 48.42%  CER 33.96%  PER 36.11%
== seed 19 against stock
Check: yours 0.6526, results.json 0.6526
n = 37
Stock WER 65.26%  (95% CI 46.40% to 88.41%)
Fine-tuned WER 48.42%
Delta 16.84 points  (95% CI 7.22 to 28.99)
== seed 19 against the bottom-8 model
Check: yours 0.4421, results.json 0.4421
n = 37
Stock WER 44.21%  (95% CI 30.52% to 60.83%)
Fine-tuned WER 48.42%
Delta -4.21 points  (95% CI -15.67 to 8.33)
```

In the second block of each pair the line labelled "Stock" is the bottom-8 model.

### Conclusions

The adapter's score on M02's 37 clips didn't move with the seed: 48.42% three times. Its training noise in word errors is small here, though CER and PER moved by about 1 point.

By the rule written before the runs, the adapter is consistently about 4 words behind the bottom-8 model's 44.21% on these clips. That isn't the same as the gap being real. Every interval against the bottom-8 model still includes zero, and repeating on the same 37 clips can't change that.

The missing piece is the bottom-8 model's own spread. Its three recorded runs here (43, 42 and 49 errors) span 7 words, more than the 4-word gap, and the adapter's 46 sits inside that range. Those runs differ in recipe, and a same-recipe bottom-8 repeat with new seeds hasn't been done.

Prediction vs outcome: I predicted a best-to-worst gap of about 8 points and got 0. I predicted an average within 2 to 3 points of 44.21% and got 4.21 points above, outside that band.

The dev-vs-test pattern from the first run held. The adapter's dev WER was near 30 to 32% against 46.4% for the bottom-8 model, but its test score was slightly worse. Whether that comes from fitting the earlier sessions is still untested.

Not shown: anything about M01, M04 or the Pi.

</details>

## Related

[F-2: M02, adapter after 8 blocks](../f2-m02-adapter8/notes.md), [F-2: M02, bottom 8 blocks, which parts to train](../f2-m02-bottom8-parts/notes.md), [Prototype model, severe focus, repeated with the optimizer in RAM](../prototype-severe-focus-bottom8-offload/notes.md)
