# F-2: M02, bottom 8 blocks, which parts to train

Proposal: §5.2 / F-2 recipe comparison

Status: recorded 27 Sep 2026, original run and fixed-score rerun.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

Compare training the bottom encoder blocks against the top ones, and training whole blocks against attention and convolution alone.

## Setup

M02 has 156 training clips (9 min), 24 dev clips and 37 Session2 test clips (95 reference words). The whole-block arm has 210,461,318 trainable parameters and the attention/conv arm 76,161,670.

## Plan

Train both bottom-eight arms, score each in NeMo and int8 on the same test clips, and keep the first run, the failed scoring fix and the working rerun separate.

## Why

This picks the recipe for later F-1 and F-2 work. The first dev score counted formatting, so checkpoint selection needed a rerun.

## Limits

M02 only. It can't rule out a different learning rate helping the smaller arm. The top-eight fixed-score comparison has its own note, and F-4 still needs the Pi.

## Results

| Fixed-score rerun | NeMo WER | Int8 WER |
|---|---:|---:|
| Stock | 65.26% | 60.00% |
| Bottom 8, whole blocks | 44.21% | 44.21% |
| Bottom 8, attention + conv | 65.26% | 62.11% |

With the fixed score, whole blocks in int8 beat stock by 15.79 points (95% CI 6.45 to 28.09) and beat attention/conv by 17.89 points (95% CI 7.04 to 32.57).

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

Does training the bottom eight encoder blocks help M02 more than training the top eight (the top-8 model)? And within those blocks, is training only attention and conv enough?

### Context

Every earlier run trained the top 8 encoder blocks (16-23, `--freeze-encoder-layers 16`). The transducer papers on disordered speech (Shor 2019, Green 2021, Tobin & Tomanek 2022) train the lower encoder instead. This run trains blocks 0-7 and freezes 8-23, in two arms. In the whole-blocks arm every part of blocks 0-7 is trainable (attention, conv, both feed-forward halves, layer norms). In the attention-and-conv arm only `self_attn` and `conv` inside blocks 0-7 train, and the feed-forward halves and layer norms stay frozen.

Each comparison changes one thing. Whole-blocks against attention-and-conv changes which parts inside the same eight blocks learn. Whole-blocks against the top-8 model changes bottom eight blocks for top eight.

Both arms use the same recipe as the top-8 model apart from which weights train. Data is data/public/torgo/export/M02 (156 train clips, about 9 min; test = Session2, 37 clips, 95 reference words). Frozen batch norm, normalised transcripts, batch 2 x 8, warm-up 10, lr 2e-5 (the default), seed 17, speed perturbation 0.85-1.15, early stop patience 3, 4 best checkpoints averaged. The prediction network and joint are trainable in both arms, same as the top-8 model.

Like the top-8 model, each is scored twice: the checkpoint in NeMo, and the int8 export (what the Pi runs). NeMo only gets compared with NeMo, and export with export.

### Numbers that already exist (M02 test, the same 37 clips every time)

Scored in NeMo:
  Stock 65.26%   batch-norm-frozen run 63.16%   top-8 model 57.89%
Scored after int8 export:
  Stock 60.00% (11 of 95 words deleted)   top-8 model exported 52.63% (8 deleted)
  The first F-1 run's M02-held-out model on these clips: 52.63%
  Single words against sentences, exported: stock 107.4% / 41.2%, top-8 model 88.9% / 38.2%
With 37 clips, a gain of about 7 points sat right at the edge of what ci.py could show (the top-8 model, in NeMo and exported).

### Parameter check (before either run)

train.py printed 210,461,318 trainable parameters for the whole-blocks arm and 76,161,670 for the attention-and-conv arm.

The whole-blocks arm should equal the top-8 model's count, 210,461,318 (models/parakeet-ft-M02c/run.json), because the top-8 model also trained 8 whole blocks and every block is the same size. The attention-and-conv arm should be that minus what it freezes in each of the 8 blocks. Parakeet's encoder layers have no biases, so the two feed-forward halves are 4 grids of 1024 x 4096 = 16,777,216 numbers, and the five layer norms are 5 x 2,048 = 10,240. 210,461,318 - 8 x 16,787,456 = 76,161,670.

Both match exactly.

### What I expect before running the which-parts comparison

The two new models:
  The whole-blocks model = "bottom 8, whole blocks": everything in encoder blocks 0-7 learns (210 M numbers)
  The attention-and-conv model = "bottom 8, attention + conv only": inside blocks 0-7, only the attention and
            conv parts learn; the feed-forward parts stay as stock (76 M numbers)
The model to beat so far:
  The top-8 model = "top 8, whole blocks": everything in blocks 16-23 learns (210 M numbers)

Each model is scored two ways on the same 37 M02 test clips (95 words):
  "NeMo score" = the trained checkpoint decoded directly, before any export
  "Int8 score"   = the same model after the int8 export, the version the Pi runs

Scores so far:
                          NeMo score   Int8 score
  Stock (no training)       65.26%      60.00%   (11 words deleted)
  Top 8                     57.89%      52.63%   (8 words deleted)

1. The whole-blocks model (bottom 8, whole blocks): NeMo score: 58-62%   Int8 Score: 54-57%
2. The attention-and-conv model (bottom 8, attention + conv only): NeMo score: 53-57%   Int8 Score: 50-53%
3. Bottom 8 against top 8: is the whole-blocks model better, worse, or about the same as the top-8 model? Why? I suspect bottom 8 would be worse overall, mainly due to the way the model layers are structured.
4. The whole-blocks model against the attention-and-conv model: which is better, and by roughly how much? Why? probably will see a 3-7% gap, due to training on the parts that actually matter.
5. Words the model said nothing for (deleted words), Int8 Score, out of 95: the whole-blocks model: 15-20   the attention-and-conv model: 10-15
6. Against stock on the Int8 Score, does ci.py's interval clear zero? The whole-blocks model: Likely Not   the attention-and-conv model: Yes, potentially
7. I would be surprised if: the whole-blocks model is better than the attention-and-conv model

### Commands

The flags are written out in full so an out-of-date train.py fails with an error instead of quietly running the old recipe. train.py, nemo_hyp.py and export.py run in ~/venvs/willspeak-train, and evaluate.py and ci.py in app/.venv.

Folder names keep their original letters. M02e-A and M02e-B are the whole-blocks and attention-and-conv models from the first run, M02e-A2 and M02e-B2 the failed-fix repeat, M02e-A3 and M02e-B3 the rerun with the fixed dev score, and M02f the top-8 rerun.

```bash
# Whole-blocks arm - blocks 0-7, every part
python scripts/asr/train.py --export data/public/torgo/export/M02 --out models/parakeet-ft-M02e-A --train-parts all --batch 2 --grad-accum 8 --warmup-steps 10 --freeze-batchnorm --normalize-targets
# Attention-and-conv arm - blocks 0-7, attention and conv only
python scripts/asr/train.py --export data/public/torgo/export/M02 --out models/parakeet-ft-M02e-B --train-parts self_attn,conv --batch 2 --grad-accum 8 --warmup-steps 10 --freeze-batchnorm --normalize-targets

# Score each arm (set ARM=A, then ARM=B)
ARM=A
python scripts/asr/nemo_hyp.py --nemo models/parakeet-ft-M02e-$ARM/parakeet-ft.nemo --manifest data/public/torgo/export/M02/test.jsonl --audio-root data/public/torgo/export/M02 --out reports/finetune-torgo/M02e-$ARM-test-nemo.jsonl
python scripts/asr/evaluate.py --manifest data/public/torgo/export/M02/test.jsonl --audio-root data/public/torgo/export/M02 --hyp reports/finetune-torgo/M02e-$ARM-test-nemo.jsonl --group-by kind --out reports/finetune-torgo/M02e-$ARM-test-nemo
python scripts/asr/ci.py --manifest data/public/torgo/export/M02/test.jsonl --stock reports/finetune-torgo/M02-test-stock-nemo --finetuned reports/finetune-torgo/M02e-$ARM-test-nemo
python scripts/asr/export.py --nemo models/parakeet-ft-M02e-$ARM/parakeet-ft.nemo --out models/parakeet-ft-M02e-$ARM/onnx-int8
python scripts/asr/evaluate.py --manifest data/public/torgo/export/M02/test.jsonl --audio-root data/public/torgo/export/M02 --group-by kind --model-dir models/parakeet-ft-M02e-$ARM/onnx-int8 --out reports/finetune-torgo/M02e-$ARM-test-int8
python scripts/asr/ci.py --manifest data/public/torgo/export/M02/test.jsonl --stock reports/finetune-torgo/M02-stock --finetuned reports/finetune-torgo/M02e-$ARM-test-int8
```

For whole-blocks against attention-and-conv, run ci.py with the whole-blocks model's report as --stock and the attention-and-conv model's as --finetuned. For whole-blocks against top 8: --stock reports/finetune-torgo/M02c-test-int8 --finetuned reports/finetune-torgo/M02e-A-test-int8

### Results

Machine: Yack, WSL2, 24 GB given to WSL. Run 27 Sep 2026.

The whole-blocks model (bottom 8)
  Log lines present: "only norm_feed_forward1, feed_forward1, norm_self_att, self_attn, norm_conv, conv,
    norm_feed_forward2, feed_forward2, norm_out in encoder blocks 0-7 trainable"; 210461318 trainable
  Training: best checkpoint epoch 7, dev WER 72.5% (training's own validation), ran all 10 epochs
  NeMo score 45.26%   Int8 Score 47.37%
  Words, NeMo:  wrong 33, deleted 4, inserted 6 of 95
  Words, Int8:  wrong 34, deleted 7, inserted 4 of 95
  Single words / sentences, Int8: 77.8% / 35.3%   (stock 107.4% / 41.2%, top-8 model 88.9% / 38.2%)
  ci.py against stock: NeMo Delta 20.00 (95% CI 7.62 to 35.29), Int8 Delta 12.63 (95% CI 4.88 to 23.81)
    - both intervals clear zero
The attention-and-conv model (bottom 8)
  Log lines present: "only self_attn, conv in encoder blocks 0-7 trainable"; 76161670 trainable
  Training: best checkpoint epoch 1, dev WER 81.2% (training's own validation), stopped early
  NeMo score 62.11%   Int8 Score 57.89%
  Words, NeMo:  wrong 40, deleted 8, inserted 11 of 95
  Words, Int8:  wrong 36, deleted 9, inserted 10 of 95
  Single words / sentences, Int8: 92.6% / 44.1%
  ci.py against stock: NeMo Delta 3.16 (95% CI -3.54 to 10.26), Int8 Delta 2.11 (95% CI -9.18 to 12.00)
    - neither interval clears zero
ci.py, whole-blocks against attention-and-conv (Int8 Score): attention-and-conv 10.53 points worse (95% CI -23.75 to 0.00) - the interval ends exactly at zero
ci.py, whole-blocks against top 8 (Int8 Score): whole-blocks 5.26 points better (95% CI -0.97 to 12.09) - includes zero

Prediction against actual
  1. The whole-blocks model: predicted NeMo 58-62%, Int8 54-57%; measured 45.26% and 47.37% (better than predicted)
  2. The attention-and-conv model: predicted NeMo 53-57%, Int8 50-53%; measured 62.11% and 57.89% (worse than predicted)
  3. Bottom 8 against top 8: predicted worse; measured better (47.37% against 52.63% Int8), interval includes zero
  4. Whole-blocks against attention-and-conv: predicted attention-and-conv better by 3-7 points; measured whole-blocks better by 10.53 (Int8) and 16.85 (NeMo)
  5. Deleted words, Int8: predicted whole-blocks 15-20, attention-and-conv 10-15; measured whole-blocks 7, attention-and-conv 9 (stock 11, top-8 model 8)
  6. Interval clears zero against stock, Int8: predicted whole-blocks likely not, attention-and-conv potentially; measured whole-blocks yes, attention-and-conv no
  7. Surprised if whole-blocks beats attention-and-conv: whole-blocks beat attention-and-conv

### Found after the run: the dev score that picks checkpoints counted formatting

Stock scores 57.97% on M02 dev by evaluate.py (reports/finetune-torgo/M02-dev-stock-nemo). But training's own dev score (val_wer) put the whole-blocks model's best checkpoint at 72.5% and the attention-and-conv model's at 81.2%, both worse than stock, even though the whole-blocks model then beat stock on test by 20 points.

The cause: NeMo's val_wer compares the model's raw output with the transcript, and with --normalize-targets the dev transcripts are lowercase with no punctuation. So "Tomato." against "tomato" counted as an error. val_wer picks the checkpoints, the averaging and the early stop, so both models were chosen by a score that was partly measuring capitals and full stops. The attention-and-conv model is hit hardest, since its chosen checkpoint is from epoch 1 and the run stopped early.

The test numbers above are still right, because evaluate.py scored them. What's in doubt is whether each run released its best checkpoint. The batch-norm-frozen run, the top-8 model and the first F-1 run all used the same val_wer.

The first attempt at a fix (27 Sep 2026, the failed-fix repeats) didn't take effect. The scorer was patched after the model loaded, and the model keeps its own copy of the scoring code from when it was built. The check below caught it: the log showed "stock dev WER 0.7971" instead of about 0.58, and the "WER predicted" lines still had capitals and full stops. So the two repeats ran with the old score (whole-blocks: best epoch 4, 0.739, stopped after epoch 7; attention-and-conv: best epoch 0, 0.783, 2 checkpoints averaged). In effect they repeat the first run with a different random order on the GPU, and they weren't scored on test.

The second attempt fixed it in train.py (27 Sep 2026). The scorer is now patched before the model loads, and the run stops if the loaded model isn't using it. The new options:

1. --dev-scoring evaluate (default): val_wer normalises output and transcript the way evaluate.py does.
2. --keep-stock (default): the untrained model is scored on dev first. Only checkpoints that beat it get averaged, and if none does, the stock model is released unchanged.
3. --dev-scoring nemo --no-keep-stock reproduces every run before this date.

### Rerun with the fixed dev score

Same data, seed, recipe and trainable parts. Only checkpoint selection changes. The outputs get new names because train.py never overwrites a run, and M02e-A2 and M02e-B2 were already taken by the failed attempt above.

No new predictions. The seven above stand unchanged for the rerun.

```bash
source ~/venvs/willspeak-train/bin/activate
python scripts/asr/train.py --export data/public/torgo/export/M02 --out models/parakeet-ft-M02e-A3 --train-parts all --batch 2 --grad-accum 8 --warmup-steps 10 --freeze-batchnorm --normalize-targets --dev-scoring evaluate --keep-stock
python scripts/asr/train.py --export data/public/torgo/export/M02 --out models/parakeet-ft-M02e-B3 --train-parts self_attn,conv --batch 2 --grad-accum 8 --warmup-steps 10 --freeze-batchnorm --normalize-targets --dev-scoring evaluate --keep-stock
for M in A3 B3; do
  python scripts/asr/nemo_hyp.py --nemo models/parakeet-ft-M02e-$M/parakeet-ft.nemo --manifest data/public/torgo/export/M02/test.jsonl --audio-root data/public/torgo/export/M02 --out reports/finetune-torgo/M02e-$M-test-nemo.jsonl
  python scripts/asr/export.py --nemo models/parakeet-ft-M02e-$M/parakeet-ft.nemo --out models/parakeet-ft-M02e-$M/onnx-int8
done
deactivate
source .venv/bin/activate
for M in A3 B3; do
  python scripts/asr/evaluate.py --manifest data/public/torgo/export/M02/test.jsonl --audio-root data/public/torgo/export/M02 --hyp reports/finetune-torgo/M02e-$M-test-nemo.jsonl --group-by kind --out reports/finetune-torgo/M02e-$M-test-nemo
  python scripts/asr/evaluate.py --manifest data/public/torgo/export/M02/test.jsonl --audio-root data/public/torgo/export/M02 --group-by kind --model-dir models/parakeet-ft-M02e-$M/onnx-int8 --out reports/finetune-torgo/M02e-$M-test-int8
  python scripts/asr/ci.py --manifest data/public/torgo/export/M02/test.jsonl --stock reports/finetune-torgo/M02-test-stock-nemo --finetuned reports/finetune-torgo/M02e-$M-test-nemo
  python scripts/asr/ci.py --manifest data/public/torgo/export/M02/test.jsonl --stock reports/finetune-torgo/M02-stock --finetuned reports/finetune-torgo/M02e-$M-test-int8
done
python scripts/asr/ci.py --manifest data/public/torgo/export/M02/test.jsonl --stock reports/finetune-torgo/M02e-A3-test-int8 --finetuned reports/finetune-torgo/M02e-B3-test-int8
```

Before trusting the run, check each training log for "dev WER scored like evaluate.py", "WER predicted" lines in lowercase with no punctuation, and a "stock dev WER" line near 0.58 (evaluate.py gives 57.97% on these 24 clips). If it says about 0.80, stop, because the old score is back.

Rerun result (27 Sep 2026)
  Scorer check: "WER predicted" lines lowercase with no punctuation; stock dev WER logged 0.5797,
  the same as evaluate.py's 57.97% on these 24 clips - the fixed score now matches.
  Whole-blocks model, rerun (bottom 8): selected fine-tuned, best epoch 4, dev WER 46.4%, 4 checkpoints averaged
    NeMo score 44.21%   Int8 Score 44.21%
    Words, NeMo:  wrong 31, deleted 5, inserted 6 of 95
    Words, Int8:  wrong 33, deleted 5, inserted 4 of 95
    Single words / sentences, Int8: 74.1% / 32.4%
    ci.py against stock: NeMo Delta 21.05 (95% CI 8.94 to 36.47), Int8 Delta 15.79 (95% CI 6.45 to 28.09)
  Attention-and-conv model, rerun (bottom 8): selected fine-tuned, best epoch 1, dev WER 56.5%
    (only just under stock's 57.97%; no "averaged" line, so one checkpoint beat stock and it was used alone)
    NeMo score 65.26%   Int8 Score 62.11%
    Words, NeMo:  wrong 42, deleted 8, inserted 12 of 95
    Words, Int8:  wrong 34, deleted 15, inserted 10 of 95
    Single words / sentences, Int8: 103.7% / 45.6%
    ci.py against stock: NeMo Delta 0.00 (95% CI -6.25 to 5.62), Int8 Delta -2.11 (95% CI -14.29 to 6.02)
  ci.py, whole-blocks rerun against attention-and-conv rerun (Int8): attention-and-conv 17.89 points worse (95% CI -32.57 to -7.04) - clears zero

Prediction against actual, rerun (the same seven predictions)
  1. The whole-blocks model: predicted NeMo 58-62%, Int8 54-57%; measured 44.21% and 44.21%
  2. The attention-and-conv model: predicted NeMo 53-57%, Int8 50-53%; measured 65.26% and 62.11%
  3. Bottom 8 against top 8: predicted worse; measured better (Int8 44.21% against the top-8 model's 52.63%, Delta 8.42, 95% CI 1.05 to 17.05 - clears zero)
  4. Whole-blocks against attention-and-conv: predicted attention-and-conv better by 3-7 points; measured whole-blocks better by 17.89 (Int8), and now the interval clears zero
  5. Deleted words, Int8: predicted whole-blocks 15-20, attention-and-conv 10-15; measured whole-blocks 5, attention-and-conv 15
  6. Interval clears zero against stock, Int8: predicted whole-blocks likely not, attention-and-conv potentially; measured whole-blocks yes, attention-and-conv no
  7. Surprised if whole-blocks beats attention-and-conv: whole-blocks beat attention-and-conv

### Conclusions

First run (checkpoints picked by the old val_wer):

1. Training the bottom eight blocks whole is the first single-speaker fine-tune on M02 to beat stock by more than 37 clips can hide. Int8 was 47.37% against 60.00% (Delta 12.63, 95% CI 4.88 to 23.81) and NeMo 45.26% against 65.26% (Delta 20.00, CI 7.62 to 35.29).
2. Against the top eight blocks (the top-8 model in int8), whole-blocks is 5.26 points better on Int8 (CI -0.97 to 12.09). That points toward the bottom of the encoder, like the disordered-speech papers say, but on 37 clips it isn't shown.
3. Training only attention and conv gave no measurable gain over stock (Int8 Delta 2.11, CI -9.18 to 12.00), and whole-blocks beat it by 10.53 points with an interval ending at zero. Since that model's checkpoint was picked at epoch 1 by a score that counted formatting, this comparison was provisional until the rerun.
4. Most of the gain is on single words. Whole-blocks got 77.8% on words against stock's 107.4%, and 35.3% on sentences against 41.2%. Deleted words fell for both models (whole-blocks 7, attention-and-conv 9, stock 11).
5. The int8 export cost the whole-blocks model 2.1 points (45.26% to 47.37%). For stock and the top-8 model it went the other way, about 5 points better after export. This is the first model trained in the blocks the export's quantisation is suspected of damaging (block 1's conv), so the export needs checking before it goes on the device.
6. Every prediction missed. Whole-blocks was far better than predicted and attention-and-conv worse, bottom beat top, whole-blocks beat attention-and-conv, and the interval results came out the reverse of what I predicted. The idea that the feed-forward parts matter least isn't supported here.
7. The run exposed the val_wer bug above. Fixing it changes how every future run picks its checkpoint, and the batch-norm-frozen run, the top-8 model and the first F-1 run were all picked the old way.

Rerun (checkpoints picked by the fixed score, which matches evaluate.py):

8. The fixed score picked a better whole-blocks model. Int8 was 44.21% against stock's 60.00% (Delta 15.79, CI 6.45 to 28.09) and NeMo 44.21% against 65.26% (Delta 21.05, CI 8.94 to 36.47). Both intervals clear zero by more than in the first run, and deleted words fell to 5 of 95 (stock 11).
9. The int8 export doesn't cost the whole-blocks model anything now: 44.21% in NeMo and after export. The 2.1-point loss in the first run came from that run's checkpoint. Training the bottom blocks wasn't the cause, so it's no longer a reason to keep this model off the device.
10. The attention-and-conv model was no better the second time. With a fair score only its epoch-1 checkpoint beat stock on dev, and only just (56.5% against 57.97%). On test it's level with stock (NeMo Delta 0.00, Int8 -2.11). Whole-blocks beats it by 17.89 points with an interval that clears zero. At this learning rate, training only attention and conv in blocks 0-7 doesn't improve M02, and the feed-forward parts carry the gain. A higher learning rate for that arm is the one explanation this run can't rule out.
11. Against the top eight blocks (52.63% Int8), the bottom-8 model is 8.42 points better (95% CI 1.05 to 17.05). That's the first time bottom against top clears zero. It only just clears, and the top-8 model was picked with the old score, so a top-8 rerun with the fixed score is needed for a like-for-like comparison.
12. GPU training here isn't bit-for-bit repeatable. The first run and the failed-fix repeat used the same seed and the same old score but picked different best epochs (7 and 4). Single runs can differ by a few points for that reason alone.

ci.py, top 8 against the bottom-8 model (Int8): the bottom-8 model 8.42 points better (95% CI 1.05 to 17.05)

</details>

## Related

[F-2: M02, top 8 blocks, fixed dev score](../f2-m02-top8-fixed-dev/notes.md), [F-2: severe speakers, bottom 8 blocks, fixed 5 epochs](../f2-severe-bottom8-5epochs/notes.md)
