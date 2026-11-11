# F-2: adapter levers. Stacking, regularisation, word and sentence training, bigger whole-block steps

Proposal: F-2. Stages `P`, `R`, `K` and `G` of `lab.py` (shortcut: `python lab.py queue ideas`). Prototype bar: 35%, which is 148 wrong of 425 pooled test words.

Status: run 14 October 2026 on the lab computer (stamp `1015-ideas`, code at 7b94571), read 15 October. R, K and G finished all their runs (72, 27 and 24). P stopped at its first command and was rerun later (see Results). No prediction was written first.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Background

The 13 to 14 October sweeps (stages L, T, E, S, C on M01, M02, M04; pooled wrong of 425, mean of seeds 17 to 19) left four leads. Severe-speaker pretraining before the adapter got 182.3 (42.9%) against 213.0 for the adapter alone. The adapter levels off near 200 (30 epochs 200.3, rate 1e-3 200.7). Whole-block was still improving (20 epochs 256.7, rate 8e-5 267.0, default 304.7). Single words sat near 70% WER while sentences got to about 36%.

## Tests

1. P stacks the winners (pretraining, 30 epochs, rate 6e-4): the adapter with 30 epochs and 6e-4, the pretrained adapter at default, with 30 epochs, with 6e-4 and with both, and the default adapter as control. 6 arms x 3 seeds x 3 speakers = 54 runs.
2. R tries things never varied before: weight decay 0, 1e-2 and 1e-1 (default 1e-3), warmup 30 (default 10), averaging the last 1 or 8 epochs (default 4), and trainable batch-norm statistics. 8 arms x 3 x 3 = 72 runs.
3. K trains on single words only, sentences only, or both, always testing on all clips. 3 x 3 x 3 = 27 runs.
4. G pushes whole-block further: rate 8e-5 with 20 or 40 epochs and 1.6e-4 with 20, against the default. 4 arms x 2 seeds x 3 speakers = 24 runs, seeds 17 and 18 only because the runs are long.

A change counts only when at least 2 of 3 speakers move the same way and the pooled count moves by more than about 19 words (4.5 points), the run-to-run noise repeated seeds have shown.

## Pre-run Hypotheses

No prediction of my own was recorded. The hypotheses were these. P: stacking helps less than the sum of its parts, since the adapter already plateaus near 200, and pretraining plus 30 epochs is the most likely to beat 182.3, by a few errors. R: nothing moves the result by 4.5 points, because a 1 million weight adapter trained for 15 epochs has little for regularisation to hold back. K: words-only helps words and hurts sentences, and the mixed default is best overall; if sentences-only does as well on words, the two kinds aren't competing. G: the gains continue to a point, then stop, still above the adapter's 200.

## Results

The queue started P at 16:03 on 14 October (lab computer clock), then ran U (soup note), R until 19:27, K until 20:24 and G until 22:21. Counts are pooled wrong words, mean over seeds, lower is better. Stock Parakeet gets 323 wrong (76.0%).

### P: severe pretraining helps, stacking on top barely adds

Rerun with the leftover-folder fix in the `followup` queue (15 October 15:32 to 20:27, 72 runs, none failed), three seeds per arm.

| Arm | Wrong by seed (17 / 18 / 19) | Mean | WER | Against the default adapter | Speakers moving the same way |
| --- | --- | ---: | ---: | ---: | --- |
| adapter, no pretraining, default | 205 / 216 / 215 | 212.0 | 49.9% | - | - |
| adapter, no pretraining, 30 epochs, 6e-4 | 208 / 189 / 189 | 195.3 | 46.0% | -16.7 | 2 of 3 |
| pretrained, adapter default | 188 / 181 / 186 | 185.0 | 43.5% | -27.0 | 3 of 3 |
| pretrained, adapter 30 epochs | 179 / 180 / 181 | 180.0 | 42.4% | -32.0 | 3 of 3 |
| pretrained, adapter 6e-4 | 183 / 176 / 175 | 178.0 | 41.9% | -34.0 | 3 of 3 |
| pretrained, adapter 30 epochs and 6e-4 | 189 / 185 / 196 | 190.0 | 44.7% | -22.0 | 3 of 3 |
| pretrained, whole-block 8e-5, 20 epochs | 193 / 190 / 194 | 192.3 | 45.3% | - | - |
| pretrained, whole-block 1.6e-4, 20 epochs | 191 / 195 / 192 | 192.7 | 45.3% | - | - |

Pretraining on the other severe speakers is the one change that clearly moves the adapter: 185.0 against 212.0, better on all three speakers. That matches stage C's 182.3, though I had to retrain since stage C's weights were gone. More epochs or a higher rate on top move it at most 7 words (185.0 to 178.0), inside the noise, and both together is worse (190.0). Whole-block from the pretrained model lands near 192.

Split by kind of clip, added 21 October from each run's `results.json` (mean of seeds 17 to 19; the 425 words are 122 single-word words and 303 sentence words):

| Arm | Single words wrong (WER) | Sentence words wrong (WER) | Substitutions / deletions / insertions |
| --- | ---: | ---: | --- |
| adapter, no pretraining, default | 94.7 (77.6%) | 117.3 (38.7%) | 163.3 / 18.3 / 30.3 |
| pretrained, adapter default | 90.0 (73.8%) | 95.0 (31.4%) | 149.0 / 12.3 / 23.7 |
| pretrained, adapter 6e-4 | 92.3 (75.7%) | 85.7 (28.3%) | 147.7 / 10.7 / 19.7 |
| pretrained, whole-block 8e-5, 20 epochs | 85.0 (69.7%) | 107.3 (35.4%) | 154.3 / 16.3 / 21.7 |
| stock, for reference | 135 (110.7%) | 188 (62.0%) | 195 / 46 / 82 |

Almost all of pretraining's gain is on sentences. From the default adapter to the pretrained one at 6e-4, sentence errors drop 31.6 words and single-word errors 2.4. Every seed of that arm is under 30% on sentences: 87, 84 and 86 of 303 (28.7%, 27.7%, 28.4%). Per speaker that's M01 31.4%, M02 22.1%, M04 28.3% (38.0 of 122, 15.7 of 68, 32.0 of 113). Insertions fell from 30.3 to 19.7 and deletions from 18.3 to 10.7, while substitutions fell by 15.6.

Single words stay between 70% and 78% in every arm. With sentences at 85.7, reaching 148 needs single words at 62 or fewer (51% WER), about 30 fewer than any arm. Pretrained whole-block is best on words (85.0) but worse on sentences (107.3). Combining the two would give about 170.7 (40.2%), but that's a bound from known labels, since the device can't tell which kind of clip it's hearing.

### R: no effect, and trainable batch-norm ruins it

| Arm | Wrong by seed (17 / 18 / 19) | Mean | WER | Against the default | Speakers moving the same way |
| --- | --- | ---: | ---: | ---: | --- |
| default (decay 1e-3, warmup 10, average last 4) | 205 / 216 / 215 | 212.0 | 49.9% | - | - |
| weight decay 0 | 205 / 222 / 212 | 213.0 | 50.1% | +1.0 | 2 of 3 |
| weight decay 1e-2 | 207 / 217 / 217 | 213.7 | 50.3% | +1.7 | 2 of 3 |
| weight decay 1e-1 | 209 / 222 / 215 | 215.3 | 50.7% | +3.3 | 3 of 3 |
| warmup 30 | 214 / 211 / 222 | 215.7 | 50.7% | +3.7 | 3 of 3 |
| average last 1 epoch | 205 / 216 / 213 | 211.3 | 49.7% | -0.7 | 1 of 3 |
| average last 8 epochs | 204 / 220 / 210 | 211.3 | 49.7% | -0.7 | 2 of 3 |
| batch-norm statistics trainable | 321 / 319 / 334 | 324.7 | 76.4% | +112.7 | 3 of 3 |

Weight decay, warmup and averaging all land within 4 words of the default, so none has an established effect (that doesn't prove they're equal). Trainable batch-norm statistics wipe out the whole gain: 324.7 against 323 for stock. Per speaker that's M01 121 (stock 131), M02 62 (stock 62) and M04 142 (stock 130), so M04 ends up worse than stock. Batch norm also caused problems in my 28 September fine-tune runs, on a different recipe. The default adapter scored 205 / 216 / 215 here and 208 / 219 / 212 in the earlier sweeps with the same recipe and seeds, so reruns differ by up to 3 words per seed.

### K: training on both kinds is best

| Arm | Wrong by seed (17 / 18 / 19) | Mean | WER | Against the default | Speakers moving the same way |
| --- | --- | ---: | ---: | ---: | --- |
| both kinds (default) | 205 / 216 / 215 | 212.0 | 49.9% | - | - |
| sentences only | 250 / 240 / 248 | 246.0 | 57.9% | +34.0 | 3 of 3 |
| single words only | 271 / 286 / 285 | 280.7 | 66.0% | +68.7 | 3 of 3 |

Both restricted arms are worse on every speaker. Split by clip kind from each run's `results.json` (the queue didn't print this), the default gets 117 of 303 sentence words and 95 of 122 single words wrong. Sentences-only keeps sentences at 117 but single words go to 129, more than the reference count because insertions count. Words-only takes sentences to 180 and single words to 101. So the kinds don't compete, and single words are the weak part in every arm.

### G and G2: whole-block levels off near 195 to 200

G2 (15-16 October, 20:28 to 00:22) added a third seed and two longer arms to G. Table from G2's summary.

| Arm | Wrong by seed (17 / 18 / 19) | Mean | WER | Against the default | Speakers moving the same way |
| --- | --- | ---: | ---: | ---: | --- |
| default (5 epochs, 2e-5) | 297 / 310 / 316 | 307.7 | 72.4% | - | - |
| 20 epochs, 8e-5 | 213 / 219 / 217 | 216.3 | 50.9% | -91.3 | 3 of 3 |
| 20 epochs, 1.6e-4 | 214 / 200 / 206 | 206.7 | 48.6% | -101.0 | 3 of 3 |
| 40 epochs, 8e-5 | 209 / 194 / 193 | 198.7 | 46.7% | -109.0 | 3 of 3 |
| 40 epochs, 1.6e-4 | 201 / 195 / 188 | 194.7 | 45.8% | -113.0 | 3 of 3 |
| 60 epochs, 8e-5 | 202 / 193 / 197 | 197.3 | 46.4% | -110.3 | 3 of 3 |

Longer, faster training takes whole-block from 307.7 to about 195-199 and stops; 60 epochs (197.3) is no better than 40 (198.7). The best arm (40 epochs, 1.6e-4) averages 194.7 (45.8%), best run 188. That's 17 words below the default adapter's 212.0, inside the noise, so it's level with the adapter. I'm withdrawing my earlier remark that it "had not stopped improving". A whole-block model is also a 2.4 GB file against a 1-million-number adapter.

### Against the bar

The best raw result in the project is now P's pretrained adapter at 6e-4, 178.0 wrong (41.9%), 30 words above 148. Whole-block's best is 47 words above. Both plateau well short.

### Sentences only

Secondary score, added 21 October 2026: sentence clips only, 303 words, where 35% is 106 wrong or fewer. Read from each run's `results.json` with `scripts/asr/sentence_rescore.py`; the full ranking is in `docs/measurements.md` under "Secondary score: sentences only".

| Recipe (run name) | Seeds | Sentence words wrong (of 303) | Sentence WER | M01 / M02 / M04 | All-clip WER |
| --- | ---: | ---: | ---: | --- | ---: |
| `big-p-pre-ad8-lr6e-4` | 3 | 85.7 | 28.3% | 31.1 / 23.0 / 28.3 | 41.9% |
| `big-p-pre-ad8-ep30` | 3 | 90.0 | 29.7% | 34.7 / 20.1 / 30.1 | 42.4% |
| `big-p-pre-ad8-ep30-lr6e-4` | 3 | 94.3 | 31.1% | 35.8 / 25.0 / 29.8 | 44.7% |
| `big-p-pre-ad8-default` | 3 | 95.0 | 31.4% | 32.8 / 26.0 / 33.0 | 43.5% |
| `big-p-pre-wb-lr1.6e-4-ep20` | 3 | 103.7 | 34.2% | 39.9 / 27.9 / 31.9 | 45.3% |
| `big-g-wb-lr1.6e-4-ep40` | 3 | 106.7 | 35.2% | 37.4 / 32.8 / 34.2 | 45.8% |
| `big-p-pre-wb-lr8e-5-ep20` | 3 | 107.3 | 35.4% | 40.7 / 30.9 / 32.4 | 45.3% |
| `big-p-ad8-ep30-lr6e-4` | 3 | 108.0 | 35.6% | 40.7 / 38.7 / 28.3 | 46.0% |
| `big-g-wb-lr8e-5-ep60` | 3 | 110.0 | 36.3% | 41.5 / 34.3 / 31.9 | 46.4% |
| `big-g-wb-lr8e-5-ep40` | 3 | 111.3 | 36.7% | 41.0 / 33.3 / 34.2 | 46.7% |
| `big-g-wb-lr1.6e-4-ep20` | 3 | 113.3 | 37.4% | 39.9 / 36.8 / 35.1 | 48.6% |
| `big-r-ad8-avg8` | 3 | 116.7 | 38.5% | 39.3 / 35.3 / 39.5 | 49.7% |
| `big-k-ad8-sentences-only` | 3 | 117.0 | 38.6% | 35.0 / 39.7 / 41.9 | 57.9% |
| `big-r-ad8-avg1` | 3 | 117.3 | 38.7% | 39.9 / 33.8 / 40.4 | 49.7% |
| `big-k-ad8-default` | 3 | 117.3 | 38.7% | 39.6 / 34.3 / 40.4 | 49.9% |
| `big-r-ad8-wd0` | 3 | 118.3 | 39.1% | 39.9 / 34.3 / 41.0 | 50.1% |
| `big-r-ad8-wd1e-2` | 3 | 118.7 | 39.2% | 38.8 / 35.3 / 41.9 | 50.3% |
| `big-r-ad8-wd1e-1` | 3 | 120.0 | 39.6% | 39.6 / 35.8 / 41.9 | 50.7% |
| `big-r-ad8-warmup30` | 3 | 120.0 | 39.6% | 40.2 / 36.3 / 41.0 | 50.7% |
| `big-g-wb-lr8e-5-ep20` | 3 | 120.7 | 39.8% | 40.4 / 37.7 / 40.4 | 50.9% |
| `big-g-wb-default` | 3 | 170.3 | 56.2% | 59.6 / 38.2 / 63.4 | 72.4% |
| `big-k-ad8-words-only` | 3 | 180.0 | 59.4% | 59.8 / 42.2 / 69.3 | 66.0% |
| `big-r-ad8-bn-trainable` | 3 | 212.7 | 70.2% | 63.4 / 58.8 / 84.4 | 76.4% |
| `stock 0.6B v2 (reference)` | 1 | 188.0 | 62.0% | 61.5 / 47.1 / 71.7 | 76.0% |

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

I didn't record a prediction before the queue ran on 14 October. Any later prediction has to be dated and labelled as made after the results.

### Commands and Setup

```bash
python lab.py queue ideas --stamp 1015-ideas --detach
```

Stages P, U, R, K, G in that order. Stage C's pretrained models get rebuilt on demand for P. The queue now stops a sweep after an out-of-disk failure that cleaning doesn't fix, so it no longer fails every remaining run. P was rerun in the 15 October follow-up queue (see Results).

### Follow-up queue, 15 October

Written before the run. This one was an exploration ("do all of them just to see what results"), so no predictions were written first. Changes to the stages in `lab.py`:

1. `P` now has eight arms: the six adapter arms plus whole-block from the pretrained models at 8e-5 and 20 epochs and at 1.6e-4 and 20 epochs. The fix for the leftover weightless pretrained folder is in. The "vs default" column is blank for the two whole-block arms because this grid has no `wb-default`; compare them with G and C.
2. `G2` reuses every finished G run (same names, `big-g-*`), trains the third seed (19), and adds `wb-lr1.6e-4-ep40` and `wb-lr8e-5-ep60`. It writes its own summary to `reports/lab-big-g2/summary.json` and leaves G's alone.
3. Lab test 24 (acoustic phrase scoring), the unrun lab tests 16 to 23 and stage B follow in the same queue: `python lab.py queue followup`.

L, T, E, S and C (261 runs) are still unrun and aren't in this queue.

### Full Results

The tables above are the full results. Per-run counts are in `reports/lab-big-{p,r,k,g,g2}/summary.json`, and the decodes and scores are `reports/finetune-torgo/big-<stage>-*`.

### Conclusions

Written from the logs on 15 October.

Against the hypotheses: for P (added 16 October), pretraining helped and stacking on top added at most 7 words, which is noise. R held, since nothing moved 19 words. K partly held. Words-only training did hurt sentences (+63 wrong words), but it didn't help single words either, and sentences-only was worse on single words, so the two kinds support each other. G was wrong in the direction that mattered: whole-block reaches about 201.5, not above the adapter's 200.

Regularisation knobs on the adapter aren't worth more runs, and batch-norm statistics stay frozen. Keep training on words and sentences together. Whole-block with 8e-5 and more epochs is a real second route to about 200 wrong, but it has two seeds and makes a 2.4 GB file against a 1-million-number adapter.

The plateau near 200 across unrelated changes suggests the limit is the amount and kind of training data. P and G2 (16 October) back that up: pretraining plus an adapter reaches 178 to 185 and whole-block 195 to 199 with no gain from more epochs, both 30 to 50 words above the bar.

</details>
