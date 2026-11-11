# Lab test 8: feed-forward modules only, blocks 0 to 7, then which of the two

Proposal F-2 (H3), extending the which-parts comparison. Prototype bar: 35%.

Status: stages 8 and 8b finished on 8 October 2026 on the lab RTX 4000 Ada, nine training runs. I reviewed the results on 12 October without making any new run.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

On M02, training only attention and convolution in blocks 0 to 7 gave nothing (62.11% against stock's 60.00%), while whole blocks got 44.21% on the same 37 clips. Each Conformer block also has two feed-forward modules and some normalisation layers, so maybe the gain is in the feed-forward part. Stage 8 trains both feed-forward modules of blocks 0 to 7 and nothing else. Stage 8b, run after reading stage 8, trains each one alone.

## Setup

Pooled means wrong words summed over M01 (169 test words), M02 (95) and M04 (161), 425 in all. Stock Parakeet-TDT-0.6B-v2 gets 323 wrong (76.0%), the best so far is about 201 to 210 (about 48.5%, adapter after block 8, three seeds), and the 35% bar means 148 or fewer.

Whole-block is the F-2 model that fully trains blocks 0 to 7. M01 and M04 use the fixed five-epoch runs (`models/parakeet-ft-f2-<SPK>-bottom8-5epochs`). M02 uses the dev-selected `models/parakeet-ft-M02e-A3` (42 of 95 wrong in NeMo), a different recipe.

`train.py --train-parts feed_forward1,feed_forward2` opens only those modules in blocks 0 to 7 (names in `BLOCK_PARTS`) and prints the exact trainable count at the start. The rest is the F-2 recipe: five epochs, last weights, learning rate 2e-5, batch 2 with 8 accumulation steps, frozen BatchNorm, normalised targets. I dropped a same-size LoRA arm to stick to flags `train.py` already has.

## Plan

1. Stage 8: `--train-parts feed_forward1,feed_forward2`, blocks 0 to 7, three speakers, decode and score.
2. After reading stage 8, stage 8b: `--train-parts feed_forward1`, then `--train-parts feed_forward2`, same setup.
3. `pool_table.py` against stock and whole-block.

Decision rule, set before the run: a change counts only if at least 2 of 3 speakers move with the pooled count and pooled WER moves more than 4.5 points, the seed-to-seed noise seen so far (4 to 5 points per speaker). Anything else is unclear, which isn't the same as equal. `pool_table.py` prints the verdict. Three agreeing seeds could settle a smaller gap, but this note has one run per cell unless it says otherwise.

If feed-forward only came within 4.5 points of whole-block, the gain would be in those modules, and stage 8b would ask which one. No better than stock would mean the gain needs attention, convolution or normalisation too. In between would mean the parts add up.

## Why

Fewer trainable parts means less memory, less forgetting and an export that changes less of the model, if the gain survives.

## Limits

This shows whether the feed-forward modules alone reproduce the whole-block gain on three speakers, one seed each. It can't say why, whether it holds in other blocks, or what it does to typical speech, and M02's whole-block comparator uses a different recipe.

## Results

Recorded NeMo WER on M01/M02/M04, 166 clips and 425 reference words:

| System | Wrong words | WER |
|---|---:|---:|
| stock | 323 | 76.0% |
| rebuilt whole-block baseline | 291 | 68.5% |
| both feed-forward modules | 312 | 73.4% |
| feed_forward1 only | 310 | 72.9% |
| feed_forward2 only | 314 | 73.9% |

Both stages report `failed: none`. No feed-forward recipe reaches the 35% bar or beats the rebuilt whole-block baseline. They're 2.1–3.1 points better than stock, a single seed each, and the verdict is unclear. The comparators are the rebuilt `lab-baselines` models; the older paths and counts in Setup were the plan.

Sources: stage-8 table (`reports/lab-08-feed-forward-only/table.json`), stage-8 log (`logs/pool_table-20261008-200539-155335.log`), stage-8b log (`logs/pool_table-20261008-201400-067778.log`). The stage-8 JSON uploaded 12 October matches its log. The bisect JSON was written on the lab machine and `table-bisect.json` is now here (checked 21 October: ff 312, ff1 310, ff2 314, matching the log).

### Sentences only

Added 21 October 2026 as a secondary score: the same decodes scored on sentence clips only. That's 303 of the 425 reference words, so 35% means 106 wrong or fewer. Means over the seeds listed, lower is better, read from each run's `results.json` with `scripts/asr/sentence_rescore.py`. The full ranking is in `docs/measurements.md` under "Secondary score: sentences only".

| Recipe (run name) | Seeds | Sentence words wrong (of 303) | Sentence WER | M01 / M02 / M04 | All-clip WER |
| --- | ---: | ---: | ---: | --- | ---: |
| `lab-08-ff` | 1 | 175.0 | 57.8% | 60.7 / 41.2 / 64.6 | 73.4% |
| `lab-08-ff1` | 1 | 176.0 | 58.1% | 60.7 / 42.6 / 64.6 | 72.9% |
| `lab-08-ff2` | 1 | 180.0 | 59.4% | 59.8 / 45.6 / 67.3 | 73.9% |
| `stock 0.6B v2 (reference)` | 1 | 188.0 | 62.0% | 61.5 / 47.1 / 71.7 | 76.0% |

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

No prediction was written before this run (8 October 2026), and I haven't added one after the fact.

### Commands and Setup

Run from `app/` in the WSL tree on Yack after `git pull`, with the GPU free and the PC set not to sleep. The script stops if a `train.py` run is already going, and each stage stops right away if one of its outputs already exists.

```bash
python lab.py run 8
```

At the end the script writes `logs/lab-<stamp>-summaries.txt`, which is the file recorded in these notes. The full stage output is `logs/lab-<stamp>-8.txt`. Stage 8b is `python lab.py run 8b`, and it won't start unless stage 8's decodes exist.

The commands come from `lab.py`. Its flags were checked against each script's options with a dry run (`python lab.py run --dry`, `tests/unit/test_lab_cli.py`), and each script's logic has unit tests on synthetic data, which aren't recognition results. `--train-parts feed_forward1,feed_forward2` wasn't tried on the real model before the run; only part selection is unit-tested.

### Full Results

The Results table and linked logs hold the completed October 8 comparison.

### Conclusions

Training the feed-forward modules alone didn't recover the whole-block or adapter results with this five-epoch recipe. The 35% target is still unmet.

</details>

## Related

[F-2: M02, bottom 8 blocks, which parts](../f2-m02-bottom8-parts/notes.md), [Lab test 9](../lab-09-depth-sweep/notes.md)
