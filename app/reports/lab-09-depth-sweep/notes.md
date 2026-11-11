# Lab test 9: depth sweep, how many blocks and which ones

Proposal: F-2 (H3), an extension. Prototype bar: 35%.

Status: run 7 October 2026 (a rerun, all arms finished). Results read 14 October.

See also the [Report Context](../Report%20Context.md) and the [Measurement guide](../../../docs/measurements.md).

## Idea

F-2 trained whole blocks 0 to 7, and the top eight blocks gave no gain. Nothing says 8 is the right number or the bottom the right place, so this trains blocks 0 to 1, 0 to 3, 0 to 11, 0 to 15 and a middle window, 8 to 15, with the same recipe.

## Setup

Scores are pooled wrong words over M01 (169 test words), M02 (95) and M04 (161). Stock (Parakeet-TDT-0.6B-v2) is the first row of the table. The best so far is about 201 to 210 wrong (about 48.5%, adapter after block 8, three seeds), and the 35% bar is 148 or fewer.

The 0 to 7 arm is the existing whole-block F-2 model, not retrained. M01 and M04 use the fixed five-epoch runs (`models/parakeet-ft-f2-<SPK>-bottom8-5epochs`), but M02 uses the dev-selected `models/parakeet-ft-M02e-A3` (42 of 95 wrong in NeMo), a different recipe.

`train.py --train-parts all --train-blocks N --train-first-block K` opens N whole blocks from block K, for five epochs at 2e-5 with last weights, frozen BatchNorm and normalised targets. The 12 and 16 block arms use `--offload-optimizer` and might still run out of memory on the 4060. A failed run is retried once at batch 1 with 16 accumulation steps, then listed and skipped. `ARMS9` sets the arm list (`label:first block:blocks`).

## Plan

1. Train each arm on each speaker's F-2 export, then decode and score his test set.
2. Run `pool_table.py` on stock, whole-block and every arm that finished on all three speakers.

Decision rule, set beforehand: a change counts only when at least 2 of 3 speakers move with the pooled count and pooled WER moves more than 4.5 points, the run-to-run noise seen across seeds (4 to 5 points per speaker). Anything else is unclear, which doesn't mean equal. `pool_table.py` prints the verdict. Three agreeing seeds could settle a smaller gap, but here it's one run per cell.

A gain still growing at 12 or 16 blocks would mean 8 wasn't the limit, and memory and export cost would set the depth. Blocks 0 to 3 or 0 to 1 matching 0 to 7 would put the gain in the lowest blocks, and fewer blocks would do. Blocks 8 to 15 matching 0 to 7 would mean position in the lower half doesn't matter, and the top eight failing is about the top.

## Why

It shows how the gain depends on depth before I commit to one recipe for the expo model.

## Limits

One seed per cell gives the rough shape of the curve but no optimum, since cells within 4 to 5 points aren't ordered. Learning rate at other depths and int8 behaviour of the deeper models aren't tested.

## Results

Whole blocks trained, three speakers, 425 reference words, one run per cell.

| Blocks trained | Wrong | WER | Gain over stock (95% CI, clips) |
|---|---:|---:|---|
| stock | 323 | 76.0% | - |
| 0-1 | 328 | 77.2% | -1.2 (-4.6 to +1.3), unclear |
| 0-3 | 322 | 75.8% | +0.2 (-3.1 to +2.9), unclear |
| 0-7 (whole-block) | 291 | 68.5% | +7.5 (+1.9 to +14.2), better |
| 0-11 | 292 | 68.7% | +7.3 (+1.5 to +14.0), better |
| 0-15 | 285 | 67.1% | +8.9 (+3.0 to +16.0), better |
| 8-15 | 296 | 69.6% | +6.4 (+2.6 to +11.0), better |

Two or four blocks do nothing, and eight do most of what this recipe can. 12 or 16 blocks are only 1.4 points (6 words) past 8, too little to count as a gain. Blocks 8-15 alone get most of it (M01 +7.1, M02 +5.3, M04 +6.2 points), so it isn't confined to 0-7. Nothing nears the bar, and the adapters (48.9%) are far ahead of every depth.

Source: the pool_table log (`logs/pool_table-20261007-235205-136888.log`), from runs on the RTX 4000 Ada lab computer, 6-8 Oct 2026. Results reviewed 14 October 2026.

Sentences only, a secondary score added 21 October 2026: the same decodes scored without the single-word clips. That's 303 of the 425 reference words, so 35% is 106 wrong or fewer. Means over the seeds listed, read from each run's `results.json` with `scripts/asr/sentence_rescore.py`. Every recipe's ranking is in `docs/measurements.md` under "Secondary score: sentences only".

| Recipe (run name) | Seeds | Sentence words wrong (of 303) | Sentence WER | M01 / M02 / M04 | All-clip WER |
| --- | ---: | ---: | ---: | --- | ---: |
| `lab-09-b16` | 1 | 161.0 | 53.1% | 54.9 / 35.3 / 61.9 | 67.1% |
| `lab-09-b12` | 1 | 168.0 | 55.4% | 57.4 / 38.2 / 63.7 | 68.7% |
| `lab-09-mid8` | 1 | 171.0 | 56.4% | 54.1 / 45.6 / 65.5 | 69.6% |
| `lab-09-b4` | 1 | 186.0 | 61.4% | 62.3 / 45.6 / 69.9 | 75.8% |
| `lab-09-b2` | 1 | 187.0 | 61.7% | 61.5 / 48.5 / 69.9 | 77.2% |
| `stock 0.6B v2 (reference)` | 1 | 188.0 | 62.0% | 61.5 / 47.1 / 71.7 | 76.0% |

One run only covered some speakers, so it's scored on their own sentence words:

| Recipe (run name) | Speakers | Seeds | Sentence words wrong | Sentence WER | All-clip WER |
| --- | --- | ---: | ---: | ---: | ---: |
| `lab-09-b2-first-run` | M01 | 1 | 75.0 of 122 | 61.5% | 78.1% |

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

No prediction was written before this run (7 October 2026), and none is added after the fact.

### Commands and Setup

Run from `app/` in the WSL tree on Yack after `git pull`, with the GPU free and the PC set not to sleep. The script stops if a `train.py` run is already going, and each stage stops straight away if one of its outputs already exists.

```bash
python lab.py run 9
```

At the end the script writes `logs/lab-<stamp>-summaries.txt`, which is the file these notes record. The stage's full output is `logs/lab-<stamp>-9.txt`. This is the longest stage. `ARMS9="b2:0:2 mid8:8:8"` runs only some arms.

The commands come from `lab.py`. I checked its flags against each script's own options with a dry run (`python lab.py run --dry`) (`tests/unit/test_lab_cli.py`), and each script's logic has unit tests on synthetic data. Those tests aren't recognition results. Two things weren't run on the real models or data beforehand: `--train-blocks` and `--train-first-block` on the real model, and the memory use of the 12 and 16 block arms on the 8 GB card.

### Full Results

The Results section above has the completed comparison and its sources.

### Conclusions

Interpretation added 21 October 2026 from the recorded results above.

On the pooled test set (425 reference words, stock 323 wrong, lower is better), training the first two or four encoder blocks did nothing. They got 328 and 322 wrong, with intervals including zero. Eight blocks or more gave the gain, with blocks 0-7 at 291, 0-11 at 292 and 0-15 at 285 wrong.

Going from 8 to 16 blocks saved 6 words (1.4 points). That's far under the project rule (2 of 3 speakers the same way and more than 4.5 points, about 19 words), so deeper whole-block training isn't established as better.

Blocks 8-15 alone reached 296 wrong, close to blocks 0-7, so the useful change isn't confined to the bottom of the encoder. That fits the adapter position study, where a single adapter after block 8 was the best of seven positions (three-seed mean 206 wrong, [adapter positions](../f2-adapter-position-seeds/notes.md)).

Every depth is far behind the adapters. The adapter on stock is at 208 wrong and the adapter on the severe-pretrained model at 178.0 (three-seed mean). Longer whole-block training later reached 194.7 ([stage G2](../f2-adapter-levers/notes.md)), so it was the five-epoch schedule that held these runs back. The depth didn't.

Next step: no more depth sweeps. If whole-block training gets used again, the levers are its schedule and learning rate. The number of blocks it opens isn't one.

</details>

## Related

- [Lab test 8](../lab-08-feed-forward-only/notes.md)
- [F-2: severe speakers, bottom 8 blocks, fixed 5 epochs](../f2-severe-bottom8-5epochs/notes.md)
