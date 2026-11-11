# F-2: adapter soup, averaging the weights of five seeds

Proposal: F-2 (adapter), stage `U` of `lab.py`. Prototype bar: 35%.

Status: run 14 October 2026 on the lab computer as stage `U` of queue `1015-ideas` (16:03 to 16:36, code at 7b94571). It did 15 adapter trainings and 3 averaged decodes, and none failed. I read the results on 15 October. No prediction was written first.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

The block-8 adapter is one linear layer of about 1 million numbers, and different seeds land in slightly different places (pooled errors for seeds 17 to 19 were 208, 219 and 212). Averaging five adapters' weights gives one adapter that costs nothing extra to run. If the seeds differ mostly by noise, the average should land near the better end or below it.

## Plan

For M01, M02 and M04, train the default adapter (15 epochs, 3e-4, stock base) at seeds 17 to 21 and decode each. Average the five `adapter.pt` files with `scripts/asr/adapter_soup.py` and decode that. `adapter_soup.py` won't mix adapters with different position, width or base.

The rule, set before the run, is the 4.5-point rule on the pooled count, comparing the soup with the mean of the five seeds. The soup also has to be no worse than the best single seed on at least 2 of 3 speakers.

If the soup is clearly below the seed mean, the prototype should ship an averaged adapter. If it's equal, the seeds already agree and one seed is enough. If it's worse, the adapters aren't in one basin and averaging isn't valid here.

## Limits

This only covers the stock-base 15-epoch recipe on the TORGO speakers. Pretrained bases and other recipes would need their own soup, and it says nothing about the intended user.

## Results

Wrong words on the test clips (M01 169 words, M02 95, M04 161; 425 in all). Stock gets 323 wrong.

| Speaker | Stock | Seed 17 / 18 / 19 / 20 / 21 | Mean of the five seeds | Averaged adapter (soup) |
| --- | ---: | --- | ---: | ---: |
| M01 | 131 | 86 / 95 / 86 / 93 / 85 | 89.0 | 85 |
| M02 | 62 | 41 / 41 / 41 / 41 / 44 | 41.6 | 41 |
| M04 | 130 | 78 / 80 / 88 / 82 / 81 | 81.8 | 83 |
| Pooled | 323 | 205 / 216 / 215 / 216 / 210 | 212.4 (50.0%) | 209 (49.2%) |

The soup is 3.4 words (0.8 points) below the seed mean. The rule needs more than about 19 words (4.5 points), so it isn't shown to be better. Against the best seed per speaker it ties on M01 (85) and M02 (41) and is 5 words worse on M04 (83 against 78).

It isn't worse than its parts either, so the adapters do sit in one basin. That's the middle outcome: one seed is as good. The seed spread is mostly M01 (85 to 95) and M04 (78 to 88), and M02 barely moves.

Sentences only (secondary score, added 21 October 2026). Same decodes, single-word clips left out: 303 of the 425 words, so 35% is 106 wrong or fewer. Means over the seeds listed, read from each run's `results.json` with `scripts/asr/sentence_rescore.py`. The full ranking is in `docs/measurements.md` under "Secondary score: sentences only".

| Recipe (run name) | Seeds | Sentence words wrong (of 303) | Sentence WER | M01 / M02 / M04 | All-clip WER |
| --- | ---: | ---: | ---: | --- | ---: |
| `big-u-soup` | 1 | 117.0 | 38.6% | 37.7 / 33.8 / 42.5 | 49.2% |
| `big-u-ad8` | 5 | 117.8 | 38.9% | 39.3 / 34.7 / 40.9 | 50.0% |
| `stock 0.6B v2 (reference)` | 1 | 188.0 | 62.0% | 61.5 / 47.1 / 71.7 | 76.0% |

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

No prediction was recorded before the queue ran on 14 October. Any later prediction has to be dated and marked as made after the results.

### Commands and Setup

```bash
python lab.py run U
```

In practice it ran inside `python lab.py queue ideas --stamp 1015-ideas --detach`. The 15 adapter runs (3 speakers x 5 seeds) plus 3 averaged decodes took 33 minutes on the RTX 4000 Ada. Results are in `reports/lab-big-u/summary.json`.

### Full Results

See the table above. Per-speaker counts for every seed and the soup are in `reports/lab-big-u/summary.json`, and the decodes and scores are under `reports/finetune-torgo/big-u-*`.

### Conclusions

Written from the logs on 15 October.

Averaging five seeds of the block-8 adapter gives no gain over a single seed (209 against 212.4 wrong). I'll keep one seed, and the prototype pipeline doesn't need a soup step.

The averaged adapter does work, since it's no worse than its parts. So a soup is a safe way to check that seeds agree, but it won't bring the count down toward the bar.

</details>
