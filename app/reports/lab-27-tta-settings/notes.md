# Lab test 27: test-time adaptation settings, one at a time

Proposal: device design (extension), following lab test 22. Prototype bar: 35%, which is 148 wrong of 425 pooled test words.

Status: written 20 October 2026, run in the `1020-new` queue on the lab computer (finished 02:09 on 21 October, lab clock). Results come from the printed tables in `logs/`, read 21 October 2026; the queue's own tables and reports stayed on the lab computer.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

Lab test 22's loop helped on every speaker (197 to 179) with settings picked once and never varied: four chunks mixed by hash, an agreement gate of 0.5 between F-1 and the adapter on F-1, and at most as many pseudo-labelled clips as labelled ones. Those were all guesses, so here I change one at a time on the same system and seed.

## Setup

Six variants, all on the adapter on F-1 at seed 17 with 425 pooled test words. The control is lab test 22 (179).

| Variant | Change |
| --- | --- |
| `c8` | eight chunks instead of four: more rounds, each learning from fewer clips |
| `id` | chunks in recorded order instead of hash order, the way a day's clips arrive |
| `g0.3` | looser gate |
| `g0.7` | stricter gate |
| `nogate` | every non-empty answer is trained on |
| `cap0.5` | at most half as many pseudo-labelled clips as labelled ones |

Variants with the same chunking share chunk manifests. Nothing reads the test text.

## Plan

1. For each speaker and variant: run the rounds, merge, score, and write the by-chunk table.
2. Make one pooled table with lab test 22's result as the control.

Decision rule, set before the run: the project's usual rule against the control. With one seed, a variant inside the rule is "no different", which isn't the same as "the same".

Decided before the run: `nogate` or `g0.3` better would mean the gate throws away useful answers and the loop copes with its own mistakes. `g0.7` or `cap0.5` better would mean wrong pseudo-labels cost words and the device should be stricter. `id` worse than the control would mean hash order mixes prompts in a way the device won't see, so lab 22's gain is optimistic. `c8` better would mean retraining more often on less.

## Why

The device will run this loop overnight, so its settings should come from a measurement instead of the first values I wrote down.

## Limits

This shows which settings move the pooled count on these speakers at one seed. It can't show interactions between settings, seed variation, or what happens over weeks of use.

## Results

Lab 22's gain is fragile. With its settings the loop gave 179 wrong against 197 without it. One setting changed at a time (positive means more wrong words):

| Variant | Wrong (of 425) | WER | Against the control (179) | Against no loop (197) |
| --- | --- | --- | --- | --- |
| no loop | 197 | 46.4% | +18 | 0 |
| control (4 chunks, hash order, gate 0.5, cap equal to the labelled clips) | 179 | 42.1% | 0 | −18 |
| `cap0.5` | 178 | 41.9% | −1 | −19 |
| `g0.3` | 187 | 44.0% | +8 | −10 |
| `nogate` | 191 | 44.9% | +12 | −6 |
| `g0.7` | 194 | 45.6% | +15 | −3 |
| `c8` | 197 | 46.4% | +18 | 0 |
| `id` | 204 | 48.0% | +25 | +7 |

Only the half cap matches the control, and eight chunks give nothing over no loop. Recorded order, which is how a day's clips arrive, is 25 words worse than the control and 7 worse than no loop. By the rule (about 19 words) `id` is worse, and the rest are inside the rule or at its edge (`c8` is 18). So it's the "`id` worse than the control" case: lab 22's gain is optimistic.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

No prediction was written before the `1020-new` queue started (the command `python lab.py queue 25 29 28 30 27 26 --stamp 1020-new --detach` is recorded in `app/scripts/asr/jack_notes.md`, 20 October), and I haven't added one after the fact.

Numbers from earlier runs for comparison, noted 21 October before any result here was in. The control is lab test 22's loop with default settings on the adapter on F-1 at seed 17: 179 wrong (42.1%) against 197 without the loop. Chunk 0 is unchanged by construction, and chunks 1 to 3 went from 156 to 138 wrong. In lab test 22, M01 went 78 to 73, M02 47 to 41 and M04 72 to 65, so an arm that helps only one speaker doesn't pass the rule.

Single runs of one recipe at different seeds spread by a median of 10 words in this project, so an arm has to move about 19 words (the project rule) to count.

### Commands and Setup

```bash
python lab.py run 27
```

About 66 adapter trainings of 1.5 minutes each, so roughly 2 hours. Results go to `reports/lab-27-tta-settings/table.json` and `by-chunk-<variant>-<speaker>.json`.

### Full Results

Source: `logs/pool_table-20261021-030327-352171.log`, one seed (17) per arm. Wrong words per speaker (M01 of 169, M02 of 95, M04 of 161):

| System | M01 | M02 | M04 | Pooled | WER | 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| stock | 131 | 62 | 130 | 323 | 76.0% | 66.7 to 86.3 |
| adapter on F-1, no loop | 78 | 47 | 72 | 197 | 46.4% | 40.2 to 53.2 |
| control (lab 22) | 73 | 41 | 65 | 179 | 42.1% | 36.3 to 48.9 |
| `c8` | 85 | 49 | 63 | 197 | 46.4% | 40.0 to 54.3 |
| `id` | 80 | 52 | 72 | 204 | 48.0% | 41.9 to 54.9 |
| `g0.3` | 74 | 48 | 65 | 187 | 44.0% | 38.0 to 51.4 |
| `g0.7` | 84 | 46 | 64 | 194 | 45.6% | 38.7 to 53.8 |
| `nogate` | 75 | 50 | 66 | 191 | 44.9% | 38.5 to 52.7 |
| `cap0.5` | 72 | 41 | 65 | 178 | 41.9% | 36.1 to 48.7 |

Under `id`, M01 and M02 are worse than with no loop (80 against 78, and 52 against 47), and M04 is unchanged. The by-chunk tables stayed on the lab computer.

### Conclusions

The gain lab 22 found comes from its particular settings and order, and doesn't show that self-training works in general. Recorded order is what a device would actually see, and at one seed it gives no gain. Together with lab 25 (5.7 words on the pretrained adapter), that means test-time adaptation isn't a prototype feature.

With one seed per arm and differences of 18 to 25 words, close to the project's seed spread (median 10), this shows the gain is fragile. It doesn't show the loop never helps. One possible reason, which I didn't test: in hash order, earlier chunks contain other takes of the same prompts as later chunks, and recorded order doesn't. Counting repeated prompts across chunks would check that.

</details>

## Related

[Lab test 22](../lab-22-test-time-adaptation/notes.md), [Lab test 25](../lab-25-tta-on-pretrained/notes.md)
