# Lab test 25: test-time adaptation on the pretrained adapter. Do the two gains add?

Proposal: F-2 and device design (extension), following lab test 22 and stage P. Prototype bar is 35%, which is 148 wrong of 425 pooled test words.

Status: written 20 October 2026, run in the `1020-new` queue on the lab computer (finished 02:09 on 21 October, lab clock). Results read from the printed tables in `logs/` on 21 October 2026. The tables and reports the queue wrote stayed on the lab computer and aren't in the repository.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

Two changes have each taken the block-8 adapter below 200 wrong words, and they've never been combined. Severe-speaker pretraining (stage P, 15 October) put the adapter at 6e-4 on a model first trained on the other severe speakers: 183 / 176 / 175 at seeds 17 / 18 / 19, mean 178.0 (41.9%), against 213 on stock. Test-time adaptation (lab test 22, 16 October) retrained the adapter on F-1 between chunks of test clips on its own agreed answers, no transcripts: 179 (42.1%) against 197 at one seed, with the gain growing in later chunks.

One changes where the adapter starts, the other what it learns from during use. If they fix different errors the gains should add, landing near 160. If they fix the same clips, the loop won't gain much.

## Setup

Stage 25 of `lab.py`, for M01, M02 and M04 at seeds 17, 18 and 19. The static system is stage P's `pre-ad8-lr6e-4` run, with its saved answers reused (retrained under the same name only if missing). Test clips go into four chunks by hash of clip id, as in lab test 22. The static system decodes chunk 0. Round k trains the same recipe (adapter after block 8, 15 epochs, 6e-4, on the pretrained model) on the speaker's labelled clips plus earlier chunks' answers that F-1 agrees with, then decodes chunk k. Nothing reads the test text; `tta.py` only sees id, path and duration.

Pooled scoring over 425 reference words, stock 323. Each seed's static system is compared with the same seed plus the loop, and the three-seed means too.

## Plan

1. Rebuild the phase 1 models (stage C's pretrained models) if their weights were cleaned.
2. For each speaker and seed: static answers, three rounds of build, train, decode, then merge, score and write the by-chunk table.
3. Write one pooled table with stock, F-1, the adapter on F-1, lab 22's result if present, and the static and looped systems at each seed.

Decision rule, set before the run: the project's rule on the three-seed mean, loop against static. At least 2 of 3 speakers have to move the same way and pooled WER has to move by more than about 4.5 points (19 words). The by-chunk tables also have to show the gain holding or growing in later chunks.

Outcomes, written before the run. Loop beats static by the rule, near 160: the gains add, so ship the pretrained adapter and keep lab test 22's overnight retraining queue. Loop wins by less than the rule: the two fix overlapping clips and the loop is worth less on a better start. Loop worse: the better start makes the agreement gate admit more wrong answers, so tighten `tta_agree_min` before concluding.

## Why

Both results sit about 30 words above the bar. Stacking them is the one combination of the two helpful changes I haven't tried, and it's about three hours of GPU.

## Limits

This can show whether self-training still helps when the starting system is already the best one, at three seeds instead of one. It can't show gains from a longer deployment or a real day's order of clips, anything about the intended user, or latency on the Pi.

## Results

The two gains don't add by the project's rule. Self-training moved the pretrained adapter's three-seed mean from 178.0 wrong words (41.9%) to 172.3 (40.5%), which is 5.7 words or 1.3 points. The rule needs more than 4.5 points (about 19 words). So the loop beats static by less than the rule: the two fix overlapping clips, and the loop is worth little on the better start.

| Seed | Static (stage P, 6e-4) | With the loop | Change |
| --- | --- | --- | --- |
| 17 | 183 (43.1%) | 179 (42.1%) | −4 |
| 18 | 176 (41.4%) | 173 (40.7%) | −3 |
| 19 | 175 (41.2%) | 165 (38.8%) | −10 |
| Mean | 178.0 (41.9%) | 172.3 (40.5%) | −5.7 |

Wrong words of 425 pooled reference words. Stock is 323 and the bar is 148.

Every seed moved the right way, but by 3 to 10 words, inside the 10-word median seed spread. The speakers don't agree: summed over seeds, M01 went 223 to 209, M02 111 to 113, M04 200 to 195. The best looped run (seed 19, 165, 38.8%) is still 17 words above the bar. The by-chunk tables stayed on the lab computer, so I couldn't check whether the gain holds in later chunks.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

No prediction was written before the `1020-new` queue started (the command `python lab.py queue 25 29 28 30 27 26 --stamp 1020-new --detach` is in `app/scripts/asr/jack_notes.md`, 20 October), and none is added now.

Reference points from earlier runs, noted on 21 October before any result of this test was available:

The static system (stage P's `pre-ad8-lr6e-4`) got 183, 176 and 175 at seeds 17, 18 and 19, mean 178.0 (41.9%), so the looped mean needs 159 or fewer for a 19-word gain. Lab test 22 moved the adapter on F-1 from 197 to 179 at one seed, 18 words. Simply adding would land near 160, right at the rule's edge. The bar is 148.

On average 92.3 of the static system's 178 wrong words are on the 122 single-word words (75.7%) and 85.7 on the 303 sentence words (28.3%). Lab 22's by-chunk table will show which kind the loop fixes. Across 51 recipe families the median spread of three or more seeds is 10 words (standard deviation about 4.3), so one seed moving by under about 10 words isn't a signal.

### Commands and Setup

```bash
python lab.py run 25
```

About 27 adapter trainings plus three pretrained-model rebuilds if weights were cleaned, roughly 3 hours on the RTX 4000 Ada. Results go in `reports/lab-25-tta-on-pretrained/table.json` and `by-chunk-<speaker>-s<seed>.json`.

### Full Results

Source: `logs/pool_table-20261020-214458-297331.log`, the table the stage printed. Wrong words per speaker (M01 of 169, M02 of 95, M04 of 161). Intervals resample clips, 2000 draws.

| System | M01 | M02 | M04 | Pooled | WER | 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| stock | 131 | 62 | 130 | 323 | 76.0% | 66.7 to 86.3 |
| F-1 | 91 | 56 | 101 | 248 | 58.4% | 50.0 to 67.2 |
| adapter on F-1 | 78 | 47 | 72 | 197 | 46.4% | 40.2 to 53.2 |
| adapter on F-1 with the loop (lab 22) | 73 | 41 | 65 | 179 | 42.1% | 36.3 to 48.9 |
| pretrained adapter 6e-4, seed 17 | 74 | 37 | 72 | 183 | 43.1% | 37.2 to 50.0 |
| same with the loop, seed 17 | 76 | 39 | 64 | 179 | 42.1% | 36.5 to 49.3 |
| pretrained adapter 6e-4, seed 18 | 77 | 37 | 62 | 176 | 41.4% | 35.2 to 48.9 |
| same with the loop, seed 18 | 69 | 41 | 63 | 173 | 40.7% | 34.3 to 47.7 |
| pretrained adapter 6e-4, seed 19 | 72 | 37 | 66 | 175 | 41.2% | 35.2 to 48.4 |
| same with the loop, seed 19 | 64 | 33 | 68 | 165 | 38.8% | 32.8 to 45.5 |

Every row beats stock with the interval clear of zero, and none meets the bar. Static and looped runs have almost the same intervals. Empty answers: 0 of 166 for every static run, 2, 1 and 2 with the loop. The seed-17 looped run ties lab 22's looped adapter on F-1 at 179, with different per-speaker counts.

### Conclusions

Severe-speaker pretraining is still the one change that clearly moved the adapter. Self-training adds a small step on top that this three-seed run doesn't establish (5.7 words on the mean, with M02 flat). Lab test 27 (`../lab-27-tta-settings/notes.md`) found lab 22's own gain was sensitive to the loop's settings, so on this evidence the loop isn't worth an overnight retraining queue on the device. The pretrained adapter stays the leading single system either way.

</details>

## Related

[Lab test 22](../lab-22-test-time-adaptation/notes.md), [Stage P and the adapter levers](../f2-adapter-levers/notes.md), [Lab test 11, severe pretraining](../lab-11-severe-pretraining/notes.md)
