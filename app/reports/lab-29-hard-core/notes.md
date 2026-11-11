# Lab test 29: the hard core of errors

Proposal: analysis (extension), following lab tests 3 and 23. Prototype bar: 35%, or 148 wrong of 425 pooled test words.

Status: written 20 October 2026, run in the lab computer's `1020-new` queue (done 02:09 on 21 October, lab clock). Read from the printed tables in `logs/` on 21 October 2026. The files the queue wrote stayed on the lab computer and aren't in the repository.

See also the [Report Context](../Report%20Context.md) and the [Measurement guide](../../../docs/measurements.md).

## Idea

Every improvement since 1 October has ended between 178 and 213 wrong words, and lab test 23 showed the systems fail on the same clips (oracle 159). So before another training idea I looked at the errors: which clips and words every saved system gets wrong, what kind of clip, how long. It's an after-the-fact reading of the test references and chooses nothing. It shows where the last 30 words to the bar would have to come from.

## Setup

`scripts/asr/hard_core.py` reads each system's saved answers and per-clip errors, for every system that has all three speakers (stock, F-1, whole-block, the adapters, the pretrained adapter, the TTA runs). It reports the pooled per-clip oracle, the clips every system gets fully wrong or right, errors by kind (single word or sentence) and duration, and the hardest clips with every system's answer.

## Plan

One CPU run, writing `reports/lab-29-hard-core/hard_core.json` and printing the tables to the log.

It's description, so there's no decision rule. Beforehand I noted what each outcome would mean. Mostly single words would make the bar a single-word problem, for the matcher or phrase list to fix. A few long sentences holding most shared errors would point at those recordings (noise, truncation). An oracle near the bar with a small shared core would mean the systems disagree usefully, and a confidence-based combination might work where lab 23's selector failed.

## Why

Thirty words isn't many. Knowing which thirty makes the next test specific.

## Limits

This only describes these 166 clips. The hardest-clip list is test material, so nothing can be tuned on it.

## Results

Single words are the hard core, and no sentence clip is missed by every system. Eight saved systems on the same clips, plus the per-clip oracle (the best of the eight on each clip, picked with the reference in hand):

| System | Wrong | WER |
| --- | --- | --- |
| stock | 323 | 76.0% |
| F-1 | 248 | 58.4% |
| whole-block | 291 | 68.5% |
| adapter on stock | 208 | 48.9% |
| adapter on F-1 | 197 | 46.4% |
| pretrained adapter, seed 17 | 183 | 43.1% |
| adapter on F-1 with the loop (lab 22) | 179 | 42.1% |
| pretrained adapter with the loop, seed 17 | 179 | 42.1% |
| per-clip oracle | 124 | 29.2% |

56 clips (56 words) are fully wrong in every system, all of them single words, 46% of the 122 single-word clips. Ten clips (20 words) are fully right in every system. Of the best system's 179 wrong words, another system gets 55 right and none gets the other 124.

The oracle is under the bar, where lab test 3's was 158 and lab 23's 159. But it's a bound that needs the reference. Lab 23's selector, without one, scored 205, and these eight systems are mostly one family.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

No prediction was written before the `1020-new` queue started (the command `python lab.py queue 25 29 28 30 27 26 --stamp 1020-new --detach` is recorded in `app/scripts/asr/jack_notes.md`, 20 October), and none is added after the fact.

For reading the result I took some reference points on 21 October from recorded runs, before any result of this test was available. The best single systems are near 178 wrong of 425 (stage P `pre-ad8-lr6e-4` mean 178.0, lab 22's loop 179). The per-clip oracle over five systems in lab test 3 was 158, and over the systems in lab test 23 it was 159, both above the 148 bar. By kind of clip, about half the best system's errors are on single words: 92.3 of 178 on 122 single-word words (75.7%) against 85.7 on 303 sentence words (28.3%). Every recipe in the project leaves single words between about 70% and 86%. So a large share of the unfixable core should be single words, and this test reads that directly.

### Commands and Setup

```bash
python lab.py run 29
```

It takes seconds, CPU only.

### Full Results

Source: `logs/lab-1020-new-29.txt`. The stage printed the analysis, and `hard_core.json` stayed on the lab computer.

| Kind of clip | Clips | Words | Best system wrong | Oracle wrong | Clips every system gets wrong |
| --- | --- | --- | --- | --- | --- |
| sentence | 44 | 303 | 93 (30.7%) | 64 (21.1%) | 0 |
| single word | 122 | 122 | 86 (70.5%) | 60 (49.2%) | 56 |

| Speaker | Clips | Words | Best system wrong | Oracle wrong | Clips every system gets wrong |
| --- | --- | --- | --- | --- | --- |
| M01 | 65 | 169 | 73 | 49 | 21 |
| M02 | 37 | 95 | 41 | 25 | 13 |
| M04 | 64 | 161 | 65 | 50 | 22 |

| Duration | Clips | Words | Best system wrong | Oracle wrong | Clips every system gets wrong |
| --- | --- | --- | --- | --- | --- |
| under 1.5 s | 15 | 15 | 8 | 7 | 7 |
| 1.5 to 3 s | 80 | 80 | 50 | 32 | 28 |
| over 3 s | 71 | 330 | 121 | 85 | 21 |

The best system here is the adapter on F-1 with the loop, tied at 179 with the pretrained adapter with the loop. All 20 hardest clips are single words, including `juliet`, `foxtrot`, `briar`, `pitting` and `brawn`. The stock model answered the first of them with "One, two, three, stop." These lists are test material and nothing was tuned on them.

### Conclusions

The reference point written before the run held, and more strongly than I expected: the unfixable core is entirely single words. On sentences the best system makes 30.7% errors against an oracle of 21.1%, and no sentence clip defeats every system. Single words are the exemplar matcher's job in the design, and this is the clearest case yet that the 425-word pooled bar mixes two jobs. The 124-word oracle shows selection could pass the bar in principle, but lab 23 already showed a learned selector can't get there.

</details>

## Related

- [Lab test 3](../lab-03-hypothesis-bounds/notes.md)
- [Lab test 23](../lab-23-hypothesis-selector/notes.md)
