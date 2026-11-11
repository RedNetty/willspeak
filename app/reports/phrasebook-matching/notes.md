# Phrase matching: saved answers snapped to a phrase list, and what the device would say

Proposal: T-3 and the device design (`docs/design.md`). Prototype bar: 35%.

Status: covers the sentence runs of 1 and 2 October and stage W (single words and the whole test set), which ran on the lab computer 13 to 14 October (stamp `1012-pending`, CPU only, on saved decodes). Results read 15 October. The original notes for the October runs held three predictions, but that folder was never tracked by Git, so they're missing from this checkout and I haven't reinvented them. Stage W had no prediction.

See also the [Report Context](../Report%20Context.md) and the [Measurement guide](../../../docs/measurements.md).

## Idea

The device speaks what it hears, and the caregiver gives it a list of phrases the friend is likely to say. So I took each test clip's saved model answer and swapped it for the nearest phrase on a list, by the mean of character, word and phone edit distance (`scripts/asr/phrasebook.py`). The WER of the result is spoken WER. The 35% bar applies under its second reading, device output with its list (the first reading is raw model output).

## Setup

Five models' saved test answers on M01, M02 and M04 were matched: stock Parakeet-TDT-0.6B-v2 (raw 76.0%), the F-1 model (58.4%), the block-8 adapter on stock (48.9%), the block-8 adapter on F-1 (46.4%, 197 wrong) and the whole-block model (68.5%). That's 166 clips and 425 reference words (M01 169, M02 95, M04 161). 122 clips are single words (122 words) and 44 are sentences (303 words).

The pool is every distinct prompt in all speakers' train, dev and test files, 462 in all (322 single words). Each clip's list is a random subset of the given size, averaged over 20 repeats. A closed list always holds the true phrase, and an open list has it removed. The rejection rule at t keeps the nearest phrase only if its distance is at most t, and otherwise keeps the raw answer. Coverage c is computed as c × closed + (1 − c) × open from the two measured rows. No run used an actual partly filled list.

## Results

All numbers are spoken WER in percent, pooled over M01, M02 and M04, from `device-both/results.json` unless stated.

Closed list, no rejection rule, by list size:

| System | Raw | 10 phrases | 25 | 50 | 100 | all 462 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stock | 76.0 | 29.9 | 35.5 | 36.3 | 38.9 | 34.6 |
| F-1 | 58.4 | 12.7 | 15.6 | 18.1 | 19.8 | 22.4 |
| adapter on stock | 48.9 | 8.5 | 12.7 | 14.4 | 16.6 | 20.2 |
| adapter on F-1 | 46.4 | 9.8 | 12.4 | 13.5 | 15.7 | 18.8 |
| whole-block | 68.5 | 21.8 | 25.7 | 26.7 | 29.1 | 29.6 |

With the true phrase on the list, everything is under 35% except stock at 25 to 100 phrases (35.5 to 38.9). With an open list and no rule it's 92.5 to 134.2% everywhere, worse than raw (the adapter on F-1 gets 92.9% with all 461 other phrases). Snapping without a rejection rule is only safe if the phrase is on the list.

The adapter on F-1's 18.8% (whole pool, closed, no rule) splits very unevenly. Sentence words score 2.6% and single-word words 59.0%, against 36.0% and 72.1% raw. On single words alone (`words/results.json`, a 322-word pool, closed) it gets 22.8% with 10 words, 30.7% with 25, 37.3% with 50, 44.9% with 100 and 55.7% with all 322, so a long list keeps them above 35%.

Rejection rule and partial coverage for the adapter on F-1. "No rule" always snaps.

| Coverage | No rule | t = 0.50 | 0.60 | 0.65 | 0.70 | 0.80 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 100% | 18.8 | 26.6 | 22.6 | 21.2 | 18.8 | 18.8 |
| 90% | 26.2 | 28.7 | 25.7 | 24.8 | 22.9 | 25.6 |
| 80% | 33.6 | 30.7 | 28.8 | 28.3 | 27.0 | 32.4 |
| 50% | 55.9 | 36.9 | 38.2 | 39.1 | 39.3 | 52.7 |

The adapter on stock is similar: 20.2, 27.7, 35.1 and 57.4 with no rule at 100, 90, 80 and 50% coverage, best cells 20.2, 24.8, 28.8 and 39.2. The rule matters from about 80% coverage down. The best cell at 50% is 36.9%, so 35% is crossed between 50% and 80% coverage, near 55 to 60% by interpolation.

The rule protects sentences. For the adapter on F-1 on sentence clips with an open list, t = 0.50 to 0.55 gives 35.6% against 36.0% raw, so a missing sentence barely suffers. At t = 0.80 it's 79.5%.

The earlier sentence-only runs (1 and 2 October) survive in `docs/TODO.md` and the Report Context row, which also summarises their rejection grid and its wider version. On 44 sentence clips with a 342-phrase list including the true phrase, spoken WER was 3.0% for the adapter on stock, 8.3% F-1, 8.9% adapter on F-1, 11.6% whole-block and 14.9% stock. With the true phrase removed it was 79 to 89%.

## Limits

This shows what the device says when the friend's phrase is on its list, and how fast that fails as coverage drops. How often a real list covers what he says is the unknown that decides it, and this can't measure that. Nor does it show how the device would tell a word from a sentence (the manifest says here), or whether a threshold carries to another speaker.

The closed rows are optimistic by construction, since TORGO speakers read overlapping prompts and the pool contains the test prompts. The thresholds were read off these same test clips, with no separate data and no confidence intervals, so the best cell is optimistic too. Neighbouring cells differ by a few points on 166 clips. The coverage rows are mixtures of two measured rows.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

None was written for stage W. The predictions for the 1 October run are in the original notes file, which isn't in this checkout.

### Commands and Setup

```bash
python lab.py run W
```

This runs `scripts/asr/phrasebook.py` twice (`--kinds word` into `words/`, `--kinds sentence,word` into `device-both/`) with the pool glob `data/public/torgo/export/*/{train,dev,test}.jsonl`, sizes 10, 25, 50 and 100, 20 repeats, and thresholds 0.20 to 0.80. The log is `logs/lab-1012-pending-W.txt`.

### Full Results

See `reports/phrasebook-matching/words/results.json` and `reports/phrasebook-matching/device-both/results.json`. The tables above come from the stage's log.

### Conclusions

Interpretation written from the log on 15 October.

Under the device reading of the 35% bar, the project is already under it when the phrase is on the list. The adapter on F-1 gets 18.8% and the adapter on stock 20.2%, against 46.4% and 48.9% raw.

That depends on coverage. With the rejection rule the adapter on F-1 stays under 35% down to roughly 55 to 60% coverage (interpolated between the 80% and 50% rows). At 50% the best cell is 36.9%.

Single words are the weak part, at 59.0% with a closed list of 322 words. A short list of his own words (10 to 25) gets 23 to 31%.

None of this changes the raw reading, which is 46.4% at best.

</details>
