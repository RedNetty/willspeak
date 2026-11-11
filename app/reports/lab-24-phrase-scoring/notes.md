# Lab test 24: which listed phrase the audio supports

Proposal: decoding (phrase list on a device). Follows the phrasebook matching of 1–2 October and lab test 6. Prototype bar: 35%.

Status: written and unit-tested 15 October 2026, run 16 October 2026 (`1016-retry`, lab computer). No predictions were written: I ran it as an exploration ("do all of them just to see what results"), which overrides the predict-first rule for this run only.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md), [Phrasebook matching](../phrasebook-matching/notes.md)

## Idea

The phrasebook matcher compares the model's written answer with every phrase on a list, so it throws the audio away. A muffled "tomato" and "potato" can come out as the same wrong word, and text distance can't tell which one was said. This test keeps the audio. For each clip it takes the 15 phrases nearest the model's answer by text distance, adds the true phrase and the model's own answer, and asks the Parakeet-TDT model how likely each one is given the audio (teacher-forced, summed over all alignments, as in lab test 6).

## Setup

Systems are stock, F-1, and the adapter on F-1 (the best raw system). Speakers M01, M02 and M04, all 425 test words (sentences and single words). The phrase list is every distinct prompt in the export, sentences and words.

There are three lists (see `scripts/asr/phrase_report.py`). In `closed` the true phrase is on the list but only gets scored if it's among the 15 nearest by text, and a miss counts against the method. In `closed+` the true phrase is always scored, which is what a full search of the list would reach. In `open` the true phrase isn't on the list. Spoken WER at coverage c is c × closed + (1 − c) × open.

The methods are the raw answer, the nearest phrase by text (kept only if its distance is at most t), and the phrase with the lowest `nll + lam × distance` (kept only if it costs at most tau nats more than the model's own answer).

## Plan

1. `phrase_score.py` writes one table per speaker and system (`<speaker>-<system>.json`). It checks its list-based lattice against the torch lattice on the first clip and prints the difference.
2. `phrase_report.py` picks t, or lam and tau, for each speaker using only the other two speakers (at 80% coverage), applies them to that speaker, and pools. It gives a 95% bootstrap interval over clips and the paired difference from the raw answer.
3. It also measures how much of each speaker's test prompts the rest of the data covers (own training prompts, other speakers, neither).

The decision rule is the project's usual one: at least 2 of 3 speakers move the same way and the pooled count moves by more than about 19 words (4.5 points). A gain only counts for the device if it holds at a coverage a caregiver's list can reach, which the coverage section estimates.

## Why

The raw model can't reach 35%. The best is 182 wrong of 425 (43%), and even the oracle of five systems is 158. The device reading can: the phrasebook gave 18.8% with a closed list and about 27% at 80% coverage with a rule. So the question here is whether the audio adds anything to that, especially on single words and when the true phrase isn't on the list.

## Limits

This shows whether acoustic scoring beats text matching with the same list, with settings chosen without the speaker's own test clips. It can't show how well a real caregiver's list covers real speech (TORGO prompts repeat across speakers, so coverage looks better than it may be at home), or how the thresholds behave for a speaker whose speech is changing. The `closed` rows are limited by the 15-phrase shortlist, and `closed+` shows how much that costs.

## Results

Lab run 16 October 2026, queue `1016-retry` on the RTX 4000 Ada, read from `logs/lab-1016-retry-24.txt` and `reports/lab-24-phrase-scoring/report/results.json`. 166 clips, 425 words, 15-phrase shortlist. "Chosen on others" is the cross-speaker pick from Plan step 2, and the 95% intervals resample clips. Spoken WER, lower is better.

| System | Raw | Closed list, text (chosen on others) | Closed list, acoustic (chosen on others) | 80% coverage, acoustic | 50% coverage, acoustic |
| --- | --- | --- | --- | --- | --- |
| Stock | 76.0% | 37.9% (28.8 to 48.7) | 34.1% (26.4 to 43.8) | 40.8% | 50.7% |
| F-1 | 58.4% | 26.6% (20.3 to 34.3) | 26.1% (20.3 to 33.3) | 32.9% | 43.1% |
| Adapter on F-1 | 46.4% | 28.2% (21.5 to 36.2) | 26.4% (20.1 to 33.6) | 31.8% | 40.0% |

Scoring the shortlisted phrases against the audio beats text matching by 1 to 4 points, most for the weakest model. With a list, F-1 alone (26.1%) is level with the adapter on F-1 (26.4%), so the list closes the gap between models. Sentences drop to 17 to 20% at 80% coverage. Single words barely move (adapter on F-1: 72.1% raw, 68.7% matched).

If the thresholds are read off the test clips themselves, the best cell for the adapter on F-1 is 30.1% (text, t ≤ 0.70). So the honest cross-speaker numbers above cost 1 to 2 points against that optimistic reading. The per-speaker chosen settings were text t ≤ 0.60 to 0.80, and acoustic lam 10 to 40 with tau ≤ 30 to 50.

On coverage, 100% of each speaker's test prompts appear in the other speakers' data and 0% in his own train and dev, so a list from the other speakers misses nothing. That's TORGO's shared prompt set, not a real user, so the 80% and 50% rows are the realistic ones.

On the device reading of the bar (what it says with its list), the adapter on F-1 is at 26 to 32% with settings chosen on other speakers, so under 35% down to about 80% coverage. On the raw reading nothing changes. Single words are still the open problem either way.

An earlier attempt on 15 October stopped with `MISSING models/lab-baselines/adapter-f1-M01/adapter.pt`. That's fixed in `lab.py`: `prepare_baseline` now trains a model again when a stage needs its weights and only the decodes are left.
