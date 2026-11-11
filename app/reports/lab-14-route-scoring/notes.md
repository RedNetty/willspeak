# Lab test 14: route scoring. Single words to the matcher, sentences to Parakeet

Proposal: T-3 and the device design (`docs/design.md`). Prototype bar: 35%.

Status: run 7 October 2026 (CPU, reads saved decodes). Results read 14 October.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

The pooled test set mixes single words and sentences, and every model is scored on both with one recognizer. On the device an enrolled word goes to the exemplar matcher and a sentence goes to Parakeet. This test builds the answers that routing would give and scores them on the same clips, to read the bar for the device as well as the raw model.

## Setup

Pooled is the sum of wrong words over M01 (169 test words), M02 (95) and M04 (161), 425 in all. Stock Parakeet-TDT-0.6B-v2 gets 323 wrong (76.0%). The best raw result so far is about 201 to 210 wrong (about 48.5%, adapter after block 8, three seeds). The 35% bar is 148 wrong or fewer.

The matcher is enrolled like the device would be. Every single-word take from an earlier session is an exemplar, a word's prototype is the mean of its takes, and the projection is the shipped recipe fitted on the other speakers only. The session mean uses only current and earlier clips of the test session, shrunk toward the enrolment mean. Exemplars come from the WavLM word embeddings, since the export folder leaves out earlier-session takes of the test prompts.

In `routed`, every single-word clip gets the matcher's best word and sentences get Parakeet. In `routed-min0.3` and `routed-min0.5`, the matcher's answer is kept only if its cosine score reaches that number. I ran these for stock, whole-block and the adapter after block 8. A word he never enrolled counts as wrong.

The router takes word or sentence from the manifest; the device would have to work that out from audio. The archive only has single words, so sentences can't go to the matcher offline. This covers the second reading of the 35% bar (device output with its list). The other tests measure the first reading, raw model output, and I haven't decided which applies.

## Plan

1. Make the WavLM word embeddings if missing.
2. Run `route_score.py` for each Parakeet system. `route.json` records, per speaker, how many word clips were enrolled and how often the matcher and Parakeet were right on exactly those clips.
3. `pool_table.py` for the three systems and their routed variants against stock.

Decision rule, set before the run: a change counts only when at least 2 of the 3 speakers move with the pooled count and pooled WER moves by more than 4.5 points, the seed-to-seed noise (4 to 5 points per speaker). Anything else is unclear, which doesn't mean equal. `pool_table.py` prints the verdict. This is one run per cell, so I can't lean on agreeing seeds to settle a smaller gap.

How I'd read it (written before the run): if routed beats its own Parakeet system by the rule, the matcher is worth the extra path. A routed system at or below 148 wrong would mean the device with its list reaches the bar under that reading, even if the raw model doesn't. If routing doesn't help, the matcher is no better on these words than the fine-tuned model.

## Why

It tests the device as designed instead of a model on its own, and it tells me how much the two readings of the bar matter.

## Limits

This gives routed accuracy on three speakers' last-session words with an oracle router. It says nothing about how well the device tells words from sentences, about words he didn't enrol, or about the Pi.

## Results

Same 425 reference words. "min" is the match-score threshold below which Parakeet's answer is kept.

| System | Wrong | WER | Against its unrouted self |
|---|---:|---:|---|
| stock | 323 | 76.0% | - |
| routed stock | 292 | 68.7% | +7.3 points vs stock (+3.7 to +12.0), better |
| routed stock, min 0.3 / 0.5 | 300 / 320 | 70.6% / 75.3% | +5.4 / +0.7 vs stock |
| whole-block | 291 | 68.5% | - |
| routed whole-block | 271 | 63.8% | +12.2 vs stock (+6.1 to +19.4); 20 fewer words than unrouted |
| routed whole-block, min 0.3 / 0.5 | 276 / 289 | 64.9% / 68.0% | |
| adapter on stock | 208 | 48.9% | - |
| routed adapter on stock | 216 | 50.8% | 8 words worse than unrouted |
| routed adapter on stock, min 0.3 / 0.5 | 215 / 208 | 50.6% / 48.9% | min 0.5 leaves the count unchanged |

Routing words to the matcher rescues the weak recognisers (stock -7.3 points, whole-block -4.7) and does nothing for the adapter, because on words the matcher is no better than the adapter. No routed system reaches the bar. The router is an offline oracle on word or sentence, so the real device would do somewhat worse.

Source: pool_table log (`logs/pool_table-20261007-231043-940765.log`), and `table.json` written on the lab computer.

*Source runs: RTX 4000 Ada lab computer, 6-8 Oct 2026. Results reviewed 14 October 2026.*

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

No prediction was written before this run (7 October 2026), and none is added after the fact.

### Commands and Setup

Run from `app/` in the WSL tree on Yack after `git pull`, GPU free and the PC set not to sleep. The script stops if a `train.py` run is already going, and each stage stops at once if one of its outputs exists.

```bash
python lab.py run 14
```

At the end the script writes `logs/lab-<stamp>-summaries.txt`, which is the file recorded in these notes. The stage's full output is `logs/lab-<stamp>-14.txt`. Run it before Lab test 3 if the routed systems should be in its oracle.

The commands come from `lab.py`. Its flags were checked against each script's options with a dry run (`python lab.py run --dry`) (`tests/unit/test_lab_cli.py`), and each script's logic has unit tests on synthetic data, which aren't recognition results. The archive-and-export joins on real TORGO weren't run beforehand; the router and enrolment were only tested on small invented data.

### Full Results

The Results section above has the completed comparison and its sources.

### Conclusions

*Interpretation added 21 October 2026 from the results above.*

On the pooled test set (425 reference words, lower is better), sending single words to the matcher and sentences to Parakeet helped the weak recognisers. Stock went from 323 to 292 wrong (+7.3 points, interval +3.7 to +12.0) and whole-block from 291 to 271. It didn't help the adapter on stock, which went from 208 to 216.

Once the recogniser is strong, the matcher is no better on single words than the adapter, so routing just swaps one set of errors for another. A threshold of 0.5 kept the adapter's own answers and left the count at 208. And since the router reads word or sentence from the manifest, real routing would do somewhat worse than these rows.

Single words are still the weak point under every method tried since. The adapter on F-1 is at 72.1% WER on single words raw and 68.7% with acoustic phrase scoring ([lab test 24](../lab-24-phrase-scoring/notes.md)). So routing to the matcher isn't worth adding on top of an adapter for the expo model, and single-word errors need a different lever.

</details>

## Related

[Phrase matching](../phrasebook-matching/notes.md), [Lab test 3](../lab-03-hypothesis-bounds/notes.md), [Lab test 5](../lab-05-layer-screen/notes.md)
