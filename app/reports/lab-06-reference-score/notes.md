# Lab test 6: reference score, how likely the model finds the true words and how likely it finds silence

Proposal: F-2 (H3), a diagnostic. Prototype bar: 35%.

Status: run 6 October 2026 (GPU), results read 14 October. See also the [Report Context](../Report%20Context.md) and the [Measurement guide](../../../docs/measurements.md).

## Idea

A wrong or empty answer has two possible causes. The model may give the true words almost no probability (a modelling problem), or the true words may be more likely than its answer (a search problem a different decoder could fix). So for every test clip I score the true transcript, the empty transcript and the model's own answer with its joint network, teacher-forced.

## Setup

Pooled means wrong words summed over M01 (169 test words), M02 (95) and M04 (161), 425 in all. Stock Parakeet-TDT-0.6B-v2 gets 323 wrong (76.0%), and the best raw result so far is about 201 to 210 (about 48.5%, adapter after block 8, three seeds). The 35% bar is 148 wrong or fewer.

Whole-block is the F-2 model that trains encoder blocks 0 to 7 in full: the fixed five-epoch runs for M01 and M04 (`models/parakeet-ft-f2-<SPK>-bottom8-5epochs`), and the dev-selected `models/parakeet-ft-M02e-A3` for M02 (42 of 95 wrong in NeMo), so M02's recipe differs.

Per clip the script reports `nll_ref` (negative log probability of the true words), `nll_empty` (of saying nothing) and `margin = nll_empty - nll_ref`. A margin near or below zero means silence looks as likely as the true words, which is how an empty answer looks from outside. A clip where `nll_ref < nll_hyp` is a search miss. Stock writes capitals and punctuation, which inflates its `nll_ref`, so `nll_empty` is the number to compare across models. The score uses NeMo's own TDT sigma and durations, and on a GPU it prints the largest difference from NeMo's loss on the first batch. Read that line first.

## Plan

1. Score each speaker's test set with stock and with his whole-block model, each using its own decoded answers.
2. From the `results.json` files, read the share of clips with a margin below a few nats and the share of search misses, per speaker and model, among empty answers and among wrong clips.

No decision rule, since it's a diagnostic. I just record the shares and which clips they come from.

Many empty answers with a small margin would mean the model isn't sure speech is there, the failure Lab test 10's loss targets. Many search misses would mean a better decoder (beam search, or rescoring against the phrase list) could recover words with no training. Few misses and large margins would mean the true words get little probability, and only different weights or more data can fix that.

## Why

It decides whether the next effort goes to decoding, the loss or the weights, before I spend training time on any of them.

## Limits

This shows where the probability sits for these two models on 425 test words. It doesn't show that a decoder change would recover the search misses, since a clip counted as a miss is only a candidate.

## Results

True words against an empty answer, per test clip, for stock and whole-block on each speaker. Margin is the cost of the empty answer minus the cost of the true words, in nats. Below zero means the model prefers silence.

| Speaker, model | Clips | Cost of true words | Cost of empty | Median margin | Clips preferring silence |
|---|---:|---:|---:|---:|---:|
| M01 stock | 65 | 15.0 | 6.3 | -6.6 | 94% |
| M01 whole-block | 65 | 14.2 | 6.1 | -5.5 | 92% |
| M02 stock | 37 | 16.1 | 6.1 | -8.8 | 95% |
| M02 whole-block | 37 | 10.4 | 4.6 | -4.5 | 81% |
| M04 stock | 64 | 17.1 | 5.8 | -7.9 | 100% |
| M04 whole-block | 64 | 15.9 | 5.1 | -8.1 | 97% |

Silence is cheaper than the true words by a median of 4.5 to 8.8 nats, and fine-tuning doesn't change that much. M02 improves most, from 95% to 81%. The first-batch check matches NeMo's own loss to 0.0000 nats. No clip has the true words more likely than the model's own wrong answer, which points at the model's scores and away from the search.

One caveat: the true words are scored in normalised form (lower case, no punctuation), but the model writes capitalised, punctuated text and pays extra for it. Part of the margin can be formatting, and that also muddies the zero-clip search comparison. This doesn't show that decoding changes can't help.

Sources: `reports/lab-06-reference-score/<speaker>-<model>.json`, and the logs `reference_score-20261006-2137*` to `-2139*` under `logs/`. Runs on the RTX 4000 Ada lab computer, 6-8 Oct 2026, reviewed 14 October 2026.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

No prediction was written before this run (6 October 2026), and none is added after the fact.

### Commands and Setup

Run from `app/` in the WSL tree on Yack after `git pull`, with the GPU free and the PC set not to sleep. The script stops if a `train.py` run is already going, and each stage stops at once if one of its outputs already exists.

```bash
python lab.py run 6
```

At the end the script writes `logs/lab-<stamp>-summaries.txt`, which is the file recorded in these notes. The stage's full output is `logs/lab-<stamp>-6.txt`.

The commands come from `lab.py`. Its flags were checked against each script's own options with a dry run (`python lab.py run --dry`, see `tests/unit/test_lab_cli.py`), and each script's logic has unit tests on synthetic data. Those tests aren't recognition results. Before the run, `reference_score.py` had not been run on a real model. The lattice code is checked in `tests/unit/test_tdt_lattice.py` against brute-force path enumeration on small cases and against a copy of NeMo's reference TDT recursion written into the test. It hadn't been compared with NeMo's own loss on a real model.

### Full Results

The Results section above has the completed comparison and its sources.

### Conclusions

Interpretation added 21 October 2026 from the recorded results above.

For stock and each speaker's whole-block model, an empty answer is cheaper than the true words on 81% to 100% of the test clips (median margin -4.5 to -8.8 nats, where below zero means it prefers silence). Fine-tuning only clearly changed this on M02, from 95% to 81% of clips preferring silence.

No clip had the true words more likely than the model's own wrong answer, which points at the model's scores and away from the beam search. The caveat in Results still stands. The true words were scored in normalised form while the model writes capitalised, punctuated text, so part of the margin and of the search comparison can be formatting.

A later test agrees that scoring is where the gain is. [Lab test 24](../lab-24-phrase-scoring/notes.md) scored a short list of candidate phrases against the audio with the same TDT likelihood. It beat text matching by 1 to 4 points and brought the adapter on F-1 from 46.4% raw to 26.4% with a closed list.

The preference for silence mostly belongs to stock and the whole-block model. The adapters return nothing on only 3 of 166 test clips (adapter on stock in lab test 14's table, adapter on F-1 in lab test 22's), against 34 for stock and 24 for whole-block. This test didn't score the adapters.

Next step: no decoder change for the expo model on this evidence. Rescoring only pays when there's a list of candidates to score, as on the device.

</details>

## Related

- [Lab test 10](../lab-10-silence-margin/notes.md)
- [F-2: severe speakers, bottom 8 blocks, fixed 5 epochs](../f2-severe-bottom8-5epochs/notes.md)
