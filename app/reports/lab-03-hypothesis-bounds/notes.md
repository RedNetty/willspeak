# Lab test 3: hypothesis bounds, best-of-N and a vote over the systems that exist

Proposal: F-2 (H3), an extension that puts a ceiling on routers and ensembles. Prototype bar: 35%.

Status: run 7 October 2026 (CPU, reads existing decodes), results read 14 October. See also the [Report Context](../Report%20Context.md) and the [Measurement guide](../../../docs/measurements.md).

## Idea

If no combination of the existing systems could reach 35%, there's no point building a router or ensemble from them. So per test clip I take the errors of whichever system did best. A device can't know that, so this oracle is a ceiling, and a plain word-level vote shows what a device could actually do.

## Setup

Pooled means wrong words summed over M01 (169 test words), M02 (95) and M04 (161), 425 in all. Stock Parakeet-TDT-0.6B-v2 gets 323 wrong (76.0%), the best raw result so far is about 201 to 210 (about 48.5%, adapter after block 8, three seeds), and the 35% bar is 148 or fewer.

Systems, each decoded once: stock, F-1 (leave-one-speaker-out, on the F-2 test clips), the adapter after block 8 on stock (fixed 15-epoch schedule, seed 17), the adapter on F-1 (seed 17) and whole-block. If stage 14 has run, the routed systems (words to the matcher, sentences to Parakeet) are added. Each model appears once so the vote doesn't count it twice.

The vote aligns every system to the one with the fewest pooled errors (the project's own aligner) and gives each word to the majority, ties to that pivot. No confidence scores and no realigning the others with each other, so it's simpler than full ROVER.

## Plan

1. Check that every system's hypotheses exist for all three speakers and that no output exists yet.
2. `hyp_bounds.py`: wrong words for each system alone, the oracle and the vote for every group of up to 4 systems, and the lowest oracle over all groups.

The only decision rule is the bar: report the lowest oracle as a percentage of the 425 words and whether it's at or below 148. If it's above 148, no router or ensemble of these systems can reach 35%, and gains have to come from a better model or more data. If the oracle is below 148 but the vote isn't, a combination could get there only with a confidence signal the device doesn't have yet. If the vote itself is below 148, a plain vote reaches the bar, at the cost of running several models on the Pi.

## Why

It takes seconds and closes or opens a whole family of ideas (routers, ensembles, voting) before any gets built.

## Limits

This is an upper bound from seed-17 decodes on these 425 words. It can't show what a real router achieves, since the oracle chooses with the answer in hand and errors on the same clip are correlated, or say anything about speed on the Pi. The systems share a base model, so their errors overlap more than independent ones would.

## Results

Pooled over 425 test words. The bar is 148 wrong or fewer.

| System | Wrong | WER |
|---|---:|---:|
| stock | 323 | 76.0% |
| whole-block | 291 | 68.5% |
| F-1 | 248 | 58.4% |
| adapter on stock | 208 | 48.9% |
| adapter on F-1 | 197 | 46.4% |

Picking the best of these five complete hypotheses per clip, with the reference in hand, gives 158 wrong (37.2%). That's still 10 above the bar, and no group tried does better. The best plain vote is 193 (45.4%, from adapter on stock, adapter on F-1 and whole-block), and a vote over all five gives 207 (48.7%). No vote beats the best single system (adapter on F-1, 197).

The oracle only picks among complete hypotheses, so it doesn't bound word-level combination, rescoring, phrase correction or later models. The routed systems weren't included (stage 14 ran afterwards), and there's one run per system.

Source: hyp_bounds log (`logs/hyp_bounds-20261007-230840-545548.log`). `bounds/results.json` was written on the lab computer. Runs on the RTX 4000 Ada lab computer, 6-8 Oct 2026, reviewed 14 October 2026.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

No prediction was written before this run (7 October 2026), and none is added after the fact.

### Commands and Setup

Run from `app/` in the WSL tree on Yack after `git pull`, with the GPU free and the PC set not to sleep. The script stops if a `train.py` run is already going, and each stage stops at once if one of its outputs already exists.

```bash
python lab.py run 3
```

At the end the script writes `logs/lab-<stamp>-summaries.txt`, which is the file recorded in these notes. The stage's full output is `logs/lab-<stamp>-3.txt`. Run it after stage 14 to include the routed systems. The script says which routed systems it found.

The commands come from `lab.py`. Its flags were checked against each script's own options with a dry run (`python lab.py run --dry`, see `tests/unit/test_lab_cli.py`), and each script's logic has unit tests on synthetic data. Those tests aren't recognition results. Before the run, `hyp_bounds.py` had not been run on the real hypothesis files.

### Full Results

The Results section above has the completed comparison and its sources.

### Conclusions

Interpretation added 21 October 2026 from the recorded results above.

Among the five systems of 7 October on the pooled test set (stock 323, whole-block 291, F-1 248, adapter on stock 208 and adapter on F-1 197 wrong of 425), picking the best complete answer per clip with the reference in hand gives 158 wrong (37.2%). That's the ceiling for choosing between these answers, and it's still 10 words above the 148-wrong bar.

No real selection rule got near the ceiling. The best plain vote was 193 (45.4%), 4 words better than the best single system. The project rule needs 2 of 3 speakers moving the same way and more than 4.5 points (about 19 words), so that isn't a shown gain. A vote over all five gave 207.

[Lab test 23](../lab-23-hypothesis-selector/notes.md) repeated this with four systems and selectors fitted on the labelled clips. On the same 425 test words the vote gave 196, the learned selector 205 and the oracle 159, so the result held with a fitted selector too.

Later single systems now sit close to this old ceiling. The adapter on the severe-pretrained model at 6e-4 averages 178.0 over three seeds ([stage P](../f2-adapter-levers/notes.md)), and lab test 22's test-time adaptation reached 179 at one seed. Better single models have moved the project further than choosing between answers did.

Next step: selection between whole answers is closed. Gains have to come from a better single model, word-level correction or the device's phrase list ([lab test 24](../lab-24-phrase-scoring/notes.md)).

</details>

## Related

- [Lab test 14](../lab-14-route-scoring/notes.md)
- [Phrase matching](../phrasebook-matching/notes.md)
- [F-2: adapter position seeds](../f2-adapter-position-seeds/notes.md)
