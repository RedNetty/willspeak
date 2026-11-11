# Lab test 2: weight merging, other speakers' changes without his own data

Proposal: extends F-1 (H2) and F-2 (H3). Prototype bar: 35%.

Status: run 6 October 2026 on the lab computer, results read 14 October. A later stage 2 attempt stopped on a missing checkpoint (`logs/lab-1-2.txt`), which doesn't affect these decodes.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

Each speaker's whole-block fine-tune is a set of weight changes from the same stock model, and adding several onto one base can combine their skills (task arithmetic, Ilharco et al., ICLR 2023; averaging, Wortsman et al., ICML 2022, 'model soups'). Do two severe speakers' changes help a third who gave no data, and does averaging all three cost a speaker his own gain?

## Setup

Pooled means wrong words summed over M01 (169 test words), M02 (95) and M04 (161), 425 in all. Stock Parakeet-TDT-0.6B-v2 gets 323 wrong (76.0%). The best raw result so far is about 201 to 210 wrong (about 48.5%, adapter after block 8, three seeds). The 35% bar is 148 wrong or fewer.

Whole-block is the F-2 model that trains encoder blocks 0 to 7 in full. M01 and M04 use the fixed five-epoch runs (`models/parakeet-ft-f2-<SPK>-bottom8-5epochs`). M02 uses the dev-selected `models/parakeet-ft-M02e-A3` (42 of 95 wrong in NeMo), so its recipe differs from the other two.

F-1 already trains on seven other speakers and decodes the held-out one. This is the cheap version: no training, just arithmetic on three existing checkpoints. `weights.py` first reports per block how big each change is and how well they line up (cosine). Near zero would mean a merge probably won't help anyone.

## Plan

1. `weights.py`: size and direction of each change per block, written to `geometry.json`.
2. For each speaker S, decode with the average of the other two speakers' changes (0.5 each, nothing from S) and with the average of all three (1/3 each).
3. `pool_table.py`: stock, both merges and whole-block.

Decision rule, set before the run: a change counts only if pooled WER moves more than 4.5 points (seed noise, 4 to 5 points per speaker) and at least 2 of 3 speakers agree. Otherwise it's "unclear", which doesn't mean the two are equal. `pool_table.py` prints the verdict. One run per cell; three agreeing seeds could settle a smaller gap.

How I meant to read it, written before the run: if the others-only merge beats stock on 2 of 3 speakers, the changes transfer and a new speaker could start from a merge (then the number to watch is how much of whole-block's gain it keeps). If not, what each model learned is tied to its speaker or recipe, and F-1's gain isn't available this cheaply. If the all-three merge falls short of whole-block, that's the price of sharing.

## Why

If changes transfer, the expo model could be built from existing checkpoints and the intended user would need less of his own data. If not, it's closed in an afternoon.

## Limits

This only tests plain averaging across three severe speakers, and the others-only merge averages just two models. It says nothing about other speakers or severities, about smarter merges (trimming, sign resolution), or about M02's different recipe.

## Results

NeMo, 166 clips and 425 reference words. "Others" is stock plus the average of the other two speakers' changes.

| System | M01 wrong | M02 wrong | M04 wrong | Pooled wrong | WER | Gain over stock (95% CI, clips) |
|---|---:|---:|---:|---:|---:|---|
| stock | 131/169 | 62/95 | 130/161 | 323 | 76.0% | - |
| others (test speaker left out) | 120 | 61 | 125 | 306 | 72.0% | +4.0 (+0.2 to +8.2); M01 +6.5, M04 +3.1, M02 +1.1; unclear |
| all three | 124 | 49 | 123 | 296 | 69.6% | +6.4 (+1.7 to +11.9), better |
| own whole-block | 130 | 40 | 121 | 291 | 68.5% | +7.5 (+1.9 to +14.2), better |

Cosines are 0.45 (M01/M02), 0.50 (M01/M04) and 0.39 (M02/M04). M02's change is about three times bigger (3.80 against 1.15 for M01, 1.33 for M04), so a plain average is mostly M02. All three change block 7 most and block 0 least.

The others-only merge moved stock by 4.0 points, which is under the rule, so the verdict is unclear. On M01 it got 120 wrong and even beat M01's own whole-block model (130), which had barely moved from stock (131). All three averaged landed 5 words from own whole-block. Nothing reaches the bar.

Sources: `table.json`, `geometry.json`, written 6 Oct on the RTX 4000 Ada lab computer (runs 6-8 Oct 2026, reviewed 14 October 2026).

Sentences only (secondary score, added 21 October 2026): the same decodes scored on sentence clips alone, 303 of the 425 words, where 35% is 106 wrong. Read from each run's `results.json` with `scripts/asr/sentence_rescore.py`; `docs/measurements.md` ranks every recipe under "Secondary score: sentences only".

| Recipe (run name) | Seeds | Sentence words wrong (of 303) | Sentence WER | M01 / M02 / M04 | All-clip WER |
| --- | ---: | ---: | ---: | --- | ---: |
| `lab-02-all` | 1 | 171.0 | 56.4% | 58.2 / 39.7 / 64.6 | 69.6% |
| `lab-02-others` | 1 | 175.0 | 57.8% | 54.9 / 47.1 / 67.3 | 72.0% |
| `stock 0.6B v2 (reference)` | 1 | 188.0 | 62.0% | 61.5 / 47.1 / 71.7 | 76.0% |

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

No prediction was written before this run (6 October 2026), and I haven't added one after the fact.

### Commands and Setup

Run from `app/` in the WSL tree on Yack after `git pull`, with the GPU free and the PC set not to sleep. The script stops if a `train.py` run is already going, and each stage stops right away if one of its outputs already exists.

```bash
python lab.py run 2
```

At the end the script writes `logs/lab-<stamp>-summaries.txt`, which is the file these notes are based on. The stage's full output is `logs/lab-<stamp>-2.txt`.

The commands come from `lab.py`. Its flags were checked with a dry run (`python lab.py run --dry`, `tests/unit/test_lab_cli.py`), and each script has unit tests on synthetic data, which aren't recognition results. Not tried on the real models beforehand: loading a blend through `nemo_hyp.py --mix`, and `weights.py` on the real checkpoints.

### Full Results

The completed comparison and its sources are in Results above.

### Conclusions

*Interpretation added 21 October 2026 from the recorded results above.*

The others-only gain is unclear: its interval (+0.2 to +8.2) just excludes zero, but 4.0 points is under the rule. Most of it is M01 (131 to 120); M02 barely moved (62 to 61). The 5-word gap between all three averaged and own whole-block is inside the noise, so sharing neither cost nor gained anything measurable, and any plain average is mostly M02's model anyway.

Training on other speakers does far more than averaging their weights. Lab test 11's model, trained only on the other severe speakers, got 275 wrong against this merge's 306, and F-1 (seven other speakers) got 248.

Next step: none. Merging is closed as a shortcut, and a new speaker should start from a model trained on other speakers, either F-1 or stage P's severe pretraining.

</details>

## Related

[F-1, bottom 8 blocks](../f1-loso-bottom8/notes.md), [Lab test 1](../lab-01-weight-interpolation/notes.md), [Lab test 11](../lab-11-severe-pretraining/notes.md)
