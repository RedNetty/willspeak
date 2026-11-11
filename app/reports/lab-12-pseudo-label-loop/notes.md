# Lab test 12: pseudo-label loop, fine-tune on his own recordings without ever reading what he said

Proposal: F-2 (H3), extended to a setting the device will actually face. Prototype bar: 35%.

Status: run 8 October 2026, results read 14 October. See also the [Report Context](../Report%20Context.md) and the [Measurement guide](../../../docs/measurements.md).

## Idea

On the device there won't be transcripts of what he says, only a phrase list a caregiver typed in and a model that's sometimes right. So the loop is: decode his earlier recordings, snap each answer to the nearest phrase, keep only the clips where the snap is trustworthy, fine-tune stock on them, and use the new model for the next round. His real transcripts are never read.

## Setup

Pooled means wrong words summed over M01 (169 test words), M02 (95) and M04 (161), 425 in all. Stock Parakeet-TDT-0.6B-v2 gets 323 wrong (76.0%), and the best raw result so far is about 201 to 210 (about 48.5%, adapter after block 8, three seeds). The 35% bar is 148 wrong or fewer.

Whole-block is the F-2 model that trains encoder blocks 0 to 7 in full: the fixed five-epoch runs for M01 and M04 (`models/parakeet-ft-f2-<SPK>-bottom8-5epochs`), and the dev-selected `models/parakeet-ft-M02e-A3` for M02 (42 of 95 wrong in NeMo), so M02's recipe differs.

A clip is kept when its nearest phrase is within a distance threshold (mean of character, word and phone edit distance, as in `phrasebook.py`) and the runner-up is at least a margin further away. It's labelled with the phrase, never the raw answer. The dev split holds out whole phrases, so dev WER only measures agreement with pseudo-labels. Thresholds were fixed before the run and not tuned on any score.

The list comes from the other speakers' prompts (`loso-<SPK>` train and dev), so a phrase only he read is missing, as it would be on the device. `pseudo_label.py --test-manifest` refuses a list containing any of his test prompts. `--audit` reads his true text only after the choice, to count right labels, and changes nothing used for training. A unit test runs the loop on a manifest with no text column to check the labels don't come from it.

Each of the three rounds fine-tunes stock (the F-2 whole-block recipe) on labels from the previous round's model, and round 1's labels come from stock. The upper reference is whole-block on his true transcripts.

## Plan

1. Round R: decode his train manifest with the current model, build pseudo-labels and an export, fine-tune stock, decode his test set.
2. Record clips kept, kept labels that were right (audit), and phrases in the dev split.
3. `pool_table.py`: stock, whole-block (true labels) and rounds 1 to 3.

Decision rule, set before the run: a change counts only when at least 2 of the 3 speakers move with the pooled count and pooled WER moves by more than 4.5 points, the noise seen across seeds (4 to 5 points per speaker). Anything else is unclear, which doesn't mean equal. `pool_table.py` prints the verdict. Three agreeing seeds can settle a smaller gap; this note has one run per cell.

If round 1 beats stock by the rule, his recordings plus a phrase list improve the model with no transcripts, which is the device's real situation. If rounds 2 and 3 beat round 1 the loop compounds, and if they fall back it's reinforcing its own mistakes. If it gets most of whole-block's gain, true transcripts add little over the list; if not, the audit says whether the labels were the limit.

## Why

The F-2 results use true transcripts, which the device won't have. This is the closest test of what it will have.

## Limits

This only covers TORGO's prompt-read speech with a list of other speakers' prompts and these thresholds. It doesn't cover a caregiver's real list, free speech or the intended user, and a list that covers his phrases better or worse would change things. Labels only exist where a phrase is close, so the training set is smaller than whole-block's.

## Results

Pooled over 425 reference words, true transcripts never read.

| System | Wrong | WER | Gain over stock (95% CI) |
|---|---:|---:|---|
| stock | 323 | 76.0% | - |
| round 1 | 315 | 74.1% | +1.9 (0.0 to +3.8), unclear |
| round 2 | 315 | 74.1% | +1.9 (+0.2 to +3.7), unclear |
| round 3 | 313 | 73.6% | +2.4 (+0.8 to +4.1), unclear |
| whole-block (true transcripts) | 291 | 68.5% | +7.5 (+1.9 to +14.2) |

The loop barely moves anything, and nearly all of that is M02 (62 to 55-56 wrong). M01 and M04 change by about one word. Rounds 2 and 3 add essentially nothing, and it's nowhere near training on true transcripts.

Source: pool_table log (`logs/pool_table-20261008-003935-533689.log`). Runs on the RTX 4000 Ada lab computer, 6-8 Oct 2026, reviewed 14 October 2026.

Sentences only (secondary score, added 21 October 2026): the same decodes scored without the single-word clips. That's 303 of the 425 words, and 35% is 106 wrong or fewer. Means over the seeds listed, from each run's `results.json` via `scripts/asr/sentence_rescore.py`. The full ranking is in `docs/measurements.md` under "Secondary score: sentences only".

| Recipe (run name) | Seeds | Sentence words wrong (of 303) | Sentence WER | M01 / M02 / M04 | All-clip WER |
| --- | ---: | ---: | ---: | --- | ---: |
| `lab-12-r3` | 1 | 181.0 | 59.7% | 60.7 / 41.2 / 69.9 | 73.6% |
| `lab-12-r2` | 1 | 182.0 | 60.1% | 60.7 / 41.2 / 70.8 | 74.1% |
| `lab-12-r1` | 1 | 183.0 | 60.4% | 60.7 / 42.6 / 70.8 | 74.1% |
| `stock 0.6B v2 (reference)` | 1 | 188.0 | 62.0% | 61.5 / 47.1 / 71.7 | 76.0% |

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

No prediction was written before this run (8 October 2026), and none is added after the fact.

### Commands and Setup

Run from `app/` in the WSL tree on Yack after `git pull`, with the GPU free and the PC set not to sleep. The script stops if a `train.py` run is already going, and each stage stops at once if one of its outputs already exists.

```bash
python lab.py run 12
```

At the end the script writes `logs/lab-<stamp>-summaries.txt`, which is the file recorded in these notes. The stage's full output is `logs/lab-<stamp>-12.txt`. `ROUNDS=1` runs one round only.

The commands come from `lab.py`. Its flags were checked against each script's own options with a dry run (`python lab.py run --dry`, see `tests/unit/test_lab_cli.py`), and each script's logic has unit tests on synthetic data. For `pseudo_label.py` that includes a check that the label-free guarantee holds. Those tests aren't recognition results, and before the run nothing in this stage had been run on the real models or data.

### Full Results

The Results section above has the completed comparison and its sources.

### Conclusions

Interpretation added 21 October 2026 from the recorded results above.

On the pooled test set (425 reference words, stock 323 wrong, lower is better), three rounds of fine-tuning stock on its own phrase-snapped answers reached 315, 315 and 313 wrong. Each round's gain over stock (1.9 to 2.4 points) is under the 4.5-point project rule, so the verdict is unclear. Almost all of the movement is M02.

Training on the true transcripts (whole-block, 291 wrong) did far better. Self-training that starts from stock doesn't stand in for labelled data.

[Lab test 22](../lab-22-test-time-adaptation/notes.md) later ran a different loop: test-time adaptation that retrains the adapter on F-1 between chunks of test clips, using only answers the F-1 model agrees with. It reached 179 wrong against 197 for the same system without the loop, better on all three speakers, at one seed. That's 4.2 points pooled, just under the 4.5-point rule. The likely difference is the starting point. A system near 46% WER gives answers good enough to learn from, and stock at 76% doesn't.

Next step: drop self-training from stock. Self-training on a strong starting system is open in lab tests 25 and 27, which had no results when this was written.

</details>

## Related

- [Phrase matching](../phrasebook-matching/notes.md)
- [F-2: severe speakers, bottom 8 blocks, fixed 5 epochs](../f2-severe-bottom8-5epochs/notes.md)
