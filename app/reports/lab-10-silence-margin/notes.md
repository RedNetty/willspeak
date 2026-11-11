# Lab test 10: silence margin. A loss against empty answers, and non-speech training clips

Proposal: F-2 (H3), an extension, since the device listens all day. Prototype bar: 35%.

Status: all nine stage-10 training runs finished on 8 October 2026 on the lab RTX 4000 Ada, after the empty-target reduction fix. Results reviewed 12 October. No new recognition run was made during the review.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

When a fine-tuned model is unsure of a hard word, saying nothing can look as likely as any words, so the device stays silent on speech. The flip side is a model trained only on speech writing words for breath or room tone. I tested two additions, alone and together: a margin loss that penalises a clip with words when the empty answer isn't unlikely enough, and synthetic non-speech clips with an empty transcript.

## Setup

Pooled is the sum of wrong words over M01 (169 test words), M02 (95) and M04 (161), 425 in all. Stock Parakeet-TDT-0.6B-v2 gets 323 wrong (76.0%). The best raw result so far is about 201 to 210 wrong (about 48.5%, adapter after block 8, three seeds). The 35% bar is 148 wrong or fewer.

Whole-block is the F-2 model that fully trains encoder blocks 0 to 7. M01 and M04 use the fixed five-epoch runs (`models/parakeet-ft-f2-<SPK>-bottom8-5epochs`). M02 uses `models/parakeet-ft-M02e-A3` (dev-selected, 42 of 95 wrong in NeMo), so M02's recipe differs from the other two.

The margin loss adds `weight * relu(margin - NLL_empty)` to NeMo's TDT loss for every clip with words (`margin_loss.py`, lattice in `tdt_lattice.py`), and is zero once the empty answer is at least `margin` nats less likely than certain. The non-speech clips come from `battery.py`'s generator (silence, white and pink noise, hum, room tone, breath, taps), and their loss is the negative log probability of the empty path. Untuned settings, picked beforehand: margin 5 nats, weight 1, 8 clips per kind per epoch (56 extra per epoch), changed with `MARGIN` and `KPER`. Training uses `--empty-clips-seed` 1000 and the test battery 2000, so the clips differ.

Three arms per speaker: the F-2 recipe plus the margin, the non-speech clips, or both. Each model is scored on pooled WER and empty-answer share, and on false inserts on the silence battery (Lab test 13), against stock and plain whole-block.

## Plan

1. Build the silence battery with seed 2000 if it's missing, and decode it with stock and each speaker's whole-block model.
2. Train each arm on each speaker's export, then decode his test set and the battery.
3. `pool_table.py` for test-set WER and empty answers, `battery.py score` for false inserts.

Decision rule, set before the run: a change counts only when at least 2 of the 3 speakers move with the pooled count and pooled WER moves by more than 4.5 points, the seed-to-seed noise (4 to 5 points per speaker). Anything else is reported as unclear, which doesn't mean the two are equal. `pool_table.py` prints the verdict. Three agreeing seeds could settle a smaller gap, but this is one run per cell. False inserts are counted per clip, and fewer than 5 of the roughly 200 battery clips isn't a difference.

How I'd read it (written before the run): an arm that lowers empty answers and WER without raising false inserts is a free win. If one arm trades WER against false inserts, the combined arm shows the trade. If nothing moves WER, empty answers aren't the limit, and false inserts may still decide whether an arm is worth keeping.

## Why

A device that listens all day hears far more silence than speech. A false insert means the voice says something he didn't, and nothing in the project's numbers counts that yet.

## Limits

One seed per cell, test WER and a synthetic battery only. It doesn't cover other people talking nearby, a gating problem (`docs/design.md`, Other voices), or real room noise beyond the edge clips from M02's test recordings. The margin value is one guess.

## Results

NeMo results on M01/M02/M04, 166 clips and 425 reference words:

| System | Wrong words | WER | Empty hypotheses |
|---|---:|---:|---:|
| stock | 323 | 76.0% | 34/166 |
| rebuilt whole-block baseline | 291 | 68.5% | 24/166 |
| margin only | 315 | 74.1% | 24/166 |
| non-speech clips only | 291 | 68.5% | 24/166 |
| both | 292 | 68.7% | 25/166 |

Noise and both beat stock in the logged paired comparison, but neither improves on the plain whole-block baseline, and margin alone is worse than it. No arm reaches the 35% bar or beats the earlier adapters. The extra noise clips also mean more optimizer updates under a fixed epoch schedule, so the update budget isn't equal.

The useful gain is on the battery. Stock writes words on 10/140 synthetic non-speech clips. Every noise-only and combined model writes words on 0/140, across silence, noise, hum, room tone, breath and taps. The full 200-clip totals are still 58–60 clips with words, because the 60 quiet edges from real recordings mostly still get words. Those edges were picked by RMS and haven't been checked by ear to be speech-free. This is only a decoder battery and doesn't measure false triggers in a live room.

Sources: recognition table log (`logs/pool_table-20261008-203159-756156.log`), battery JSON (`reports/lab-10-silence-margin/battery.json`) and battery log (`logs/battery-20261008-203200-058435.log`). The battery JSON uploaded 12 October matches the logged counts. The recognition `table.json` from the lab computer is now in this folder and matches the log (checked 21 October: margin 315, noise 291, both 292). All training runs completed with loss reduction `mean_volume` and empty target lengths guarded; the earlier failed preflights don't affect them.

Sentences only (secondary score, added 21 October 2026): sentence clips alone, 303 words, where 35% is 106 wrong or fewer. Read from each run's `results.json` with `scripts/asr/sentence_rescore.py`; the full ranking is in `docs/measurements.md` under "Secondary score: sentences only".

| Recipe (run name) | Seeds | Sentence words wrong (of 303) | Sentence WER | M01 / M02 / M04 | All-clip WER |
| --- | ---: | ---: | ---: | --- | ---: |
| `lab-10-noise` | 1 | 169.0 | 55.8% | 57.4 / 36.8 / 65.5 | 68.5% |
| `lab-10-both` | 1 | 171.0 | 56.4% | 57.4 / 39.7 / 65.5 | 68.7% |
| `lab-10-margin` | 1 | 175.0 | 57.8% | 60.7 / 42.6 / 63.7 | 74.1% |
| `stock 0.6B v2 (reference)` | 1 | 188.0 | 62.0% | 61.5 / 47.1 / 71.7 | 76.0% |

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

No prediction was written before this run (8 October 2026), and none is added after the fact.

### Commands and Setup

Run from `app/` in the WSL tree on Yack after `git pull`, GPU free and the PC set not to sleep. The script stops if a `train.py` run is already going, and each stage stops at once if one of its outputs exists.

```bash
python lab.py run 10
```

At the end the script writes `logs/lab-<stamp>-summaries.txt`, which is the file recorded in these notes. The stage's full output is `logs/lab-<stamp>-10.txt`. Putting `MARGIN=3 KPER=4` in front of `bash` changes the settings. Set them before the run starts and write them in these notes.

The commands come from `lab.py`. Its flags were checked against each script's options with a dry run (`python lab.py run --dry`) (`tests/unit/test_lab_cli.py`), and each script's logic has unit tests on synthetic data. Those tests aren't recognition results. Nothing was run on the real models or data beforehand, neither the margin loss inside a real training run nor the battery on real models. The empty-path score the loss relies on is checked against brute-force path enumeration in `tests/unit/test_tdt_lattice.py`. The loss wrapper is tested for its hinge, gradient, the fused-joint calling pattern and the first-batch check (`tests/unit/test_margin_loss.py`). On a GPU the run prints the largest difference from NeMo's loss on the first batch, so read that line before the results.

### Full Results

The Results table, battery JSON and linked logs record the completed October 8 run.

### Conclusions

Non-speech training cleaned up the outputs on generated non-speech, but didn't improve pooled speech recognition over whole-block training. The margin didn't improve recognition or the synthetic battery compared with noise alone. The next thing worth trying is non-speech training on a stronger recognizer. These models shouldn't be adopted for the prototype on their WER.

</details>

## Related

[Lab test 6](../lab-06-reference-score/notes.md), [Lab test 13](../lab-13-silence-false-inserts/notes.md)
