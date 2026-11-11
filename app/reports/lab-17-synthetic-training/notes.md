# Lab test 17: training on synthetic dysarthric speech, three ways

Proposal: not a proposal item. It extends H3 / F-2 in the [Report Context](../Report%20Context.md). Prototype bar: 35%.

Status: written and unit-tested 14 October 2026. Blocked on 16 October by the missing TORGO control speakers. Run 27 October 2026 in queue `1027-1530` (finished 18:37) with no prediction written first. See Results.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

If lab test 16's synthetic speech is any use, it should mean fewer wrong words on each speaker's real test clips. I compared three ways of using it on the usual 425 words.

## Setup

1. Arm 1, mix: his own training clips plus synthetic ones (`synth_scale`, default 2 per own clip), adapter after block 8 on stock (the 48.9% recipe).
2. Arm 2, synthetic then own: whole-block on synthetic clips only (`synth_pre_scale`, default 6 per own clip), then the usual recipe on his sessions. That's lab test 11 with synthetic speech in place of the other severe speakers.
3. Arm 3, severe plus synthetic, then own: lab test 11's leave-one-out export of the other severe speakers with synthetic clips added, then his sessions. This checks whether synthetic adds anything to the real severe data.

Severity is lab test 16's calibration for that speaker, or 0.7 if it hasn't run (`synth_severity` overrides). The clips are made from control recordings whose prompts never appear in any dev or test set, and the build refuses any that would. Seed 17 throughout, so one seed per arm.

## Plan

1. Build the three exports per speaker with `synth_dysarthria.py build` (reused if present).
2. Train and decode the nine models, trimming weights as the disk fills.
3. Pool the wrong words and compare with stock (323), whole-block (291), the adapter (208) and, if present, lab test 11's severe-others-then-own (248).

Decision rule, set before the run: the usual 4.5-point rule from `pool_table.py`. Under about 19 wrong words (4.5 points of 425) isn't a difference, and one seed can't rank close arms.

How I'd read it (written before the run): arm 1 beating the adapter's 208 by more than that would mean mixing helps, and lab test 18 would use it. Within the noise is no evidence of help, just longer training. Arm 1 or 2 clearly worse than its base would mean synthetic speech teaches habits real speech doesn't have. Arm 3 under lab test 11's 248 would mean it adds to real severe speech. If only M01 or only M04 moves, I'd check his calibrated severity in lab test 16 first.

## Why

It's the only new source of training data I have. If it helps, it could serve other speakers too. If not, my friend's own sessions are the only data, and that sets the recording plan.

## Limits

This only shows whether one of three uses lowers wrong words on the 425 common words. It can't show a gain for a new speaker with no recordings, on another day's speech, or beyond one seed. The control speakers read the same word lists, so the synthetic data and test words share one narrow vocabulary.

## Results

Run 27 October 2026 (queue `1027-1530`, source `logs/lab-1027-1530-17.txt`, `table.json` in this folder on the lab computer). Seed 17, severity from lab 16, pooled wrong of 425 (stock 323, bar 148). The first four rows are earlier runs for comparison.

| System | M01 | M02 | M04 | Pooled | WER | Sentences only (of 303) |
|---|---:|---:|---:|---:|---:|---:|
| stock | 131 | 62 | 130 | 323 | 76.0% | 188 (62.0%) |
| whole-block | 130 | 40 | 121 | 291 | 68.5% | 167 (55.1%) |
| adapter on stock (arm 1's base) | 87 | 42 | 79 | 208 | 48.9% | 112 (37.0%) |
| other severe speakers, then own (lab 11) | 98 | 47 | 103 | 248 | 58.4% | 142 (46.9%) |
| synthetic speech only | 122 | 42 | 125 | 289 | 68.0% | 166 (54.8%) |
| adapter + synthetic speech | 93 | 49 | 89 | 231 | 54.4% | 136 (44.9%) |
| synthetic, then own | 122 | 41 | 119 | 282 | 66.4% | 164 (54.1%) |
| severe + synthetic, pretraining only | 98 | 52 | 107 | 257 | 60.5% | 157 (51.8%) |
| severe + synthetic, then own | 89 | 46 | 107 | 242 | 56.9% | 140 (46.2%) |

No arm gets to 189, which arm 1 needed to count as help (208 less 19). Synthetic speech in the adapter's training makes it 23 words worse (231 against 208), on all three speakers. Whole-block on synthetic speech alone (289) is level with whole-block on his own clips (291), and 282 once his clips are added after. So for that recipe the synthetic clips teach about as much as real ones, and nothing the adapter can use. On the severe-speaker route they give 242 against 248, inside seed noise.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

No prediction has been written and the stage hasn't run. One should be written before `python lab.py run 17` and recorded here.

For comparison, these are numbers from earlier recorded runs, taken on 21 October before any result of this test existed. Arm 1's base is the adapter on stock at seed 17: 208 wrong of 425 (48.9%), reproduced exactly in three studies. Under the project rule a change has to move about 19 words, so arm 1 needs 189 or fewer to count as help. Arm 2's bases are whole-block (291) and lab test 11's other-severe-speakers-then-own (248). Arm 3 compares with lab test 11's 248, but stage P later reached 178.0 with severe pretraining under the adapter, so synthetic speech would have to add to that to change the plan. Lab test 21 (speech synthesiser audio added to the adapter on F-1) didn't help: 204 and 214 against 197.

### Commands and Setup

Run from `app/` in the WSL tree on Yack after `git pull`, GPU free and the PC set not to sleep. It needs the controls export and the leave-one-out severe exports, which get made if missing. With `trim_models` on, each model's weights are deleted after its decode.

```bash
python lab.py run 17
```

At the end the script writes `logs/lab-<stamp>-summaries.txt`, which is the file recorded in these notes. The stage's full output is `logs/lab-<stamp>-<stage>.txt`.

The commands come from `lab.py`. Its flags were checked against each script's options with a dry run (`python lab.py run --dry`) (`tests/unit/test_lab_cli.py`, `tests/unit/test_lab_hardening.py`), and each script's logic has unit tests on synthetic data, which aren't recognition results. At setup on 14 October 2026 no stage had run on the real models or data, and there was no GPU to validate on.

### Full Results

See Results above.

### Conclusions

*Interpretation added 29 October 2026 from the table above.* Synthetic dysarthric speech doesn't help the adapter and isn't a way past the 35% bar. It's one seed per arm, so a difference under about 10 words is noise.

</details>

## Related

[Lab test 16](../lab-16-synthetic-realism/notes.md), [Lab test 11](../lab-11-severe-pretraining/notes.md), [Adapter on stock](../f2-m01-m04-adapter8/notes.md)
