# Lab test 16: synthetic dysarthric speech, and how hard it is for the stock model

Proposal: not a proposal item; extends H3 / F-2 in the [Report Context](../Report%20Context.md). Prototype bar: 35%.

Status: written and unit-tested 14 October 2026. Blocked on 16 October by the missing TORGO control speakers. Ran 27 October 2026 in queue `1027-1530` (finished 17:34), no prediction written first.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

The severe speakers have a few hundred recordings each and there's no more of their speech. The TORGO control speakers (healthy, FC and MC) read the same prompts, so I make their recordings sound dysarthric with signal processing and check how hard that is for the stock model, and at what strength it matches each real speaker. Nothing is trained.

## Setup

`scripts/asr/synth_dysarthria.py` splits each clip into syllable-sized units, slows them unevenly with pitch kept (WSOLA), adds silences between them, smears, weakens or drops consonants, flattens the spectral envelope toward its long-term average, adds hypernasality and breathiness, and applies a 4-7 Hz tremor and slow loudness drift. One severity number scales everything; 0 leaves the clip unchanged. Clip lengths match the real speakers' pace (seconds per letter from their training exports), and every random choice is seeded.

This uses the controls' `test` split, which is fine since nothing is trained. Lab test 17 trains on the controls' `train` and `dev` splits, which share no prompt with any speaker's test set (prompts are split globally by hash), and it refuses clips whose prompt is in the target's dev or test set. The target is the real speakers' stock WER on the 425 common words (323 wrong, 76.0%).

## Plan

1. Make the controls export if missing (2.5 GB download, fetched by `python lab.py doctor --fix --allow-download`).
2. Build 150 control clips at each severity in `synth_severities` (default 0.3, 0.5, 0.7, 0.9, 1.1), plus clean copies.
3. Decode each folder with stock and score it.
4. `synth_dysarthria.py summarize` gives WER per severity and the severity where it matches each real speaker and the pooled three (`calibration` in `summary.json`).

No pooled decision rule. Lab tests 17 and 18 read the calibrated severity from `summary.json`, or use 0.7 if it's missing.

Before the run I wrote down how I'd read it. A curve that rises and crosses the real WER in range gives lab test 17 a usable severity. One that stays far below means the processing is too mild. One that hits the target early and then explodes means the audio is going unintelligible and the value can't be trusted. Very different values for M01, M02 and M04 mean the per-speaker value gets used.

## Why

Synthetic training data is only worth it if it's hard the way real speech is. This is minutes of decoding before hours of training.

## Limits

Equal WER is a weak match. The model can fail on very different sounds and land on the same number, so this says nothing about whether the audio sounds like the speakers, whether the errors fall on the same words, or whether training on it helps. That's lab test 17. The controls also have no tremor or weakness of their own and read cleanly.

## Results

My lab run, 27 October 2026 (queue `1027-1530`, source `logs/lab-1027-1530-16.txt`, `summary.json` in this folder on the lab computer). Stock WER on processed control clips, 370 reference words each:

| Severity | 0.3 | 0.5 | 0.7 | 0.9 | 1.1 | Clean controls (no change) |
|---|---:|---:|---:|---:|---:|---:|
| Stock WER | 25.7% | 35.1% | 61.4% | 90.8% | 103.5% | 11.4% |

Real stock WER is M01 77.5% (131 of 169), M02 65.3% (62 of 95), M04 80.7% (130 of 161). Interpolating linearly, the curve meets M01 at 0.81, M02 at 0.73, M04 at 0.83, and the pooled mean of 74.5% at 0.79. Clean clips start at 11.4%, close to the 12.4% typical-speech check from 30 September. So stock finds the synthetic clips about as hard as the real ones at those settings, which isn't the same as them sounding alike. Labs 17 and 18 used this calibration.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

None was written before `python lab.py run 16`.

Numbers from earlier runs for comparison, collected 21 October before any result existed. The target is M01 131 of 169 (77.5%), M02 62 of 95 (65.3%), M04 130 of 161 (80.7%), pooled 323 (76.0%). Stock on the controls (30 September check) is 12.40%, so the curve has to climb about 64 points. Lab test 21's espeak-ng speech gave no training gain (204 and 214 wrong against 197), which is why realism gets checked first here.

### Commands and Setup

Run from `app/` in the WSL tree on Yack after `git pull`, GPU free, PC set not to sleep. The stage reuses the stock decodes of M01, M02 and M04 (or rebuilds them with `rebuild_missing`) and stops if `reports/lab-16-synthetic-realism/summary.json` exists.

```bash
python lab.py run 16
```

The script ends by writing `logs/lab-<stamp>-summaries.txt`, the file these notes record. Full output is in `logs/lab-<stamp>-<stage>.txt`.

The commands come from `lab.py`. Flags were dry-run checked (`python lab.py run --dry`) (`tests/unit/test_lab_cli.py`, `tests/unit/test_lab_hardening.py`), and script logic has unit tests on synthetic data, which aren't recognition results. At setup on 14 October 2026 nothing had run on real models or data and no GPU was free for validation.

### Full Results

See Results above.

### Conclusions

Added 29 October 2026 from the table; I hadn't written one. The degradation can be tuned so stock fails about as often as on the severe speakers (severity near 0.8), and the curve is smooth and monotone, so it's usable as a training-data generator. Whether it teaches the model anything is lab 17.

</details>

## Related

[Lab test 17](../lab-17-synthetic-training/notes.md), [Lab test 11](../lab-11-severe-pretraining/notes.md)
