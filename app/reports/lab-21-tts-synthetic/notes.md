# Lab test 21: text-to-speech as a source of synthetic speech

Proposal: data augmentation, following lab tests 16 to 18. Prototype bar: 35%.

Status: run on 16 October 2026 (03:21 to 04:48) in the `followup` queue, one seed (17), no prediction written first. The results below come from the recorded run log.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

Lab tests 17 and 18 degrade healthy speakers reading TORGO prompts, words the severe speakers already have. A synthesiser can say anything, including phrases a caregiver types in. This checks whether synthesiser audio, degraded the same way, helps the adapter on the F-1 model.

## Setup

`tts_synth.py make` speaks the other speakers' prompts (`loso-<speaker>` train and dev) in several voices and speeds at 16 kHz mono, skipping his own dev and test prompts. `synth_dysarthria.py build` degrades them (lab test 16's calibrated severity, else 0.7) and mixes them in at `synth_scale` per own clip, and the adapter on F-1 trains on that with the best system's recipe. A second arm at severity 0.2 barely degrades, to separate new words and voices from the degrading.

`tts_extra_text` names a file of extra phrases, like a caregiver's list. The engine is espeak-ng (`sudo apt install espeak-ng`, small robotic voices) or piper (`pip install piper-tts`, `.onnx` voices in `tts_voices`). Without either the stage stops with install instructions.

## Plan

1. Build one TTS export per speaker, then two mixes each (calibrated severity and 0.2).
2. Train the adapter on each mix, decode the test clips, run lab test 13's silence battery.
3. Pool against stock, F-1, the adapter on F-1, and lab 18's recorded-controls arm if it has run.

Decision rule, set before the run: the 4.5-point noise rule against the adapter on F-1, and a gain that brings more false inserts on no-speech clips counts against the arm. If only the calibrated arm helps, it's the degrading. If both help equally, it's the words and voices, and typed phrases matter more than the degrading code. No change means synthesiser audio is too far from dysarthric speech, so stay with lab 18's recorded controls. More false inserts means the clips teach the model to hear speech in noise and need the silence loss.

## Why

His phrases won't be in TORGO. If typed phrases spoken by a synthesiser help, the device could learn his vocabulary before he records much.

## Limits

Doesn't show that the audio sounds like him, or what a better synthesiser than espeak-ng or these piper voices would do.

## Results

Wrong words pooled over M01, M02 and M04 (425 test words). The 35% bar is 148. One run per arm.

| System | M01 | M02 | M04 | Pooled | WER | 95% CI (clips resampled) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| stock | 131/169 | 62/95 | 130/161 | 323 | 76.0% | 66.7 to 86.3 |
| F-1 | 91 | 56 | 101 | 248 | 58.4% | 50.0 to 67.2 |
| adapter on F-1 | 78 | 47 | 72 | 197 | 46.4% | 40.2 to 53.2 |
| adapter on F-1 + TTS (severity 0.7) | 84 | 45 | 75 | 204 | 48.0% | 41.2 to 55.8 |
| adapter on F-1 + TTS (barely degraded, 0.2) | 74 | 56 | 84 | 214 | 50.4% | 43.7 to 58.2 |

Neither arm beats the adapter on F-1. They're +7 and +17 wrong words, inside the 19-word run-to-run noise, so no benefit and no shown harm. The degraded arm is 10 words ahead of the barely-degraded one, also noise. Both beat stock (+28.0 and +25.6 points, intervals above zero), but so does the adapter alone.

Silence battery (200 no-speech clips, stock 70), calibrated and barely degraded: M01 37 and 47, M02 46 and 48, M04 46 and 40. Most are breath clips (7 to 14 of 20) and quiet recording edges (20 to 40 of 60). Hum, pink and white noise and silence stay near zero except taps (9 of 20, M04 calibrated). The F-1 models got 24, 25 and 19 in lab test 13, so TTS adds false inserts. The log had no battery row for the adapter on F-1, so the comparison is with F-1 itself. By the decision rule this counts against the arm.

Sentences only (secondary score, added 21 October 2026): the same decodes on sentence clips alone, 303 of the 425 words, where 35% is 106 wrong. From each run's `results.json` via `scripts/asr/sentence_rescore.py`. All recipes are ranked in `docs/measurements.md` under "Secondary score: sentences only".

| Recipe (run name) | Seeds | Sentence words wrong (of 303) | Sentence WER | M01 / M02 / M04 | All-clip WER |
| --- | ---: | ---: | ---: | --- | ---: |
| `lab-21-tts` | 1 | 114.0 | 37.6% | 40.2 / 36.8 / 35.4 | 48.0% |
| `lab-21-ttslow` | 1 | 128.0 | 42.2% | 36.1 / 50.0 / 44.2 | 50.4% |
| `stock 0.6B v2 (reference)` | 1 | 188.0 | 62.0% | 61.5 / 47.1 / 71.7 | 76.0% |

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

None. I started the `followup` queue as an exploration ("run every proposed follow-up just to see"), which overrides the predict-first rule for this batch, so read the numbers that way.

### Commands and Setup

Run from `app/` in the WSL tree on Yack after `git pull`, with the GPU free and the PC set not to sleep.

```bash
python lab.py run 21
```

At the end the script writes `logs/lab-<stamp>-summaries.txt`, which is the file recorded in these notes. The stage's full output is `logs/lab-<stamp>-<stage>.txt`.

The commands come from `lab.py`. A dry run (`python lab.py run --dry`) checks its flags against each script's options (`tests/unit/test_lab_cli.py`, `tests/unit/test_lab_hardening.py`), and each script's logic has unit tests on synthetic data, which aren't recognition results. At setup on 14 October 2026 nothing had run on the real models or data, and there was no GPU or TORGO audio to validate on.

### Full Results

`reports/lab-21-tts-synthetic/table.json` and `battery.json`, log `logs/lab-1016-followup-21.txt`.

### Conclusions

Interpretation written from the log on 16 October.

Synthesiser audio mixed in at two clips per own clip didn't help the adapter on F-1 at either severity (one seed). It also raised the number of non-speech clips that get words, which matters for hands-free use. So both the "no change or worse" and "more false inserts" outcomes from the plan apply.

Not tested: a better voice, more synthetic data, typed caregiver phrases, or lab test 18's silence loss.

</details>

## Related

[Lab test 16](../lab-16-synthetic-realism/notes.md), [Lab test 17](../lab-17-synthetic-training/notes.md), [Lab test 18](../lab-18-best-system-hardened/notes.md)
