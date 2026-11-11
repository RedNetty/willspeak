# Lab test 13: silence false inserts. What each model writes for clips with no speech

Proposal: device behaviour (`docs/design.md`), not a proposal item. Prototype bar: 35%.

Status: run 7 October 2026 (decode only). Results read 14 October.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

A device that listens all day hears far more silence, breathing and room noise than speech. Anything a model writes for those is a false insert: the voice says something he didn't. Nothing in the project counts this yet, so I built a battery of non-speech clips and counted what each model writes.

## Setup

The battery (`battery.py build`, seed 2000) has near-digital silence, white and pink noise, mains hum, room tone, breath-like bursts and decaying taps, 20 clips each, plus up to 60 edge clips cut from the quiet start and end of M02's real test recordings (real room, real microphone). The same seed rebuilds the same clips; Lab test 10's training uses other seeds. Other people's speech is left out as a gating problem (`docs/design.md`, Other voices).

Models: stock, and each speaker's whole-block and F-1 models, reusing existing decodes. `battery.py score` reports per kind how many clips got any words and how many words in total.

## Plan

1. Build the battery if it's missing.
2. Decode it with each model and score false inserts per kind.

No pooled decision rule, just counts per model and kind. They're small, so a few clips either way means nothing.

How I'd read it (written before the run): if the fine-tuned models write words for noise that stock doesn't, training on speech alone teaches them to hear speech everywhere, and the device needs a guard (Lab test 10, or a stricter energy gate). If no model writes anything, the battery is too easy or the models cope, and the edge clips matter most. If stock writes more, fine-tuning reduces a stock-model problem.

## Why

The device will hit this constantly and it's never been measured. This is a quick first number.

## Limits

This only covers synthetic and edge clips on existing models. It can't show false inserts in a real room over hours, with speech nearby, or behind the device's energy gate (which would remove some of these clips).

## Results

Clips where each model writes any words, on the 200-clip battery (seed 2000): 140 synthetic and 60 real quiet edges.

| System | Any words, of 200 | Synthetic, of 140 | Edges, of 60 | Silence clips, of 20 |
|---|---:|---:|---:|---:|
| stock | 70 | 10 | 60 | 2 |
| whole-block M01 / M02 / M04 | 64 / 59 / 64 | 4 / 0 / 4 | 60 / 59 / 60 | 0 / 0 / 0 |
| F-1 for M01 / M02 / M04 | 24 / 25 / 19 | 2 / 2 / 2 | 22 / 23 / 17 | 0 / 0 / 0 |

Stock writes words for 10 of 140 synthetic clips and for all 60 real edges. The F-1 models cut the edges to about a third and write nothing on synthetic silence. The edges were picked by RMS and haven't been checked by ear for speech. This is only a decoder battery, so it doesn't give a live-room false-trigger rate. A battery from a different seed (3000, in `f2-long-adapter`) gives stock 3 of 140, so counts across seeds can't be compared.

Source: battery log (`logs/battery-20261007-230944-273496.log`).

*Source runs: RTX 4000 Ada lab computer, 6-8 Oct 2026. Results reviewed 14 October 2026.*

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

No prediction was written before this run (7 October 2026), and none is added after the fact.

### Commands and Setup

Run from `app/` in the WSL tree on Yack after `git pull`, GPU free and the PC set not to sleep. The script stops if a `train.py` run is already going, and each stage stops at once if one of its outputs exists.

```bash
python lab.py run 13
```

At the end the script writes `logs/lab-<stamp>-summaries.txt`, which is the file recorded in these notes. The stage's full output is `logs/lab-<stamp>-13.txt`.

The commands come from `lab.py`. Its flags were checked against each script's options with a dry run (`python lab.py run --dry`) (`tests/unit/test_lab_cli.py`), and each script's logic has unit tests on synthetic data, which aren't recognition results. The battery decodes on real models weren't run beforehand. The generator is unit-tested for determinism and for none of the kinds being silent.

### Full Results

The Results section above has the completed comparison and its sources.

### Conclusions

*Interpretation added 21 October 2026 from the results above.*

On the 200-clip battery (seed 2000, fewer clips with words is better), stock wrote words on 70 clips: 10 of the 140 synthetic ones and all 60 real edges. The F-1 models wrote words on 19 to 25 clips. They cut the edges to 17 to 23 of 60 and wrote nothing on the 20 silence clips. The whole-block models cut synthetic noise (0 to 4 of 140) but still wrote words on 59 or 60 of the 60 edges.

Training against non-speech (lab test 10) brought whole-block models to 0 of 140 synthetic clips but left the edges at almost all 60. Lab test 21's models, trained with synthesised speech, wrote words on more battery clips than F-1 (37 to 48 of 200 against 19 to 25).

The adapters were measured on a different battery (seed 3000, 140 synthetic clips, [long adapter](../f2-long-adapter/notes.md)). There they wrote words on up to 20 of 140 clips (M02, seed 17), and the count varied by seed. The two batteries can't be compared directly, but the adapters aren't free of words on non-speech.

Next step: the expo model needs its own battery run before I trust it near silence. Lab test 18 (the best system with silence protection) is set up for that and hasn't run.

</details>

## Related

[Lab test 10](../lab-10-silence-margin/notes.md), [Typical-speech check](../typical-speech-check/notes.md)
