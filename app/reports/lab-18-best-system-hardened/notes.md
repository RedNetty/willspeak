# Lab test 18: the best system, hardened (silence protection and synthetic speech on the adapter on F-1)

Proposal: device behaviour (`docs/design.md`, Design and safety), not a proposal item. Prototype bar: 35%.

Status: written and unit-tested 14 October 2026, blocked on 16 October by the missing TORGO control speakers. Ran 27 to 28 October 2026 in queue `1027-1530` (finished 20:11), with no prediction written first.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

The best system so far is the adapter on the F-1 model. In lab test 13 stock wrote words on 10 of 140 synthetic non-speech clips and all 60 quiet real edges, and the F-1 models still did on 17 to 23 of those edges. In lab test 10, training with non-speech clips stopped all writing on the 140 synthetic clips. This retrains it with the two changes that might make it usable on a device and counts wrong words and false inserts.

## Setup

Each speaker starts from his F-1 model, adapter after block 8, seed 17. sil adds lab test 10's silence margin loss (`--empty-margin`, variables `margin` and `kper`), synth adds synthetic clips (lab test 17's mix, severity from lab test 16), and synthsil does both. The plain adapter comes from the first-round runs. Every model is also decoded on lab test 13's 200-clip silence battery (seed 2000).

## Plan

1. Check the F-1 models, adapters and decodes, and rebuild anything missing.
2. Train and decode nine models (3 arms x 3 speakers), each also on the battery.
3. Pool wrong words against stock and the plain adapter, and score the battery with `battery.py score`.

Decision rule, set before the run: only the 4.5-point noise rule on the pooled count. A few battery clips either way don't count.

If sil matches the plain adapter and writes less on the battery, it's free and goes in the device model. If it costs more than the noise, it trades recognition for fewer false inserts and the device uses the energy gate instead. If synth wins, lab test 17 holds on the stronger base. If nothing moves, the 35% bar needs another idea.

## Why

A model that scores well but types words into every quiet moment can't be given to him, so both numbers go in one table.

## Limits

Nothing here about a real room over hours, nearby speech, or the energy gate in front. One seed per arm.

## Results

From `logs/lab-1027-1530-18.txt`, with `table.json` and `battery.json` in this folder on the lab computer. Arms add silence clips (`+sil`), synthetic dysarthric speech (`+synth`), or both. Wrong words of 425 pooled (stock 323, bar 148):

| System | M01 | M02 | M04 | Pooled | WER | Sentences only (of 303, bar 106) |
|---|---:|---:|---:|---:|---:|---:|
| F-1 | 91 | 56 | 101 | 248 | 58.4% | 142 (46.9%) |
| adapter on F-1 | 78 | 47 | 72 | 197 | 46.4% | 109 (36.0%) |
| adapter on F-1 + silence | 73 | 48 | 71 | 192 | 45.2% | 105 (34.7%) |
| adapter on F-1 + synthetic | 84 | 39 | 75 | 198 | 46.6% | 108 (35.6%) |
| adapter on F-1 + both | 83 | 42 | 79 | 204 | 48.0% | 115 (38.0%) |

No arm moves the pooled count by the 19-word rule. Silence clips are 5 words better than the plain adapter (inside noise) and bring the sentence-only score to 105, one under the 106 line, at one seed. Synthetic speech makes no difference to the pooled count.

Battery clips that got any words: stock 70 of 200, plain adapters 23, 27 and 26 (11.5% to 13.5%), silence M01 11, M02 8, M04 5, synthetic 20, 27 and 34, both 4, 9 and 5. The synthetic arm's extras are mostly breath clips (11, 14 and 17 of 20, against 0 to 2 elsewhere). Silence protection roughly halves false words and removes those breath inserts. The rest are nearly all on the 60 quiet real-recording edges, never checked to be speech-free.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

No prediction was written before `python lab.py run 18`.

Numbers from earlier runs for comparison, taken on 21 October before any result of this test existed.

The plain adapter on F-1 at seed 17 got 197 wrong of 425 (46.4%), with 3 empty answers on 166 clips.

On the battery (clips that get any words), the F-1 models got 24, 25 and 19 of 200 (M01, M02, M04) in lab test 13, and stock got 70 of 200. Lab test 10's noise and both models wrote nothing on the 140 synthetic clips but still 58 to 60 of 200 overall, almost all on the 60 quiet real edges.

The 15-epoch adapter on stock writes words on 20 of 140 synthetic clips for M02 at seed 17 (long-adapter and acoustic-only studies) and 0 to 4 for M01 and M04. The acoustic-only study's head-frozen 100-epoch model wrote nothing on all 140 for M01 and M02. So the adapter is the noisy part and sil has something to remove.

### Commands and Setup

Run from `app/` in the WSL tree on Yack after `git pull`, with the GPU free and the PC set not to sleep. The synth arms need the controls export. The silence protection arms need the battery, which gets built if it's missing.

```bash
python lab.py run 18
```

At the end the script writes `logs/lab-<stamp>-summaries.txt`, which is the file recorded in these notes. The stage's full output is `logs/lab-<stamp>-<stage>.txt`.

The commands come from `lab.py`. A dry run (`python lab.py run --dry`) checks its flags against each script's options (`tests/unit/test_lab_cli.py`, `tests/unit/test_lab_hardening.py`), and each script's logic has unit tests on synthetic data, which aren't recognition results. At setup on 14 October 2026 nothing had run on the real models or data and there was no GPU to validate on.

### Full Results

See Results above.

### Conclusions

*Interpretation added 29 October 2026 from the tables above.* Silence protection costs nothing in recognition and cuts false words on non-speech by about half, so it's worth keeping in whichever model gets chosen. Synthetic speech isn't: it adds breath false inserts and gives no gain. One seed.

</details>

## Related

[Lab test 10](../lab-10-silence-margin/notes.md), [Lab test 13](../lab-13-silence-false-inserts/notes.md), [Lab test 17](../lab-17-synthetic-training/notes.md), [Adapter on F-1](../f2-m01-m04-adapter8-on-f1/notes.md)
