# Lab test 34: adapter position on stage P's pretrained model

Proposal: F-2 (extension) and the device's silence behaviour. Follows [lab test 33](../lab-33-pooled-acoustic-adapters/notes.md) and [stage P](../f2-adapter-levers/notes.md). Prototype bar: 35%, which is 148 wrong of 425 pooled test words.

Status: written 29 October 2026 and run in the `1029-night` queue on the lab computer (14:24 to 16:04 on 29 October, lab clock). Nothing failed, and no prediction was written first. I read the results on 30 October from `logs/lab-1029-night-34.txt` on branch `lab-results-1030-100827-454341`.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

Stage P is the most accurate system so far: a model trained on the other severe speakers, plus an adapter after encoder block 8 trained on the target speaker. It scored 183, 176 and 175 wrong of 425 (mean 178.0, 41.9%). Its false words on noise and silence have never been measured.

In lab test 33 (a base trained on every dysarthric speaker), adapters after block 12, 16 or 24 almost stopped false words on lab test 13's battery, and blocks 4 and 8 didn't. Accuracy didn't suffer, and block 12 had the lowest mean. Here I check whether that holds on stage P's model, changing only the adapter's position.

## Setup

The base is stage P's pretrained model, `models/parakeet-ft-big-c-pre-<speaker>`: F-1's recipe (every parameter trainable, 2e-5, dev-selected) on the other severe speakers, seed 17. If its weights were cleaned it gets retrained (about 8 minutes per speaker), which doesn't repeat exactly: lab test 28's retrain of seed 17 scored 176 against the original 183.

Adapters use stage P's recipe (15 epochs, 6e-4, scoring-form targets) after blocks 8, 12, 16 and 24 (`python lab.py set pretrained_positions "..."`), seeds 17, 18 and 19, so 36 runs. Block 8 is retrained as the control on the same base. The base alone, never scored by itself before, is decoded on test, dev and the battery.

The battery is lab test 13's 200 clips: 140 synthetic (20 each of near-digital silence, white noise, pink noise, mains hum, room tone, breath-like bursts and taps) and 60 quiet starts and ends cut from real M02 recordings ("edges"), which may hold faint speech. The base and every seed-17 adapter are run on it. It shows what a model writes with no speech, which isn't a live-room false-trigger rate.

Numbers from earlier runs, for comparison (wrong words of 425, three-seed means unless stated, lower is better):

| Base | Block 8 | Block 12 | Block 16 | Block 24 |
| --- | --- | --- | --- | --- |
| Stage P's pretrained model, 6e-4 (stage P) | 178.0 | not run | not run | not run |
| Stock model, 3e-4 (position sweep, 2 Oct) | 206.0 | 263.3 | 254.3 | 293.7 |
| Pooled acoustic model, 6e-4 (lab test 33) | 216.0 | 210.0 | 215.3 | 223.3 |

Battery, clips that got any words at seed 17, summed over the three speakers' models (of 600):

| Base | Block 8 | Block 12 | Block 16 | Block 24 |
| --- | --- | --- | --- | --- |
| Pooled acoustic model (lab test 33) | 85 | 11 | 9 | 18 |

On the battery (of 200, one model), stock wrote words on 70 clips, the pooled acoustic base alone on 56, and the adapter on F-1 on 23 to 27 per speaker (lab test 18). Stage P gets 85.7 of 303 sentence words wrong (28.3%) and 92.3 of 122 single words (75.7%).

## Plan

1. Rebuild the pretrained models if needed.
2. Decode each base alone on test, dev and the battery.
3. Train and decode the 36 adapters seed by seed (seed 17 first), with the battery for each seed-17 adapter.
4. Release each base once everything that needs it is saved.
5. Write `table.json`, three-seed means against block 8 (`seeds.json`) and `battery.json`.

Rules, set before the run. Accuracy uses the project's rule on three-seed means against this test's block 8 (at least 2 of 3 speakers the same way, more than about 4.5 points or 19 words), with stage P's 178.0 alongside. A position clearly has fewer false words if its seed-17 battery total is at most half of block 8's and lower for at least 2 of 3 speakers. It's one seed and one battery, so only a screen. A deeper position is worth using only if it passes both.

If one passes, confirm it with the int8 export (lab test 28's path) and on the Pi. If it cuts false words but loses accuracy, that's my call, and the silence gate and matcher might do it more cheaply. If none cuts false words, lab test 33's result came from its base. If a deeper one also wins on accuracy, block 8's lead was specific to stock.

## Why

A device that listens all day hears far more silence, breathing and room noise than speech, and every false word gets spoken aloud in his voice. Moving the adapter would be an easy safety win if it worked.

## Limits

All positions use 6e-4 and there's one base seed, so per-position learning rates and base-to-base variation aren't tested. Nothing here covers a real room, the live gate, int8, the Pi or the intended user. The test clips have been scored many times before, so this is exploratory.

## Results

Block 8 stays. Rebuilt here it scored 176, 172 and 175 wrong of 425 (mean 174.3, 41.0%), within noise of stage P's 178.0. Blocks 12, 16 and 24 are 50 to 76 words worse on all three speakers, far past the 19-word rule, and none clearly cuts false words.

| Adapter after block | Seeds 17 / 18 / 19 | Mean | WER | vs block 8 | Sentences (of 303) | Single words (of 122) | Battery clips with words (of 600) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none (pretrained model alone, seed 17) | 244 | 244.0 | 57.4% | +69.7 | 141 | 103 | 98 |
| 8 | 176, 172, 175 | 174.3 | 41.0% | | 83.3 | 91.0 | 98 |
| 12 | 227, 225, 222 | 224.7 | 52.9% | +50.4 | 124.3 | 100.3 | 82 |
| 16 | 243, 232, 229 | 234.7 | 55.2% | +60.4 | 134.7 | 100.0 | 57 |
| 24 | 247, 245, 260 | 250.7 | 59.0% | +76.4 | 137.7 | 113.0 | 133 |
| *stage P (reference)* | 183, 176, 175 | 178.0 | 41.9% | +3.7 | 85.7 | 92.3 | not measured |

On false words, block 16 is lowest (57 against 98, lower for all three speakers), but the screen needed 49 or fewer. Block 24 has more (133). So no deeper position qualifies, and lab test 33's result came from its base: there, block 12 or deeper wrote words on 9 to 18 of 600 clips, against 57 to 133 here.

Almost all false words here come from the 60 edge clips, which may hold faint speech. On the 140 synthetic clips the base and blocks 8 to 16 write words on 0 to 6 per model, mostly breath-like bursts. Block 24 on M04 writes words on 18.

Block 8 also repeats stage P: 174.3 on a fresh base against 178.0, with a seed range of 4 words. With lab test 28's retrain (176 at seed 17), three independent builds now land between 172 and 183. On sentences it's under the 35% line (106 of 303) at every seed, with 84, 81 and 85.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

No prediction was written before the `1029-night` queue was started (`python lab.py queue positions --stamp 1029-night --detach`), and none is added afterwards.

### Commands and Setup

From the lab computer's `app/`:

```bash
python lab.py run 34                                        # this test alone, about 2.5 hours
python lab.py queue positions --stamp 1029-night --detach   # lab test 34, then the speaker-replay rerun
```

Estimated cost was about 2.5 hours in all: three pretraining rebuilds if needed (about 25 minutes), 36 adapters with their decodes (lab test 33's 45 adapters took about 2 hours), and the battery decodes.

On disk, the three pretrained bases (2.5 GB each) are kept until every result that needs them is saved. Adapters are 4 MB each. Records are `runs.json` in this folder and storage snapshots in `logs/lab-<stamp>-storage.jsonl`. `finished.json` is written only when nothing failed, so a rerun resumes.

### Full Results

Sources are `logs/pool_table-20261029-210436-505421.log`, `logs/battery-20261029-210437-834631.log` and the stage log `logs/lab-1029-night-34.txt` (log file names use the lab computer's UTC clock). The three pretrained models were rebuilt at the start of the stage (F-1's recipe on `loso-severe-<speaker>`, seed 17), so every row except stage P sits on the same new bases. Wrong words per speaker (M01 of 169, M02 of 95, M04 of 161):

| System and seed | M01 | M02 | M04 | Pooled | WER | Sentences (of 303) |
| --- | --- | --- | --- | --- | --- | --- |
| stock | 131 | 62 | 130 | 323 | 76.0% | 188 |
| adapter on stock, 17 / 18 / 19 | 87 / 95 / 86 | 42 / 41 / 43 | 79 / 83 / 83 | 208 / 219 / 212 | 50.1% mean | 112 / 120 / 121 |
| stage P, 17 / 18 / 19 | 74 / 77 / 72 | 37 / 37 / 37 | 72 / 62 / 66 | 183 / 176 / 175 | 41.9% mean | 87 / 84 / 86 |
| pretrained model alone, 17 | 95 | 55 | 94 | 244 | 57.4% | 141 |
| block 8, 17 | 68 | 39 | 69 | 176 | 41.4% | 84 |
| block 8, 18 | 76 | 37 | 59 | 172 | 40.5% | 81 |
| block 8, 19 | 71 | 38 | 66 | 175 | 41.2% | 85 |
| block 12, 17 | 99 | 50 | 78 | 227 | 53.4% | 128 |
| block 12, 18 | 93 | 48 | 84 | 225 | 52.9% | 123 |
| block 12, 19 | 90 | 48 | 84 | 222 | 52.2% | 122 |
| block 16, 17 | 93 | 58 | 92 | 243 | 57.2% | 141 |
| block 16, 18 | 90 | 53 | 89 | 232 | 54.6% | 135 |
| block 16, 19 | 90 | 53 | 86 | 229 | 53.9% | 128 |
| block 24, 17 | 96 | 63 | 88 | 247 | 58.1% | 135 |
| block 24, 18 | 90 | 59 | 96 | 245 | 57.6% | 136 |
| block 24, 19 | 97 | 60 | 103 | 260 | 61.2% | 142 |

Every row beats stock with the clip-bootstrap interval clear of zero, and none meets the pooled bar (148). Mean wrong words per speaker over the three seeds: block 8 71.7 / 38.0 / 64.7, block 12 94.0 / 48.7 / 82.0, block 16 91.0 / 54.7 / 89.0, block 24 94.3 / 60.7 / 95.7.

Battery at seed 17, clips with any words (of 200 per model: 60 edges, 140 synthetic):

| Model | M01 | M02 | M04 | Total of 600 | Of which edges (of 180) |
| --- | --- | --- | --- | --- | --- |
| stock (one model) | | | | 70 of 200 | 60 of 60 |
| pretrained model alone | 30 | 36 | 32 | 98 | 95 |
| block 8 | 29 | 27 | 42 | 98 | 95 |
| block 12 | 28 | 22 | 32 | 82 | 76 |
| block 16 | 19 | 14 | 24 | 57 | 52 |
| block 24 | 38 | 31 | 64 | 133 | 113 |

Disk was 20.3 GB free before the stage and 24.8 GB after (storage snapshots in `logs/lab-1029-night-storage.jsonl`).

### Conclusions

Written 30 October 2026, after the results. Block 8 stays the adapter position for stage P's recipe and for the device model. Moving the adapter deeper costs 50 to 76 words and doesn't halve false words, so the silence gate and lab test 18's battery-trained models are still the ways to deal with false words. The rebuilt stage P (174.3) supports treating 174 to 178 as this recipe's level.

</details>

## Related

[Stage P and the adapter levers](../f2-adapter-levers/notes.md), [Lab test 33](../lab-33-pooled-acoustic-adapters/notes.md), [Lab test 13](../lab-13-silence-false-inserts/notes.md), [Adapter position seeds](../f2-adapter-position-seeds/notes.md), [Lab test 28](../lab-28-deployable-export/notes.md)
