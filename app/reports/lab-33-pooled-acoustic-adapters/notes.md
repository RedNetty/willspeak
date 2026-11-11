# Lab test 33: one acoustic model on every dysarthric speaker, then per-speaker adapters at five positions

Proposal: F-2 (extension), following the [acoustic-only study](../f2-acoustic-only/notes.md) and the [adapter position sweep](../f2-adapter-position-seeds/notes.md). Prototype bar: 35%, or 148 wrong of 425 pooled test words.

Status: my idea, written 28 October. Ran 28 October 23:18 to 29 October 02:06 on the lab RTX 4000 Ada (queue `1029-acoustic2`, after lab tests 31 and 32), with no prediction written first. Read 29 October. Attempts at 16:04 and 20:14 didn't start (GPU busy, then disk under the 14 GB minimum, see [lab test 31](../lab-31-adapter-on-acoustic-only/notes.md)). All 45 adapters and every decode finished.

See also the [Report Context](../Report%20Context.md) and the [Measurement guide](../../../docs/measurements.md).

## Idea

Train the acoustic-only recipe once on every dysarthric TORGO speaker together, then give each target their own adapter at several encoder positions. It differs from lab test 32 in two ways. One base serves all three targets, like a single device model before personalisation, and it has also heard the targets' own training sessions and the moderate and mild speakers.

## Setup

The training set is `export/pooled-M01-M02-M04`, built by `torgo_manifest.py --pooled`. It has every dysarthric speaker's clips whose prompts are in TORGO's training split (F01, F03, F04, M01, M02, M03, M04, M05), and only the training sessions of M01, M02 and M04. Dev is the targets' own dev clips, and there's no test split since each target is scored on `export/<speaker>/test.jsonl`. The script refuses to write the export if any training or dev clip shares a prompt with a test clip, and logs the composition, sizes and that check in `pooled.json`. The F-1 export for M01 already has 1,849 clips (104.8 min), so I expected about 1,900 clips and 110 minutes.

The base is the acoustic recipe at seed 17, reused from the study without its eight-recipe search. It trains 30 epochs by default (`python lab.py set acoustic_pooled_epochs N`), about 3,600 updates at about 120 an epoch. That's my pick, unmeasured, and 100 epochs would take roughly three times as long. Adapters go after blocks 4, 8, 12, 16 and 24 (`python lab.py set pooled_positions "..."`) for 15 epochs at 6e-4 with seeds 17, 18 and 19, so 45 runs. Every position runs at seed 17 first, so a stopped run still leaves one full curve. I also decode the base alone on each target, and run the silence battery on the base and every seed-17 adapter.

From earlier runs, F-1 alone (targets never heard) got 248. The adapter on stock at 3e-4 had three-seed means of 254.0, 215.0, 206.0, 251.3, 263.3, 254.3 and 293.7 after blocks 4, 6, 8, 10, 12, 16 and 24. Block 8 won there, but on the stock model, so the best spot could move on a retrained encoder.

## Plan

1. Build the pooled export if it's missing and log its composition.
2. Train the base, then decode it on each target's test and dev clips and on the battery.
3. Train and decode the 45 adapters seed by seed.
4. Release the base once everything that needs it is saved, then write `table.json`, `seeds.json` (three-seed means against stage P) and `battery.json`.

The decision rule is the project's usual one: three-seed means, at least 2 of 3 speakers the same way, more than about 4.5 points (19 words). Against stage P's 178.0, a mean of 159 or fewer is a gain and 197 or more is a loss. One position only beats another by 19 or more words with 2 of 3 speakers agreeing, never from one seed.

Beating stage P would make shared model plus adapter the better design, and cheaper for a new person. Within the rule, I'd choose on cost and robustness. A best position away from block 8 would mean the sweep's answer was specific to the stock model. The base alone near stage P would mean most of the gain comes from hearing the speaker's own sessions inside a big pool.

## Why

It's a different shape of system: train once on everyone, then personalise cheaply. If it works, a new speaker needs a few minutes of adapter training instead of a new pretraining run.

## Limits

A gain can't be pinned on the targets' own sessions or the extra speakers, since both change at once (lab test 32, on the other severe speakers only, helps there). There's one base seed, and every position uses block 8's 6e-4 with no per-position tuning. Nothing here covers int8, the Pi or the intended user. The test clips have been scored many times before, so this is exploratory.

## Results

It lost at every position, by 32 to 45 words. The best, after block 12, averaged 210.0 wrong of 425 (49.4%) against stage P's 178.0 (41.9%), with all three speakers worse. Every mean is past the loss line of 197.

| Adapter position | Seeds 17 / 18 / 19 | Mean | vs stage P | Sentences (of 303) | Single words (of 122) |
| --- | --- | --- | --- | --- | --- |
| none (shared base alone, seed 17) | 222 | 222.0 | +44.0 | 120 | 102 |
| after block 4 | 218, 213, 225 | 218.7 | +40.7 | 121.7 | 97.0 |
| after block 8 | 223, 213, 212 | 216.0 | +38.0 | 125.3 | 90.7 |
| after block 12 | 214, 201, 215 | **210.0** | +32.0 | 123.3 | 86.7 |
| after block 16 | 212, 217, 217 | 215.3 | +37.3 | 120.3 | 95.0 |
| after block 24 | 231, 225, 214 | 223.3 | +45.3 | 116.7 | 106.7 |
| stage P (control) | 183, 176, 175 | 178.0 | | 85.7 | 92.3 |
| acoustic-only study (own clips only) | 192, 191, 192 | 191.7 | +13.7 | 101.0 | 90.7 |

Wrong words, lower is better. Stock is 323 and the bar is 148. "vs stage P" is the difference in three-seed means, and at every position all 3 of 3 speakers are worse.

No position beats another by the rule. Block 12 (lowest) and block 24 (highest) are only 13.3 words apart, so the sweep's block-8 answer is neither confirmed nor overturned here. The adapter adds at most 12 words over the shared base alone.

Pooling everyone diluted the targets. The shared base had heard each target's own sessions and still got 222, against 191.7 for the same recipe on that speaker alone. The targets' clips were only 20.8 of the 92.8 training minutes. So the base alone was 44 words behind stage P, nowhere near it.

Adapters after block 12 or later nearly stopped false words. On the battery the shared base wrote words on 56 of 200 clips (45 of the 60 quiet edges). After block 12, 16 or 24 each speaker's model did it on 0 to 8 clips, and after block 4 or 8 on 22 to 52. Some quiet edges may hold faint real speech, so this isn't purely a gain.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

No prediction was written before the run, and I haven't added one after the fact. The reference points and decision rules above were written on 28 October, before any result.

### Commands and Setup

From the lab computer's `app/`:

```bash
python lab.py run 33
python lab.py --set pooled_positions="6 8 10" --set acoustic_pooled_epochs=60 run 33   # other positions or length
```

Cost is one base and 45 adapters. The base time is an estimate. It has about three times the data of lab test 32's pretraining, so roughly 1.5 to 2.5 hours at 30 epochs. The adapters and their decodes take about 2 hours, for about 4.5 hours in all. The export adds about 250 MB of audio under `data/public/torgo/export/`.

Disk and records work as in [lab test 31](../lab-31-adapter-on-acoustic-only/notes.md#commands-and-setup). There's one full-encoder training (about 12.5 GB peak), and checkpoints are deleted once the .nemo is saved. The base is kept until all 48 results that need it are saved. `runs.json` goes in this folder, storage snapshots go in `logs/lab-<stamp>-storage.jsonl`, and `finished.json` is only written when nothing failed.

### Full Results

Sources: `logs/lab-1029-acoustic2-33.txt`, `logs/lab-1029-acoustic2-summaries.txt`, `logs/pool_table-20261029-070557-865071.log` and `logs/battery-20261029-070559-509895.log`, plus JSON in `reports/lab-33-pooled-acoustic-adapters/` (`table.json`, `battery.json`) on the lab computer.

Pooled export (`export/pooled-M01-M02-M04/pooled.json`): 1,641 training clips, 92.78 min, 175 MB, and 0 clips share a prompt with a test clip. By speaker that's F01 85 clips (3.45 min), F03 363 (18.02), F04 275 (13.10), M01 72 (5.72), M02 156 (8.96), M03 270 (12.97), M04 82 (6.07) and M05 338 (24.48). Dev is the targets' own 37 clips (M01 4, M02 24, M04 9). My estimate in Setup (about 1,900 clips, 110 min) was too high. NeMo loaded 1,640 of the clips, with one dropped, probably for being under its 0.1 s minimum.

Base: 30 epochs at about 103 updates an epoch, 3,090 in all, taking 48.5 minutes, with a released dev WER of 45.54%. The whole stage took 2 h 48 min against the 4.5-hour estimate.

Wrong words per speaker (M01 of 169, M02 of 95, M04 of 161). The intervals resample clips with 2000 draws.

| System | M01 | M02 | M04 | Pooled | WER | 95% CI | Empty of 166 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| stock | 131 | 62 | 130 | 323 | 76.0% | 66.7 to 86.3 | 34 |
| shared base alone, s17 | 87 | 44 | 91 | 222 | 52.2% | 44.4 to 60.8 | 0 |
| block 4, s17 | 86 | 53 | 79 | 218 | 51.3% | 43.9 to 59.8 | 3 |
| block 4, s18 | 86 | 45 | 82 | 213 | 50.1% | 42.3 to 58.9 | 5 |
| block 4, s19 | 91 | 56 | 78 | 225 | 52.9% | 44.9 to 62.0 | 2 |
| block 8, s17 | 80 | 48 | 95 | 223 | 52.5% | 45.9 to 59.4 | 3 |
| block 8, s18 | 83 | 42 | 88 | 213 | 50.1% | 43.8 to 57.2 | 4 |
| block 8, s19 | 85 | 47 | 80 | 212 | 49.9% | 43.2 to 57.2 | 3 |
| block 12, s17 | 87 | 46 | 81 | 214 | 50.4% | 43.6 to 57.9 | 4 |
| block 12, s18 | 83 | 45 | 73 | 201 | 47.3% | 40.0 to 55.6 | 4 |
| block 12, s19 | 83 | 43 | 89 | 215 | 50.6% | 42.9 to 59.1 | 2 |
| block 16, s17 | 84 | 49 | 79 | 212 | 49.9% | 43.3 to 57.1 | 1 |
| block 16, s18 | 91 | 43 | 83 | 217 | 51.1% | 44.3 to 59.1 | 1 |
| block 16, s19 | 87 | 46 | 84 | 217 | 51.1% | 43.9 to 59.3 | 1 |
| block 24, s17 | 91 | 47 | 93 | 231 | 54.4% | 47.1 to 62.9 | 0 |
| block 24, s18 | 93 | 45 | 87 | 225 | 52.9% | 45.5 to 61.5 | 0 |
| block 24, s19 | 85 | 43 | 86 | 214 | 50.4% | 43.0 to 58.9 | 0 |

Every row beats stock with the interval clear of zero, and none meets the bar. On sentences only (of 303, where 35% is 106) no row gets under the line. The lowest is block 24 at seed 19, with 112 (37.0%).

Battery (lab test 13's 200 clips): clips that got any words at seed 17, listed as M01, M02, M04.

| System | Breath (20) | Edge (60) | Other synthetic (120) | Total (200) |
| --- | --- | --- | --- | --- |
| stock | 1 | 60 | 9 | 70 |
| shared base alone | 2 | 45 | 9 | 56 |
| block 4 | 14, 0, 5 | 23, 52, 35 | 0, 0, 0 | 37, 52, 40 |
| block 8 | 3, 0, 0 | 24, 33, 15 | 0, 3, 7 | 27, 36, 22 |
| block 12 | 0, 0, 0 | 8, 0, 2 | 0, 0, 1 | 8, 0, 3 |
| block 16 | 0, 0, 0 | 2, 7, 0 | 0, 0, 0 | 2, 7, 0 |
| block 24 | 0, 0, 0 | 4, 6, 8 | 0, 0, 0 | 4, 6, 8 |

### Conclusions

Interpretation added 29 October 2026 from the results above. A shared acoustic model personalised with a small adapter is clearly worse than stage P's per-speaker pretraining, and also worse than the acoustic recipe on the speaker's own clips. Mixing the target's few minutes into 93 minutes of other speakers, with the text head frozen, seems to wash out what the per-speaker model learns, and the adapter can't get it back. The design is cheaper to personalise, but at this cost in accuracy it isn't worth adopting. The one useful side result is that adapters after block 12 or later almost stopped false words on noise and quiet edges. Whether that holds on stage P's model is untested.

</details>

## Related

- [Acoustic-only study](../f2-acoustic-only/notes.md)
- [Adapter positions, three seeds](../f2-adapter-position-seeds/notes.md)
- [Lab test 31](../lab-31-adapter-on-acoustic-only/notes.md)
- [Lab test 32](../lab-32-acoustic-severe-pretraining/notes.md)
