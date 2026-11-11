# Lab test 31: an adapter on each speaker's own acoustic-only model

Proposal: F-2 (extension), following the [acoustic-only study](../f2-acoustic-only/notes.md). Prototype bar: 35%, which is 148 wrong of 425 pooled test words.

Status: written 28 October 2026. Ran 28 October 2026, 20:19 to 21:47, on the lab RTX 4000 Ada (queue `1029-acoustic2`, stages 31, 32, 33), with no prediction written first. Read 29 October.

Earlier tries failed. At 16:04 (`--stamp 1028-acoustic`) all three stopped within seconds because queue `1028-rest` had the GPU, and at 20:07 that queue reached stage 31 itself and I stopped it at 20:13. At 20:14 queue `1029-acoustic` skipped all three for disk (12 GB free, under the 14 GB minimum).

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

The acoustic-only study's recipe trains all 24 encoder blocks for 100 epochs at 1e-5 with the prediction network and joint frozen, on one speaker's own clips: M01 72 clips (5.7 min), M02 156 (9.0 min), M04 82 (6.1 min). On 27-28 October it got 192, 191 and 192 wrong of 425 at seeds 17, 18 and 19 (mean 191.7, 45.1%), against 208, 219 and 212 (mean 213.0) for the adapter on stock. None of its nine models wrote words on any of the 140 noise and silence clips.

Here I put the block-8 adapter on top, trained on the same clips, to see if a second small stage adds anything.

## Setup

The base is the study's recipe on `export/<speaker>`, seed 17 only, 100 epochs (`--train-parts all --train-blocks 24 --freeze-transducer-head --lr 1e-5 --target-form stock`, last four epochs averaged). The adapter is stage P's recipe (after block 8, 15 epochs, 6e-4, scoring-form targets) at seeds 17, 18 and 19 on the same speaker's clips.

The base alone is decoded too, repeating the study's seed 17 (84, 36 and 72 wrong) with current code to show how exactly reruns repeat on this GPU. The silence battery (lab test 13's 200 clips) is decoded for the base and the seed-17 adapter.

For comparison, the study got 101.0 of 303 (33.3%) on sentences only, and stage P (severe pre-training, then the same adapter) got 178.0 (183, 176, 175). No earlier run puts an adapter on a whole-block model of the same speaker. The closest is stage P, where more adapter epochs changed the result by 7 words at most.

## Plan

1. Per speaker, train the base (seed 17), decode it on test and dev, and decode the battery.
2. Train the three adapters on that base and decode each on test and dev. Decode the battery for seed 17.
3. Release the base once every result that needs it is saved.
4. Build the pooled table (`table.json`), three-seed means against the acoustic-only study (`seeds.json`), and the battery table (`battery.json`).

Rule set before the run: the usual one on the three-seed mean against 191.7, at least 2 of 3 speakers the same way and more than about 4.5 points (19 words). So 172 or fewer is a gain and 211 or more a loss. A gain would make the adapter worth stacking on the device model. Within the rule means the base already took what the clips can give. A loss means the adapter overfits. More battery inserts than the base would mean it undoes the base's silence handling.

## Why

The acoustic-only model is the most stable recipe so far (seed range 1 word) and the only one with no false inserts, so an accuracy gain on top would give both. It's also the quickest of the three acoustic tests (31, 32, 33).

## Limits

One base seed per speaker, so no base-to-base variation. The adapter recipe is stage P's, untuned for this base. Nothing on int8, the Pi or the intended user. The test clips have been scored in many earlier runs, so treat this as exploratory.

## Results

Adding the adapter didn't help. Across three seeds the stacked model got 196.3 of 425 words wrong (46.2%). The acoustic-only models got 191.7 (45.1%), and the base I rebuilt here got 192. That's 4.6 words worse, well inside the 19-word noise margin, so it counts as no change. The full-encoder model seems to have already learned what a few minutes of each speaker's audio can teach it.

| Seed | Base alone | Base + adapter | Change |
| --- | --- | --- | --- |
| 17 | 192 | 189 | −3 |
| 18 | (same base) | 194 | +2 |
| 19 | (same base) | 206 | +14 |
| Mean | 192.0 | 196.3 | +4.3 |

Wrong words of 425 pooled reference words, lower is better. Stock is 323, the bar 148, stage P 178.0.

The rebuilt base matched the study's seed 17 to the word on every speaker. The adapter shifted errors around: sentence words got worse (101.0 → 112.0 of 303, 33.3% → 37.0%) and single words a bit better (90.7 → 84.3 of 122). The base is under the 35% sentence line (106) and the stacked model over it at every seed. Its seed range is 17 words, against 1 for the study.

It also brought back some breath inserts. The bases wrote nothing on the synthetic clips, but the seed-17 adapters wrote words on 4, 8 and 1 of the 20 breath clips (M01, M02, M04). On the 60 quiet edges of real recordings the bases already wrote words on 60, 59 and 56, and with the adapter 60, 60 and 31.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

None was written before the run, and none added after. The comparison numbers and decision rule above were written on 28 October, before any result.

### Commands and Setup

From the lab computer's `app/`:

```bash
python lab.py run 31                          # this test alone
python lab.py queue acoustic --detach         # labs 31, 32 and 33 in order
```

Estimated cost, from the acoustic-only study's times on the RTX 4000 Ada (18 to 25 min per 100-epoch base): three bases, nine adapters and their decodes, about 2 hours.

Disk: one full-encoder training at a time, about 12.5 GB at peak (four checkpoints plus the final .nemo, by train.py's estimate). Checkpoints are deleted as soon as the averaged .nemo is saved and reads back intact. A base is only released after every result that needs it is saved. Before each base the stage checks for 14.5 GB free and runs `lab_space.py clean` if short. If it's still short it stops with a disk message that the queue retries on.

Records written during the run:

1. `reports/lab-31-adapter-on-acoustic-only/runs.json` has each attempt's code hashes and settings, plus each run's status, minutes, wrong words, free disk afterwards and the base checkpoint it used.
2. `logs/lab-<stamp>-storage.jsonl` and the stage log record disk, temp folder, RAM, folder sizes and the biggest model folders at the start and end of the stage, after each base, and after any failure.
3. `finished.json` is only written when nothing failed. Rerunning the stage resumes it: finished decodes are reused and the tables rebuilt (older tables get moved aside).

### Full Results

Sources: `logs/lab-1029-acoustic2-31.txt` (the stage log) and `logs/pool_table-20261029-024742-050080.log`. The table JSON is `reports/lab-31-adapter-on-acoustic-only/table.json` on the lab computer. Wrong words per speaker (M01 of 169, M02 of 95, M04 of 161), intervals resample clips with 2000 draws.

| System | M01 | M02 | M04 | Pooled | WER | 95% CI | Sentences (of 303) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| stock | 131 | 62 | 130 | 323 | 76.0% | 66.7 to 86.3 | 188 |
| adapter on stock, seeds 17/18/19 | 87, 95, 86 | 42, 41, 43 | 79, 83, 83 | 208, 219, 212 | 50.1% mean | | 112, 120, 121 |
| stage P, seeds 17/18/19 | 74, 77, 72 | 37, 37, 37 | 72, 62, 66 | 183, 176, 175 | 41.9% mean | | 87, 84, 86 |
| acoustic-only study, seeds 17/18/19 | 84, 86, 84 | 36, 34, 36 | 72, 71, 72 | 192, 191, 192 | 45.1% mean | | 102, 100, 101 |
| base rebuilt here, seed 17 | 84 | 36 | 72 | 192 | 45.2% | 38.2 to 53.1 | 102 |
| base + adapter, seed 17 | 88 | 38 | 63 | 189 | 44.5% | 37.6 to 52.3 | 109 |
| base + adapter, seed 18 | 90 | 39 | 65 | 194 | 45.6% | 38.5 to 53.6 | 110 |
| base + adapter, seed 19 | 85 | 43 | 78 | 206 | 48.5% | 41.4 to 56.5 | 117 |

Three-seed means against the study's 191.7: adapter on stock +21.3 (3 of 3 speakers agree), stage P −13.7 (2 of 3), base + adapter +4.6 (2 of 3). By speaker, the adapter helped M04 at two seeds (63 and 65 against 72) and hurt M01 at every seed (85 to 90 against 84). Empty answers were 0 of 166 for the base and 0, 1 and 1 with the adapter.

Battery (lab test 13's 200 clips: 140 synthetic, 60 quiet edges of real recordings), clips that got any words, seed 17:

| System | Breath | Edge | Other synthetic (120) | Total |
| --- | --- | --- | --- | --- |
| stock | 1/20 | 60/60 | 9/120 | 70/200 |
| base M01 / M02 / M04 | 0, 0, 0 | 60, 59, 56 | 0, 0, 0 | 60, 59, 56 |
| base + adapter M01 / M02 / M04 | 4, 8, 1 | 60, 60, 31 | 0, 0, 0 | 64, 68, 32 |

Times: the bases took 17, 25 and 19 minutes (100 epochs, 500 updates for M01) and each adapter about 1.5 minutes. The stage took 1 h 28 min, under the 2-hour estimate.

### Conclusions

*Added 29 October 2026, from the results above.* A second training stage on clips the full-encoder model already fits doesn't help. The pooled count stays put, sentences get worse, and results vary more between seeds. The acoustic-only model alone is the better of the two, and it keeps its clean silence behaviour. Stage P is still the best single system (178.0).

</details>

## Related

[Acoustic-only study](../f2-acoustic-only/notes.md), [Stage P and the adapter levers](../f2-adapter-levers/notes.md), [Lab test 32](../lab-32-acoustic-severe-pretraining/notes.md), [Lab test 33](../lab-33-pooled-acoustic-adapters/notes.md)
