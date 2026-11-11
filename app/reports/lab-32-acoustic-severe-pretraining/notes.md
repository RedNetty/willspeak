# Lab test 32: the acoustic recipe as severe-speaker pretraining, then the adapter

Proposal: F-2 (extension), combining stage P and the [acoustic-only study](../f2-acoustic-only/notes.md). Prototype bar is 35%, which is 148 wrong of 425 pooled test words.

Status: written 28 October 2026. Ran 28 October 2026, 21:47 to 23:17, on the lab RTX 4000 Ada (queue `1029-acoustic2`, after lab test 31) at the default 30 pretraining epochs, no prediction written. Read 29 October. Tries at 16:04 and 20:14 didn't start (GPU busy, then disk under the 14 GB minimum; see [lab test 31](../lab-31-adapter-on-acoustic-only/notes.md)). No failures in the 21:47 run.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

The best system so far is stage P: pretrain on the other severe speakers (stage C's phase 1, F-1's recipe, every parameter trainable at 2e-5, dev-selected), then the block-8 adapter at 6e-4 on the target. It got 183, 176 and 175 wrong of 425 (mean 178.0, 41.9%). Lab test 26 tried four other pretrainings and none beat it.

The acoustic-only recipe (all 24 encoder blocks, prediction network and joint frozen, 1e-5, a long fixed schedule) is my most stable recipe and the only one with no false inserts on the noise clips, but it's only been trained on one speaker's own few minutes. Here it becomes the pretraining on the other severe speakers, with stage P's adapter unchanged.

## Setup

Pretraining uses `export/loso-severe-<speaker>`, the other three severe speakers (for M01: F01, M02 and M04, 604 clips, 39.4 min). TORGO's prompt split is the same for every speaker, so the target's test prompts are never in it. Seed 17, one model per target.

Default is 30 epochs (`python lab.py set acoustic_severe_epochs N`), about 1,140 updates at about 38 an epoch, a bit more than the acoustic-only M02 model's roughly 1,000. The study's 100 epochs would be about 3,800 updates and triple the time. 30 is my pick, not a measured optimum.

The adapter is stage P's recipe exactly: after block 8, 15 epochs, 6e-4, seeds 17, 18 and 19.

The pretrained model alone is decoded on the target too, as a reference for a model that's never heard the speaker, next to F-1 alone (every other dysarthric speaker, 248 wrong in lab test 18's table). The silence battery is decoded for the pretrained model and the seed-17 adapter.

Numbers from earlier runs, for comparison: stage P (the control) 178.0 (183, 176, 175), sentences only 85.7 of 303 (28.3%). Lab test 26's other pretraining recipes with the same adapter: 180.3 to 200.3. Acoustic-only study on one speaker's own clips: 191.7.

## Plan

1. Build the leave-one-out severe exports if missing.
2. Per speaker, train the pretrained model (seed 17) and decode it on the target's test and dev clips and on the battery.
3. Train the three adapters on it and decode each, plus the battery for seed 17.
4. Free the pretrained model once everything that needs it is saved.
5. Write the pooled table (`table.json`), three-seed means against stage P (`seeds.json`) and the battery (`battery.json`).

Decision rule, set before the run: the project's rule on the three-seed mean against stage P's 178.0 (at least 2 of 3 speakers the same way, more than about 4.5 points, which is 19 words). A mean of 159 or fewer is a gain and 197 or more is a loss.

Outcomes, written before the run. A gain means how pretraining is done matters after all and the acoustic recipe is better. Within the rule means pretraining is saturated, as lab test 26 hinted, and what's in the set matters more than how it's trained. A loss means freezing the text head, or the shorter low-rate schedule, gives the adapter a worse start than training every parameter. Fewer battery inserts than stage P would mean the recipe keeps its quiet on silence in the stacked system.

## Why

Stage P is the best result and the acoustic recipe is the most stable, so this is the direct way to see if they combine. It costs about the same as lab test 26.

## Limits

It shows whether this pretraining moves the stacked system's pooled count at three adapter seeds, how the pretrained model does on an unheard speaker, and false inserts. It can't show variation between pretrained models (one seed each), the best length (one length unless changed), int8 or Pi behaviour, or anything about the intended user. The test clips have been scored many times before, so this is exploratory.

## Results

It's a loss, and a big one. With the same adapter on top, the three-seed mean is 228.3 wrong of 425 (53.7%) against stage P's 178.0 (41.9%). That's 50.3 words worse, all three speakers worse, well past the loss line of 197. By the outcomes set beforehand, freezing the text head with the shorter low-rate schedule gives the adapter a worse start than training every parameter.

| | Pretrained model alone (seed 17) | + adapter, seeds 17 / 18 / 19 | Mean |
| --- | --- | --- | --- |
| This test (acoustic recipe on the other severe speakers) | 234 | 231, 220, 234 | 228.3 |
| Stage P (F-1's recipe, every parameter trainable, on the same speakers) | not decoded alone | 183, 176, 175 | 178.0 |

Wrong words of 425, lower is better. Stock 323, bar 148.

The adapter barely helps on this base: 234 alone, 228.3 with it, 5.7 words. On stage P's base it's worth much more, but stage P's pretrained model was never decoded alone, so the exact size isn't recorded.

The pretrained model alone is the best "never heard this speaker" model so far. It got 234 (55.1%), against F-1 alone at 248 (seven other dysarthric speakers, F-1's recipe) and lab test 11's whole-block pretraining on the same severe speakers at 275. M02 gains little from it (56 of 95, stock 62). Both the data and the recipe differ from F-1, so this doesn't isolate the recipe.

On sentences it got 136.3 of 303 (45.0%) with the adapter, against stage P's 85.7 (28.3%). Single words were 92.0 of 122, level with stage P's 92.3.

None of these models wrote words on most of the synthetic noise. With the adapter, M02 wrote words on 7 of 20 tap clips and 4 of 20 pink-noise clips (details below). Stage P's models have no battery decode to compare with.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

No prediction was written before the run, and I'm not adding one after. The reference points and decision rule above were written on 28 October, before any result.

### Commands and Setup

From the lab computer's `app/`:

```bash
python lab.py run 32                                        # 30 epochs
python lab.py --set acoustic_severe_epochs=100 run 32       # the study's length instead
```

Cost: three pretraining runs and nine adapters. Pretraining time was estimated from stage C's phase 1 on the same data (about 45 seconds an epoch, and all 24 blocks is slower per step): 25 to 45 minutes each at 30 epochs, about 2.5 hours in all.

Disk and records as in [lab test 31](../lab-31-adapter-on-acoustic-only/notes.md#commands-and-setup): one full-encoder training at a time (about 12.5 GB peak), checkpoints deleted once the .nemo is saved, `runs.json` here, storage snapshots in `logs/lab-<stamp>-storage.jsonl`, `finished.json` only if nothing failed.

### Full Results

Sources: `logs/lab-1029-acoustic2-32.txt` and `logs/pool_table-20261029-041750-903978.log`, with the table JSON in `reports/lab-32-acoustic-severe-pretraining/table.json` on the lab computer. Wrong words per speaker (M01 of 169, M02 of 95, M04 of 161). Intervals resample clips, 2000 draws.

| System | M01 | M02 | M04 | Pooled | WER | 95% CI | Sentences (of 303) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| stock | 131 | 62 | 130 | 323 | 76.0% | 66.7 to 86.3 | 188 |
| stage P, seeds 17/18/19 | 74, 77, 72 | 37, 37, 37 | 72, 62, 66 | 183, 176, 175 | 41.9% mean | | 87, 84, 86 |
| acoustic-only study, seeds 17/18/19 | 84, 86, 84 | 36, 34, 36 | 72, 71, 72 | 192, 191, 192 | 45.1% mean | | 102, 100, 101 |
| severe acoustic model alone, seed 17 | 86 | 56 | 92 | 234 | 55.1% | 47.5 to 63.4 | 130 |
| + adapter, seed 17 | 94 | 45 | 92 | 231 | 54.4% | 47.5 to 62.1 | 137 |
| + adapter, seed 18 | 86 | 49 | 85 | 220 | 51.8% | 45.1 to 59.1 | 129 |
| + adapter, seed 19 | 88 | 51 | 95 | 234 | 55.1% | 48.6 to 62.4 | 143 |

Three-seed mean against stage P: +50.3 words, 3 of 3 speakers worse. Empty answers were 1 of 166 alone, and 1, 2 and 3 with the adapter. Every row beats stock with the interval clear of zero.

Pretraining sets (`export/loso-severe-<speaker>`): M01 604 clips (39.4 min), M02 592 (37.6 min), M04 589 (34.2 min). 30 epochs, 1,140 updates for M01, 21, 21 and 21 minutes each. Adapters took about 1.5 (M01) to 3 minutes (M02). The stage took 1 h 30 min, under the 2.5-hour estimate.

Battery (lab test 13's 200 clips), clips that got any words, seed 17:

| System | Breath | Edge (60) | Hum | Pink | Taps | White | Total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| stock | 1 | 60 | 1 | 2 | 1 | 1 | 70/200 (also room 2, silence 2) |
| pretrained M01 / M02 / M04 | 1, 3, 3 | 46, 54, 46 | 0 | 0 | 0 | 0 | 47, 57, 49 |
| + adapter M01 | 5 | 41 | 0 | 0 | 1 | 0 | 47 |
| + adapter M02 | 3 | 33 | 2 | 4 | 7 | 1 | 50 |
| + adapter M04 | 4 | 9 | 0 | 0 | 1 | 1 | 15 |

No model wrote words on the room-noise or digital-silence clips.

### Conclusions

Interpretation added 29 October 2026. How the pretrained model is trained does matter, but every change so far has been a loss: lab test 26's four variants got 180.3 to 200.3 and this one 228.3. Training every parameter on the other severe speakers (stage P) is still the best start for the adapter, and the acoustic recipe (text head frozen, 1e-5, 30 epochs) is much worse. The acoustic model is good on its own for a speaker it's never heard (234), but it leaves the adapter little to add. I only tried one pretraining length (30 epochs, one seed), so a longer schedule could change the base, but it's unlikely to close a 50-word gap.

</details>

## Related

[Stage P and the adapter levers](../f2-adapter-levers/notes.md), [Lab test 26](../lab-26-pretraining-recipe/notes.md), [Acoustic-only study](../f2-acoustic-only/notes.md), [Lab test 31](../lab-31-adapter-on-acoustic-only/notes.md), [Lab test 33](../lab-33-pooled-acoustic-adapters/notes.md)
