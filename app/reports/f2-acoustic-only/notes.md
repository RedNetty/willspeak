# F-2: longer acoustic adaptation with the text head protected

Status: designed 9 October 2026. First attempt 12-13 October 2026 on the lab computer (stamp `1012-pending`, now in `1012-pending-previous/`) hit the free-disk guard after the dev search and one confirmation seed, so no test result. Second attempt, same stamp, evening of 13 October to 14:12 on 14 October: dev search done, seed-17 test for the control (all three speakers) and the chosen recipe (M01 and M02 only), then it stopped because `long_runs.py` changed mid-run. Seeds 18 and 19 never ran. Read 21 October. Third attempt (stamp `1027-1530`, 27 to 28 October) finished all speakers and seeds. Script: `scripts/asr/acoustic_only.sh`, shared coordinator `long_runs.py`.

## Idea

In the finished lab comparison the adapter on stock got 48.9% and whole-block training 68.5%. In Lab 9, opening 16 blocks instead of eight only went from 68.5% to 67.1%. So more trainable capacity hasn't been shown to help. But the whole-block recipes also train the prediction network and joint, and a few minutes of his speech might teach that head the training vocabulary instead of teaching the encoder his voice.

That's a guess, not a diagnosis. It comes from [Shor et al., Interspeech 2019](https://www.isca-archive.org/interspeech_2019/shor19_interspeech.pdf), whose RNN-T experiments favoured adapting the lower encoder, with or without the joint. Their ALS data and architecture differ from TORGO/Parakeet, so their WER figures don't predict ours. Freezing both prediction network and joint is a stricter, separate test.

## Setup

The new `train.py --freeze-transducer-head` freezes the prediction network and joint while gradients still pass through them to the encoder. Without it training is unchanged. Batch-norm statistics stay frozen.

Eight 100-epoch recipes cover every combination:

| Factor | Settings |
|---|---|
| Encoder blocks opened | bottom 8; all 24 conformer blocks |
| Prediction network and joint | trainable; frozen |
| Peak learning rate | 3e-6; 1e-5 |

The subsampling front end stays frozen. All eight use stock-form targets, the same augmentation, effective batch 16, and average the last four epochs. The control is a fresh 15-epoch scoring-form adapter. Head comparisons have matched budgets; the adapter keeps its usual shorter one.

The seed-17 search uses pooled dev errors over M01/M02/M04 to lock one global recipe, which is then repeated with the control at seeds 18/19 before test decoding. That's at most 39 full runs plus nine two-step smoke runs. No single speaker or test result picks settings. Learning curves show whether longer runs lower training loss while dev gets worse.

Going in, I thought freezing the head was more likely to help than just opening blocks, that the all-24-block models might still overfit, and that the adapter might stay best. No prediction recorded.

```bash
bash scripts/asr/acoustic_only.sh 1009-acoustic --dry-run
bash scripts/asr/acoustic_only.sh 1009-acoustic --offload-optimizer
```

Run in the lab checkout's `app/`. `--offload-optimizer` puts non-adapter AdamW state in system RAM (skip it if training fits in VRAM; it doesn't cut activation memory). A two-step smoke run tests every capacity/head/rate combination first, and `--phase smoke` stops there. Don't run it next to another training job.

Results: `reports/f2-acoustic-only/<stamp>/`. Models: `models/parakeet-ft-long-acoustic-<stamp>/`. Live log: `logs/lab-long-acoustic-<stamp>.txt`. Protocol, cleanup, resume, paired intervals and the synthetic non-speech check are in the [long-adapter note](../f2-long-adapter/notes.md). The selected recipe and control keep all three seeds' models for export.

The practical criterion is a gain above 4.5 points, repeated over seeds and helping at least two speakers. The full dev matrix gets reported even if the adapter wins, which would rule out these recipes but not every kind of acoustic adaptation. Past test inspection, small dev sets and untested Pi/int8 parity limit what this shows.

## Results

### First attempt (12-13 October 2026, `1012-pending`, lab RTX 4000 Ada; dev only, read 14 October)

It stopped at 08:43 on 13 Oct, before control seed 19, when free disk fell under the 14 GiB guard (13 GB free of 150 GB; queue output at `9c2c21d:app/logs/lab-queue-1012-pending.out`). No test result.

Seed-17 search, wrong of 112 dev words (M01 18, M02 69, M04 25). "open"/"frozen" is the head trainable or frozen, "b8"/"b24" the blocks opened, low/high 3e-6 and 1e-5.

| Recipe | Dev wrong | Recipe | Dev wrong |
|---|---:|---|---:|
| b24-open-high-lr | **41** | b24-open-low-lr | 53 |
| b24-frozen-high-lr | 42 | b24-frozen-low-lr | 55 |
| b8-open-high-lr | 45 | b8-open-low-lr | 62 |
| control (15-epoch adapter) | 46 | b8-frozen-high-lr | 51 |
| | | b8-frozen-low-lr | 63 |

The rule picked b24-open-high-lr (all 24 blocks, head trainable, 1e-5), 5 words ahead of the control. Seed-18 confirmation dev was 39 vs 43 for the control. At the high rate every arm is within 5 words of the control, which a 112-word dev set can't separate. At the low rate all are clearly worse (53 to 63), so 3e-6 underfits; I didn't try higher. Freezing the head barely mattered at the high rate (b24 42 vs 41, b8 51 vs 45).

About 12 hours of GPU and no tested number. The bar is 148 wrong or fewer on test. Dev files are in `1012-pending-previous/`.

### Second attempt (13-14 October 2026, same stamp, read 21 October from `reports/f2-acoustic-only/1012-pending/`)

It restarted from scratch because the saved fingerprint no longer matched the code, redid the seed-17 search, retrained the finalists, and decoded test and the 140-clip synthetic battery for what finished. At 14:13 on 14 October the next run refused to start because `scripts/asr/long_runs.py` had changed during the study (`failures.jsonl`: 13 final runs, all "Code changed during the experiment"). So M04 for the chosen recipe and all seed-18 and seed-19 runs are missing, and `summary.json` and `battery.json` were never written.

Seed-17 dev, wrong of 112. Rerunning moved recipes by 1 to 2 words, enough to change the pick:

| Recipe | First attempt | Second attempt |
|---|---:|---:|
| b24-frozen-high-lr | 42 | **40** |
| b24-open-high-lr | **41** | 42 |
| b8-open-high-lr | 45 | 45 |
| control (15-epoch adapter) | 46 | 46 |
| b8-frozen-high-lr | 51 | 50 |
| b24-open-low-lr | 53 | 54 |
| b24-frozen-low-lr | 55 | 53 |
| b8-open-low-lr | 62 | 62 |
| b8-frozen-low-lr | 63 | 63 |

This time it picked b24-frozen-high-lr: all 24 encoder blocks at 1e-5 for 100 epochs, head frozen, stock-form targets. Confirmation dev matched the search exactly (control 4 + 26 + 16, b24-frozen 8 + 18 on M01 and M02), so one seed on the same code repeats itself.

Test at seed 17, wrong words (lower is better):

| Speaker (reference words) | Stock | Control adapter | b24-frozen-high-lr |
|---|---:|---:|---:|
| M01 (169) | 131 | 87 (51.5%) | 84 (49.7%) |
| M02 (95) | 62 | 42 (44.2%) | 36 (37.9%) |
| M04 (161) | 130 | 79 (49.1%) | not reached |
| M01 + M02 (264) | 193 | 129 (48.9%) | 120 (45.5%) |

The control's 208 pooled (87 + 42 + 79) equals the 7 October adapter rebuild and the long-adapter control at seed 17, so it reproduces across three studies.

On the two finished speakers the frozen-head model has 9 fewer wrong (3 on M01, 6 on M02). M02 alone gains 6.3 points, but resampling its 37 test clips (2,000 draws, seed 0) gives a 95% interval of -2.9 to +16.8, so it's not established. With one seed, two speakers and no M04 the project's rule can't be applied.

M02's 36 is among the lowest M02 counts on these clips (stage P pretrained adapter at 6e-4: 37 at every seed; its 30-epoch version 33 to 35; whole-block from pretrained 34 to 39). M02's gain is mostly sentences (23 to 18 of 68 sentence words) with single words flat (19 to 18 of 27). M01 is the reverse: single words 41 to 37 of 47, sentences level at 46 vs 47 of 122.

On the synthetic non-speech battery (140 clips: silence, white and pink noise, hum, room tone, breaths, taps) the frozen-head model wrote words on 0 of 140 for M01 and for M02. The control did on 1 (M01), 20 (M02) and 0 (M04). M02's 20 matches the long-adapter control at seed 17 exactly, mostly short words like "Okay.", "Mm-hmm." and "Cool." on taps (8), breaths (5) and pink noise (4). This is a decoder test on generated audio, not a live-room false-trigger rate.

My reading then: the head-frozen full-depth fine-tune at 1e-5 looks at least as good as the adapter on two speakers and much quieter on non-speech, but it's one seed without M04. Finishing properly needs a new stamp (`python lab.py run acoustic-only --stamp <new>`), which redoes the search and might pick another recipe by a word or two. Cheaper: train b24-frozen-high-lr directly for M04 at seed 17 and all speakers at seeds 18 and 19, with the `train.py` flags from the log's final runs (line 32,535), and compare with the control's long-adapter seeds (208, 219, 212).

Sentences only (secondary score, added 21 October 2026): same decodes, sentence clips only. That's 303 of the 425 reference words, and 35% or lower is 106 wrong or fewer. Means over the listed seeds, lower is better, read from each `results.json` with `scripts/asr/sentence_rescore.py`. The full ranking is in `docs/measurements.md` (Secondary score: sentences only).

| Recipe (run name) | Seeds | Sentence words wrong (of 303) | Sentence WER | M01 / M02 / M04 | All-clip WER |
| --- | ---: | ---: | ---: | --- | ---: |
| `f2-acoustic-only/1012-pending/control/` | 1 | 112.0 | 37.0% | 37.7 / 33.8 / 38.1 | 48.9% |
| `stock 0.6B v2 (reference)` | 1 | 188.0 | 62.0% | 61.5 / 47.1 / 71.7 | 76.0% |

Runs on only some speakers, scored on their own sentence words:

| Recipe (run name) | Speakers | Seeds | Sentence words wrong | Sentence WER | All-clip WER |
| --- | --- | ---: | ---: | ---: | ---: |
| `f2-acoustic-only/1012-pending/b24-frozen-high-lr/` | M01 + M02 | 1 | 65.0 of 190 | 34.2% | 45.5% |

Sources: acoustic-only log (`logs/lab-long-acoustic-1012-pending.txt`) (second attempt, final runs from line 32,470), `1012-pending/selection.json`, `1012-pending/failures.jsonl`, each `*-seed17-test` folder's `results.json`, and `1012-pending/battery-hypotheses/`. Prediction still blank.

### Completed study (27 to 28 October 2026, stamp `1027-1530`)

My lab run, sources `logs/lab-long-acoustic-1027-1530.txt`, `logs/lab-1027-1530-acoustic-only.txt`, `reports/f2-acoustic-only/1027-1530/summary.json` and `selection.json` on the lab results branch.

The seed-17 search picked b24-frozen-high-lr again (all 24 encoder blocks, head frozen, 1e-5, 100 epochs, stock-form targets), and confirmation dev matched. Test, wrong of 425 (stock 323, bar 148):

| Seed | Control (15-epoch adapter) | b24-frozen-high-lr | Gain over control, pooled (95% CI, prompts resampled) |
| --- | --- | --- | --- |
| 17 | 208 (48.9%) | **192** (45.2%) | +3.8 points (-0.9 to +8.4) |
| 18 | 219 (51.5%) | **191** (44.9%) | +6.6 points (+1.7 to +11.4) |
| 19 | 212 (49.9%) | **192** (45.2%) | +4.7 points (-1.4 to +11.1) |
| Mean | 213.0 (50.1%) | **191.7** (45.1%) | about +5.0 points |

At seed 17, M01 84 vs 87, M02 36 vs 42, M04 72 vs 79. It beats the control on every speaker at every seed. Only seed 18's interval excludes zero, so the rule (2 of 3 speakers agree, more than about 4.5 points) is met on the mean but not seed by seed.

Sentences only (303 words, line at 106): 102, 100 and 101 wrong (33.7%, 33.0%, 33.3%), under the line every seed, vs the control's 112, 120 and 121 (37.0%, 39.6%, 39.9%). Gains of +3.3, +6.6 and +6.6 points, seed 18's interval again excluding zero.

None of the nine frozen-head models wrote anything on the 140 non-speech clips (battery files byte-identical). The control wrote words on some (0 to 20). Seeds 17 and 19 give identical per-speaker counts (84, 36, 72) and pooled 192; seed 18 gives 191. I didn't compare 17 and 19 clip by clip. Seed 17 matches the 14 October attempt exactly (M01 84, M02 36).

It's still behind the stage P pretrained adapter (178.0, sentences 28.3%), and the gain over the control is small next to the gap to the bar.
