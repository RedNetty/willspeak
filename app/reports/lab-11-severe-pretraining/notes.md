# Lab test 11: severe pre-training, fine-tune on the other severe speakers first, then on his own sessions

Proposal: F-1 (H2) and F-2 (H3) combined. Prototype bar: 35%.

Status: run 7-8 October 2026, results read 14 October.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

F-1 trains on every other dysarthric speaker and decodes the held-out one. Severe speech isn't like mild speech, so maybe only the other severe speakers are useful teachers. Here I pre-train on the other severe speakers (F01 and two of M01, M02, M04), then fine-tune on the speaker's own earlier sessions, and compare both steps with F-1, whole-block and stock.

## Setup

Pooled means total wrong words over M01 (169 test words), M02 (95) and M04 (161), 425 in all. Stock Parakeet-TDT-0.6B-v2 gets 323 wrong. The best raw result so far is about 201 to 210 (about 48.5%, adapter after block 8, three seeds), and the 35% bar is 148 or fewer.

Whole-block is the F-2 model that fully trains blocks 0 to 7: the fixed five-epoch runs for M01 and M04 (`models/parakeet-ft-f2-<SPK>-bottom8-5epochs`), and for M02 `models/parakeet-ft-M02e-A3` (dev-selected, 42 of 95 wrong in NeMo), a different recipe.

`torgo_manifest.py --severe-loso` writes `export/loso-severe-<SPK>/`, with the other severe speakers' prompts as train and dev and this speaker's test prompts as test (prompt-disjoint as always). It checks the three speakers' own manifests stay byte-identical, since earlier results depend on them. Phase 1 is the F-1 recipe (blocks 0 to 7, dev-selected, stock kept as a candidate). Phase 2 continues with the F-2 recipe (five epochs, last weights) on the speaker's own export.

F-1 trains on seven other speakers, severe and mild, so this changes both who is in the pre-training set and how many. A difference can't be pinned on severity alone. The adapter on F-1 also beat the adapter on stock for M04 every time but showed no clear difference for M01 or M02, so the effect may vary by speaker.

## Plan

1. Make the severe-only exports if missing and check the three speakers' manifests didn't change.
2. Per speaker, run phase 1 and decode, then phase 2 from that model and decode again.
3. Run `pool_table.py` over the five systems: stock, F-1, severe pre-training alone, whole-block, and severe pre-training then his own sessions.

Rule set before the run: a change only counts if at least 2 of 3 speakers move the same way as the pooled count and pooled WER moves more than 4.5 points, the seed-to-seed noise seen so far (4 to 5 points per speaker). Anything else is reported as unclear, and unclear doesn't mean equal. `pool_table.py` prints the verdict. Three agreeing seeds could settle a smaller gap, but this note has one run per cell unless stated.

Phase 1 beating F-1 on 2 of 3 speakers would mean severe speakers teach more than the bigger mixed set. Phase 2 beating whole-block would make severe pre-training worth doing before the intended user's recordings. Otherwise whole-block on his own sessions stays best.

## Why

The intended user will only have a few recordings at first, so a good start from other severe speakers would let us begin before his own audio exists.

## Limits

Three speakers, one seed each. Nothing about transfer to the intended user, and the amount of pre-training data isn't held fixed.

## Results

Pooled over M01/M02/M04, 425 reference words, intervals resample clips.

| System | M01 | M02 | M04 | Wrong | WER | Gain over stock (95% CI) |
|---|---:|---:|---:|---:|---:|---|
| stock | 131 | 62 | 130 | 323 | 76.0% | - |
| severe others only (target speaker never heard) | 104 | 60 | 111 | 275 | 64.7% | +11.3 (+4.1 to +18.9) |
| F-1 | 91 | 56 | 101 | 248 | 58.4% | +17.6 (+9.4 to +25.9) |
| severe others, then his own earlier sessions | 98 | 47 | 103 | 248 | 58.4% | +17.6 (+10.8 to +25.0) |
| whole-block (own data only) | 130 | 40 | 121 | 291 | 68.5% | +7.5 (+1.9 to +14.2) |

Per speaker, the severe-only gain was M01 +16.0, M04 +11.8 and M02 only +2.1. The follow-up on his own sessions fixes M02 (47) but doesn't beat F-1 elsewhere. Everything here is well behind the adapters (208 and 197).

Source: pool_table log (`logs/pool_table-20261008-002503-340709.log`). Runs were on the RTX 4000 Ada lab computer, 6-8 Oct 2026, and reviewed 14 October 2026.

Sentences only (secondary score, added 21 October 2026). Same decodes, scored on the sentence clips alone. That's 303 of the 425 words, so 35% means 106 wrong or fewer. Means over the listed seeds, lower is better, from each run's `results.json` via `scripts/asr/sentence_rescore.py`. The full ranking is in `docs/measurements.md` under "Secondary score: sentences only".

| Recipe (run name) | Seeds | Sentence words wrong (of 303) | Sentence WER | M01 / M02 / M04 | All-clip WER |
| --- | ---: | ---: | ---: | --- | ---: |
| `lab-11-then` | 1 | 142.0 | 46.9% | 44.3 / 38.2 / 54.9 | 58.4% |
| `lab-11-pre` | 1 | 162.0 | 53.5% | 50.0 / 50.0 / 59.3 | 64.7% |
| `stock 0.6B v2 (reference)` | 1 | 188.0 | 62.0% | 61.5 / 47.1 / 71.7 | 76.0% |

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

None written before the run (7-8 October 2026), and none added since.

### Commands and Setup

Run from `app/` in the WSL tree on Yack after `git pull`, GPU free, PC set not to sleep. The script stops if a `train.py` run is already going, and each stage stops if one of its outputs already exists.

```bash
python lab.py run 11
```

The script writes `logs/lab-<stamp>-summaries.txt` at the end, which is what these notes record. Full output is in `logs/lab-<stamp>-11.txt`. The first step rewrites the export manifests, and the script stops if M01, M02 or M04's own manifests change.

Commands come from `lab.py`. The flags were checked against each script with a dry run (`python lab.py run --dry`, `tests/unit/test_lab_cli.py`) and the logic has unit tests on synthetic data, which aren't recognition results. Two things hadn't been run on real models or data: the severe-only exports on the real TORGO tree (only tested on a small made-up tree), and training from a local `.nemo` through `--base`.

### Full Results

The completed comparison and its sources are in Results above.

### Conclusions

*Added 21 October 2026, from the recorded results above.*

On the pooled set (425 words, stock 323 wrong, lower is better), whole-block training on just the other severe speakers got 275 wrong (64.7%) without ever hearing the target speaker. That's an 11.3-point gain over stock, interval +4.1 to +18.9. It helped M01 and M04 most and barely moved M02 (62 to 60).

Continuing that model on the speaker's own earlier sessions got 248 wrong (58.4%), the same as F-1. Both beat the speaker's own whole-block model (291). With this recipe, hearing other dysarthric voices first seems worth more than the speaker's own data alone.

The idea worked much better with an adapter. Severe pre-training followed by the block-8 adapter averaged 182.3 wrong over three seeds in stage C, and 178.0 (41.9%) in stage P at a rate of 6e-4. That beat the adapter alone (212.0) on all three speakers ([stage P](../f2-adapter-levers/notes.md)) and is the best three-seed result in the project.

Lab test 26 varies this pre-training under the adapter and had no result yet when this was written. Severe pre-training plus the adapter is the leading candidate for the expo model, but it's still 30 words above the 148-wrong bar.

</details>

## Related

[F-1: bottom 8 blocks](../f1-loso-bottom8/notes.md), [Lab test 2](../lab-02-weight-merging/notes.md)
