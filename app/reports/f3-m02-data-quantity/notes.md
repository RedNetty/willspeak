# F-3: data-quantity curve, M02, adapter after block 8

Proposal: F-3 (data quantity, §5.2 / H3): how the score changes with minutes of the speaker's own training audio.

Status: Set up 2 Oct 2026 and run 14 Oct 2026 (see Results). This is stage F of `lab.py`: 24 runs (four subset sizes, two schedules, three seeds), an estimated 1.5 hours of GPU. The pre-run hypotheses are recorded below. I didn't record a prediction of my own.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

If wrong words are still falling at M02's whole 9.0-minute training set, recording more from the client is a real way to cut errors, and the recording sessions come first. If the curve is flat from about 5 minutes, more audio of the same kind won't get under the 35% bar, and the work is in the model and the phrase list. This tests H3 (adaptation improves with the speaker's own data) as a whole curve.

## Setup

The adapter is the 1,049,600-number layer `h + W h + b` after encoder block 8, trained alone on the downloaded model. The fixed schedule trains a set number of epochs and averages the last 4 checkpoints. One update is one optimiser step, after 8 batches of 2 clips. M02 has 9.0 minutes of training audio (156 clips) and 95 test words. Its three block-8 results under the fixed schedule are 38, 44 and 42 wrong (mean 41.3, 43.5%), and stock gets 62 wrong.

`torgo_manifest.py --export --speakers M02 --curve M02 --curve-minutes 1 3 5 7` writes `export/M02-min1`, `-min3`, `-min5` and `-min7`. Each takes M02's training clips in a fixed order (by a hash of the clip id) and stops after that many minutes, so each subset sits inside the next. Dev and test are unchanged, so every subset is scored on the same 95 words. Add the whole set (9.0 minutes) and stock (0 minutes) and that's six points. The three seeds change only the training randomness (clip order, speed draws, masks), never the clips.

Runs are only a few minutes each, so I ran both schedules. The epochs arm uses 15 epochs everywhere, like the whole-set runs, so a 1-minute subset gets one or two updates per epoch, 15 to 30 in all. The updates arm runs enough epochs to match the whole set's updates in 15 epochs, capped at 150 (the script counts the clips and prints the plan). With so few updates per epoch, its last-four-epoch average covers far fewer updates than in the whole-set runs. That's an expected difference between the arms and the whole-set point, and isn't a bug.

The whole-set point reuses existing runs: seed 17 is `f2-M02-adapter8-pos`, seeds 18 and 19 are `f2-M02-adapter8-fixed-seed18` and `19`.

The manifest step rewrites M02's own export files, which every earlier M02 result depends on, so the script hashes M02's train, dev and test manifests before and after and stops if they differ. It also won't run if a training run is going or the raw TORGO audio isn't at `data/public/torgo/TORGO`.

## Plan

1. Check the whole-set results exist and no output does, write the four subsets, and compare the hashes.
2. Count the clips, work out the equal-updates epochs, then train, decode and score the 24 runs in seed order so an interrupted job leaves complete seeds. The summary shows updates per epoch and updates actually run next to each result.
3. Print wrong words on the 95 test words per subset, arm and seed, with the three-seed mean and range, each subset's share of the whole-set gain over stock, and the 5- and 7-minute means' distance from the whole set.

How I'd read it, decided before the run: a gap between two three-seed means only counts above 6 words. M02's whole-set seeds span 6 (38, 44, 42), so a mean of three is uncertain by about 2 and a difference of two means by about 3. In the equal-updates arm, a 5-minute mean within 6 words of the whole set means the curve is flat from 5 minutes. More than 6 words above means it's still falling and recording more is worth it. The 7-minute point, being close to 9.0 minutes, should land near the whole set either way. One speaker and one session split says nothing about another speaker or the client.

## Why

H3 has only been tested at one data size, and the prototype's bar is out of reach of every model change tried so far. Whether recording more helps decides how much of the semester goes to the recording protocol and the consent steps the client phase needs.

## Limits

This gives the curve for one speaker, with a noise estimate, under two schedules. It doesn't cover another speaker, another position or the F-1 base, audio from the client's own sessions (which differ from TORGO's), or anything past 9.0 minutes. It can't resolve a difference under 6 words, and which schedule wins here may not hold in general.

## Reading and Video

None added.

## Results

Lab run on 14 October 2026 (queue stamp `1012-pending`, stage F), read from `logs/lab-1012-pending-F.txt`. M02's block-8 adapter, 95 test words, stock 62 wrong. Wrong words at seeds 17 / 18 / 19, and the mean:

| Minutes of training audio | Equal epochs (15) | Equal updates |
| --- | --- | --- |
| 1 | 55 / 53 / 58, mean 55.3 | 56 / 63 / 52, mean 57.0 |
| 3 | 56 / 51 / 46, mean 51.0 | 55 / 54 / 51, mean 53.3 |
| 5 | 46 / 45 / 46, mean 45.7 | 47 / 42 / 42, mean 43.7 |
| 7 | 49 / 44 / 50, mean 47.7 | 41 / 41 / 46, mean 42.7 |

One minute already beats stock (62) by 5 to 9 words. About 5 minutes reaches the level of the whole 9-minute set (38 to 46 wrong in earlier runs), and 7 minutes adds nothing more. With equal updates the 5 and 7 minute points are 2 to 5 words better than with equal epochs, but that's inside the seed spread of 4 to 9 words. It's one speaker and 95 words, so this only sizes the recording burden for M02.

Sentences only (secondary score, added 21 October 2026). The same decodes, scored on the sentence clips alone. Across the three speakers that's 303 of the 425 reference words, and 35% or lower means 106 wrong or fewer. Means are over the seeds listed, lower is better, read from each run's `results.json` with `scripts/asr/sentence_rescore.py`. Every recipe's ranking is in `docs/measurements.md` under "Secondary score: sentences only".

These runs only cover some speakers, so each is scored on its own sentence words (M02 alone has 68, M02 and M04 have 181):

| Recipe (run name) | Speakers | Seeds | Sentence words wrong | Sentence WER | All-clip WER |
| --- | --- | ---: | ---: | ---: | ---: |
| `f3-min7-updates` | M02 | 3 | 25.0 of 68 | 36.8% | 44.9% |
| `f3-min5-updates` | M02 | 3 | 25.7 of 68 | 37.7% | 46.0% |
| `f3-min5-epochs` | M02 | 3 | 25.7 of 68 | 37.7% | 48.1% |
| `f3-min7-epochs` | M02 | 3 | 28.3 of 68 | 41.7% | 50.2% |
| `f3-min1-epochs` | M02 | 3 | 29.7 of 68 | 43.6% | 58.2% |
| `f3-min3-epochs` | M02 | 3 | 31.3 of 68 | 46.1% | 53.7% |
| `f3-min3-updates` | M02 | 3 | 32.0 of 68 | 47.1% | 56.1% |
| `f3-min1-updates` | M02 | 3 | 33.0 of 68 | 48.5% | 60.0% |

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Pre-run Hypotheses

Recorded on 2 Oct 2026, before any run of this test. The results are scored against these as written. Prediction: none recorded. All hypotheses use three-seed means and the equal-updates arm unless stated.

1. The 5-minute mean is within 6 words of the whole-set mean (about 41 wrong). Low confidence. A one-layer adapter should saturate early, and M02's score was already level across adapter positions 4, 6 and 8, but that doesn't measure it.
2. The 1-minute mean recovers between 30 and 70% of the whole-set gain over stock. With stock at 62 and the whole set about 41, that's a 1-minute mean between about 47 and 56 wrong.
3. At 1 and 3 minutes the equal-updates arm is lower than the equal-epochs arm by more than 6 words. With only 15 to 30 updates in all, the equal-epochs arm should be under-trained.

### Commands and Setup

Run from `app/` in the WSL tree on Yack after `git pull`, with the GPU free, nothing else training, and the laptop plugged in and set not to sleep. Stage F alone:

```bash
python lab.py run F
```

The script is described in the [base-model note](../f2-base-model-ceiling/notes.md#commands-and-setup). I dry-ran it on a stub copy of the tree with fake training and manifest programs. That checks the shell logic, the epoch arithmetic and the summary code, and nothing about real training, the real manifest step or real scores. The equal-updates epoch counts are worked out from the real clip counts at run time, and the plan is printed first.

### Full Results

See Results above. The per-run log is `logs/lab-1012-pending-F.txt`.

### Conclusions

*Written 21 October 2026 from the recorded results above.*

The comparison is M02's block-8 adapter trained on 1, 3, 5 and 7 minutes of its 9.0-minute training set, scored on M02's 37 test clips (95 reference words, lower is better). Stock gets 62 wrong. The whole set under the same fixed schedule averaged 42.0 wrong over seeds 17 to 19 in the L, T, E, S and C sweeps (`reports/finetune-torgo/big-l-ad8-default-M02-s*-test-nemo`).

Hypothesis 1 held. The 5-minute equal-updates mean (43.7) is within 6 words of the whole set, so the curve is flat from about 5 minutes. The 7-minute means (42.7 equal updates, 47.7 equal epochs) are no lower.

Hypothesis 2 didn't hold, because one minute recovers less than predicted. The 1-minute equal-updates mean of 57.0 is 5 words better than stock. That's about a quarter of the whole-set gain of 20 words, and just outside the predicted 47 to 56.

Hypothesis 3 didn't hold either. At 1 and 3 minutes the equal-updates arm is 1.7 and 2.3 words worse than equal epochs, where hypothesis 3 had it more than 6 better. At 5 and 7 minutes it's 2 to 5 words better. All of these gaps are inside the 4 to 9 word seed spread, so the schedule choice isn't shown to matter. No interval was computed in this stage.

For the recording plan: on this speaker about 5 minutes of audio is enough for the adapter, and more audio of the same kind isn't what stands between the adapter and the 35% bar. It's one speaker and 95 words, so it sizes a first recording session and doesn't set a ceiling.

</details>

## Related

[F-2: adapter position seeds](../f2-adapter-position-seeds/notes.md), [F-2: base-model ceiling](../f2-base-model-ceiling/notes.md), [F-2: augmentation variants](../f2-augmentation-variants/notes.md), [F-2: M02, adapter after 8 blocks, seeds 18 and 19](../f2-m02-adapter8-seeds/notes.md)
