# F-2: augmentation variants, adapter after block 8, M01, M02 and M04

Proposal F-2 (single-speaker fine-tuning, §5.2 / H3), a follow-up to the [adapter position seeds](../f2-adapter-position-seeds/notes.md). Same adapter as before. Only the augmentation of the training clips changes.

Status: set up 2 Oct 2026, run 13 to 14 Oct 2026 (see Results). Two stages of `lab.py`, each 27 runs (three variants, three speakers, three seeds) and about 1.5 hours of GPU. Stage A changes the two augmentations every run already has. Stage N adds two that `train.py` didn't have before 2 Oct: tempo stretch and white noise. The pre-run hypotheses are below. I didn't record a prediction of my own.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

The prototype bar is 35% word error rate or lower. The best pooled figure so far is 48.5% (206 wrong of 425 test words, block 8, three seeds). The plan listed "add augmentation" as a way to get more out of 5 to 9 minutes of audio per speaker. But reading `train.py` on 2 Oct showed two augmentations are already on in every run: speed perturbation between 0.85 and 1.15 of normal speed, and the base model's SpecAugment. So there are really two questions:

1. Stage A: how much do the two existing settings contribute, and does a wider speed range help?
2. Stage N: do tempo change without pitch change, and white noise, help? Both were added to `train.py` on 2 Oct as flags that are off by default.

## Setup

Some terms. Speed perturbation resamples a clip to a random speed in a range, so the pitch moves too, with a new draw every epoch. Tempo stretch changes speed with a phase vocoder and keeps the pitch; 1.25 makes a clip 25 percent faster and 0.75 makes it 25 percent slower. White noise has equal energy at every frequency. Its level is in dBFS (decibels below the loudest value a recording can hold), so -70 is very quiet and -40 is a faint hiss. SpecAugment blanks random stretches of time and frequency in each training clip's spectrogram; dev and test are never masked. A seed fixes clip order and random draws. The fixed schedule trains exactly 15 epochs and averages the last 4 checkpoints. Pooled means summed wrong words over M01 (169 test words), M02 (95) and M04 (161), 425 in all.

The baseline is the block-8 adapter on stock with the fixed schedule and default augmentation, already measured at pooled 201, 207 and 210 (mean 206.0). M02's three seeds were 38, 44 and 42. Stock is 323. Seed 17 for M01 and M04 came from an older `train.py`, while seeds 18 and 19 and every run here use the current one, so the baseline row mixes the two (same caveat as the position seeds note).

Otherwise every variant uses the baseline flags (`--adapter-after-blocks 8 --lr 3e-4 --batch 2 --grad-accum 8 --warmup-steps 10 --epochs 15 --select last --no-keep-stock`). Stage A has speedwide (`--speed-perturb 0.7 1.3`, wider than 0.85 to 1.15), nospeed (`--no-speed-perturb`, SpecAugment only) and nospec (`--no-spec-augment`, speed perturbation only).

Stage N keeps the defaults on and adds to them. Each added augmentation is applied to half the clips each time a clip is used (`--extra-aug-prob`, default 0.5), after the speed change. The variants are tempo (`--time-stretch 0.75 1.25`), noise (`--white-noise -70 -40`, a whole-number level from -70 to -41 dBFS, since NeMo draws up to but not including the second number) and tempo-noise (both).

Whether -70 to -41 is a sensible range depends on how loud the clips are. Noise at level L gives a clip a signal-to-noise ratio of its own level minus L. So stage N first runs `scripts/asr/clip_levels.py` on the training clips and prints each speaker's loudness percentiles and the ratios for the quietest, a typical and the loudest clip. The tempo range of 0.75 to 1.25 is wider than speed perturbation's and hasn't been tested in this project.

The new flags use NeMo's own `TimeStretchPerturbation` and `WhiteNoisePerturbation`, with keyword names taken from NeMo's main branch (read 2 Oct). Yack's NeMo may be older, so `train.py` runs each added augmentation on a second of fake audio before loading the model. A bad keyword or a library that can't run it stops the run in under a minute instead of inside a data-loader worker. In an earlier CPU-only test (not on Yack), against a copy of NeMo's perturbation code with its phase vocoder (librosa 1.0.0, numba 0.68), a one-second clip came out 0.80 to 1.33 seconds long and noise measured -70 to -45 dBFS over eight draws. A deliberately wrong keyword was caught and the unit tests pass. No training run had used the flags yet.

## Plan

1. The script checks every baseline result exists and no output does, then trains, decodes and scores each stage's 27 runs in seed order (everything at seed 17, then 18, then 19), so an interrupted job still leaves complete seeds. Stage N stops at once if `scripts/asr/train.py` has no `--time-stretch`, which means `git pull` wasn't run in that folder.
2. Each stage prints pooled wrong words per variant and seed, the three-seed mean and range, the difference from default, per-speaker means, and a verdict per variant. Stage N prints the loudness table first.

The rules, written before the run. A variant is better only if its three-seed pooled mean is more than 20 words lower (about twice the pooled seed noise, the rule from the position seeds note) and it's lower on at least two of three speakers. Worse is the mirror image, and anything else is not different. tempo-noise only beats tempo and noise if it's also more than 20 words below the lower of the two.

If nothing gets below 148 pooled (35% of 425 is 148.75), augmentation isn't what stands between this model and the bar, and the lowest mean is printed with its percentage. A variant better pooled but worse on one speaker is reported per speaker, and no setting goes on the device from this test alone. If noise is worse and the loudness table shows the loud end gives under about 6 dB signal-to-noise on typical clips (median clip quieter than -35 dBFS), the range was too loud and says nothing about quieter noise.

## Why

`train.py`'s help text calls speed perturbation the strongest augmentation for low-intelligibility speech in the literature, and the project has never measured it against its absence. Stage A does that. Stage N is the next cheapest way to add variety without recording anything. Dysarthric speakers vary a lot in speaking rate, which tempo stretch imitates, and noise gives each clip a different background. Earlier reading suggested rate changes helped other models with more data, but I didn't recheck those papers for this note. If neither stage helps, what's left is the base model and more data, which the other stages of the same script measure.

## Limits

This can show whether speed perturbation and SpecAugment each help, whether 0.7 to 1.3 beats 0.85 to 1.15, and whether tempo stretch or white noise at these settings help on top of the default, on three speakers and three seeds.

It can't show anything about gain changes, heavier SpecAugment, real room noise or reverberation, since none of those are in `train.py`. Each new augmentation is one setting (one range, one level, one probability). It says nothing about the F-1 base, other adapter positions or other learning rates, can't see differences under about 20 words pooled, and can't say whether augmentation would help more with more recordings.

## Reading and Video

None added.

## Results

Lab run 13 to 14 October 2026 (queue stamp `1012-pending`, RTX 4000 Ada), read from `logs/lab-1012-pending-A.txt` and `-N.txt`. Block-8 adapter on stock, fixed 15-epoch schedule, seeds 17, 18, 19, pooled wrong words of 425 on M01, M02 and M04. Stock is 323.

| Variant | Seeds 17 / 18 / 19 | Mean | Against default |
| --- | --- | --- | --- |
| Default | 208 / 219 / 212 | 213.0 | |
| Wider speed range (stage A) | 214 / 223 / 220 | 219.0 | +6.0 |
| No speed perturbation (A) | 209 / 218 / 205 | 210.7 | -2.3 |
| No SpecAugment (A) | 207 / 221 / 223 | 217.0 | +4.0 |
| Tempo stretch (N) | 213 / 229 / 215 | 219.0 | +6.0 |
| White noise (N) | 212 / 224 / 208 | 214.7 | +1.7 |
| Tempo stretch and noise (N) | 221 / 223 / 220 | 221.3 | +8.3 |

Nothing moved the pooled count by the 4.5-point rule (about 19 words) either way. The default's own three seeds span 11 words, which is more than every difference except tempo-and-noise. Removing speed perturbation or SpecAugment didn't hurt measurably, and adding tempo stretch or noise didn't help. Only seeds 17 to 19 were run, and the figures come from the logs, not a rescored table.

Sentences only (secondary score, added 21 October 2026). These are the same decodes scored on sentence clips alone, without the single-word clips. That's 303 of the 425 reference words, and 35% or lower means 106 wrong or fewer. Means over the listed seeds, lower is better. Read from each run's `results.json` with `scripts/asr/sentence_rescore.py`. The full ranking is in `docs/measurements.md` under "Secondary score: sentences only".

| Recipe (run name) | Seeds | Sentence words wrong (of 303) | Sentence WER | M01 / M02 / M04 | All-clip WER |
| --- | ---: | ---: | ---: | --- | ---: |
| `f2-adapter8-aug-nospeed` | 3 | 118.0 | 38.9% | 38.8 / 34.3 / 41.9 | 49.6% |
| `f2-adapter8-aug-noise` | 3 | 119.7 | 39.5% | 38.8 / 38.7 / 40.7 | 50.5% |
| `f2-adapter8-aug-speedwide` | 3 | 121.7 | 40.2% | 39.3 / 37.7 / 42.5 | 51.5% |
| `f2-adapter8-aug-nospec` | 3 | 123.0 | 40.6% | 43.4 / 38.2 / 38.9 | 51.1% |
| `f2-adapter8-aug-tempo-noise` | 3 | 124.0 | 40.9% | 39.9 / 39.7 / 42.8 | 52.1% |
| `f2-adapter8-aug-tempo` | 3 | 124.3 | 41.0% | 38.8 / 37.7 / 45.4 | 51.5% |
| `stock 0.6B v2 (reference)` | 1 | 188.0 | 62.0% | 61.5 / 47.1 / 71.7 | 76.0% |

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Pre-run Hypotheses

Written 2 Oct 2026 before any run. Results are scored against these as written. No prediction recorded. All hypotheses are three-seed pooled means against the default's 206.0.

1. speedwide is not different from the default (within 20 words either way). The adapter has 1 M numbers and sees 5 to 9 minutes of audio. A wider speed range changes how the clips look but adds no information about the speaker.
2. nospeed is worse by 20 words or more (mean 226 or higher). Without speed perturbation the adapter sees the same 72 to 156 clips 15 times unchanged, apart from SpecAugment masks.
3. nospec is not different (within 20 words). Masking matters for a model that updates all its weights, and should matter less with only the adapter free.
4. tempo is not different (within 20 words). The default already varies speed by 15 percent either way, and tempo stretch widens that to 25 percent on half the clips, a modest change for an adapter with 1 M numbers. The recorded estimate was about a 60% chance, with "better" the likeliest alternative.
5. noise is not different (within 20 words), as long as the loudness table shows typical clips louder than -35 dBFS. A faint hiss 20 dB or more below the speech is a small change.
6. tempo-noise is not different, and no variant gets a three-seed mean of 148 or lower. The lowest of the six means lands between 180 and 215 words, 42% to 51% pooled. Reaching 148 would need more than 55 words below the default from augmentation alone.

### Commands and Setup

Run from `app/` in the WSL tree on Yack after `git pull`, with the GPU free and the laptop plugged in and set not to sleep. `train.py` changed on 2 Oct (the new flags), so the hash in the first lines of the log is no longer 032323d84c43. Stage N stops at once if the pull didn't bring the new file. Stages A and N only:

```bash
python lab.py run A N
```

All five stages together are in the [base-model note](../f2-base-model-ceiling/notes.md#commands-and-setup). I dry-ran the script on a stub copy of the tree first, with made-up manifests and fake training and decoding programs. That checks the shell logic, file names and summary code, but nothing about training, NeMo or real scores. The new `train.py` was tested separately and its unit tests pass with and without NeMo's perturbation classes.

### Full Results

See Results above. Per-run logs are `logs/lab-1012-pending-A.txt` and `-N.txt`.

The loudness table at the start of stage N (`logs/lab-1012-pending-N.txt`, lines 10 to 21) gives median training-clip levels of -11.2 dBFS (M01), -26.5 dBFS (M02) and -21.6 dBFS (M04). At the loudest noise (-41 dBFS) the median clip keeps 30, 15 and 19 dB of signal-to-noise. M02's quietest clip drops to 9 dB.

### Conclusions

*Interpretation added 21 October 2026 from the recorded results above.*

Against the default block-8 adapter on stock (pooled 213.0 of 425 wrong, three seeds), every variant lands between 210.7 and 221.3. The biggest move is 8.3 words (about 2 percentage points). That's well inside the 20-word rule set before the run and inside the default's own 11-word seed spread, so no variant is shown better or worse. It doesn't prove the settings are equal either.

Against the hypotheses: speedwide, nospec, tempo and noise were "not different", as expected. Hypothesis 2 didn't hold. Removing speed perturbation was supposed to cost 20 words or more, but it scored 210.7, 2.3 words below the default. Hypothesis 6 held, with the lowest mean (210.7, 49.6%) inside the 180 to 215 range and nowhere near 148. Note the hypotheses assumed a default of 206.0, while this run measured 213.0.

The noise result counts under hypothesis 5's condition, since every speaker's median clip is louder than -35 dBFS. M02's quietest clips got as little as 9 dB signal-to-noise at the loud end, so the noise wasn't trivially faint for that speaker.

Training-clip augmentation isn't what stands between the adapter and the 35% bar (148 wrong). Later stages agree: the only change that moved the adapter by more than the rule was severe-speaker pretraining (stage P, 178.0 at rate 6e-4, `reports/f2-adapter-levers`). Keep the default augmentation for the expo model (speed perturbation 0.85 to 1.15 plus SpecAugment). Nothing here justifies another augmentation sweep.

</details>

## Related

[F-2: adapter position seeds](../f2-adapter-position-seeds/notes.md), [F-2: adapter position sweep](../f2-adapter-position-sweep/notes.md), [F-2: base-model ceiling](../f2-base-model-ceiling/notes.md), [F-3: M02 data quantity](../f3-m02-data-quantity/notes.md)
