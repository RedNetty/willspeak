# Check: cleaned-up train.py, reruns of recorded results

Proposal: none, this is a software check supporting F-2 and the choice of the prototype model.

Status: run 1 Oct 2026, 12:43 to 13:08, on commit 9991cd5. Every score landed inside its pass band. The pre-run hypotheses were written before the reruns, with no prediction of my own.

See also the [Report Context](../Report%20Context.md) and the [Measurement guide](../../../docs/measurements.md).

## Idea

Does `train.py` still give the same scores after the cleanup commits of 1 Oct? I reran three results already recorded in notes, on the commit in the WSL tree, and checked each against a pass band set in advance. That's the range of word errors (wrong, deleted and inserted words) a rerun can land in and still count as a reproduction. GPU runs on Yack aren't bit-for-bit repeatable, so I didn't expect exact matches.

## Setup

Between af07966 (11:41) and the pulled commit, the cleanup touched `train.py`, `ci.py`, `_cli.py`, `evaluate.py`, `audio.py`, the corpus export, the app server and pages, and smaller scripts. `train.py` got a new log format with a progress line every 30 seconds, a check that the saved inputs didn't change during the run, argument checks, a check that the loss stays finite, averaging of floating-point values only, and a dev score of the averaged model before release. `ci.py` now stops if a WER disagrees with `results.json`.

The adapter runs of 12:13 to 12:20 (the [M01 and M04 adapter test](../f2-m01-m04-adapter8/notes.md)) used 9aafe98. Nothing after it (167598f, b5e0589, 8863650, ddb10d5 or later) had been run. The big change is the release step. In 9aafe98 the averaged model only got a dev score in dev-selected runs. Since 167598f it gets one whenever checkpoints are averaged, fixed-schedule runs included, and the log shows `Released model dev WER`. That code had never run on real NeMo. The unit tests pass on b5e0589, but only on a CPU-only machine with stand-in models, with one test skipped for lack of a recogniser.

| Check | Recorded result | What it exercises |
| --- | --- | --- |
| 1 | Adapter after block 7 on M01 and M04, 15 epochs on a fixed schedule: 49.70% (84 of 169 words wrong) and 49.07% (79 of 161), NeMo | the fixed-schedule release path with an adapter, including the new dev score of the averaged model |
| 2 | Adapter after block 7 on M02, default recipe (checkpoints picked on dev, stock kept, up to 10 epochs, seed 17): 48.42% (46 of 95), the same on seeds 17, 18 and 19; the log's first score line is `stock dev WER 0.5797` | the dev-selection release path, the stock guard, early stopping |
| 3 | Whole-block bottom-8 model on M02 (the recipe the prototype models use): 44.21% (42 of 95); two earlier whole-block runs scored 45.26% (43) and 51.58% (49) | training 210,461,318 numbers, the pruned checkpoints, saving the `.nemo` file |

I also scored one extra run. At 12:34 I'd trained the M02 adapter by hand on the same code with `--epoch 15` (read as `--epochs 15`) into `models/parakeet-ft-f2-M02-adapter8b`. That raises the planned updates from 100 to 150, and the learning rate decays over the planned total, so it's a different recipe from the default 10 epochs. It has no pass band. The script printed the `train.py` hash in its `run.json` next to the pulled file's hash.

## Plan

1. Pull once and print the commit. A pull in the middle would change `train.py` under the runs.
2. Run checks 1, 2 and 3. After each, decode the 37, 65 or 64 test clips in NeMo and score with `evaluate.py` and `ci.py`.
3. Compare each score with its band and check the log lines listed in the commands.

Pass bands:

- Check 1, seed 17: within 4 points of the record, so 77 to 91 word errors on M01 and 73 to 85 on M04. This is the real code check, since it repeats the recorded run exactly. The severe-focus repeat of 30 Sep landed 3 to 4 points apart on single speakers, hence 4 points. The log must show `Checking the averaged model on dev before release...` and a released dev WER.
- Check 1, seeds 18 and 19: no band. They measure the M01 and M04 seed spread, which the M01 and M04 note lists as not shown. So far only M02's is known (three seeds, 46 errors each).
- Check 2: 44 to 48 (within 2 words of 46), and `stock dev WER 0.5797` in the log, since the zero adapter has to give the stock score.
- Check 3: 41 to 50, the earlier whole-block range (42 to 49) plus a word each side, and `210461318 trainable` in the log.

On a failure I'd trace the cause with the `train.py` hash in each run's `run.json` and a `git diff` between 9aafe98 and the pulled commit. On a crash, read the log before decoding.

## Why

The next tests and the prototype model are built on `train.py`. The M01 and M04 result (adapter ahead of the F-1 model on both speakers) rests on one run of the code at 9aafe98, and the code has changed since.

## Limits

From the method review on 1 Oct: the bands are engineering tolerances, with a 4-point allowance borrowed from another recipe. They aren't equivalence tests or confidence intervals. Passing shows nothing changed beyond run-to-run spread, and a miss would call for a closer look without proving a regression. I report every planned seed. The test sets were already looked at, so this checks software behavior and isn't independent confirmation of a new method.

The server, pages, corpus export, int8 export (`scripts/asr/export.py`, unchanged by the cleanup), the Pi and any later commit aren't covered.

## Reading and Video

Nothing added. The reading on why two runs of one recipe differ is in the [seeds note](../f2-m02-adapter8-seeds/notes.md#reading-and-video).

## Results

All three checks passed and every planned log line was there.

| Check | Run | WER | Word errors | Recorded | Band | Delta against the recorded run |
| --- | --- | ---: | ---: | ---: | --- | --- |
| 1 | M01 adapter, seed 17 | 49.70% | 84 of 169 | 84 | 77 to 91 | 0.00 (0.00 to 0.00) |
| 1 | M01 adapter, seed 18 | 51.48% | 87 |  | spread only | -1.78 (-6.56 to 2.65) |
| 1 | M01 adapter, seed 19 | 50.89% | 86 |  | spread only | -1.18 (-6.06 to 3.83) |
| 1 | M04 adapter, seed 17 | 49.07% | 79 of 161 | 79 | 73 to 85 | 0.00 (0.00 to 0.00) |
| 1 | M04 adapter, seed 18 | 47.20% | 76 |  | spread only | 1.86 (-2.91 to 6.43) |
| 1 | M04 adapter, seed 19 | 50.93% | 82 |  | spread only | -1.86 (-8.98 to 4.17) |
| 2 | M02 adapter, default recipe | 48.42% | 46 of 95 | 46 | 44 to 48 | 0.00 (0.00 to 0.00) |
| extra | M02 adapter, 15-epoch cap (manual run) | 48.42% | 46 | 46 | none | 0.00 (-4.11 to 4.21) |
| 3 | M02 whole-block bottom-8 | 44.21% | 42 | 42 | 41 to 50 | 0.00 (-4.49 to 4.21) |

A positive delta means the rerun did better. Seeds 18 and 19 are compared with the recorded seed-17 run.

The seed-17 reruns on M01, M04 and the M02 default recipe matched the recorded runs clip by clip. An interval of 0.00 to 0.00 needs every clip to have the same error count in both runs. The M02 log also repeated the [M02 adapter note](../f2-m02-adapter8/notes.md): `stock dev WER 0.5797`, a best dev WER of 31.88% at `best-epoch=5`, and a last-epoch loss of 1.4593 after 90 updates. The bands allowed for run-to-run differences, but there weren't any.

The whole-block model matched the recorded `M02e-A3` run on total errors (42) and on dev WER after averaging 4 checkpoints (see the [bottom-8 parts note](../f2-m02-bottom8-parts/notes.md)). Some errors moved to different clips, though. Only a second same-seed run could separate GPU noise from the code change, and I didn't make one.

The manual 15-epoch run also got 46 errors, but on different clips, so it's a different model. Its loss matches the default run in epoch 1 (4.4812) and splits at epoch 2 (3.2898 against 3.2936), where warmup ends and decay starts. That fits a different decay length, though I didn't test it. Early stopping ended it after epoch 10 of 15, best at epoch 7. Its stored `train.py` hash (c696b4873186) matches the pulled file.

The seed spread was 1.78 points (3 words) on M01 and 3.73 points (6 words) on M04, both under the 4-point guide. The three-seed means are 50.69% on M01 and 49.07% on M04. Each seed's delta interval is 9 to 13 points wide, so the seed moves the score less than clip sampling does. Every seed beats the F-1 model on both speakers. On M04 the interval excludes zero every time (+16.15, +18.01, +14.29) and on M01 it includes zero every time (+7.69, +5.92, +6.51). That's the first run's pattern, so the [M01 and M04 note](../f2-m01-m04-adapter8/notes.md) conclusion stands.

The new release step ran in all six check-1 runs without a crash and printed a `Released model dev WER` each time (see Full Results). With dev sets of 18 words (M01) and 25 (M04), those move in steps of 5.6 and 4.0 points and say little about quality.

Training took 1m 56s to 2m 18s for each of the seven adapter runs and 3m 20s for the whole-block run. The script took about 25 minutes, against 35 planned.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Pre-run Hypotheses

Recorded 1 Oct 2026 at 12:35, before any run of this check. No prediction of my own was recorded.

1. Check 1, seed 17: M01 about 84 word errors (49.7%), most likely 80 to 88. M04 about 79 (49.1%), most likely 75 to 83. Across seeds 17, 18 and 19 the best and worst on each speaker differ by 2 to 4 points, which is 4 to 7 words on M01 and 3 to 6 on M04. The three-seed mean is within 3 points of today's score. All three M04 runs beat the F-1 model (65.22%) and at least two of three M01 runs beat it (57.40%). The averaged model's dev WER is about 44% on M01 (7 to 9 of 18 words wrong) and 60 to 68% on M04 (15 to 17 of 25).
2. Check 2: 46 word errors (48.42%), most likely 44 to 48. The log starts with `stock dev WER 0.5797`. The best dev WER is 30 to 33%, training stops before epoch 10, and the averaged model's dev WER is within 2 points of the best single checkpoint's.
3. Check 3: about 45 word errors (47.4%), most likely 42 to 49, so within 7 words of the recorded 44.21%.
4. Reasoning: the cleanup changed logging, validation, input checks and the release step. It didn't touch the loss, the optimizer, or the order and augmentation of the clips, so scores should fall inside the spread already seen (no spread on the M02 adapter, 42 to 49 errors for the whole-block recipe). M01 and M04 have less training data than M02, so their seed spread should be bigger. The unit tests pass, so I don't expect a crash. If something does fail, the likeliest place is the one piece of code that has never run on real NeMo, the dev score of the averaged model after a fixed-schedule or adapter run.

### Commands and Setup

As run: the commands below, in one script (`logs/recheck.sh` in the WSL tree). The output went to `logs/recheck-full-1001-1243.txt` and a summary to `logs/recheck-summary-1001-1243.txt`. The order was check 1 with seed 17 (M01, M04), check 2, scoring the manual 12:34 run, check 3, then check 1 with seeds 18 and 19. The manual 12:34 run was:

```bash
python scripts/asr/train.py --export data/public/torgo/export/M02 --out models/parakeet-ft-f2-M02-adapter8b --adapter-after-blocks 8 --lr 3e-4 --batch 2 --grad-accum 8 --warmup-steps 10 --freeze-batchnorm --normalize-targets --dev-scoring evaluate --keep-stock --epoch 15
```

Run from `~/willspeak` in WSL on Yack, with the GPU free and Windows apps closed. Training and decoding use `~/venvs/willspeak-train`, and `evaluate.py` and `ci.py` use `app/.venv`. I budgeted about 35 minutes. Check 1 is about 20 minutes (six runs of about 3.5 minutes each with decoding), check 2 about 4, and check 3 about 5 to 10 (not timed before). The training and scoring blocks stop at their first error. Pull once, here, and not again until the last block is done:

```bash
cd ~/willspeak && git pull && cd app
git log -1 --format='code: %h %ad %s' --date=format:%H:%M
git status --short | head
```

Check 1, train and decode (adapter after block 7, M01 and M04, three seeds):

```bash
( set -e
source ~/venvs/willspeak-train/bin/activate
for SEED in 17 18 19; do
  for S in M01 M04; do
    python scripts/asr/train.py --export data/public/torgo/export/$S --out models/parakeet-ft-f2-$S-adapter8-recheck-seed$SEED --adapter-after-blocks 8 --lr 3e-4 --batch 2 --grad-accum 8 --warmup-steps 10 --epochs 15 --select last --no-keep-stock --freeze-batchnorm --normalize-targets --dev-scoring evaluate --seed $SEED
    python scripts/asr/nemo_hyp.py --stock --adapter models/parakeet-ft-f2-$S-adapter8-recheck-seed$SEED/adapter.pt --manifest data/public/torgo/export/$S/test.jsonl --audio-root data/public/torgo/export/$S --out reports/finetune-torgo/f2-$S-adapter8-recheck-seed$SEED-test-nemo.jsonl
  done
done )
```

Check 1, log lines (the six newest training logs, oldest first):

```bash
for f in $(ls -t logs/train_parakeet-*.log | head -6 | sort); do echo "== $f"; grep -hE "RUN parakeet|parameters exact|fixed schedule|Checking the averaged|averaged|Released|Elapsed|failed|Traceback" "$f"; done
```

Each should show `1049600 trainable`, `kept epochs 12, 13, 14, 15`, `Checking the averaged model on dev before release...`, `averaged 4 checkpoints` and a `Released model dev WER` with an actual number (instead of "not measured").

Check 1, score and compare (each rerun against the first adapter run of today, and against the F-1 model):

```bash
( set -e
source .venv/bin/activate
for SEED in 17 18 19; do
  for S in M01 M04; do
    M=data/public/torgo/export/$S; R=f2-$S-adapter8-recheck-seed$SEED-test-nemo
    python scripts/asr/evaluate.py --manifest $M/test.jsonl --audio-root $M --hyp reports/finetune-torgo/$R.jsonl --group-by kind --out reports/finetune-torgo/$R
    echo "== $S seed $SEED against the first adapter run"
    python scripts/asr/ci.py --manifest $M/test.jsonl --stock reports/finetune-torgo/f2-$S-adapter8-test-nemo --finetuned reports/finetune-torgo/$R
    echo "== $S seed $SEED against the F-1 model"
    python scripts/asr/ci.py --manifest $M/test.jsonl --stock reports/finetune-torgo/f1-loso-$S-bottom8-on-F2-test-nemo --finetuned reports/finetune-torgo/$R
  done
done
python3 -c "import json;[print(n, round(r['wer']*100,2), r['word']['sub']+r['word']['dele']+r['word']['ins'], 'of', r['word']['n_ref']) for n in ['f2-%s-adapter8%s-test-nemo'%(s,t) for s in ('M01','M04') for t in ('','-recheck-seed17','-recheck-seed18','-recheck-seed19')] for r in [json.load(open('reports/finetune-torgo/%s/results.json'%n))]]" )
```

Check 2 and check 3, train and decode:

```bash
( set -e
source ~/venvs/willspeak-train/bin/activate
python scripts/asr/train.py --export data/public/torgo/export/M02 --out models/parakeet-ft-f2-M02-adapter8-recheck --adapter-after-blocks 8 --lr 3e-4 --batch 2 --grad-accum 8 --warmup-steps 10 --freeze-batchnorm --normalize-targets --dev-scoring evaluate --keep-stock --seed 17
python scripts/asr/nemo_hyp.py --stock --adapter models/parakeet-ft-f2-M02-adapter8-recheck/adapter.pt --manifest data/public/torgo/export/M02/test.jsonl --audio-root data/public/torgo/export/M02 --out reports/finetune-torgo/f2-M02-adapter8-recheck-test-nemo.jsonl
python scripts/asr/train.py --export data/public/torgo/export/M02 --out models/parakeet-ft-f2-M02-bottom8-recheck --train-parts all --batch 2 --grad-accum 8 --warmup-steps 10 --freeze-batchnorm --normalize-targets --dev-scoring evaluate --keep-stock
python scripts/asr/nemo_hyp.py --nemo models/parakeet-ft-f2-M02-bottom8-recheck/parakeet-ft.nemo --manifest data/public/torgo/export/M02/test.jsonl --audio-root data/public/torgo/export/M02 --out reports/finetune-torgo/f2-M02-bottom8-recheck-test-nemo.jsonl )
```

Check 2 and check 3, log lines (the two newest training logs, check 2 first):

```bash
for f in $(ls -t logs/train_parakeet-*.log | head -2 | sort); do echo "== $f"; grep -hE "RUN parakeet|stock dev WER|parameters exact|best dev WER|Checking the averaged|Averaged dev|averaged|Released|Elapsed|stopped|failed|Traceback" "$f"; done
```

Check 2 should show `1049600 trainable` and `stock dev WER 0.5797`. Check 3 should show `210461318 trainable`.

Check 2 and check 3, score and compare:

```bash
( set -e
source .venv/bin/activate
M=data/public/torgo/export/M02
for R in f2-M02-adapter8-recheck f2-M02-bottom8-recheck; do
  python scripts/asr/evaluate.py --manifest $M/test.jsonl --audio-root $M --hyp reports/finetune-torgo/$R-test-nemo.jsonl --group-by kind --out reports/finetune-torgo/$R-test-nemo
done
echo "== adapter rerun against stock"
python scripts/asr/ci.py --manifest $M/test.jsonl --stock reports/finetune-torgo/M02-test-stock-nemo --finetuned reports/finetune-torgo/f2-M02-adapter8-recheck-test-nemo
echo "== adapter rerun against the first adapter run"
python scripts/asr/ci.py --manifest $M/test.jsonl --stock reports/finetune-torgo/f2-M02-adapter8-test-nemo --finetuned reports/finetune-torgo/f2-M02-adapter8-recheck-test-nemo
echo "== bottom-8 rerun against stock"
python scripts/asr/ci.py --manifest $M/test.jsonl --stock reports/finetune-torgo/M02-test-stock-nemo --finetuned reports/finetune-torgo/f2-M02-bottom8-recheck-test-nemo
echo "== bottom-8 rerun against the recorded bottom-8 model"
python scripts/asr/ci.py --manifest $M/test.jsonl --stock reports/finetune-torgo/M02e-A3-test-nemo --finetuned reports/finetune-torgo/f2-M02-bottom8-recheck-test-nemo
git log -1 --format='code at the end: %h %ad %s' --date=format:%H:%M )
```

In every `ci.py` block a positive Delta means the line labelled "Fine-tuned" is better than the line labelled "Stock". In the second block of each pair the line labelled "Stock" is an earlier model of the same kind, so a Delta near 0 means the rerun reproduced it.

### Full Results

The scores are in the Results section. These are the log lines:

| Check | Run | Log lines present | Released model dev WER |
| --- | --- | --- | ---: |
| 1 | M01, seed 17 | `1049600 trainable`, `kept epochs 12, 13, 14, 15`, `Checking the averaged model on dev before release...`, `averaged 4 checkpoints` | 44.44% |
| 1 | M01, seed 18 | the same four | 22.22% |
| 1 | M01, seed 19 | the same four | 38.89% |
| 1 | M04, seed 17 | the same four | 64.00% |
| 1 | M04, seed 18 | the same four | 60.00% |
| 1 | M04, seed 19 | the same four | 60.00% |
| 2 | M02 adapter, default recipe | `1049600 trainable`, `stock dev WER 0.5797`, best 31.88% at `best-epoch=5`, `averaged 4 checkpoints`, stopped after epoch 9 | 33.33% |
| extra | M02 adapter, 15-epoch cap | `1049600 trainable`, `stock dev WER 0.5797`, best 31.88% at `best-epoch=6`, `averaged 4 checkpoints`, stopped after epoch 10 | 33.33% |
| 3 | M02 whole-block bottom-8 | `210461318 trainable`, `stock dev WER 0.5797`, best 44.93% at `best-epoch=3`, `averaged 4 checkpoints`, stopped after epoch 7, `parakeet-ft.nemo` saved (2472.2 MB) | 46.38% |

The commit was 9991cd5 at the start and at the end. Between 9991cd5 and 9a6c1cb (16:15), `train.py`, `nemo_hyp.py`, `block_adapter.py` and `evaluate.py` didn't change. `ci.py` did change after the run (13 lines), so the `ci.py` that scored these runs isn't the latest one.

### Conclusions

1. The cleaned-up `train.py` (commit 9991cd5, hash c696b4873186) passed all three checks. Every score is inside its pass band and every listed log line is there. The adapter recipe reproduced the recorded results exactly at seed 17 on three speakers. The whole-block recipe reproduced the recorded total (42 errors) and dev WER (46.4%), with some errors on different clips.
2. The one piece of code that had never run on NeMo, the dev score of the averaged model after a fixed-schedule or adapter run, ran six times without crashing.
3. The same-recipe seed spread that the M01 and M04 note listed as not shown is now measured: 3 words (1.78 points) on M01 and 6 words (3.73 points) on M04 over three seeds. The adapter beats the F-1 model on every seed of both speakers. The interval excludes zero on every M04 seed and on no M01 seed. The M01 and M04 conclusions don't change.
4. A 15-epoch cap gives a different model from the default recipe. It has the same total of 46 errors, but on different clips, and its loss departs from epoch 2. I didn't show which of the two does better on new clips.
5. What held: M01 about 84 (got 84) and M04 about 79 (got 79). The three-seed means were within 3 points of the first run (+0.99 on M01, 0.00 on M04). Every M04 seed beat the F-1 model, and so did every M01 seed. Released dev WER was near 44% on M01 and in 60 to 68% on M04 (44.44%, 64.00%). Check 2 hit 46 errors with `stock dev WER 0.5797`, a best dev WER of 31.88%, a stop before epoch 10, and an averaged dev WER 1.45 points above the best. Check 3 landed inside 42 to 49 (got 42, the low end). What missed: the M01 spread was 3 words (1.78 points), under the 4 to 7 words predicted. The plan also assumed GPU runs would differ from each other, and the three adapter reruns didn't.
6. Not shown: anything about the app server and pages, the corpus export, the int8 export or the Pi, since this check didn't run them. It also doesn't show whether the whole-block run's different clips come from GPU run-to-run differences or from the code change, or why the 15-epoch run took a different path (the learning-rate explanation wasn't tested). It says nothing about commits after 9991cd5 beyond the four script files named above. These are reruns on test sets that were already looked at, so they check software behavior and aren't independent confirmation of the adapter.

</details>

## Related

- [F-2: M01 and M04, adapter after 8 blocks](../f2-m01-m04-adapter8/notes.md)
- [F-2: M02, adapter after 8 blocks](../f2-m02-adapter8/notes.md)
- [F-2: M02, adapter after 8 blocks, seeds 18 and 19](../f2-m02-adapter8-seeds/notes.md)
- [F-2: M02, bottom 8 blocks, which parts to train](../f2-m02-bottom8-parts/notes.md)
