# F-2: severe speakers, bottom 8 blocks, fixed 5 epochs

Proposal: §5.2 / H3

Status: F-2 recorded 29 Sep, F-1 comparison 30 Sep 2026, interpretation added 21 October 2026. See also the [Report Context](../Report%20Context.md) and the [Measurement guide](../../../docs/measurements.md).

## Idea

Try the bottom-eight recipe on every severe speaker, then compare each personal model with F-1 on the same later-session clips.

## Setup

F-2 speakers are M01, M02 and M04, with 5.7, 9.0 and 6.1 training minutes. F01 has one session and is context only. The dev sets are tiny, so the schedule is fixed: five epochs, averaging the last four checkpoints.

## Plan

Compare stock, F-2 NeMo and F-2 int8 per speaker, then score each bottom-eight F-1 model on that speaker's F-2 test set for a paired H3 comparison.

## Why

The proposal asks whether one person's data beats other people's on his future sessions. The release rule was fixed before the runs.

## Limits

This compares training choices at 5.7–9 min on the later sessions. F-1 wins for M01 and M04, and M02 is inconclusive (that isn't proof of a tie). It can't separate limited data from limited training steps, prove H3 false given enough data, or put F01 in a session-disjoint F-2.

## Results

| Speaker | F-2 test clips | Stock int8 | F-1 bottom-8 int8 | F-2 fixed-rule int8 | F-2 gain over F-1, points (95% CI) |
|---|---:|---:|---:|---:|---|
| M01 | 65 | 70.41% | 58.58% | 72.78% | −14.20 (−22.93 to −6.84) |
| M02 | 37 | 60.00% | 56.84% | 51.58% | 5.26 (−8.42 to 16.30) |
| M04 | 64 | 88.20% | 65.22% | 83.23% | −18.01 (−31.34 to −6.41) |

Positive favors F-2. These are F-2 test sets; the F-1 note uses different LOSO ones.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

Does the bottom-eight recipe that worked on M02 (the bottom-8 model got Int8 44.21% against stock 60.00%) help the other severe speakers too? This fills the F-2 column: each speaker's own earlier sessions in, the last session out, prompt-disjoint. H3 itself (single-speaker beats multi-speaker) is F-2 against F-1 on the same clips, in the section after the result.

### The rule for picking the released model (written before any of these runs)

The bottom-8 model was picked on M02's dev set (24 clips). The other speakers' dev sets are much smaller. M01's is about 4 clips, too few to choose a checkpoint on, because one word decides the "best" epoch and --keep-stock could release stock by chance. So in this set of runs dev chooses nothing, for any speaker. Train exactly 5 epochs, average the last 4 (epochs 1-4) and release that model (`--epochs 5 --select last --no-keep-stock`).

The 5 and 4 come from the bottom-8 model's own run. Its best epoch was epoch 4 (the fifth pass over the data), and it averaged 4 checkpoints. Dev WER is still printed every epoch, just for the record.

The same rule runs on M02 too (f2-M02-bottom8-5epochs), so the F-2 column is one recipe for every speaker. M02 at fixed 5 epochs against the bottom-8 model also shows what the fixed rule costs or gains compared with picking on dev. GPU runs aren't bit-for-bit repeatable, so a difference of a few points there is noise.

Everything else is the bottom-8 model's recipe: blocks 0-7 whole (--train-parts all, 210,461,318 trainable), frozen batch norm, normalised transcripts, batch 2 x 8, warm-up 10, lr 2e-5, seed 17.

F01 has one session, so it isn't an F-2 speaker. The proposal (4.1) gives the single-speaker runs to the severe speakers with two or more sessions. F01 is in the commands below, but its row is context only, outside the F-2 column and H3.

Prediction (Int8 score, before the run):

- M02 (fixed 5 epochs): 44-50%, about the same as the bottom-8 model (44.21%) or a few points worse; the interval against the bottom-8 model includes zero. Same recipe, only the checkpoint choice differs, and single runs move a few points on their own.
- M01: stock 65-75%, F-2 58-66%. Better than stock, interval probably includes zero. Only 5.7 min of training (about 22 optimizer steps in 5 epochs against M02's ~50), so a smaller gain than M02.
- M04: stock 85-95%, F-2 70-80%. Better than stock, interval clears zero. Most room to improve (stock is mostly insertions and wrong single words), and the first F-1 run gave M04 the biggest gain of any speaker (17 points).
- F01: stock 73.91% (the same 23 test clips as the export check), F-2 62-70%. Better than stock, interval includes zero, given the least training data (3.4 min) and only 23 test clips. Same-session test, so easier than the others.

I would be surprised if: any speaker gets worse than stock, as from previous tests they did not.

### Split sizes (from the --check lines below, before training)

| speaker | train clips (min) | dev clips | test clips |
| --- | --- | --- | --- |
| M02 | 156 (9.0) | 24 | 37 |
| M01 | 72 (5.7) | 4 | 65 |
| M04 | 82 (6.1) | 9 | 64 |
| F01 | 85 (3.4) | 10 | 23 |


### Commands

Run from app/ in WSL, after pulling the train.py that has --select. Each block stops at its first error.

```bash
( set -e
source ~/venvs/willspeak-train/bin/activate
for S in M01 M04 F01; do python scripts/asr/train.py --export data/public/torgo/export/$S --check; done
for S in M02 M01 M04 F01; do
  python scripts/asr/train.py --export data/public/torgo/export/$S --out models/parakeet-ft-f2-$S-bottom8-5epochs --train-parts all --epochs 5 --select last --no-keep-stock --batch 2 --grad-accum 8 --warmup-steps 10 --freeze-batchnorm --normalize-targets --dev-scoring evaluate
  python scripts/asr/nemo_hyp.py --nemo models/parakeet-ft-f2-$S-bottom8-5epochs/parakeet-ft.nemo --manifest data/public/torgo/export/$S/test.jsonl --audio-root data/public/torgo/export/$S --out reports/finetune-torgo/f2-$S-bottom8-5epochs-test-nemo.jsonl
  python scripts/asr/export.py --nemo models/parakeet-ft-f2-$S-bottom8-5epochs/parakeet-ft.nemo --out models/parakeet-ft-f2-$S-bottom8-5epochs/onnx-int8
done
for S in M01 M04 F01; do
  python scripts/asr/nemo_hyp.py --stock --manifest data/public/torgo/export/$S/test.jsonl --audio-root data/public/torgo/export/$S --out reports/finetune-torgo/$S-test-stock-nemo.jsonl
done )

( set -e
source .venv/bin/activate
for S in M01 M04 F01; do
  python scripts/asr/evaluate.py --manifest data/public/torgo/export/$S/test.jsonl --audio-root data/public/torgo/export/$S --group-by kind --out reports/finetune-torgo/$S-stock
  python scripts/asr/evaluate.py --manifest data/public/torgo/export/$S/test.jsonl --audio-root data/public/torgo/export/$S --hyp reports/finetune-torgo/$S-test-stock-nemo.jsonl --group-by kind --out reports/finetune-torgo/$S-test-stock-nemo
done
for S in M02 M01 M04 F01; do
  python scripts/asr/evaluate.py --manifest data/public/torgo/export/$S/test.jsonl --audio-root data/public/torgo/export/$S --hyp reports/finetune-torgo/f2-$S-bottom8-5epochs-test-nemo.jsonl --group-by kind --out reports/finetune-torgo/f2-$S-bottom8-5epochs-test-nemo
  python scripts/asr/evaluate.py --manifest data/public/torgo/export/$S/test.jsonl --audio-root data/public/torgo/export/$S --group-by kind --model-dir models/parakeet-ft-f2-$S-bottom8-5epochs/onnx-int8 --out reports/finetune-torgo/f2-$S-bottom8-5epochs-test-int8
  python scripts/asr/ci.py --manifest data/public/torgo/export/$S/test.jsonl --stock reports/finetune-torgo/$S-test-stock-nemo --finetuned reports/finetune-torgo/f2-$S-bottom8-5epochs-test-nemo
  python scripts/asr/ci.py --manifest data/public/torgo/export/$S/test.jsonl --stock reports/finetune-torgo/$S-stock --finetuned reports/finetune-torgo/f2-$S-bottom8-5epochs-test-int8
done
python scripts/asr/ci.py --manifest data/public/torgo/export/M02/test.jsonl --stock reports/finetune-torgo/f2-M02-bottom8-5epochs-test-int8 --finetuned reports/finetune-torgo/M02e-A3-test-int8 )
```

Check each training log before trusting it. It should say "only norm_feed_forward1, ... in encoder blocks 0-7 trainable", "210461318 trainable", "fixed schedule (--select last): kept epochs 1, 2, 3, 4" and "averaged 4 checkpoints". ci.py overwrites ci.json in the fine-tuned folder, so copy each printed line here. The last ci.py line is M02 (fixed 5 epochs) against the bottom-8 model, where a positive Delta means the bottom-8 model (picked on dev) is better.

### Results

Run 29 Sep 2026 on Yack. A positive Delta means F-2 is better than stock.

| speaker | stock NeMo | F-2 NeMo | stock Int8 | F-2 Int8 | Int8 Delta (95% CI) | deleted words, stock / F-2 (Int8) |
| --- | --- | --- | --- | --- | --- | --- |
| M02 | 65.26% | 51.58% | 60.00% | 51.58% | 8.42 (2.59 to 16.09) | 11 / 8 |
| M01 | 77.51% | 74.56% | 70.41% | 72.78% | -2.37 (-4.78 to -0.52) | 8 / 7 |
| M04 | 80.75% | 82.61% | 88.20% | 83.23% | 4.97 (-3.33 to 13.64) | 31 / 21 |
| F01* | 73.91% | 71.74% | 73.91% | 73.91% | 0.00 (-4.17 to 8.70) | 7 / 7 |

The star on F01 means it isn't F-2 (one session, proposal 4.1), so it's context only. The rest of the numbers, with single words / sentences as error rates and words as wrong / deleted / inserted:

| speaker | NeMo Delta (95% CI) | single words / sentences, Int8 stock | single words / sentences, Int8 F-2 | words, Int8, stock to F-2 |
| --- | --- | --- | --- | --- |
| M02 | 13.68 (5.26 to 24.47) | 107.4% / 41.2% | 88.9% / 36.8% | 57 errors, 11 deleted to 36/8/5 of 95 |
| M01 | 2.96 (0.00 to 6.82) | 108.5% / 55.7% | 112.8% / 57.4% | 88/8/23 to 90/7/26 of 169 |
| M04 | -1.86 (-21.62 to 14.39) | 112.5% / 77.9% | 104.2% / 74.3% | 73/31/38 to 71/21/42 of 161 |
| F01 | 2.17 (-8.70 to 6.98) | 110.0% / 46.2% | 105.0% / 50.0% | 23/7/4 to 23/7/4 of 46 |

M02 (fixed 5 epochs) against the bottom-8 model (Int8): Delta 7.37 (95% CI -0.96 to 17.72). The bottom-8 model (picked on dev) is better, but the interval includes zero.

What the numbers show, before interpreting them. M02 gains under the fixed rule, and the interval clears zero in both NeMo and Int8. M01 doesn't gain. It's 2.96 better in NeMo with the interval touching zero, and 2.37 worse in Int8 with the interval clearing zero on the worse side. M04 shows no measurable change in either direction, and F01 (context only) shows no change.

A limit of the rule that I found after the run: "5 epochs" was taken from M02, and M01 and M04 have about half of M02's clips, so 5 epochs gave them about half the training steps. M01 gets 72 clips / (2 x 8) = about 4.5 steps per epoch, about 22 steps in all, against about 50 for M02, and the first 10 steps are warm-up. This run can't separate "too little data" from "too little training". The numbers above stay the F-2 result, because the rule was written before the run. Any run that changes the rule now is a follow-up and is labelled that way (see F-3: M02's 5-minute subset is about M01's and M04's size).

### Checks on the runs (before trusting the table)

This is the first use of `--select last` with NeMo. Run from app/ in WSL:

```bash
for f in $(grep -l -- "fixed schedule (--select last)" logs/train_parakeet-20260929-*.log); do echo "== $f"; grep -hE "train [0-9]+ clips|parameters exact|kept epochs|averaged|saved .*parakeet-ft-" "$f"; done
python3 -c "import json;[print(n,'wrong',r['word']['sub'],'deleted',r['word']['dele'],'inserted',r['word']['ins'],'of',r['word']['n_ref'],{g:round(v['rate']*100,1) for g,v in r['by_group'].items()}) for n in ['f2-M02-bottom8-5epochs-test-int8','M01-stock','f2-M01-bottom8-5epochs-test-int8','M04-stock','f2-M04-bottom8-5epochs-test-int8','F01-stock','f2-F01-bottom8-5epochs-test-int8'] for r in [json.load(open(f'reports/finetune-torgo/{n}/results.json'))]]"
```

Each of the four logs should show 210461318 trainable, "kept epochs 1, 2, 3, 4", "averaged 4 checkpoints" and a saved model. The second line fills in the deleted-word and single-word / sentence numbers above.

Checks (29 Sep): all four logs show all of that, so --select last did what it should and the runs are sound.

### F-2 against F-1 on the same clips (H3) - after the F-1 rerun

The proposal compares F-2 with T-1 and F-1 for each speaker. T-1 is stock on these clips (above). F-1 is the F-1 rerun's model for that speaker, the one that never heard him, scored on his F-2 test clips. The first lines of the block rename folders made before the naming change (the F-1 rerun's `loso-<SPK>-b` and this run's earlier names) to the names used in these notes. The F-1 rerun ([bottom-eight notes](../f1-loso-bottom8/notes.md)) retrains F-1 with this same bottom-eight recipe, so both sides train the bottom eight and the only difference is one speaker against many.

ci.py calls the first folder "Stock", but here it's the F-1 model. A positive Delta means F-2 is better.

```bash
( set -e
source .venv/bin/activate
for S in F01 M01 M02 M04; do
  if [ -d models/parakeet-ft-loso-$S-b ]; then mv models/parakeet-ft-loso-$S-b models/parakeet-ft-f1-loso-$S-bottom8; fi
  if [ -d reports/finetune-torgo/loso-$S-b ]; then mv reports/finetune-torgo/loso-$S-b reports/finetune-torgo/f1-loso-$S-bottom8; fi
  if [ -d models/parakeet-ft-$S-c3 ]; then mv models/parakeet-ft-$S-c3 models/parakeet-ft-f2-$S-bottom8-5epochs; fi
  for T in test-nemo test-nemo.jsonl test-int8; do
    if [ -e reports/finetune-torgo/$S-c3-$T ]; then mv reports/finetune-torgo/$S-c3-$T reports/finetune-torgo/f2-$S-bottom8-5epochs-$T; fi
  done
done
for S in M01 M02 M04; do
  python scripts/asr/evaluate.py --manifest data/public/torgo/export/$S/test.jsonl --audio-root data/public/torgo/export/$S --group-by kind --model-dir models/parakeet-ft-f1-loso-$S-bottom8/onnx-int8 --out reports/finetune-torgo/f1-loso-$S-bottom8-on-F2-test
  python scripts/asr/ci.py --manifest data/public/torgo/export/$S/test.jsonl --stock reports/finetune-torgo/f1-loso-$S-bottom8-on-F2-test --finetuned reports/finetune-torgo/f2-$S-bottom8-5epochs-test-int8
done )
```

| speaker | T-1 (stock) | F-1 (bottom 8) | F-2 (bottom 8) | F-2 against F-1: Delta (95% CI) |
| --- | --- | --- | --- | --- |
| M01 | 70.41% | 58.58% | 72.78% | -14.20 (-22.93 to -6.84) |
| M02 | 60.00% | 56.84% | 51.58% | 5.26 (-8.42 to 16.30) |
| M04 | 88.20% | 65.22% | 83.23% | -18.01 (-31.34 to -6.41) |

All Int8, on each speaker's F-2 test clips (M01 65, M02 37, M04 64). Run 30 Sep 2026 on Yack.

What the numbers show, before interpreting them. For M01 and M04 the F-1 model (other speakers' speech, about 110 minutes) is better than the F-2 model (his own earlier sessions, about 6 minutes) on his later-session clips, and both intervals clear zero on the F-1 side. For M02, F-2 is 5.26 points better but the interval includes zero. M02 has the most F-2 training data (9 minutes). H3 says single-speaker beats multi-speaker "with enough data". At 5.7-9 minutes of one speaker, trained for a fixed 5 epochs, it doesn't: multi-speaker wins twice and ties once.

### Conclusions

Interpretation added 21 October 2026 from the recorded results above.

Against my written prediction (all Int8 on each speaker's F-2 test clips, lower WER is better), the stock ranges held for M01 (70.41%, predicted 65 to 75%) and M04 (88.20%, predicted 85 to 95%). The F-2 ranges missed for every speaker. M02 got 51.58% against a predicted 44 to 50%, M01 72.78% against 58 to 66%, M04 83.23% against 70 to 80%, and F01 73.91% against 62 to 70%. The surprise I named did happen: M01 got worse than stock in Int8, by 2.37 points, with the interval clearing zero on the worse side.

On the same clips, the F-1 model beat the personal model for M01 (by 14.20 points) and M04 (by 18.01 points), with both intervals clearing zero. M02's 5.26-point lead for F-2 has an interval that includes zero, so it isn't settled either way. At 5.7 to 9 minutes with this recipe, H3 isn't supported.

The limit noted after the run (about 22 training steps for M01 against about 50 for M02) turned out to matter. On the pooled lab set (M01, M02 and M04, 425 test words, NeMo decoder, seeds 17 to 19), the same whole-block training scored 304.7 wrong at 5 epochs, 282.0 at 8, 269.3 at 12 and 256.7 at 20. That's sweep E in `docs/measurements.md`, under Sweeps L, T, E, S, C. Stage G reached 198.7 at 40 epochs and rate 8e-5 (`reports/f2-adapter-levers`). So the fixed 5-epoch rule under-trained the smaller speakers. It didn't show that their own data can't help.

F-1 as a base kept its lead in later work. In lab test 20 the adapter on F-1 beat the adapter on stock on 879 held-out words (265 against 337 wrong, interval clear of zero). Severe-speaker pretraining before the adapter gave the best raw result so far (stage P, 178.0 of 425).

For the expo model, a personal whole-block model trained for 5 epochs isn't a candidate. These results point toward other speakers' data first, then a short personal stage on top.

</details>

## Related

- [F-1: severe speakers, leave-one-speaker-out, bottom 8 blocks](../f1-loso-bottom8/notes.md)
- [F-2: M02, bottom 8 blocks, which parts to train](../f2-m02-bottom8-parts/notes.md)
- [Check: TORGO controls, typical-speech forgetting](../typical-speech-check/notes.md)
