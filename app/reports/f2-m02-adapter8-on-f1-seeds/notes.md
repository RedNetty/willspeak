# F-2: M02, adapter after 8 blocks on the F-1 model, seeds 17, 18 and 19

Proposal: F-2 (single-speaker fine-tuning, §5.2 / H3). This repeats the [adapter on F-1 test](../f2-m01-m04-adapter8-on-f1/notes.md) on a third speaker, M02, with three seeds. Not a new method.

Status: run 1 Oct 2026, about 19:52 to 20:01. On M02 the stack is level with the adapter on stock. Seeds scored 49, 45 and 42 wrong against 46 on every adapter-on-stock run, and no interval excludes zero. The seeds spread 7 words. Of five pre-run hypotheses, three held and two missed. I recorded no prediction of my own.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

Train the adapter on top of M02's F-1 model three times with different seeds and score it on M02's 37 test clips. On M04 this beat the adapter on stock by about 6 points over three seeds, and on M01 it made no difference. M02 is the third speaker, so this checks whether the F-1 base helps here and how far the seeds spread.

## Setup

The adapter is the small layer after encoder block 7 (`h + W h + b`, 1,049,600 numbers, starting at zero), trained with everything else frozen. The adapter on stock sits on the downloaded model. The adapter on F-1, or the stack, sits on the speaker's F-1 model, the shared model trained on the other speakers that never heard this one.

A seed fixes clip order, speed perturbation and which spectrogram patches get blanked. It doesn't touch the adapter's zero start or the F-1 model (one fixed file per speaker). The dev set is 24 clips on M02, scored each epoch. A word error is a wrong, deleted or inserted word. A gain is baseline WER minus candidate WER, so positive favours the candidate. The interval comes from `ci.py` reshuffling the test clips 2,000 times, and a gain counts as shown only if the whole interval is on one side of zero.

M02's 37 test clips hold 95 reference words, so one word is 1.05 points of WER:

| Model | Word errors (WER) |
| --- | --- |
| Stock, NeMo | 62 (65.26%) |
| F-1 model alone, int8 export | 54 (56.84%); no NeMo score is recorded |
| Whole-block bottom-8 model trained on M02's earlier sessions, NeMo | 42 (44.21%); two earlier whole-block runs scored 43 and 49 |
| Adapter on stock, seeds 17, 18 and 19, NeMo | 46 (48.42%) on every seed |
| Adapter on stock, 15-epoch cap, manual run, NeMo | 46, on different clips |

The F-1 int8 score is from the [severe-speakers note](../f2-severe-bottom8-5epochs/notes.md). Int8 and NeMo can differ by a few words on this CPU (stock got 57 wrong in int8, 62 in NeMo), so I decode the F-1 file in NeMo too.

M02's adapter on stock used the default recipe: dev picks checkpoints, the untrained model is kept as a guard that a checkpoint must beat on dev to be released, and training runs up to 10 epochs. Those runs ended after epoch 9 or 10, best dev epoch 5 to 7. M01 and M04 used a fixed 15-epoch schedule since their dev sets (18 and 25 words) are too small to choose on. Using M02's own recipe means the baseline already exists (46 on every run on record), but M02's stack only compares with M02's adapter on stock, not with the other speakers.

With `--base` set to the F-1 file, the guard is the F-1 model's dev score, logged on the `stock dev WER` line. If nothing beats it, `train.py` releases an all-zeros adapter that scores like the F-1 model. I'd read that path in the code but it had never run with an F-1 base, since the M01 and M04 stack runs had the guard off.

M02 has 156 training clips (9.0 minutes), against 5.7 minutes for M01 and 6.1 for M04.

`ci.py` prints "repeated transcripts are resampled separately" when a prompt repeats among the test clips. The old M02 outputs came from a `ci.py` without that warning, so I didn't know if M02's 37 clips repeat prompts. The script runs each same-seed comparison both ways and counts the warnings.

## Plan

1. Check that the earlier outputs exist and the new ones don't (the script does this first), and report whether the F-1 model already has a NeMo score on these clips.
2. Decode M02's 37 test clips with the F-1 model alone in NeMo if that score doesn't exist yet, and score it with `evaluate.py`.
3. For seeds 17, 18 and 19, train the adapter with `--base` set to M02's F-1 file and M02's own recipe: learning rate 3e-4, batch 2 with 8 accumulated, 10 warmup updates, normalised transcripts, batch norm frozen, dev scoring with `evaluate.py`, guard on, default epoch cap. Decode the test clips with the F-1 file plus the adapter.
4. Score with `evaluate.py`. For each seed, use `ci.py` to compare the stack with stock, with the adapter on stock at the same seed (by clip, and with prompts kept together), with the F-1 model alone and with the whole-block model.
5. Print the nine scores, spreads and means, what each training log released, and a yes or no per prediction.
6. Check in each training log that the base model is the F-1 file.

How I'd read it (written before the run). One word is 1.05 points, so the usual 4-point noise guide is about 4 words. Against 46, a seed (or the mean) is below at 42 or fewer, level at 43 to 49, above at 50 or more. Below would suggest the F-1 base helps M02 too (two of three speakers), level means no effect found (one of three), above means it may cost a little. A best-to-worst spread over 4 words, against the adapter on stock's 0, means one run can't settle a comparison on M02. If a log says nothing beat the F-1 model on dev, that seed is all zeros and its score is just the F-1 model's.

A difference is shown only if its interval is wholly on one side of zero, and if prompt grouping pulls it back over zero the "shown" is withdrawn. The whole-block model's 42 is the best M02 score so far, and I won't claim a seed beats it unless the interval shows it. Three seeds per side is thin and a fourth could move it.

## Why

M04 gained from the F-1 base and M01 didn't, with no clue why. M02 has the most data and the most earlier adapter runs, so it's the cheapest place to check. The device is meant to be a shared model plus a small personal layer, so the next experiments shouldn't rest on one speaker. About 15 minutes of GPU time.

## Limits

This covers M02 on its last session in NeMo: where three stack seeds land against 46, their spread, and the F-1 model's first NeMo score on these clips.

Earlier 4-word (4.21-point) M02 comparisons had intervals about 11 points either side, so a gap under about 8 words probably won't show. Spread from the F-1 model's own training isn't covered (one F-1 file, only the adapter seed changes). Three speakers can't separate speaker, base quality and chance as reasons M04, M01 and M02 differ, and the fixed-schedule M01 and M04 stack runs aren't like-for-like. Nothing here about whole blocks on the F-1 base, other learning rates or positions, the Pi or int8 (the adapter has no export), and 24 dev clips say little. The test clips have been scored many times, and no variant is picked by its test score.

## Reading and Video

None added. See the lists in the [M02 seeds note](../f2-m02-adapter8-seeds/notes.md#reading-and-video).

## Results

Wrong words out of 95 (WER), NeMo, M02's 37 test clips:

| Model | Seed 17 | Seed 18 | Seed 19 | Mean |
| --- | --- | --- | --- | --- |
| Adapter on stock | 46 (48.42%) | 46 (48.42%) | 46 (48.42%) | 46.0 (48.42%) |
| Adapter on F-1 (the stack) | 49 (51.58%) | 45 (47.37%) | 42 (44.21%) | 45.3 (47.72%) |

Stock got 62 (65.26%), the F-1 model alone 52 (54.74%; the int8 export got 54) and the whole-block model 42 (44.21%).

Gains of the stack at the same seed (positive means the stack is better; interval by clip, then with prompts kept together):

| Compared with | Seed 17 | Seed 18 | Seed 19 |
| --- | --- | --- | --- |
| Adapter on stock | -3.16 (-11.63 to 6.41; -12.90 to 6.74) | 1.05 (-11.29 to 14.00; -12.50 to 16.09) | 4.21 (-6.12 to 12.68; -6.90 to 13.27) |
| F-1 model alone | 3.16 (-9.76 to 15.96) | 7.37 (-4.82 to 19.00) | 10.53 (-3.85 to 22.43) |
| Stock | 13.68 (1.58 to 27.71) | 17.89 (3.60 to 34.31) | 21.05 (7.87 to 35.38) |
| Whole-block model | -7.37 (-18.42 to 4.81) | -3.16 (-14.71 to 7.69) | 0.00 (-12.50 to 12.36) |

Against the adapter on stock the stack is level: 49 (level), 45 (level) and 42 (below, exactly on the 4-word line), mean 45.3. Every interval includes zero, with clips or prompts resampled. The seeds are 7 words (7.37 points) apart where the adapter on stock moved 0, so by my pre-run reading one run can't settle a comparison on M02.

`ci.py` printed the repeated-transcripts warning on all 12 clip-level comparisons, and the grouped ones used 35 prompt units for 37 clips. Grouped intervals are a bit wider, no verdict changed.

The F-1 model's dev WER is 44.93% (stock 57.97%). All three logs released a fine-tuned adapter (released dev WER 33.33%, 40.58%, 30.43%), with best dev epoch 1 for seeds 17 and 18 and epoch 4 for seed 19.

The stack beats the F-1 model alone on all three seeds (3, 7 and 10 words) but it's shown on none. It beats stock on all three, shown on all. Seed 19 ties the whole-block model's 42 and seed 17 is 7 behind.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Pre-run Hypotheses

Recorded on 1 Oct 2026 before any run, at 19:37. Results are scored against them as written. No prediction of my own was recorded. Anchors are the M02 scores in the Setup table.

1. Each stack seed lands at 43 to 49 wrong, level with 46, mean about 46. F-1 helped stock least on M02: on the F-1 test sets stock went from 59.52% to 52.86% on M02, 70.05% to 59.42% on M01 and 91.63% to 70.44% on M04, and the stack's effect followed that order (none on M01, about 6 points on M04). If the base helps in proportion, M02 gets close to nothing, though three speakers can't establish that. The adapter on stock also got exactly 46 on every M02 run on record (three seeds, the seed 17 rerun, the 15-epoch run), which hints at a floor for this recipe.
2. The F-1 model alone in NeMo gets 51 to 60 wrong. Int8 got 54, and NeMo scored stock 5 words worse than int8 (62 against 57), so a few above 54 is plausible.
3. The three seeds land within 4 words. The adapter on stock moved 0 on M02. The stack moved 7 and 8 on M01 and M04, but those have 169 and 161 test words and about half M02's training clips, and here a 24-clip dev set picks checkpoints.
4. All three logs release a fine-tuned adapter, none the F-1 model unchanged. The adapter on stock took M02's dev WER from 57.97% to 30% to 34%, and F-1 should start not far below stock's, so checkpoints should beat it easily.
5. No same-seed interval against the adapter on stock excludes zero, by clip or prompt. 37 clips, and earlier 4-word gains had intervals about 11 points either side.

1 and 5 together say no difference on M02. 2 checks part of the reasoning in 1 without isolating it. 1 and 3 can both hold, for example seeds at 44, 46 and 47.

### Commands and Setup

Run from `app/` in the WSL tree on Yack after `git pull`, GPU free and Windows apps closed. Training and decoding use `~/venvs/willspeak-train`, `evaluate.py` and `ci.py` use `app/.venv`. Data is `data/public/torgo/export/M02`, the same 37 test clips as always. Each training run should take about 3 to 4 minutes (earlier ones took about 3, plus about 25 seconds to load the F-1 file), plus about 25 seconds to load and decode, so about 15 minutes total. The script stops at its first error after the input check.

I dry-ran it on a stub copy of the tree. That checks the shell logic, the arguments (right models paired, the F-1 file passed as `--base` and `--nemo`), the parsing of `ci.py` output and the summary on made-up numbers, but nothing about results. The input check and the no-overwrite refusal were both exercised.

The script, saved as `logs/adapter-on-f1-m02.sh` and run with its output copied to `logs/adapter-on-f1-m02-full-<date and time>.txt`:

```bash
cd ~/willspeak/app
EX=data/public/torgo/export
FT=reports/finetune-torgo
SC=.venv/bin/python
M=$EX/M02
F1=models/parakeet-ft-f1-loso-M02-bottom8/parakeet-ft.nemo
F1R=f1-loso-M02-bottom8-on-F2-test-nemo
CIALL=$(mktemp)
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
echo "code: $(git log -1 --format='%h %ad' --date=format:%H:%M)"
echo "train.py sha256: $(sha256sum scripts/asr/train.py | cut -c1-12)   (the file the cleanup check passed: c696b4873186)"
for p in $F1 $M/test.jsonl $FT/M02-test-stock-nemo/results.json $FT/M02e-A3-test-nemo/results.json $FT/f2-M02-adapter8-test-nemo/results.json $FT/f2-M02-adapter8-seed18-test-nemo/results.json $FT/f2-M02-adapter8-seed19-test-nemo/results.json; do
  [ -e "$p" ] || { echo "MISSING $p, stopping"; exit 1; }
done
for SEED in 17 18 19; do
  for p in models/parakeet-ft-f2-M02-adapter8-on-f1-seed$SEED $FT/f2-M02-adapter8-on-f1-seed$SEED-test-nemo; do
    [ -e "$p" ] && { echo "ALREADY EXISTS $p, stopping"; exit 1; }
  done
done
echo "inputs ok"
if [ -e $FT/$F1R/results.json ]; then echo "the F-1 model's score on these clips: found, reusing $FT/$F1R"; else echo "the F-1 model's score on these clips: not found, it is decoded and scored below"; fi
ci() {
  echo "== $1" | tee -a $CIALL
  $SC scripts/asr/ci.py --manifest $M/test.jsonl --stock $FT/$2 --finetuned $FT/$3 $4 > $CIALL.one 2>&1 || { cat $CIALL.one; echo "ci.py failed, stopping"; exit 1; }
  cat $CIALL.one | tee -a $CIALL
}
source ~/venvs/willspeak-train/bin/activate
set -e
if [ ! -e $FT/$F1R/results.json ]; then
  echo; echo "######## the F-1 model alone on M02's test clips"
  python scripts/asr/nemo_hyp.py --nemo $F1 --manifest $M/test.jsonl --audio-root $M --out $FT/$F1R.jsonl
fi
for SEED in 17 18 19; do
  N=f2-M02-adapter8-on-f1-seed$SEED
  echo; echo "######## M02 seed $SEED, started $(date +%H:%M:%S)"
  python scripts/asr/train.py --export $M --base $F1 --out models/parakeet-ft-$N --adapter-after-blocks 8 --lr 3e-4 --batch 2 --grad-accum 8 --warmup-steps 10 --freeze-batchnorm --normalize-targets --dev-scoring evaluate --keep-stock --seed $SEED
  f=$(ls -t logs/train_parakeet-*.log | head -1)
  echo "-- log lines from $f"
  grep -hE "Base model|loading|parameters exact|stock dev WER|best dev WER|no checkpoint beat|did not beat base|stopped|Checking the averaged|averaged|Base dev WER|Best checkpoint dev|Released|Elapsed|saved .*adapter8-on-f1" "$f" || true
  python scripts/asr/nemo_hyp.py --nemo $F1 --adapter models/parakeet-ft-$N/adapter.pt --manifest $M/test.jsonl --audio-root $M --out $FT/$N-test-nemo.jsonl
done
deactivate
if [ ! -e $FT/$F1R/results.json ]; then
  echo; echo "######## scoring the F-1 model alone"
  $SC scripts/asr/evaluate.py --manifest $M/test.jsonl --audio-root $M --hyp $FT/$F1R.jsonl --group-by kind --out $FT/$F1R
fi
for SEED in 17 18 19; do
  R=f2-M02-adapter8-on-f1-seed$SEED-test-nemo
  if [ $SEED = 17 ]; then A=f2-M02-adapter8-test-nemo; else A=f2-M02-adapter8-seed$SEED-test-nemo; fi
  echo; echo "######## scoring M02 seed $SEED"
  $SC scripts/asr/evaluate.py --manifest $M/test.jsonl --audio-root $M --hyp $FT/$R.jsonl --group-by kind --out $FT/$R
  ci "M02 stack seed $SEED against stock" M02-test-stock-nemo $R
  ci "M02 stack seed $SEED against the adapter on stock, seed $SEED" $A $R
  ci "M02 stack seed $SEED against the adapter on stock, seed $SEED, resampled by prompt" $A $R "--cluster-by text"
  ci "M02 stack seed $SEED against the F-1 model" $F1R $R
  ci "M02 stack seed $SEED against the whole-block model" M02e-A3-test-nemo $R
done
echo; echo "######## summary"
CIALL=$CIALL python3 - <<'PY'
import json, os, re
ft = "reports/finetune-torgo/"
W = 95
pt = 100.0 / W
def get(n):
    r = json.load(open(ft + n + "/results.json")); w = r["word"]
    return r["wer"] * 100, w["sub"] + w["dele"] + w["ins"]
rows = [("stock", "M02-test-stock-nemo"),
        ("F-1 model alone", "f1-loso-M02-bottom8-on-F2-test-nemo"),
        ("whole-block model (A3)", "M02e-A3-test-nemo"),
        ("adapter on stock, seed 17", "f2-M02-adapter8-test-nemo"),
        ("adapter on stock, seed 18", "f2-M02-adapter8-seed18-test-nemo"),
        ("adapter on stock, seed 19", "f2-M02-adapter8-seed19-test-nemo"),
        ("adapter on F-1, seed 17", "f2-M02-adapter8-on-f1-seed17-test-nemo"),
        ("adapter on F-1, seed 18", "f2-M02-adapter8-on-f1-seed18-test-nemo"),
        ("adapter on F-1, seed 19", "f2-M02-adapter8-on-f1-seed19-test-nemo")]
print("== M02 (%d words, one word is %.2f points)" % (W, pt))
res = {}
for label, n in rows:
    wer, e = get(n)
    res[label] = e
    print("%-28s %6.2f%% %4d wrong" % (label, wer, e))
st = [res["adapter on F-1, seed %d" % k] for k in (17, 18, 19)]
ad = [res["adapter on stock, seed %d" % k] for k in (17, 18, 19)]
print("adapter on F-1 spread: %d words, %.2f points; adapter on stock spread: %d words, %.2f points" % (max(st) - min(st), (max(st) - min(st)) * pt, max(ad) - min(ad), (max(ad) - min(ad)) * pt))
print("adapter on F-1 mean %.1f wrong (%.2f%%); adapter on stock mean %.1f wrong (%.2f%%); whole-block model %d wrong" % (sum(st) / 3.0, sum(st) / 3.0 * pt, sum(ad) / 3.0, sum(ad) / 3.0 * pt, res["whole-block model (A3)"]))
for k in (17, 18, 19):
    e = res["adapter on F-1, seed %d" % k]
    where = "below (42 or fewer)" if e <= 42 else ("above (50 or more)" if e >= 50 else "level (43 to 49)")
    print("seed %d: %d wrong, %s against the adapter on stock's 46" % (k, e, where))
print("-- what the training logs released")
sel = {}
for k in (17, 18, 19):
    run = json.load(open("models/parakeet-ft-f2-M02-adapter8-on-f1-seed%d/run.json" % k))
    sel[k] = run.get("selected")
    print("seed %d: selected %s, F-1 model's dev WER %s, best dev WER %s, released dev WER %s" % (k, run.get("selected"), run.get("stock_dev_wer"), run.get("best_dev_wer"), run.get("released_dev_wer")))
print("-- same-seed comparisons against the adapter on stock (positive: the stack is better)")
blocks = {}
cur = None
for line in open(os.environ["CIALL"], encoding="utf-8", errors="replace"):
    line = line.rstrip("\n")
    if line.startswith("== "):
        cur = line[3:]
        blocks[cur] = None
        continue
    m = re.search(r"(?:Gain \(baseline - candidate\)|Delta)\s+(-?[\d.]+)\s+points\s+\(95% CI\s+(-?[\d.]+)\s+to\s+(-?[\d.]+)\)", line)
    if m and cur is not None:
        blocks[cur] = tuple(float(x) for x in m.groups())
warned = sum(1 for l in open(os.environ["CIALL"], encoding="utf-8", errors="replace") if l.startswith("Warning: repeated transcripts"))
print("the repeated-transcripts warning was printed in %d of the 12 comparisons resampled by clip" % warned)
excluded = 0
for label, v in blocks.items():
    if "against the adapter on stock" not in label:
        continue
    if v is None:
        print("%s: no Delta line found" % label)
        continue
    d, lo, hi = v
    ex = not (lo <= 0.0 <= hi)
    excluded += 1 if ex else 0
    print("%s: %.2f (%.2f to %.2f), %s zero" % (label.replace("M02 stack ", ""), d, lo, hi, "excludes" if ex else "includes"))
print("-- the five predictions")
print("prediction 1, all three stack seeds between 43 and 49 wrong: %s" % ("yes" if all(43 <= e <= 49 for e in st) else "no"))
f1 = res["F-1 model alone"]
print("prediction 2, the F-1 model alone between 51 and 60 wrong (it got %d): %s" % (f1, "yes" if 51 <= f1 <= 60 else "no"))
print("prediction 3, the three stack seeds within 4 words of each other: %s" % ("yes" if max(st) - min(st) <= 4 else "no"))
print("prediction 4, all three released models fine-tuned, none the F-1 model unchanged: %s" % ("yes" if all(sel[k] and sel[k].startswith("fine-tuned") for k in sel) else "no"))
print("prediction 5, no same-seed interval against the adapter on stock excludes zero (clip or prompt): %s (%d of 6 exclude)" % ("yes" if excluded == 0 else "no", excluded))
PY
rm -f $CIALL $CIALL.one
echo "code at the end: $(git log -1 --format='%h %ad' --date=format:%H:%M)"
echo "finished $(date '+%Y-%m-%d %H:%M:%S')"
```

Before trusting the scores, check each training log for a `Base model` line and a `loading` line naming `models/parakeet-ft-f1-loso-M02-bottom8/parakeet-ft.nemo` (if either says `nvidia/parakeet-tdt-0.6b-v2`, the run used stock and isn't this test), `1049600 trainable`, a `stock dev WER` line (here the F-1 model's dev score; the adapter on stock's was 0.5797), a `best dev WER` line, `averaged 4 checkpoints`, a `Released model dev WER` that's a number, and a saved `adapter.pt`. A line saying no checkpoint beat the F-1 model, or that the averaged model didn't, goes in the notes. The decode should print `Added the adapter ... after encoder blocks 0-7` and 37 clips. The `train.py` hash on the third output line should be c696b4873186, the file the cleanup check passed, and a different hash means the code changed and should be recorded. In the `ci.py` blocks a positive gain means the second model (the stack) is better, and in the F-1 and whole-block blocks that model is the baseline.

As run, the script finished with no error, from about 19:52 (the time in the saved file's name, `adapter-on-f1-m02-full-1001-1952.txt`) to 20:01:36. The copied console output starts partway through seed 17's training, at its `averaged 4 checkpoints` line. I didn't see the first lines (start time, starting commit, `train.py` hash, input check, whether the F-1 score was found), the F-1 decode, or seed 17's header and epoch lines, though its grep block after training is complete. The commit at the end was 442cc73 (dated 19:11), not the 3c2585e of the M01 and M04 runs, so the code may differ from those runs. Since the `train.py` hash wasn't seen, I don't know whether `train.py` changed. The scoring blocks and the summary are complete.

### Full Results

Log lines seen (seed 17 from its grep block only):

| Line | Seed 17 | Seed 18 | Seed 19 |
| --- | --- | --- | --- |
| Base model, and the `loading` line | `models/parakeet-ft-f1-loso-M02-bottom8/parakeet-ft.nemo` | the same | the same |
| `parameters exact` | 618875526 total, 1049600 trainable | the same | the same |
| `stock dev WER` (the F-1 model's) | 0.4493 | 0.4493 | 0.4493 |
| Best dev WER | 30.43% (epoch 1) | 36.23% (epoch 1) | 27.54% (epoch 4) |
| Training stopped | not seen | after epoch 5 (patience 3) | after epoch 8 |
| Averaging | `averaged 4 checkpoints`, then the check on dev | the same | the same |
| Released, and released dev WER | fine-tuned, 33.33% | fine-tuned, 40.58% | fine-tuned, 30.43% |
| Elapsed | 1m 38s | 1m 38s | 2m 24s |
| Decode | adapter after encoder blocks 0-7; 37 clips | the same | the same |

Training audio was 156 clips (8.96 minutes) and the dev set 24 clips (1.57 minutes).

The summary, as printed:

```text
== M02 (95 words, one word is 1.05 points)
stock                         65.26%   62 wrong
F-1 model alone               54.74%   52 wrong
whole-block model (A3)        44.21%   42 wrong
adapter on stock, seed 17     48.42%   46 wrong
adapter on stock, seed 18     48.42%   46 wrong
adapter on stock, seed 19     48.42%   46 wrong
adapter on F-1, seed 17       51.58%   49 wrong
adapter on F-1, seed 18       47.37%   45 wrong
adapter on F-1, seed 19       44.21%   42 wrong
adapter on F-1 spread: 7 words, 7.37 points; adapter on stock spread: 0 words, 0.00 points
adapter on F-1 mean 45.3 wrong (47.72%); adapter on stock mean 46.0 wrong (48.42%); whole-block model 42 wrong
seed 17: 49 wrong, level (43 to 49) against the adapter on stock's 46
seed 18: 45 wrong, level (43 to 49) against the adapter on stock's 46
seed 19: 42 wrong, below (42 or fewer) against the adapter on stock's 46
-- what the training logs released
seed 17: selected fine-tuned, F-1 model's dev WER 0.4492753744125366, best dev WER 0.30434781312942505, released dev WER 0.3333333432674408
seed 18: selected fine-tuned, F-1 model's dev WER 0.4492753744125366, best dev WER 0.36231884360313416, released dev WER 0.4057970941066742
seed 19: selected fine-tuned, F-1 model's dev WER 0.4492753744125366, best dev WER 0.2753623127937317, released dev WER 0.30434781312942505
-- same-seed comparisons against the adapter on stock (positive: the stack is better)
the repeated-transcripts warning was printed in 12 of the 12 comparisons resampled by clip
seed 17 against the adapter on stock, seed 17: -3.16 (-11.63 to 6.41), includes zero
seed 17 against the adapter on stock, seed 17, resampled by prompt: -3.16 (-12.90 to 6.74), includes zero
seed 18 against the adapter on stock, seed 18: 1.05 (-11.29 to 14.00), includes zero
seed 18 against the adapter on stock, seed 18, resampled by prompt: 1.05 (-12.50 to 16.09), includes zero
seed 19 against the adapter on stock, seed 19: 4.21 (-6.12 to 12.68), includes zero
seed 19 against the adapter on stock, seed 19, resampled by prompt: 4.21 (-6.90 to 13.27), includes zero
-- the five predictions
prediction 1, all three stack seeds between 43 and 49 wrong: no
prediction 2, the F-1 model alone between 51 and 60 wrong (it got 52): yes
prediction 3, the three stack seeds within 4 words of each other: no
prediction 4, all three released models fine-tuned, none the F-1 model unchanged: yes
prediction 5, no same-seed interval against the adapter on stock excludes zero (clip or prompt): yes (0 of 6 exclude)
code at the end: 442cc73 19:11
finished 2026-10-01 20:01:36
```

Each `ci.py` block started with `Paired percentile bootstrap` (37 clip units, or 35 text units when grouped), `2000 resamples, seed 0`, and on the clip-level blocks the repeated-transcripts warning. The gains and intervals are in the tables under Results.

### Conclusions

No effect of the F-1 base shows on M02. Seeds got 49, 45 and 42 against 46 (mean 45.3 against 46.0), same-seed gains were -3.16, 1.05 and 4.21 points, and every interval includes zero by clip or prompt.

The seeds spread 7 words (7.37 points) where the adapter on stock moved 0. The stack's spread over the three speakers is 4.14 (M01), 4.97 (M04) and 7.37 (M02), always above the adapter on stock's, and that's without the F-1 model's own training spread.

The stack beats stock on all three seeds, shown on all (lower ends 1.58, 3.60 and 7.87), and beats the F-1 model alone on all three, shown on none. It's level with the whole-block model: seed 19 ties at 42, seeds 18 and 17 are 3 and 7 behind, no interval shows a difference.

Across speakers, the F-1 base helped the adapter on M04 (about 6 points) and nothing shows on M01 or M02. The F-1 model's gain over stock goes in the same order (M04 21 points, M01 11, M02 7), matching the pre-run hypothesis, but three points can't test that and M01 and M02 can't be told apart.

Dev ranked seed 19 best, which test confirmed, but put 17 above 18 where test did the reverse. Seeds 17 and 18 peaked on dev at epoch 1, so which early checkpoints get averaged matters, and 24 dev clips is thin for picking. Prompt grouping changed no verdict (35 distinct prompts in 37 clips).

Hypotheses: 1 missed, since seed 19's 42 was one word outside 43 to 49, though the 45.3 mean is "about 46". 2 held (52). 3 missed (7 words, not 4 or fewer). 4 held. 5 held (0 of 6 intervals exclude zero). The "level" reading was right and the spread guess too low.

The three-seed mean (47.72%) is about 20 points above the high-20s goal. Not shown: the F-1 model's own spread, why speakers differ, a comparison with the fixed-schedule M01 and M04 stack runs, whole blocks on an F-1 base, other learning rates or positions, the Pi, and whatever was in the first lines of the log I didn't see.

Possible next tests, none chosen: new seeds for the M02 whole-block model (spread unmeasured, earlier runs ranged 42 to 49, as wide as the stack's); a fourth and fifth stack seed; whole blocks retrained on an F-1 base, which needs a learning rate and save path decided; F-3, waiting on equal epochs versus equal updates; F-4 on the Pi.

</details>

## Related

[F-2: M01 and M04, adapter after 8 blocks, on the F-1 model](../f2-m01-m04-adapter8-on-f1/notes.md), [F-2: M01 and M04, adapter on the F-1 model, seeds 18 and 19](../f2-m01-m04-adapter8-on-f1-seeds/notes.md), [F-2: M02, adapter after 8 blocks](../f2-m02-adapter8/notes.md), [F-2: M02, adapter after 8 blocks, seeds 18 and 19](../f2-m02-adapter8-seeds/notes.md), [F-2: M02, bottom 8 blocks, which parts to train](../f2-m02-bottom8-parts/notes.md), [F-1: leave-one-speaker-out, bottom 8 blocks](../f1-loso-bottom8/notes.md), [F-2: severe speakers, bottom 8 blocks, fixed 5 epochs](../f2-severe-bottom8-5epochs/notes.md), [Check: cleaned-up train.py](../check-train-py-cleanup/notes.md)
