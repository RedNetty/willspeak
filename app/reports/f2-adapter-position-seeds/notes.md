# F-2: adapter block 8 and its neighbours, more seeds, blocks 6 and 10 (M01, M02 and M04)

Proposal F-2 (single-speaker fine-tuning, §5.2 / H3). This is a follow-up to the [adapter position sweep](../f2-adapter-position-sweep/notes.md), not a new method.

Status: ran overnight, 1 Oct 2026 22:36 to 2 Oct 01:06 for part E, 40 runs and no failed job. Block 8 has the lowest three-seed mean (206 wrong of 425; seeds 201, 207, 210), but block 6 is only 9 words behind (215), inside the noise. Blocks 4, 10, 12, 16 and 24 trail by 45 to 88 words. M02's block-8 adapter scored 38, 44 and 42 on seeds 17, 18 and 19, under the default recipe's 46 each time. Five of six pre-run hypotheses held; I wrote no prediction of my own. Part D (phrase rejection) is in the phrase-matching note.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

The sweep ran each position once (seed 17). Block 8 won on all three speakers with 201 wrong of 425 pooled, against 252 for block 4 and 245 to 291 for blocks 12, 16 and 24. M02's block-8 adapter on the fixed schedule got 38 wrong, where M02's default recipe got 46 on every earlier run. I wanted to know if that holds across seeds, and whether blocks 6 or 10, which the sweep skipped, do better.

## Setup

A seed fixes a run's random choices: clip order, speed changes, and which spectrogram patches get blanked. The fixed schedule trains exactly 15 epochs and averages the last 4 checkpoints, so the tiny dev sets pick nothing. The adapter is the 1,049,600-number layer `h + W h + b` after encoder block N, trained alone on the downloaded model. Pooled means wrong words summed over M01 (169 test words), M02 (95) and M04 (161), 425 in all.

In the sweep (full table there) stock pooled 323, block 4 252, block 8 201, block 12 261, block 16 245 and block 24 291. Earlier seed repeats of block 8 moved each speaker by 0 to 8 words. The whole-block model scored 42 on M02.

Part C (about 20 minutes, eight runs) is M02's block-8 adapter on seeds 18 and 19, plus blocks 6 and 10 on all three speakers at seed 17. Part E (about 1.5 hours, forty runs) is seeds 18 and 19 for blocks 4, 6, 8, 10, 12, 16 and 24 on all three speakers (M02 block 8 is in part C). With seed 17 that makes three seeds for each of seven positions. Part E carries on past failures and lists them, and the whole job writes `logs/overnight-<stamp>-summaries.txt` at the end.

This uses the sweep's `train.py` (sha256 032323d84c43, commit f8fb3ff). The seed-17 block-8 runs for M01 and M04 came from older code, so that row mixes the two.

## Plan

1. Check inputs and refuse to overwrite outputs (each script does this first).
2. Part C: train, decode and score the eight runs, then use `ci.py` to compare M02's new seeds with seed 17, the default recipe, the whole-block model and stock, and blocks 6 and 10 with block 8.
3. Part E: train, decode and score the forty runs, then print pooled wrong words per position and seed, the three-seed mean and range, the order, and yes or no per hypothesis.

The rule, set before the run: a position beats another only if its three-seed mean is more than 20 words lower (about twice the pooled seed noise) and it's lower on at least two of three speakers. Nothing gets picked from one seed or one speaker. If M02's seeds 18 and 19 both score 44 or fewer, the fixed schedule beats the default recipe and the dev-picked checkpoints were costing words. If either scores 46 or more, seed 17's 38 was luck. If block 6 or 10 pools under 201, the curve is flatter than the sweep suggested. If block 8's seed range is above 15 words, its 44-word lead over block 16 is only about three times the noise. Some shrinkage is expected anyway, since block 8 was tried first and the other positions got no tuning of learning rate or epochs.

## Why

Every later test puts the adapter at block 8, but the curve rests on one run per position, and the best M02 result (38) is one seed. Three seeds per position cost about 2 hours of GPU time overnight and give the curve a noise estimate.

## Limits

This gives seed ranges, a three-seed ranking of seven positions, a repeat of M02's 38, and a check of blocks 6 and 10. It says nothing about narrower or multiple adapters, per-position learning rates or epochs, or the F-1 base (every run starts from stock). It can't separate the code change between old and new block-8 runs beyond adding seeds on current code, and it can't resolve differences smaller than the seed range. A position adopted for the device still gets checked on clips these models haven't been scored on.

## Reading and Video

None added.

## Results

Pooled wrong words (M01 + M02 + M04, 425 words; stock 323), adapter on stock, fixed 15-epoch schedule. Each cell is the three speakers' counts, then the total. Seed 17 is from the sweep (M01 and M04 block 8 on older code).

| Block | Seed 17 | Seed 18 | Seed 19 | Mean | Range |
| --- | --- | --- | --- | --- | --- |
| 4 | 113 41 98 = 252 | 109 42 104 = 255 | 104 40 111 = 255 | 254.0 | 3 |
| 6 | 90 41 84 = 215 | 77 40 97 = 214 | 80 48 88 = 216 | 215.0 | 2 |
| **8** | 84 38 79 = 201 | 87 44 76 = 207 | 86 42 82 = 210 | **206.0** | 9 |
| 10 | 96 55 98 = 249 | 97 51 108 = 256 | 99 51 99 = 249 | 251.3 | 7 |
| 12 | 100 55 106 = 261 | 99 58 109 = 266 | 96 55 112 = 263 | 263.3 | 5 |
| 16 | 102 49 94 = 245 | 107 50 99 = 256 | 108 52 102 = 262 | 254.3 | 17 |
| 24 | 121 56 114 = 291 | 121 59 112 = 292 | 121 55 122 = 298 | 293.7 | 7 |

Mean wrong words per speaker over the three seeds:

| | Stock | 4 | 6 | 8 | 10 | 12 | 16 | 24 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M01 | 131 | 108.7 | **82.3** | 85.7 | 97.3 | 98.3 | 105.7 | 121.0 |
| M02 | 62 | **41.0** | 43.0 | 41.3 | 52.3 | 56.0 | 50.3 | 56.7 |
| M04 | 130 | 104.3 | 89.7 | **79.0** | 101.7 | 109.0 | 98.3 | 116.0 |

On M02 the block-8 adapter got 38, 44 and 42 wrong of 95 (40.00%, 46.32%, 44.21%; mean 41.3), against 46 for the default recipe and 42 for the whole-block model. Block 24's gain over stock is 25% of block 8's (29.3 against 117.0 words).

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Pre-run Hypotheses

No prediction was recorded, only these hypotheses.

Part C (printed as C1 to C3):

1. M02's block-8 adapter scores 44 or fewer on both seeds 18 and 19. The fixed schedule's 38 should repeat within a few words, and since the default recipe's 46 never moved, M02's dev-picked checkpoints probably cost something.
2. The three fixed-schedule seeds on M02 land within 6 words of each other.
3. Blocks 6 and 10 both pool above block 8's 201. Low confidence, since they're block 8's nearest neighbours and the sweep's 44-word lead was from one run each.

Part E (printed as E1 to E3):

1. Block 8 has the lowest three-seed mean of the seven positions.
2. Block 8's mean is more than 20 words below the next lowest. The sweep's lead was 44 on one seed, and averaging should shrink it.
3. Block 8's three pooled counts land within 15 words. Seeds moved each speaker 0 to 8 words before, and noise adds in quadrature across speakers.

### Commands and Setup

Run from `app/` in the WSL tree on Yack after `git pull`, GPU free, Windows set not to sleep, laptop plugged in. The command block writes four scripts, starts them in the background and returns. `logs/overnight.sh` runs the phrase-matching rejection test first (seconds; see its [note](../phrasebook-matching/notes.md)), then parts C and E, writing `logs/overnight-<stamp>-C.txt`, `-E.txt` and a short `logs/overnight-<stamp>-summaries.txt`.

Part C, saved as `logs/m02-seeds-and-blocks.sh`:

```bash
cd ~/willspeak/app
EX=data/public/torgo/export
FT=reports/finetune-torgo
SC=.venv/bin/python
POS="M01:6 M01:10 M02:6 M02:10 M04:6 M04:10"
SEEDS="18 19"
CIALL=$(mktemp)
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
echo "code: $(git log -1 --format='%h %ad' --date=format:%H:%M)"
echo "train.py sha256: $(sha256sum scripts/asr/train.py | cut -c1-12)   (the 21:46 sweep ran 032323d84c43)"
for S in M01 M02 M04; do
  for p in $EX/$S/train.jsonl $EX/$S/test.jsonl $FT/$S-test-stock-nemo/results.json; do
    [ -e "$p" ] || { echo "MISSING $p, stopping"; exit 1; }
  done
  for N in 4 12; do
    [ -e $FT/f2-$S-adapter$N-pos-test-nemo/results.json ] || { echo "MISSING $FT/f2-$S-adapter$N-pos-test-nemo/results.json, stopping"; exit 1; }
  done
done
for p in $FT/f2-M01-adapter8-test-nemo/results.json $FT/f2-M04-adapter8-test-nemo/results.json $FT/f2-M02-adapter8-pos-test-nemo/results.json $FT/f2-M02-adapter8-test-nemo/results.json $FT/M02e-A3-test-nemo/results.json; do
  [ -e "$p" ] || { echo "MISSING $p, stopping"; exit 1; }
done
for j in $POS; do
  S=${j%%:*}; N=${j##*:}
  for p in models/parakeet-ft-f2-$S-adapter$N-pos $FT/f2-$S-adapter$N-pos-test-nemo; do
    [ -e "$p" ] && { echo "ALREADY EXISTS $p, stopping"; exit 1; }
  done
done
for SEED in $SEEDS; do
  for p in models/parakeet-ft-f2-M02-adapter8-fixed-seed$SEED $FT/f2-M02-adapter8-fixed-seed$SEED-test-nemo; do
    [ -e "$p" ] && { echo "ALREADY EXISTS $p, stopping"; exit 1; }
  done
done
echo "inputs ok"
ci() {
  echo "== $1" | tee -a $CIALL
  $SC scripts/asr/ci.py --manifest $EX/$S/test.jsonl --stock $FT/$2 --finetuned $FT/$3 > $CIALL.one 2>&1 || { cat $CIALL.one; echo "ci.py failed, stopping"; exit 1; }
  cat $CIALL.one | tee -a $CIALL
}
train() {
  # $1 speaker, $2 block, $3 model name, $4 seed
  python scripts/asr/train.py --export $EX/$1 --out models/parakeet-ft-$3 --adapter-after-blocks $2 --lr 3e-4 --batch 2 --grad-accum 8 --warmup-steps 10 --epochs 15 --select last --no-keep-stock --freeze-batchnorm --normalize-targets --dev-scoring evaluate --seed $4
  f=$(ls -t logs/train_parakeet-*.log | head -1)
  echo "-- log lines from $f"
  grep -hE "parameters exact|kept epochs|averaged|Released|Elapsed|saved .*adapter" "$f" || true
  python scripts/asr/nemo_hyp.py --stock --adapter models/parakeet-ft-$3/adapter.pt --manifest $EX/$1/test.jsonl --audio-root $EX/$1 --out $FT/$3-test-nemo.jsonl
}
source ~/venvs/willspeak-train/bin/activate
set -e
for SEED in $SEEDS; do
  echo; echo "######## M02, adapter after 8 blocks, fixed schedule, seed $SEED, started $(date +%H:%M:%S)"
  train M02 8 f2-M02-adapter8-fixed-seed$SEED $SEED
done
for j in $POS; do
  S=${j%%:*}; N=${j##*:}
  echo; echo "######## $S, adapter after $N blocks, started $(date +%H:%M:%S)"
  train $S $N f2-$S-adapter$N-pos 17
done
deactivate
for SEED in $SEEDS; do
  M=$EX/M02; R=f2-M02-adapter8-fixed-seed$SEED-test-nemo
  echo; echo "######## scoring M02, fixed schedule, seed $SEED"
  $SC scripts/asr/evaluate.py --manifest $M/test.jsonl --audio-root $M --hyp $FT/$R.jsonl --group-by kind --out $FT/$R
done
for j in $POS; do
  S=${j%%:*}; N=${j##*:}
  M=$EX/$S; R=f2-$S-adapter$N-pos-test-nemo
  echo; echo "######## scoring $S, adapter after $N blocks"
  $SC scripts/asr/evaluate.py --manifest $M/test.jsonl --audio-root $M --hyp $FT/$R.jsonl --group-by kind --out $FT/$R
done
S=M02
for SEED in $SEEDS; do
  R=f2-M02-adapter8-fixed-seed$SEED-test-nemo
  ci "M02 fixed seed $SEED against fixed seed 17" f2-M02-adapter8-pos-test-nemo $R
  ci "M02 fixed seed $SEED against the default recipe, seed 17" f2-M02-adapter8-test-nemo $R
  ci "M02 fixed seed $SEED against the whole-block model" M02e-A3-test-nemo $R
  ci "M02 fixed seed $SEED against stock" M02-test-stock-nemo $R
done
for S in M01 M02 M04; do
  if [ $S = M02 ]; then P8=f2-M02-adapter8-pos-test-nemo; else P8=f2-$S-adapter8-test-nemo; fi
  for N in 6 10; do
    ci "$S: after $N blocks against after 8 blocks" $P8 f2-$S-adapter$N-pos-test-nemo
  done
done
echo; echo "######## summary"
CIALL=$CIALL python3 - <<'PY'
import json, os, re
ft = "reports/finetune-torgo/"
words = {"M01": 169, "M02": 95, "M04": 161}
def get(n):
    r = json.load(open(ft + n + "/results.json")); w = r["word"]
    return r["wer"] * 100, w["sub"] + w["dele"] + w["ins"]
def name(s, n):
    if n == 8 and s != "M02":
        return "f2-%s-adapter8-test-nemo" % s
    return "f2-%s-adapter%d-pos-test-nemo" % (s, n)
print("== M02 (95 words, one word is 1.05 points)")
rows = [("stock", "M02-test-stock-nemo"), ("whole-block model (A3)", "M02e-A3-test-nemo"),
        ("adapter 8, default recipe, seed 17", "f2-M02-adapter8-test-nemo"),
        ("adapter 8, fixed schedule, seed 17", "f2-M02-adapter8-pos-test-nemo"),
        ("adapter 8, fixed schedule, seed 18", "f2-M02-adapter8-fixed-seed18-test-nemo"),
        ("adapter 8, fixed schedule, seed 19", "f2-M02-adapter8-fixed-seed19-test-nemo")]
res = {}
for label, n in rows:
    wer, e = get(n); res[label] = e
    print("%-36s %6.2f%% %4d wrong" % (label, wer, e))
fx = [res["adapter 8, fixed schedule, seed %d" % k] for k in (17, 18, 19)]
print("fixed schedule: mean %.1f wrong (%.2f%%), spread %d words (%.2f points); the default recipe scored 46 on every run on record" % (sum(fx) / 3.0, sum(fx) / 3.0 * 100 / 95, max(fx) - min(fx), (max(fx) - min(fx)) * 100 / 95))
print("== wrong words per speaker and position (fixed schedule, seed 17, adapter on stock)")
pos = (4, 6, 8, 10, 12)
print("%-8s " % "" + " ".join("%7s" % ("blk %d" % n) for n in pos))
tot = {n: 0 for n in pos}
wrong = {}
for s in words:
    line = "%-8s " % s
    for n in pos:
        wrong[(s, n)] = get(name(s, n))[1]; tot[n] += wrong[(s, n)]
        line += "%7d " % wrong[(s, n)]
    print(line)
print("%-8s " % "pooled" + " ".join("%7d" % tot[n] for n in pos) + "   (of 425 words; stock 323)")
print("pooled WER: " + "; ".join("block %d %.1f%%" % (n, 100.0 * tot[n] / 425) for n in pos))
print("lowest pooled count among these five: block %d" % min(pos, key=lambda n: tot[n]))
print("-- comparisons by clip (positive gain: the second model is better; shown only if the interval excludes zero)")
cur = None
for line in open(os.environ["CIALL"], encoding="utf-8", errors="replace"):
    line = line.rstrip("\n")
    if line.startswith("== "):
        cur = line[3:]; continue
    m = re.search(r"Gain \(baseline - candidate\)\s+(-?[\d.]+)\s+points\s+\(95% CI\s+(-?[\d.]+)\s+to\s+(-?[\d.]+)\)", line)
    if m and cur:
        d, lo, hi = (float(x) for x in m.groups())
        print("%-62s %6.2f (%6.2f to %6.2f) %s" % (cur, d, lo, hi, "excludes zero" if not (lo <= 0.0 <= hi) else "includes zero"))
print("-- the three predictions")
print("C1, both new fixed-schedule seeds at 44 wrong or fewer: %s (seeds 18, 19: %d, %d)" % ("yes" if res["adapter 8, fixed schedule, seed 18"] <= 44 and res["adapter 8, fixed schedule, seed 19"] <= 44 else "no", res["adapter 8, fixed schedule, seed 18"], res["adapter 8, fixed schedule, seed 19"]))
print("C2, the three fixed-schedule seeds within 6 words of each other: %s (spread %d)" % ("yes" if max(fx) - min(fx) <= 6 else "no", max(fx) - min(fx)))
print("C3, blocks 6 and 10 both above block 8's pooled count of %d: %s (block 6 %d, block 10 %d)" % (tot[8], "yes" if tot[6] > tot[8] and tot[10] > tot[8] else "no", tot[6], tot[10]))
PY
rm -f $CIALL $CIALL.one
echo "code at the end: $(git log -1 --format='%h %ad' --date=format:%H:%M)"
echo "finished $(date '+%Y-%m-%d %H:%M:%S')"
```

Part E, saved as `logs/seeds-all-positions.sh`:

```bash
cd ~/willspeak/app
EX=data/public/torgo/export
FT=reports/finetune-torgo
SC=.venv/bin/python
SEEDS="18 19"
JOBS=""
for SEED in $SEEDS; do
  for N in 8 6 10 4 12 16 24; do
    for S in M01 M02 M04; do
      [ $N = 8 ] && [ $S = M02 ] && continue
      JOBS="$JOBS $S:$N:$SEED"
    done
  done
done
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
echo "code: $(git log -1 --format='%h %ad' --date=format:%H:%M)"
echo "train.py sha256: $(sha256sum scripts/asr/train.py | cut -c1-12)   (the 21:46 sweep ran 032323d84c43)"
echo "jobs: $(echo $JOBS | wc -w)"
for S in M01 M02 M04; do
  for p in $EX/$S/train.jsonl $EX/$S/test.jsonl $FT/$S-test-stock-nemo/results.json; do
    [ -e "$p" ] || { echo "MISSING $p, stopping"; exit 1; }
  done
done
for j in $JOBS; do
  S=${j%%:*}; r=${j#*:}; N=${r%%:*}; SEED=${r##*:}
  for p in models/parakeet-ft-f2-$S-adapter$N-pos-seed$SEED $FT/f2-$S-adapter$N-pos-seed$SEED-test-nemo; do
    [ -e "$p" ] && { echo "ALREADY EXISTS $p, stopping"; exit 1; }
  done
done
echo "inputs ok"
train() {
  # $1 speaker, $2 block, $3 model name, $4 seed
  python scripts/asr/train.py --export $EX/$1 --out models/parakeet-ft-$3 --adapter-after-blocks $2 --lr 3e-4 --batch 2 --grad-accum 8 --warmup-steps 10 --epochs 15 --select last --no-keep-stock --freeze-batchnorm --normalize-targets --dev-scoring evaluate --seed $4
  f=$(ls -t logs/train_parakeet-*.log | head -1)
  echo "-- log lines from $f"
  grep -hE "parameters exact|kept epochs|averaged|Released|Elapsed|saved .*adapter" "$f" || true
  python scripts/asr/nemo_hyp.py --stock --adapter models/parakeet-ft-$3/adapter.pt --manifest $EX/$1/test.jsonl --audio-root $EX/$1 --out $FT/$3-test-nemo.jsonl
}
source ~/venvs/willspeak-train/bin/activate
FAILED=""
for j in $JOBS; do
  S=${j%%:*}; r=${j#*:}; N=${r%%:*}; SEED=${r##*:}
  echo; echo "######## $S, adapter after $N blocks, seed $SEED, started $(date +%H:%M:%S)"
  if ! ( set -e; train $S $N f2-$S-adapter$N-pos-seed$SEED $SEED ); then
    echo "FAILED $j, continuing with the next job"; FAILED="$FAILED $j"
  fi
done
deactivate
for j in $JOBS; do
  S=${j%%:*}; r=${j#*:}; N=${r%%:*}; SEED=${r##*:}
  R=f2-$S-adapter$N-pos-seed$SEED-test-nemo
  [ -e $FT/$R.jsonl ] || { echo "no decode for $j, not scored"; continue; }
  echo; echo "######## scoring $S, adapter after $N blocks, seed $SEED"
  $SC scripts/asr/evaluate.py --manifest $EX/$S/test.jsonl --audio-root $EX/$S --hyp $FT/$R.jsonl --group-by kind --out $FT/$R || echo "scoring failed for $j"
done
echo; echo "######## summary"
echo "failed jobs:${FAILED:- none}"
python3 - <<'PY'
import json, os
ft = "reports/finetune-torgo/"
words = {"M01": 169, "M02": 95, "M04": 161}
pos = (4, 6, 8, 10, 12, 16, 24)
def name(s, n, seed):
    if seed == 17:
        if n == 8:
            return "f2-M02-adapter8-pos-test-nemo" if s == "M02" else "f2-%s-adapter8-test-nemo" % s
        return "f2-%s-adapter%d-pos-test-nemo" % (s, n)
    if n == 8 and s == "M02":
        return "f2-M02-adapter8-fixed-seed%d-test-nemo" % seed
    return "f2-%s-adapter%d-pos-seed%d-test-nemo" % (s, n, seed)
def get(n):
    p = ft + n + "/results.json"
    if not os.path.exists(p):
        return None
    w = json.load(open(p))["word"]
    return w["sub"] + w["dele"] + w["ins"]
stock = {s: get("%s-test-stock-nemo" % s) for s in words}
st = sum(stock.values())
print("== wrong words per position, adapter on stock, fixed 15-epoch schedule (stock %d of 425 pooled)" % st)
print("seed 17 of block 8 on M01 and M04 is the earlier run (older train.py); seeds 18 and 19 are this run's")
tab = {}
for n in pos:
    row = {}
    for seed in (17, 18, 19):
        per = [get(name(s, n, seed)) for s in words]
        row[seed] = None if any(v is None for v in per) else per
    tab[n] = row
print("%-7s %-28s %-28s %-28s %6s %6s" % ("block", "seed 17 (M01 M02 M04 pooled)", "seed 18", "seed 19", "mean", "range"))
means = {}
for n in pos:
    cells = []; totals = []
    for seed in (17, 18, 19):
        r = tab[n][seed]
        if r is None:
            cells.append("%-28s" % "missing")
        else:
            cells.append("%-28s" % ("%d %d %d  %d" % (r[0], r[1], r[2], sum(r)))); totals.append(sum(r))
    if totals:
        means[n] = (sum(totals) / float(len(totals)), max(totals) - min(totals), len(totals))
        print("%-7s %s %6.1f %6d   (%d seeds)" % (n, " ".join(cells), means[n][0], means[n][1], means[n][2]))
    else:
        print("%-7s %s" % (n, " ".join(cells)))
full = {n: m for n, m in means.items() if m[2] == 3}
if full:
    order = sorted(full, key=lambda n: full[n][0])
    print("positions with all three seeds, by mean pooled wrong words: " + ", ".join("block %d (%.1f)" % (n, full[n][0]) for n in order))
    best = order[0]
    gap = full[order[1]][0] - full[best][0] if len(order) > 1 else float("nan")
    print("lowest mean: block %d; the next is %.1f words higher" % (best, gap))
    if 8 in full:
        g8 = st - full[8][0]
        print("block 8: three seeds pooled %s, range %d words" % (", ".join(str(sum(tab[8][s])) for s in (17, 18, 19)), full[8][1]))
        if 24 in full:
            print("block 24 gain over stock is %.0f%% of block 8's (%.1f against %.1f words)" % (100.0 * (st - full[24][0]) / g8, st - full[24][0], g8))
    print("-- per speaker, mean over the three seeds")
    for i, s in enumerate(words):
        line = "%-4s stock %3d " % (s, stock[s])
        for n in pos:
            r = tab[n]
            vals = [r[k][i] for k in (17, 18, 19) if r[k] is not None]
            line += " blk%-2d %5.1f" % (n, sum(vals) / len(vals)) if vals else " blk%-2d  n/a" % n
        print(line)
    print("-- the three predictions")
    print("E1, block 8 has the lowest three-seed mean pooled count of the seven positions: %s" % ("yes" if best == 8 else "no (block %d)" % best))
    if 8 in full:
        others = [full[n][0] for n in full if n != 8]
        print("E2, block 8's mean is more than 20 words below the next lowest mean: %s (%.1f words)" % ("yes" if min(others) - full[8][0] > 20 else "no", min(others) - full[8][0]))
        print("E3, block 8's three pooled counts lie within 15 words of each other: %s (range %d)" % ("yes" if full[8][1] <= 15 else "no", full[8][1]))
else:
    print("no position has all three seeds; see the table")
PY
echo "code at the end: $(git log -1 --format='%h %ad' --date=format:%H:%M)"
echo "finished $(date '+%Y-%m-%d %H:%M:%S')"
```

The wrapper, saved as `logs/overnight.sh`:

```bash
cd ~/willspeak/app
STAMP=$1
echo "overnight run $STAMP started $(date '+%Y-%m-%d %H:%M:%S')"
bash logs/phrasebook-rejection.sh </dev/null > logs/overnight-$STAMP-D.txt 2>&1
echo "D (phrase rejection) finished $(date +%H:%M:%S)"
bash logs/m02-seeds-and-blocks.sh </dev/null > logs/overnight-$STAMP-C.txt 2>&1
echo "C (M02 seeds, blocks 6 and 10) finished $(date +%H:%M:%S)"
bash logs/seeds-all-positions.sh </dev/null > logs/overnight-$STAMP-E.txt 2>&1
echo "E (seeds for every position) finished $(date +%H:%M:%S)"
{
  echo "######## D: phrase matching with a rejection rule"; cat logs/overnight-$STAMP-D.txt
  echo; echo "######## C: head of output"; head -8 logs/overnight-$STAMP-C.txt; sed -n '/######## summary/,$p' logs/overnight-$STAMP-C.txt
  echo; echo "######## E: head of output"; head -8 logs/overnight-$STAMP-E.txt; grep -n "FAILED\|ALREADY\|MISSING" logs/overnight-$STAMP-E.txt; sed -n '/######## summary/,$p' logs/overnight-$STAMP-E.txt
} > logs/overnight-$STAMP-summaries.txt
echo "SUMMARY FILE: ~/willspeak/app/logs/overnight-$STAMP-summaries.txt"
echo "overnight run finished $(date '+%Y-%m-%d %H:%M:%S')"
```

In the logs, check the `train.py` hash on the third line (032323d84c43 expected), `jobs: 40` in part E, `1049600 trainable`, and any `FAILED` lines.

### Full Results

Part E's summary block, copied from the console. The copy started partway through the last training run, so the first lines (input check, `train.py` hash) and parts C and D are missing. The code line at the end read `a1851ca 22:36`.

```text
######## summary
failed jobs: none
== wrong words per position, adapter on stock, fixed 15-epoch schedule (stock 323 of 425 pooled)
seed 17 of block 8 on M01 and M04 is the earlier run (older train.py); seeds 18 and 19 are this run's
block   seed 17 (M01 M02 M04 pooled) seed 18                      seed 19                        mean  range
4       113 41 98  252               109 42 104  255              104 40 111  255               254.0      3   (3 seeds)
6       90 41 84  215                77 40 97  214                80 48 88  216                 215.0      2   (3 seeds)
8       84 38 79  201                87 44 76  207                86 42 82  210                 206.0      9   (3 seeds)
10      96 55 98  249                97 51 108  256               99 51 99  249                 251.3      7   (3 seeds)
12      100 55 106  261              99 58 109  266               96 55 112  263                263.3      5   (3 seeds)
16      102 49 94  245               107 50 99  256               108 52 102  262               254.3     17   (3 seeds)
24      121 56 114  291              121 59 112  292              121 55 122  298               293.7      7   (3 seeds)
positions with all three seeds, by mean pooled wrong words: block 8 (206.0), block 6 (215.0), block 10 (251.3), block 4 (254.0), block 16 (254.3), block 12 (263.3), block 24 (293.7)
lowest mean: block 8; the next is 9.0 words higher
block 8: three seeds pooled 201, 207, 210, range 9 words
block 24 gain over stock is 25% of block 8's (29.3 against 117.0 words)
-- per speaker, mean over the three seeds
M01  stock 131  blk4  108.7 blk6   82.3 blk8   85.7 blk10  97.3 blk12  98.3 blk16 105.7 blk24 121.0
M02  stock  62  blk4   41.0 blk6   43.0 blk8   41.3 blk10  52.3 blk12  56.0 blk16  50.3 blk24  56.7
M04  stock 130  blk4  104.3 blk6   89.7 blk8   79.0 blk10 101.7 blk12 109.0 blk16  98.3 blk24 116.0
-- the three predictions
E1, block 8 has the lowest three-seed mean pooled count of the seven positions: yes
E2, block 8's mean is more than 20 words below the next lowest mean: no (9.0 words)
E3, block 8's three pooled counts lie within 15 words of each other: yes (range 9)
```

Part C's summary. The training log and first lines weren't copied, but the `train.py` hash on the third line was 032323d84c43 (same file as the sweep) and the code was commit a1851ca.

```text
######## summary
== M02 (95 words, one word is 1.05 points)
stock                                 65.26%   62 wrong
whole-block model (A3)                44.21%   42 wrong
adapter 8, default recipe, seed 17    48.42%   46 wrong
adapter 8, fixed schedule, seed 17    40.00%   38 wrong
adapter 8, fixed schedule, seed 18    46.32%   44 wrong
adapter 8, fixed schedule, seed 19    44.21%   42 wrong
fixed schedule: mean 41.3 wrong (43.51%), spread 6 words (6.32 points); the default recipe scored 46 on every run on record
== wrong words per speaker and position (fixed schedule, seed 17, adapter on stock)
           blk 4   blk 6   blk 8  blk 10  blk 12
M01          113      90      84      96     100
M02           41      41      38      55      55
M04           98      84      79      98     106
pooled       252     215     201     249     261   (of 425 words; stock 323)
pooled WER: block 4 59.3%; block 6 50.6%; block 8 47.3%; block 10 58.6%; block 12 61.4%
lowest pooled count among these five: block 8
-- comparisons by clip (positive gain: the second model is better; shown only if the interval excludes zero)
M02 fixed seed 18 against fixed seed 17                         -6.32 (-11.96 to  -2.20) excludes zero
M02 fixed seed 18 against the default recipe, seed 17            2.11 ( -4.48 to   9.52) includes zero
M02 fixed seed 18 against the whole-block model                 -2.11 (-12.22 to   8.97) includes zero
M02 fixed seed 18 against stock                                 18.95 (  7.23 to  32.94) excludes zero
M02 fixed seed 19 against fixed seed 17                         -4.21 (-10.00 to   0.00) includes zero
M02 fixed seed 19 against the default recipe, seed 17            4.21 ( -2.15 to  11.60) includes zero
M02 fixed seed 19 against the whole-block model                  0.00 (-10.35 to  10.98) includes zero
M02 fixed seed 19 against stock                                 21.05 (  9.84 to  35.63) excludes zero
M01: after 6 blocks against after 8 blocks                      -3.55 (-11.31 to   3.85) includes zero
M01: after 10 blocks against after 8 blocks                     -7.10 (-18.19 to   2.48) includes zero
M02: after 6 blocks against after 8 blocks                      -3.16 (-13.49 to   5.75) includes zero
M02: after 10 blocks against after 8 blocks                    -17.89 (-27.94 to  -8.97) excludes zero
M04: after 6 blocks against after 8 blocks                      -3.11 (-10.37 to   3.40) includes zero
M04: after 10 blocks against after 8 blocks                    -11.80 (-17.06 to  -6.25) excludes zero
-- the three predictions
C1, both new fixed-schedule seeds at 44 wrong or fewer: yes (seeds 18, 19: 44, 42)
C2, the three fixed-schedule seeds within 6 words of each other: yes (spread 6)
C3, blocks 6 and 10 both above block 8's pooled count of 201: yes (block 6 215, block 10 249)
code at the end: a1851ca 22:36
finished 2026-10-01 23:04:40
```

### Conclusions

1. Block 8 has the best mean, 206.0 against 215.0 for block 6, but that 9-word gap is within the noise. My rule needed more than 20 words and a lead on two speakers. Block 8 leads on M02 (by 1.7) and M04 (by 10.7) and trails on M01 (by 3.4), so it isn't shown to beat block 6. It does beat blocks 4, 10, 12, 16 and 24, which are 45 to 88 words behind, well past the seed ranges of 2 to 17.
2. The curve is low at blocks 6 to 8 and rises on both sides: 254 (block 4), 215 (6), 206 (8), 251 (10), 254 (16 and 4), 263 (12), 294 (24). Block 10 is 45 words worse than 8 and block 12 is worse than 16, so the curve isn't smooth and the middle positions aren't reliably ranked. Blocks 6 to 8 are a low region; I can't pin a single minimum.
3. The speakers differ at the low end. M01 is best at block 6, M04 at block 8, and M02 is level across 4, 6 and 8 (41.0, 43.0, 41.3). Block 4 is 25 words worse than 8 for M04 and 23 worse for M01. Three speakers can't say why.
4. Seed noise is smaller than the sweep's single-run lead suggested it might be. Block 8's pooled range was 9 (201 to 210), and per speaker M01 84 to 87, M02 38 to 44, M04 76 to 82. Other positions ranged 2 to 17. The older-code seed-17 runs for M01 and M04 (84 and 79) sit inside the new seeds' range (87 and 86; 76 and 82), so no code effect shows.
5. On M02 all three fixed-schedule seeds (38, 44, 42) are under the default recipe's 46, and the mean of 41.3 (43.5%) matches the whole-block model's 42. By clip, though, seeds 18 and 19 against the default recipe's seed 17 are 2.11 points (-4.48 to 9.52) and 4.21 points (-2.15 to 11.60), both including zero. Against the whole-block model they also include zero (-2.11 and 0.00), and against stock they exclude it (18.95 and 21.05). Seed 17's own comparison with the default recipe wasn't run. So the fixed schedule isn't shown to beat the default recipe on M02.
6. Block 24 keeps about a quarter of block 8's gain (25%), repeating the sweep's 26%.
7. Hypotheses: C1 held (44 and 42), C2 held (range 6, right at the limit), C3 held (215 and 249, both above 201), E1 held, E2 missed (a 9-word lead where it needed over 20), E3 held (range 9). Five of six. Part C's per-speaker clip comparisons against block 8 include zero for block 6 everywhere and for block 10 on M01. They exclude it for block 10 on M02 (-17.89 points) and M04 (-11.80), both favouring block 8.
8. Still open: whether blocks 5 to 7 beat 8, narrower or multi-position adapters, per-position learning rates, and the F-1 base here. The level is still far from the high-20s goal, with pooled 48.5% at block 8 and M02 at 43.5%.

</details>

## Related

[F-2: adapter position sweep](../f2-adapter-position-sweep/notes.md), [F-2: M02, adapter after 8 blocks](../f2-m02-adapter8/notes.md), [F-2: M02, adapter after 8 blocks, seeds 18 and 19](../f2-m02-adapter8-seeds/notes.md), [F-2: M01 and M04, adapter after 8 blocks](../f2-m01-m04-adapter8/notes.md)
