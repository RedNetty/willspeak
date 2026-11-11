# F-2: adapter position sweep, M01, M02 and M04, after 4, 8, 12, 16 and 24 blocks

Proposal: F-2 (single-speaker fine-tuning, §5.2 / H3). Same adapter method as before, but moved away from its one position after encoder block 8.

Status: run 1 Oct 2026, 21:46 to 22:28. Block 8 had the fewest wrong words on every speaker, and every other position was worse. Two of the three pre-run hypotheses held. I picked this test from a list of options at 21:33 and didn't write a prediction of my own.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

The adapter is one layer, `h + W h + b`, after block N of the encoder's 24 blocks, trained alone with everything else frozen. So far N = 8 every time, chosen on stock for M02 and copied to M01 and M04. Here I train it at N = 4, 12, 16 and 24 on all three speakers, plus N = 8 for M02 on the shared recipe, and report the whole curve.

## Setup

The encoder is 24 blocks that turn audio into feature vectors, and the adapter adds a learned correction after block N. `W` is 1,024 by 1,024 and `b` has 1,024 entries, so 1,049,600 numbers, all starting at zero so the model starts unchanged. After block 4 it changes what the 20 later blocks and the decoder get. After block 24 it only changes what the decoder gets. A word error is a wrong, deleted or inserted word.

Earlier adapter-after-8 results (on stock, NeMo). M01 and M04 used a fixed 15-epoch schedule, and M02 its default recipe where dev picks the checkpoints.

| Speaker | Test words | Stock | Adapter after 8, range over seeds 17 to 19 |
| --- | --- | --- | --- |
| M01 | 169 | 131 (77.51%) | 84 to 87 (fixed schedule) |
| M02 | 95 | 62 (65.26%) | 46 on every run (default recipe) |
| M04 | 161 | 130 (80.75%) | 76 to 82 (fixed schedule) |

Pooled (425 words), stock made 323 errors and the adapter after 8 about 210 (206 to 215 over seeds), a gain of about 110. The summary recomputes pooled counts from the result files.

All 13 new runs (M01 4 positions, M04 4, M02 5) use the fixed schedule: 15 epochs, last 4 checkpoints averaged, untrained-model guard off, lr 3e-4, batch 2 with 8 accumulated, 10 warmup updates, normalised transcripts, batch norm frozen, seed 17. About 3 minutes each plus decoding, around 40 minutes. M01 and M04 reuse their adapter-after-8 results (`f2-<SPK>-adapter8-test-nemo`), trained on this schedule with an earlier `train.py`. M02's block 8 is retrained so all its positions share one recipe, and its old 46 isn't used in the curve.

Adapter size is a later test. A narrower layer or several adapters needs new code, since `block_adapter.py` supports one full-size adapter at one position.

## Plan

1. Check that the inputs exist and the new outputs don't (the script does this first).
2. For each speaker and position, train the adapter on stock with `--adapter-after-blocks N` and decode the test clips with `nemo_hyp.py --stock --adapter`.
3. Score with `evaluate.py`. Use `ci.py` to compare each position with the adapter after 8 and with stock, by clip, per speaker.
4. Print one table of wrong words per speaker and position with pooled totals, the best and worst position, the spread, block 24's gain as a share of block 8's, and a yes or no for each prediction.

No position gets crowned from test scores. If one is adopted later, that needs a test the models haven't been scored on. Chance alone moves one speaker's count 4 to 7 words (4 on M02, 7 on M01, 6 on M04), so roughly 10 pooled, and a pooled gap under about 10 isn't a finding. If the whole curve is within 10, position doesn't matter at this size and block 8 stays. A steady trend with N would show first at the ends (4 and 24). A position only beats block 8 if most speakers agree and the pooled gap is over 10.

Block 24 can only remap what the decoder gets. If it keeps over half of block 8's gain, the correction is mostly at the output, which would point to a cheaper adapter. A comparison is shown only when its interval sits wholly on one side of zero.

## Why

The adapter is the practical result, one small file per speaker, but its position was picked on one speaker's stock model and never varied. The literature puts adapters after every block, which is a bigger change. This checks whether position matters first, for about 40 minutes of GPU time.

## Limits

On three speakers' last-session test clips, this shows how the pooled count changes over five positions, whether the ends differ from the middle, and whether M02's block 8 on the fixed schedule matches its 46.

There's no seed spread, since each position is one run (seed 17), and earlier seed repeats at block 8 moved 0 to 8 words per speaker. It doesn't cover per-position learning rates, other adapter sizes or several adapters, or the F-1 base (every run starts from stock). These clips have also been scored many times, so a position picked by test score wouldn't mean much.

## Reading and Video

None added. See the lists in the [M02 seeds note](../f2-m02-adapter8-seeds/notes.md#reading-and-video).

## Results

Wrong words (substitutions, deletions and insertions) on each speaker's test clips, adapter on stock, fixed 15-epoch schedule, seed 17:

| | Stock | After 4 | After 8 | After 12 | After 16 | After 24 |
| --- | --- | --- | --- | --- | --- | --- |
| M01 (169 words) | 131 | 113 | 84 | 100 | 102 | 121 |
| M02 (95 words) | 62 | 41 | 38 | 55 | 49 | 56 |
| M04 (161 words) | 130 | 98 | 79 | 106 | 94 | 114 |
| Pooled (425 words) | 323 | 252 | 201 | 261 | 245 | 291 |
| Pooled WER | 76.0% | 59.3% | 47.3% | 61.4% | 57.6% | 68.5% |

Block 8 is lowest and block 24 highest on all three speakers. Pooled, the gain over stock is 122 words at block 8 and 32 at block 24 (26% of block 8's). By clip, 10 of the 12 comparisons with block 8 exclude zero, all in block 8's favour. The two that don't are M01 block 16 (interval ends at 0.00) and M02 block 4 (3 words, 3.16 points). Against stock, every position excludes zero except M02 blocks 12 and 24 and M04 block 24.

The middle positions aren't in a smooth order (block 12 is worse than block 16 on every speaker but M01), and each is one run, so I only read the big differences.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Pre-run Hypotheses

No prediction was recorded, but these hypotheses were written down before the run.

1. The lowest pooled count is at block 12. Dysarthric speech changes what the early blocks see, so a very early adapter has little depth after it to correct things, and a very late one can only remap the output. The middle seemed like the compromise. Confidence was low: with five positions and about 10 words of noise, chance alone hits any named position about one time in three.
2. The highest and lowest pooled counts are more than 30 words apart. Position changes which part of the model can respond, and block 24 should be clearly worse than block 8.
3. Block 24's pooled gain over stock is less than half of block 8's. The 23 earlier blocks produce the representation the decoder reads, and an adapter after block 24 can't change how they heard the audio.

### Commands and Setup

Run from `app/` in the WSL tree on Yack after `git pull`, with the GPU free and Windows apps closed. Training and decoding run in `~/venvs/willspeak-train`, and `evaluate.py` and `ci.py` in `app/.venv`. I dry-ran the script on a stub copy of the tree first. That checks the shell logic, the arguments each call gets (the right `--adapter-after-blocks` per job, `--stock` on decode, seed 17), the parsing of `ci.py`'s output and the summary on made-up numbers. It says nothing about the real models.

The script, saved as `logs/adapter-position-sweep.sh` and run with its output copied to `logs/adapter-position-sweep-full-<date and time>.txt`:

```bash
cd ~/willspeak/app
EX=data/public/torgo/export
FT=reports/finetune-torgo
SC=.venv/bin/python
JOBS="M01:4 M01:12 M01:16 M01:24 M02:4 M02:8 M02:12 M02:16 M02:24 M04:4 M04:12 M04:16 M04:24"
CIALL=$(mktemp)
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
echo "code: $(git log -1 --format='%h %ad' --date=format:%H:%M)"
echo "train.py sha256: $(sha256sum scripts/asr/train.py | cut -c1-12)   (the file the cleanup check passed: c696b4873186)"
for S in M01 M02 M04; do
  for p in $EX/$S/train.jsonl $EX/$S/test.jsonl $FT/$S-test-stock-nemo/results.json; do
    [ -e "$p" ] || { echo "MISSING $p, stopping"; exit 1; }
  done
done
for S in M01 M04; do
  [ -e $FT/f2-$S-adapter8-test-nemo/results.json ] || { echo "MISSING $FT/f2-$S-adapter8-test-nemo/results.json, stopping"; exit 1; }
done
for j in $JOBS; do
  S=${j%%:*}; N=${j##*:}
  for p in models/parakeet-ft-f2-$S-adapter$N-pos $FT/f2-$S-adapter$N-pos-test-nemo; do
    [ -e "$p" ] && { echo "ALREADY EXISTS $p, stopping"; exit 1; }
  done
done
echo "inputs ok"
ci() {
  echo "== $1" | tee -a $CIALL
  $SC scripts/asr/ci.py --manifest $EX/$S/test.jsonl --stock $FT/$2 --finetuned $FT/$3 > $CIALL.one 2>&1 || { cat $CIALL.one; echo "ci.py failed, stopping"; exit 1; }
  cat $CIALL.one | tee -a $CIALL
}
source ~/venvs/willspeak-train/bin/activate
set -e
for j in $JOBS; do
  S=${j%%:*}; N=${j##*:}
  R=f2-$S-adapter$N-pos
  echo; echo "######## $S, adapter after $N blocks, started $(date +%H:%M:%S)"
  python scripts/asr/train.py --export $EX/$S --out models/parakeet-ft-$R --adapter-after-blocks $N --lr 3e-4 --batch 2 --grad-accum 8 --warmup-steps 10 --epochs 15 --select last --no-keep-stock --freeze-batchnorm --normalize-targets --dev-scoring evaluate --seed 17
  f=$(ls -t logs/train_parakeet-*.log | head -1)
  echo "-- log lines from $f"
  grep -hE "parameters exact|fixed schedule|kept epochs|Checking the averaged|averaged|Released|Elapsed|saved .*adapter" "$f" || true
  python scripts/asr/nemo_hyp.py --stock --adapter models/parakeet-ft-$R/adapter.pt --manifest $EX/$S/test.jsonl --audio-root $EX/$S --out $FT/$R-test-nemo.jsonl
done
deactivate
for j in $JOBS; do
  S=${j%%:*}; N=${j##*:}
  M=$EX/$S; R=f2-$S-adapter$N-pos-test-nemo
  echo; echo "######## scoring $S, adapter after $N blocks"
  $SC scripts/asr/evaluate.py --manifest $M/test.jsonl --audio-root $M --hyp $FT/$R.jsonl --group-by kind --out $FT/$R
done
for S in M01 M02 M04; do
  if [ $S = M02 ]; then P8=f2-M02-adapter8-pos-test-nemo; else P8=f2-$S-adapter8-test-nemo; fi
  for N in 4 12 16 24; do
    ci "$S: after $N blocks against after 8 blocks" $P8 f2-$S-adapter$N-pos-test-nemo
  done
  for N in 4 12 16 24; do
    ci "$S: after $N blocks against stock" $S-test-stock-nemo f2-$S-adapter$N-pos-test-nemo
  done
done
echo; echo "######## summary"
CIALL=$CIALL python3 - <<'PY'
import json, os, re
ft = "reports/finetune-torgo/"
words = {"M01": 169, "M02": 95, "M04": 161}
W = sum(words.values())
def get(n):
    r = json.load(open(ft + n + "/results.json")); w = r["word"]
    return r["wer"] * 100, w["sub"] + w["dele"] + w["ins"]
def name(s, n):
    if n == 8 and s != "M02":
        return "f2-%s-adapter8-test-nemo" % s
    return "f2-%s-adapter%d-pos-test-nemo" % (s, n)
pos = (4, 8, 12, 16, 24)
wrong = {}
stock = {}
print("== wrong words per speaker and position (fixed 15-epoch schedule, seed 17, adapter on stock)")
print("%-8s %6s " % ("", "stock") + " ".join("%7s" % ("blk %d" % n) for n in pos))
for s in words:
    stock[s] = get("%s-test-stock-nemo" % s)[1]
    line = "%-8s %6d " % (s, stock[s])
    for n in pos:
        wrong[(s, n)] = get(name(s, n))[1]
        line += "%7d " % wrong[(s, n)]
    print(line)
tot = {n: sum(wrong[(s, n)] for s in words) for n in pos}
st = sum(stock.values())
print("%-8s %6d " % ("pooled", st) + " ".join("%7d" % tot[n] for n in pos) + "   (of %d words)" % W)
print("pooled WER: stock %.1f%%; " % (100.0 * st / W) + "; ".join("block %d %.1f%%" % (n, 100.0 * tot[n] / W) for n in pos))
best = min(pos, key=lambda n: tot[n]); worst = max(pos, key=lambda n: tot[n])
print("lowest pooled count: block %d (%d wrong); highest: block %d (%d wrong); spread %d words" % (best, tot[best], worst, tot[worst], tot[worst] - tot[best]))
g8 = st - tot[8]; g24 = st - tot[24]
print("gain over stock, pooled: block 8 %d words, block 24 %d words, block 24 is %.0f%% of block 8" % (g8, g24, 100.0 * g24 / g8 if g8 else float("nan")))
for s in words:
    print("%s: lowest at block %d (%d wrong), highest at block %d (%d wrong)" % (s, min(pos, key=lambda n: wrong[(s, n)]), min(wrong[(s, n)] for n in pos), max(pos, key=lambda n: wrong[(s, n)]), max(wrong[(s, n)] for n in pos)))
print("-- comparisons by clip (positive gain: the second model is better; shown only if the interval excludes zero)")
cur = None; shown = 0; total = 0
for line in open(os.environ["CIALL"], encoding="utf-8", errors="replace"):
    line = line.rstrip("\n")
    if line.startswith("== "):
        cur = line[3:]
        continue
    m = re.search(r"Gain \(baseline - candidate\)\s+(-?[\d.]+)\s+points\s+\(95% CI\s+(-?[\d.]+)\s+to\s+(-?[\d.]+)\)", line)
    if m and cur:
        d, lo, hi = (float(x) for x in m.groups())
        ex = not (lo <= 0.0 <= hi)
        if "against after 8 blocks" in cur:
            total += 1; shown += int(ex)
        print("%-46s %6.2f (%6.2f to %6.2f) %s" % (cur, d, lo, hi, "excludes zero" if ex else "includes zero"))
print("positions compared with block 8 whose interval excludes zero: %d of %d" % (shown, total))
print("-- the three predictions")
print("A1, lowest pooled count at block %d (the pre-run hypothesis named block 12)" % best)
sp = tot[worst] - tot[best]
print("A2, spread of highest and lowest pooled count: %d words, %s" % (sp, "under 10" if sp < 10 else ("10 to 30" if sp <= 30 else "over 30")))
print("A3, block 24's gain over stock is less than half of block 8's: %s" % ("yes" if g24 * 2 < g8 else "no"))
PY
rm -f $CIALL $CIALL.one
echo "code at the end: $(git log -1 --format='%h %ad' --date=format:%H:%M)"
echo "finished $(date '+%Y-%m-%d %H:%M:%S')"
```

Before trusting the scores, check each training log for `1049600 trainable`, `kept epochs 12, 13, 14, 15; dev chose nothing`, `averaged 4 checkpoints` and a saved `adapter.pt`. The decode should print `Added the adapter ... after encoder blocks 0-N-1` for the right N. The `train.py` hash on the third output line should be c696b4873186. A different hash means the code changed and should be recorded.

### Full Results

This is the summary block of the run's output, copied from the console. The copy started partway through the M04 adapter-after-12 run, so the first lines aren't recorded here. The inputs line and the `train.py` hash were seen in an earlier copy of the output.

```text
######## summary
== wrong words per speaker and position (fixed 15-epoch schedule, seed 17, adapter on stock)
          stock   blk 4   blk 8  blk 12  blk 16  blk 24
M01         131     113      84     100     102     121
M02          62      41      38      55      49      56
M04         130      98      79     106      94     114
pooled      323     252     201     261     245     291   (of 425 words)
pooled WER: stock 76.0%; block 4 59.3%; block 8 47.3%; block 12 61.4%; block 16 57.6%; block 24 68.5%
lowest pooled count: block 8 (201 wrong); highest: block 24 (291 wrong); spread 90 words
gain over stock, pooled: block 8 122 words, block 24 32 words, block 24 is 26% of block 8
M01: lowest at block 8 (84 wrong), highest at block 24 (121 wrong)
M02: lowest at block 8 (38 wrong), highest at block 24 (56 wrong)
M04: lowest at block 8 (79 wrong), highest at block 24 (114 wrong)
-- comparisons by clip (positive gain: the second model is better; shown only if the interval excludes zero)
M01: after 4 blocks against after 8 blocks     -17.16 (-28.24 to  -7.23) excludes zero
M01: after 12 blocks against after 8 blocks     -9.47 (-20.11 to  -0.70) excludes zero
M01: after 16 blocks against after 8 blocks    -10.65 (-22.82 to   0.00) includes zero
M01: after 24 blocks against after 8 blocks    -21.89 (-35.06 to -10.99) excludes zero
M01: after 4 blocks against stock               10.65 (  2.22 to  21.01) excludes zero
M01: after 12 blocks against stock              18.34 (  8.93 to  29.69) excludes zero
M01: after 16 blocks against stock              17.16 (  9.62 to  26.25) excludes zero
M01: after 24 blocks against stock               5.92 (  0.57 to  11.45) excludes zero
M02: after 4 blocks against after 8 blocks      -3.16 (-11.88 to   5.95) includes zero
M02: after 12 blocks against after 8 blocks    -17.89 (-27.37 to -10.46) excludes zero
M02: after 16 blocks against after 8 blocks    -11.58 (-19.61 to  -3.77) excludes zero
M02: after 24 blocks against after 8 blocks    -18.95 (-30.77 to  -9.17) excludes zero
M02: after 4 blocks against stock               22.11 (  9.70 to  38.71) excludes zero
M02: after 12 blocks against stock               7.37 (  0.00 to  16.85) includes zero
M02: after 16 blocks against stock              13.68 (  4.87 to  25.36) excludes zero
M02: after 24 blocks against stock               6.32 ( -1.73 to  16.44) includes zero
M04: after 4 blocks against after 8 blocks     -11.80 (-21.09 to  -2.50) excludes zero
M04: after 12 blocks against after 8 blocks    -16.77 (-26.18 to  -7.14) excludes zero
M04: after 16 blocks against after 8 blocks     -9.32 (-15.13 to  -3.62) excludes zero
M04: after 24 blocks against after 8 blocks    -21.74 (-30.38 to -12.42) excludes zero
M04: after 4 blocks against stock               19.88 (  5.80 to  37.89) excludes zero
M04: after 12 blocks against stock              14.91 (  3.15 to  28.78) excludes zero
M04: after 16 blocks against stock              22.36 ( 10.00 to  36.46) excludes zero
M04: after 24 blocks against stock               9.94 ( -2.69 to  24.00) includes zero
positions compared with block 8 whose interval excludes zero: 10 of 12
-- the three predictions
A1, lowest pooled count at block 8 (the pre-run hypothesis named block 12)
A2, spread of highest and lowest pooled count: 90 words, over 30
A3, block 24's gain over stock is less than half of block 8's: yes
```

Code: commit 79d017f (21:46), `train.py` sha256 032323d84c43. That's not the c696b4873186 file that passed the cleanup check. Commit f8fb3ff at 21:46 changed `train.py`: more logging, plus changes to how a capped run stops and how validation is recorded (this schedule has no cap).

The logs I saw show the fixed schedule, `kept epochs 12, 13, 14, 15; dev chose nothing`, `averaged 4 checkpoints`, 1,049,600 trainable numbers and the adapter after the intended blocks (0-3 and 0-11 checked in the log). Each run took about 2 to 2.5 minutes. The block-8 results for M01 and M04 come from the earlier code on the same schedule. M01's dev WER was 50.00% (block 4) and 61.11% (block 12), against 44.44% at block 8 in the earlier note, but the dev set is only 4 clips.

### Conclusions

1. Position matters, and block 8 was the best of the five on all three speakers. Pooled over the 425 test words, block 8 made 201 errors, against 245 to 291 for blocks 12, 16 and 24 and 252 for block 4. The gap to the next best (block 16) is 44 words. That's above the 10-word noise guide and above the 0 to 8 words per speaker that seeds moved the adapter after 8 in earlier notes. Block 24, which only remaps what the decoder receives, kept 26% of block 8's gain.
2. The curve isn't smooth, and its shape between the ends isn't established. Block 12 was worse than block 16 pooled, and one run per position can't rank neighbours. What the result supports is "8 is better than 4, 12, 16 and 24 on these clips". It doesn't show a smooth peak.
3. The comparison does favour block 8 in one way. Block 8 was the position first tried on M02 and carried over to the others, and positions 4, 12, 16 and 24 weren't tuned (learning rate, epochs). The M01 and M04 block-8 runs also came from earlier code. Neither seems big enough to explain 44 pooled words, but neither is ruled out.
4. M02's block 8 got 38 wrong words (40.00%) on the fixed schedule, against 46 on M02's default recipe. That's below every M02 score so far, including the whole-block model's 42. The eight-word gap is above M02's 4-word noise guide, but it's one seed on one recipe, while the earlier 46s were three seeds that never moved. If it repeats, the dev-picked checkpoints on M02's 24-clip dev set were costing about 8 words.
5. Predictions: 1 missed, since the lowest pooled count was at block 8 and the hypothesis said block 12. 2 held, with the highest and lowest pooled counts 90 words apart. 3 held, with block 24's gain over stock at 26% of block 8's.
6. Not shown: whether a bigger or narrower adapter, several adapters at once, a different learning rate per position, or the F-1 base would change the picture.

</details>

## Related

[F-2: M01 and M04, adapter after 8 blocks](../f2-m01-m04-adapter8/notes.md), [F-2: M02, adapter after 8 blocks](../f2-m02-adapter8/notes.md), [F-2: M02, adapter after 8 blocks, seeds 18 and 19](../f2-m02-adapter8-seeds/notes.md), [F-2: M02, adapter on the F-1 model, seeds 17, 18 and 19](../f2-m02-adapter8-on-f1-seeds/notes.md), [F-2: M01 and M04, adapter on the F-1 model, seeds 18 and 19](../f2-m01-m04-adapter8-on-f1-seeds/notes.md)
