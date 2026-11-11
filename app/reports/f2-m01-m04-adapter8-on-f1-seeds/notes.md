# F-2: M01 and M04, adapter after 8 blocks on the F-1 model, seeds 18 and 19

Proposal: F-2 (single-speaker fine-tuning, §5.2 / H3). A repeat of the [adapter on F-1 run](../f2-m01-m04-adapter8-on-f1/notes.md) with two more seeds, to measure its run-to-run spread and check whether M04's lead holds. No new method.

Status: Run 1 Oct 2026, about 19:11 to 19:25. M04's lead over the adapter on stock repeated on both new seeds, and M01 showed no difference from the adapter on stock. Keeping prompts together withdrew no result. All three predictions held under the readings written before the run.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

Seed 17 of the adapter on F-1 scored 40.99% on M04 and 53.25% on M01. Against the adapter on stock that was 13 words better on M04 (a gain of 8.07 points, interval 1.38 to 15.43) and 6 words worse on M01 (a gain of -3.55, interval -13.07 to 6.59, which includes zero). Here I train it twice more per speaker, changing only the seed, to see whether those results come from the F-1 base or from seed 17. I also re-score the seed-17 comparisons with each prompt's takes kept together, which the first run didn't do.

## Setup

The adapter on F-1 is the small layer after block 7 (`h + W h + b`, 1,049,600 numbers, starting at zero) trained on top of the speaker's F-1 model, the shared model that never heard him. The adapter on stock is the same layer on the downloaded model. The seed sets the clip order, the speed changes and which spectrogram patches get masked. It doesn't touch the adapter's zero start or the F-1 model, which is one fixed file per speaker.

A gain is one model's WER minus another's, so positive means the second model is better. Its interval comes from `ci.py` reshuffling the test clips 2,000 times, and a gain only counts as shown when the whole interval is on one side of zero.

Earlier scores on the same clips, as word errors with WER in brackets. M01 has 169 test words and M04 has 161, so one word is 0.59 points on M01 and 0.62 on M04:

| Model | M01 | M04 |
| --- | --- | --- |
| F-1 shared model | 97 (57.40%) | 105 (65.22%) |
| Adapter on stock, seed 17 | 84 (49.70%) | 79 (49.07%) |
| Adapter on stock, seed 18 | 87 (51.48%) | 76 (47.20%) |
| Adapter on stock, seed 19 | 86 (50.89%) | 82 (50.93%) |
| Adapter on F-1, seed 17 | 90 (53.25%) | 66 (40.99%) |

In the [check of the cleaned-up `train.py`](../check-train-py-cleanup/notes.md) the adapter on stock moved 3 words (1.78 points) across its seeds on M01 and 6 words (3.73 points) on M04, and the same seed reproduced the runs on Yack clip for clip. So any gap between seeds here comes from the seed and not the machine.

`ci.py` warned on every comparison that some prompts appear more than once among the test clips. Repeated takes of one prompt aren't independent, so intervals that resample single clips can be too narrow. `--cluster-by text` resamples whole prompts instead. That matters most for M04's lower end of 1.38.

## Plan

1. Check that the earlier outputs exist and the new output folders don't (the script does this first).
2. For M01 and M04, seeds 18 and 19, train the adapter with `--base` set to the speaker's F-1 model and the seed-17 settings (learning rate 3e-4, batch 2 with 8 accumulated, 10 warmup updates, 15 epochs, normalised transcripts, `--select last --no-keep-stock`). Decode the test clips with the F-1 model plus the adapter.
3. Score with `evaluate.py`. Use `ci.py` to compare each new run with the adapter on F-1 at seed 17 (spread), the adapter on stock at the same seed (does the lead hold) and the F-1 model alone. Repeat the same-seed comparison with `--cluster-by text`.
4. Re-score the six seed-17 comparisons (against stock, the adapter on stock and the F-1 model, per speaker) with prompts kept together.
5. Print each speaker's three scores for both adapters with spread and mean, and check that each training log names the F-1 file as the base.

How I'd read it, decided before the run. One word is about 0.6 points, so the adapter on stock's spreads of 3 and 6 words set the scale. If the adapter on F-1's spread is more than about 4 points on a speaker, one run per speaker can't separate the two recipes there.

M04's lead repeats if both new seeds get 75 or fewer wrong, below the adapter on stock's best seed (76). One out of two is mixed. Neither would mean seed 17 was a good draw and the 8.07 shouldn't be relied on. For M01, a new seed inside the adapter on stock's 84 to 87, or one word outside, means no difference. Above on both would suggest the F-1 base costs M01 a little (the interval would still have to show it), and below on both would mean seed 17 was a bad draw and the base helps M01 too.

Any "shown" that disappears once prompts are kept together is withdrawn, and both versions get reported. Three seeds per side is still a thin range, and a fourth could move either one.

## Why

The speakers disagreed at seed 17, and each figure was one run. The device is meant to be a shared model plus a small personal layer, and the next experiments (F-3, and whole blocks on top of F-1) shouldn't rest on a lead that might be luck. It costs about 13 minutes of GPU time.

## Limits

For M01 and M04 on their last sessions in NeMo, this shows how far the adapter on F-1 moves across three seeds, whether each speaker's result repeats, and whether the intervals survive grouping by prompt.

It can't show spread from the F-1 model's own training (one F-1 model per speaker), or why the two speakers differ, since speaker, base quality and chance aren't separated. It doesn't test whether 3e-4 and the block-7 position, both chosen on stock, suit an F-1 base. It doesn't cover M02, whole blocks on the F-1 base, the prototype model as the base, or the Pi. The dev WERs rest on 18 and 25 words, so I read nothing from them. These test clips have been scored many times now, so I claim nothing beyond the comparisons above and pick no variant by its test score.

## Reading and Video

None added. See the lists in the [M02 seeds note](../f2-m02-adapter8-seeds/notes.md#reading-and-video).

## Results

Run on 1 Oct 2026, about 19:11 to 19:25, as one script (14 minutes). The copied output starts partway through the M04 seed-18 training. A later grep of the saved file showed the `train.py` hash the cleanup check passed, the input check, and the F-1 base on the `Base model` and `loading` lines of all four trainings. I didn't see the rest of the M01 training and decode output. The scoring, comparisons and summary are complete, with details under Run Record.

Wrong words out of 169 (M01) and 161 (M04), with WER in brackets. The F-1 model alone scored 97 (57.40%) on M01 and 105 (65.22%) on M04.

| Seed | M01 adapter on stock | M01 adapter on F-1 | M04 adapter on stock | M04 adapter on F-1 |
| --- | --- | --- | --- | --- |
| 17 | 84 (49.70%) | 90 (53.25%) | 79 (49.07%) | 66 (40.99%) |
| 18 | 87 (51.48%) | 85 (50.30%) | 76 (47.20%) | 74 (45.96%) |
| 19 | 86 (50.89%) | 83 (49.11%) | 82 (50.93%) | 68 (42.24%) |
| Mean of the three | 85.7 (50.69%) | 86.0 (50.89%) | 79.0 (49.07%) | 69.3 (43.06%) |
| Highest minus lowest | 3 words (1.78 points) | 7 words (4.14 points) | 6 words (3.73 points) | 8 words (4.97 points) |
| Words better than the adapter on stock at the same seed | | -6, +2, +3 | | +13, +2, +14 |

Same-seed gains of the adapter on F-1 over the adapter on stock (positive means the adapter on F-1 is better). "Prompts together" uses `--cluster-by text`, where M01's 65 clips hold 54 different prompts and M04's 64 clips hold 45:

| Seed | M01 gain | M01 interval, clips alone | M01 interval, prompts together | M04 gain | M04 interval, clips alone | M04 interval, prompts together |
| --- | ---: | --- | --- | ---: | --- | --- |
| 17 | -3.55 | -13.07 to 6.59 | -14.87 to 7.21 | 8.07 | 1.38 to 15.43 | 0.45 to 16.00 |
| 18 | 1.18 | -6.94 to 9.81 | -7.41 to 9.55 | 1.24 | -5.84 to 9.02 | -6.31 to 9.16 |
| 19 | 1.78 | -7.53 to 10.43 | -8.29 to 11.39 | 8.70 | 2.42 to 15.17 | 1.77 to 15.79 |

M04's lead repeats. The new seeds scored 74 and 68, and all three adapter-on-F-1 seeds (66, 68, 74) beat all three adapter-on-stock seeds (76, 79, 82). The means are 43.06% and 49.07%, 6.01 points or about 10 words apart, which is less than seed 17's 8.07 suggested. Seed 18 led by only 2 words, with an interval that includes zero. On seeds 17 and 19 the interval excludes zero.

M01 shows no difference. Seeds 18 and 19 scored 85 and 83, inside and one word below the adapter on stock's 84 to 87, and the means are 86.0 and 85.7 words (50.89% and 50.69%). Seed 17's -3.55 came from the worst of the three adapter-on-F-1 seeds. Against the F-1 model alone the adapter on F-1 was ahead on all three seeds, by 7, 12 and 14 words, but shown on none. Seed 18's interval touches zero.

The adapter on F-1's spread was 7 words (4.14 points) on M01 and 8 words (4.97 points) on M04, against 3 and 6 for the adapter on stock. Both are over the 4-point line, M01 only just, so single runs can't settle comparisons at this scale and the three-seed numbers are the ones to use. The spread leaves out the F-1 model's own training randomness.

Grouping by prompt withdrew nothing. Every verdict stayed the same, and the grouped intervals got a bit wider: M04's seed-17 lower end went from 1.38 to 0.45 and seed 19's from 2.42 to 1.77. The grouped figures are the more careful ones to quote. All 22 comparisons are under Full Results.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

Written on 1 Oct 2026 at 18:58, before any run of this test. At 18:56 I pinned down what "rate", "gain" and "clear of zero" mean and rewrote the questions in plain words, and those are the questions below. "The stack" means the adapter on F-1. The answers are my words as written, except that a doubled "2." before answer 2 is removed. The bracketed notes record how each answer was read before the run, without changing it.

1. M04 (161 words): the stack got 66 wrong on seed 17. About how many will it get wrong on seeds 18 and 19? (The adapter on stock got 79, 76 and 82.)

I expect seeds 18/17 to stay within a 5 point range on M04

[Read as: across seeds 17, 18 and 19, the adapter on F-1's M04 scores land within 5 points of each other, so 8 words or fewer (8 words is 4.97 points, 9 is 5.59). "18/17" is taken as a slip for the new seeds 18 and 19, with 17 as the third. The question asked for a count and the answer gives a spread, so it's scored as a spread and no count is scored. With seed 17 at 66 wrong, the new seeds would sit between 58 and 74. For scale, the adapter on stock's three seeds were 6 words (3.73 points) apart on M04 and 3 words (1.78) on M01.]

2. M01 (169 words): the stack got 90 wrong on seed 17, and the adapter on stock got 84, 87 and 86. Will it land better than 84, near 90, or worse than 90?

I expect the stack to land near 84 on M01

[Read as: seeds 18 and 19 each land near 84 wrong on M01. "Near" uses the 4-point guide from the earlier notes, so 77 to 91 wrong (7 words either side of 84, 45.56% to 53.85%). Two things about that. Seed 17 of the adapter on F-1 scored 90, 6 words above 84 and already inside the band, so the band is wide and a repeat of 90 would count as a hit. And landing near 84 means the adapter on F-1 does about what the adapter on stock does on M01, so this answer predicts no effect of the F-1 base on M01.]

3. In one sentence: does M04's lead come from the F-1 base or from luck on seed 17, and why?

I think the lead came from the F-1 base.

[Read as: M04's lead over the adapter on stock (13 words, 8.07 points at seed 17) comes from the F-1 base and not from seed 17, so it repeats on seeds 18 and 19. It's tested as both new M04 seeds scoring 75 or fewer wrong, below the adapter on stock's best seed (76). No reason was given with the answer. Answers 1 and 3 can both hold, since new M04 seeds between 58 and 74 satisfy both. Answers 2 and 3 together say the F-1 base helps M04 and does nothing on M01, and nothing in this test explains why. For scale, the adapter on stock moved 6 words on M04 across its seeds, and the lead at seed 17 is 13 words.]

### Commands and Setup

Run from `app/` in the WSL tree on Yack after `git pull`, with the GPU free and Windows apps closed. Training and decoding run in `~/venvs/willspeak-train`, and `evaluate.py` and `ci.py` in `app/.venv`. The data is `data/public/torgo/export/M01` and `data/public/torgo/export/M04`, the same test clips as before.

In the first run each speaker took about 3 minutes from start of training to end of decode (2m 03s and 2m 19s of training, the rest loading the F-1 file and decoding). So the four runs take about 12 minutes, plus about a minute of scoring. After the input check the script stops at its first error. I dry-ran it on a stub copy of the tree first. That checks the shell logic and the arguments each call gets (the right models paired in every comparison, and the summary logic on made-up numbers), and nothing about the results.

The script, saved as `logs/adapter-on-f1-seeds.sh` and run with its output copied to `logs/adapter-on-f1-seeds-full-<date and time>.txt`:

```bash
cd ~/willspeak/app
EX=data/public/torgo/export
FT=reports/finetune-torgo
SC=.venv/bin/python
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
echo "code: $(git log -1 --format='%h %ad' --date=format:%H:%M)"
echo "train.py sha256: $(sha256sum scripts/asr/train.py | cut -c1-12)   (the file the cleanup check passed: c696b4873186)"
for S in M01 M04; do
  for p in models/parakeet-ft-f1-loso-$S-bottom8/parakeet-ft.nemo $EX/$S/test.jsonl $FT/$S-test-stock-nemo/results.json $FT/f1-loso-$S-bottom8-on-F2-test-nemo/results.json $FT/f2-$S-adapter8-test-nemo/results.json $FT/f2-$S-adapter8-recheck-seed18-test-nemo/results.json $FT/f2-$S-adapter8-recheck-seed19-test-nemo/results.json $FT/f2-$S-adapter8-on-f1-test-nemo/results.json; do
    [ -e "$p" ] || { echo "MISSING $p, stopping"; exit 1; }
  done
  for SEED in 18 19; do
    [ -e models/parakeet-ft-f2-$S-adapter8-on-f1-seed$SEED ] && { echo "ALREADY EXISTS models/parakeet-ft-f2-$S-adapter8-on-f1-seed$SEED, stopping"; exit 1; }
  done
done
echo "inputs ok"
source ~/venvs/willspeak-train/bin/activate
set -e
for S in M01 M04; do
  F1=models/parakeet-ft-f1-loso-$S-bottom8/parakeet-ft.nemo
  for SEED in 18 19; do
    N=f2-$S-adapter8-on-f1-seed$SEED
    echo; echo "######## $S seed $SEED, started $(date +%H:%M:%S)"
    python scripts/asr/train.py --export $EX/$S --base $F1 --out models/parakeet-ft-$N --adapter-after-blocks 8 --lr 3e-4 --batch 2 --grad-accum 8 --warmup-steps 10 --epochs 15 --select last --no-keep-stock --freeze-batchnorm --normalize-targets --dev-scoring evaluate --seed $SEED
    f=$(ls -t logs/train_parakeet-*.log | head -1)
    echo "-- log lines from $f"
    grep -hE "Base model|loading|parameters exact|fixed schedule|kept epochs|Checking the averaged|averaged|Released|Elapsed|saved .*adapter8-on-f1" "$f" || true
    python scripts/asr/nemo_hyp.py --nemo $F1 --adapter models/parakeet-ft-$N/adapter.pt --manifest $EX/$S/test.jsonl --audio-root $EX/$S --out $FT/$N-test-nemo.jsonl
  done
done
deactivate
for S in M01 M04; do
  M=$EX/$S
  for SEED in 18 19; do
    R=f2-$S-adapter8-on-f1-seed$SEED-test-nemo
    A=f2-$S-adapter8-recheck-seed$SEED-test-nemo
    echo; echo "######## scoring $S seed $SEED"
    $SC scripts/asr/evaluate.py --manifest $M/test.jsonl --audio-root $M --hyp $FT/$R.jsonl --group-by kind --out $FT/$R
    echo "== $S stack seed $SEED against the stack seed 17"
    $SC scripts/asr/ci.py --manifest $M/test.jsonl --stock $FT/f2-$S-adapter8-on-f1-test-nemo --finetuned $FT/$R
    echo "== $S stack seed $SEED against the adapter on stock, seed $SEED"
    $SC scripts/asr/ci.py --manifest $M/test.jsonl --stock $FT/$A --finetuned $FT/$R
    echo "== $S stack seed $SEED against the F-1 model"
    $SC scripts/asr/ci.py --manifest $M/test.jsonl --stock $FT/f1-loso-$S-bottom8-on-F2-test-nemo --finetuned $FT/$R
    echo "== $S stack seed $SEED against the adapter on stock, seed $SEED, resampled by prompt"
    $SC scripts/asr/ci.py --manifest $M/test.jsonl --stock $FT/$A --finetuned $FT/$R --cluster-by text
  done
done
for S in M01 M04; do
  M=$EX/$S; R=f2-$S-adapter8-on-f1-test-nemo
  for pair in "stock|$S-test-stock-nemo" "the adapter on stock, seed 17|f2-$S-adapter8-test-nemo" "the F-1 model|f1-loso-$S-bottom8-on-F2-test-nemo"; do
    echo; echo "== $S stack seed 17 against ${pair%%|*}, resampled by prompt"
    $SC scripts/asr/ci.py --manifest $M/test.jsonl --stock $FT/${pair##*|} --finetuned $FT/$R --cluster-by text
  done
done
echo; echo "######## summary"
python3 - <<'PY'
import json
ft = "reports/finetune-torgo/"
words = {"M01": 169, "M04": 161}
def get(n):
    r = json.load(open(ft + n + "/results.json")); w = r["word"]
    return r["wer"] * 100, w["sub"] + w["dele"] + w["ins"]
for s in ("M01", "M04"):
    pt = 100.0 / words[s]
    rows = [("F-1 model alone", "f1-loso-%s-bottom8-on-F2-test-nemo" % s)]
    for k, n in ((17, "f2-%s-adapter8-test-nemo"), (18, "f2-%s-adapter8-recheck-seed18-test-nemo"), (19, "f2-%s-adapter8-recheck-seed19-test-nemo")):
        rows.append(("adapter on stock, seed %d" % k, n % s))
    for k, n in ((17, "f2-%s-adapter8-on-f1-test-nemo"), (18, "f2-%s-adapter8-on-f1-seed18-test-nemo"), (19, "f2-%s-adapter8-on-f1-seed19-test-nemo")):
        rows.append(("adapter on F-1, seed %d" % k, n % s))
    print("== %s (%d words)" % (s, words[s]))
    res = {}
    for label, n in rows:
        wer, e = get(n)
        res[label] = e
        print("%-28s %6.2f%% %4d wrong" % (label, wer, e))
    st = [res["adapter on F-1, seed %d" % k] for k in (17, 18, 19)]
    ad = [res["adapter on stock, seed %d" % k] for k in (17, 18, 19)]
    print("adapter on F-1 spread: %d words, %.2f points; adapter on stock spread: %d words, %.2f points" % (max(st) - min(st), (max(st) - min(st)) * pt, max(ad) - min(ad), (max(ad) - min(ad)) * pt))
    print("adapter on F-1 mean %.2f%%; adapter on stock mean %.2f%%" % (sum(st) / 3 * pt, sum(ad) / 3 * pt))
    for k in (18, 19):
        e = res["adapter on F-1, seed %d" % k]
        where = "below" if e < min(ad) else ("above" if e > max(ad) else "inside")
        print("seed %d: %d wrong, %s the adapter on stock's range %d to %d" % (k, e, where, min(ad), max(ad)))
    if s == "M04":
        print("answer 1, all three seeds within 5 points of each other: %s" % ("yes" if (max(st) - min(st)) * pt <= 5 else "no"))
        for k in (18, 19):
            e = res["adapter on F-1, seed %d" % k]
            print("answer 3, seed %d at 75 wrong or fewer: %s" % (k, "yes" if e <= 75 else "no"))
    else:
        for k in (18, 19):
            e = res["adapter on F-1, seed %d" % k]
            print("answer 2, seed %d between 77 and 91 wrong: %s" % (k, "yes" if 77 <= e <= 91 else "no"))
PY
echo "code at the end: $(git log -1 --format='%h %ad' --date=format:%H:%M)"
echo "finished $(date '+%Y-%m-%d %H:%M:%S')"
```

Before trusting the scores, check each training log. There should be a `Base model` line and a `loading` line naming `models/parakeet-ft-f1-loso-<SPK>-bottom8/parakeet-ft.nemo`. If either says `nvidia/parakeet-tdt-0.6b-v2`, the run used the stock model and isn't this test. Then look for `1049600 trainable`, `fixed schedule (--select last): kept epochs 12, 13, 14, 15`, `Checking the averaged model on dev before release...`, `averaged 4 checkpoints`, a `Released model dev WER` that's a number, and a saved `adapter.pt`. There's no `stock dev WER` line, because `--no-keep-stock` skips it. The decode should print `Added the adapter ... after encoder blocks 0-7` and 65 clips for M01 or 64 for M04. The `train.py` hash on the third line of output should be c696b4873186, the file the cleanup check passed. A different hash means the code changed and should be recorded.

As run, the script finished without an error, from about 19:11 (the time in the saved file's name, `adapter-on-f1-seeds-full-1001-1911.txt`) to 19:24:57. The commit at the end was 3c2585e, the same as the first adapter-on-F-1 run, so there's no known code change between the two runs. The copied console output starts at epoch 12 of the M04 seed-18 training. A grep of the saved file afterwards (about 19:35) showed the third line (the `train.py` hash, c696b4873186, the file the cleanup check passed), the `inputs ok` line, and the `Base model` and `loading` lines of all four trainings.

I still haven't seen the start time and starting commit lines, the rest of the M01 seed 18 and 19 training and decode output (parameters, schedule, averaging, dev WER, elapsed time, saved adapter, clip count), or the live console lines of M04 seed 18 before epoch 12. That run's grep block after training is complete, with all 15 epochs. The saved file on Yack holds all of it. The scoring blocks and the summary are complete.

For the base-model check, the `Base model` and `loading` lines of all four trainings name the speaker's F-1 file. For M01 that's `models/parakeet-ft-f1-loso-M01-bottom8/parakeet-ft.nemo` (seed 18 at 19:11:48, seed 19 at 19:15:01), and for M04 it's `models/parakeet-ft-f1-loso-M04-bottom8/parakeet-ft.nemo` (seed 18 at 19:18:04, seed 19 at 19:21:27). The M04 lines were also in the copied output, with 1049600 trainable. I didn't see the M01 trainable count. As a second check, `nemo_hyp.py` stops if an adapter's recorded base isn't the model it's attached to, and both M01 adapters decoded with the F-1 file.

### Full Results

Log lines that were seen. They come from the grep block after each M04 training run in the copied output, and for M01's `Base model` and `loading` lines, from the later grep of the saved file. The `train_parakeet-20261002-001803-915790.log` and `train_parakeet-20261002-002127-830275.log` file names are in UTC:

| Line | M04 seed 18 | M04 seed 19 | M01 seeds 18 and 19 |
| --- | --- | --- | --- |
| Base model, and the `loading` line | `models/parakeet-ft-f1-loso-M04-bottom8/parakeet-ft.nemo` | the same | `models/parakeet-ft-f1-loso-M01-bottom8/parakeet-ft.nemo` on both (seen afterwards) |
| `parameters exact` | 618875526 total, 1049600 trainable | the same | not seen |
| Seed in the run header | not seen | 19 | not seen |
| Schedule line | kept epochs 12, 13, 14, 15; dev chose nothing | the same | not seen |
| Averaging | `averaged 4 checkpoints`, then `Checking the averaged model on dev before release...` | the same | not seen |
| `Released model dev WER` | 56.00% | 56.00% | not seen |
| Elapsed | 2m 28s | 2m 28s | not seen |
| Saved | `models/parakeet-ft-f2-M04-adapter8-on-f1-seed18/adapter.pt` (4.2 MB) | `...seed19/adapter.pt` (4.2 MB) | not seen |
| `nemo_hyp.py` | `Added the adapter ... after encoder blocks 0-7`; 64 clips | the same | not seen |

The grep of the saved file, run on Yack afterwards. Each log line appears twice in the saved file, once live and once in the grep block after training, and the numbers are line numbers in that file:

```
grep -nE "train.py sha256|inputs ok|Base model|loading" ~/willspeak/app/logs/adapter-on-f1-seeds-full-1001-1911.txt
3:train.py sha256: c696b4873186   (the file the cleanup check passed: c696b4873186)
4:inputs ok
8:[19:11:48] [train]   Base model       : models/parakeet-ft-f1-loso-M01-bottom8/parakeet-ft.nemo
27:[19:12:07] [train] loading models/parakeet-ft-f1-loso-M01-bottom8/parakeet-ft.nemo
99:[19:11:48] [train]   Base model       : models/parakeet-ft-f1-loso-M01-bottom8/parakeet-ft.nemo
100:[19:12:07] [train] loading models/parakeet-ft-f1-loso-M01-bottom8/parakeet-ft.nemo
135:[19:15:01] [train]   Base model       : models/parakeet-ft-f1-loso-M01-bottom8/parakeet-ft.nemo
154:[19:15:10] [train] loading models/parakeet-ft-f1-loso-M01-bottom8/parakeet-ft.nemo
226:[19:15:01] [train]   Base model       : models/parakeet-ft-f1-loso-M01-bottom8/parakeet-ft.nemo
227:[19:15:10] [train] loading models/parakeet-ft-f1-loso-M01-bottom8/parakeet-ft.nemo
262:[19:18:04] [train]   Base model       : models/parakeet-ft-f1-loso-M04-bottom8/parakeet-ft.nemo
281:[19:18:14] [train] loading models/parakeet-ft-f1-loso-M04-bottom8/parakeet-ft.nemo
353:[19:18:04] [train]   Base model       : models/parakeet-ft-f1-loso-M04-bottom8/parakeet-ft.nemo
354:[19:18:14] [train] loading models/parakeet-ft-f1-loso-M04-bottom8/parakeet-ft.nemo
389:[19:21:27] [train]   Base model       : models/parakeet-ft-f1-loso-M04-bottom8/parakeet-ft.nemo
408:[19:21:37] [train] loading models/parakeet-ft-f1-loso-M04-bottom8/parakeet-ft.nemo
480:[19:21:27] [train]   Base model       : models/parakeet-ft-f1-loso-M04-bottom8/parakeet-ft.nemo
481:[19:21:37] [train] loading models/parakeet-ft-f1-loso-M04-bottom8/parakeet-ft.nemo
```

Order in the file: M01 seed 18, M01 seed 19, M04 seed 18, M04 seed 19. The `train.py` hash matches the cleanup check, and no run used the stock model.

For comparison, seed 17 on M04 released at a dev WER of 48.00% (25 dev words, so one word is 4.0 points) and took 2m 19s.

Test scores of each run, as printed by `evaluate.py`:

| Run | WER | CER | PER |
| --- | --- | --- | --- |
| M01 adapter on F-1, seed 17 | 53.25% | 35.10% | 33.77% |
| M01 adapter on F-1, seed 18 | 50.30% | 34.42% | 34.44% |
| M01 adapter on F-1, seed 19 | 49.11% | 32.52% | 31.95% |
| M04 adapter on F-1, seed 17 | 40.99% | 32.54% | 31.43% |
| M04 adapter on F-1, seed 18 | 45.96% | 34.47% | 35.66% |
| M04 adapter on F-1, seed 19 | 42.24% | 32.99% | 31.99% |

All comparisons, from `ci.py` (95% paired bootstrap, 2000 resamples, seed 0, over 65 clips for M01 and 64 for M04, or 54 and 45 prompts when prompts are kept together). Gain is the "against" model's WER minus the adapter on F-1's, so positive means the adapter on F-1 is better. The seed 17 rows with clips alone come from the first adapter-on-F-1 run. I only re-scored the three seed-17 comparisons and each new seed's same-seed comparison with prompts together, so the other rows say "not run":

| Speaker | Seed of the adapter on F-1 | Against | Gain | Interval, each clip alone | Interval, prompts together |
| --- | --- | --- | ---: | --- | --- |
| M01 | 17 | stock | 24.26 | 14.88 to 36.13 | 15.87 to 33.94 (54 prompts) |
| M01 | 17 | the adapter on stock, same seed | -3.55 | -13.07 to 6.59 | -14.87 to 7.21 (54 prompts) |
| M01 | 17 | the F-1 model | 4.14 | -4.26 to 13.66 | -5.26 to 13.29 (54 prompts) |
| M01 | 18 | the adapter on F-1, seed 17 | 2.96 | -5.00 to 10.66 | not run |
| M01 | 18 | the adapter on stock, same seed | 1.18 | -6.94 to 9.81 | -7.41 to 9.55 (54 prompts) |
| M01 | 18 | the F-1 model | 7.10 | 0.00 to 15.21 | not run |
| M01 | 19 | the adapter on F-1, seed 17 | 4.14 | -1.94 to 9.80 | not run |
| M01 | 19 | the adapter on stock, same seed | 1.78 | -7.53 to 10.43 | -8.29 to 11.39 (54 prompts) |
| M01 | 19 | the F-1 model | 8.28 | -1.28 to 17.61 | not run |
| M04 | 17 | stock | 39.75 | 25.55 to 56.50 | 25.34 to 55.50 (45 prompts) |
| M04 | 17 | the adapter on stock, same seed | 8.07 | 1.38 to 15.43 | 0.45 to 16.00 (45 prompts) |
| M04 | 17 | the F-1 model | 24.22 | 13.73 to 35.86 | 13.71 to 36.52 (45 prompts) |
| M04 | 18 | the adapter on F-1, seed 17 | -4.97 | -11.91 to 1.74 | not run |
| M04 | 18 | the adapter on stock, same seed | 1.24 | -5.84 to 9.02 | -6.31 to 9.16 (45 prompts) |
| M04 | 18 | the F-1 model | 19.25 | 8.53 to 31.52 | not run |
| M04 | 19 | the adapter on F-1, seed 17 | -1.24 | -8.03 to 4.73 | not run |
| M04 | 19 | the adapter on stock, same seed | 8.70 | 2.42 to 15.17 | 1.77 to 15.79 (45 prompts) |
| M04 | 19 | the F-1 model | 22.98 | 12.71 to 34.93 | not run |

The summary, as printed:

```text
######## summary
== M01 (169 words)
F-1 model alone               57.40%   97 wrong
adapter on stock, seed 17     49.70%   84 wrong
adapter on stock, seed 18     51.48%   87 wrong
adapter on stock, seed 19     50.89%   86 wrong
adapter on F-1, seed 17       53.25%   90 wrong
adapter on F-1, seed 18       50.30%   85 wrong
adapter on F-1, seed 19       49.11%   83 wrong
adapter on F-1 spread: 7 words, 4.14 points; adapter on stock spread: 3 words, 1.78 points
adapter on F-1 mean 50.89%; adapter on stock mean 50.69%
seed 18: 85 wrong, inside the adapter on stock's range 84 to 87
seed 19: 83 wrong, below the adapter on stock's range 84 to 87
answer 2, seed 18 between 77 and 91 wrong: yes
answer 2, seed 19 between 77 and 91 wrong: yes
== M04 (161 words)
F-1 model alone               65.22%  105 wrong
adapter on stock, seed 17     49.07%   79 wrong
adapter on stock, seed 18     47.20%   76 wrong
adapter on stock, seed 19     50.93%   82 wrong
adapter on F-1, seed 17       40.99%   66 wrong
adapter on F-1, seed 18       45.96%   74 wrong
adapter on F-1, seed 19       42.24%   68 wrong
adapter on F-1 spread: 8 words, 4.97 points; adapter on stock spread: 6 words, 3.73 points
adapter on F-1 mean 43.06%; adapter on stock mean 49.07%
seed 18: 74 wrong, below the adapter on stock's range 76 to 82
seed 19: 68 wrong, below the adapter on stock's range 76 to 82
answer 1, all three seeds within 5 points of each other: yes
answer 3, seed 18 at 75 wrong or fewer: yes
answer 3, seed 19 at 75 wrong or fewer: yes
code at the end: 3c2585e 17:06
finished 2026-10-01 19:24:57
```

Training curves of the three M04 runs. `curve.csv` counts epochs from 0, and this table counts from 1 like the console. Loss is the logged training loss. I didn't see the M01 seed 18 and 19 curves:

| Epoch | Seed 17 dev WER | Seed 17 loss | Seed 18 dev WER | Seed 18 loss | Seed 19 dev WER | Seed 19 loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 72.00% | 2.4493 | 72.00% | 2.4449 | 68.00% | 2.4369 |
| 2 | 72.00% | 2.1016 | 68.00% | 2.1268 | 72.00% | 2.2077 |
| 3 | 56.00% | 1.9299 | 68.00% | 1.8990 | 76.00% | 2.1042 |
| 4 | 64.00% | 1.6899 | 56.00% | 1.7435 | 64.00% | 1.6681 |
| 5 | 60.00% | 1.4947 | 60.00% | 1.4898 | 64.00% | 1.5886 |
| 6 | 56.00% | 1.5128 | 56.00% | 1.5683 | 64.00% | 1.5280 |
| 7 | 52.00% | 1.5180 | 60.00% | 1.5298 | 64.00% | 1.3517 |
| 8 | 60.00% | 1.4890 | 52.00% | 1.3024 | 64.00% | 1.4006 |
| 9 | 60.00% | 1.2424 | 52.00% | 1.2205 | 60.00% | 1.3704 |
| 10 | 56.00% | 1.2439 | 56.00% | 1.2494 | 68.00% | 1.1804 |
| 11 | 52.00% | 1.2376 | 56.00% | 1.2972 | 60.00% | 1.2488 |
| 12 | 48.00% | 1.1341 | 56.00% | 1.1013 | 56.00% | 1.0180 |
| 13 | 48.00% | 1.2762 | 56.00% | 1.3284 | 56.00% | 1.2659 |
| 14 | 52.00% | 1.0921 | 56.00% | 1.2285 | 56.00% | 1.0253 |
| 15 | 56.00% | 1.1671 | 56.00% | 1.2116 | 56.00% | 1.1499 |

### Conclusions

M04's lead over the adapter on stock repeats. By the rule set before the run both new seeds had to score 75 or fewer, and they scored 74 and 68. All three adapter-on-F-1 seeds beat all three adapter-on-stock seeds, and the three-seed means are 43.06% and 49.07%. The lead is about 6 points, smaller than the 8.07 from seed 17 alone, and seed 18's same-seed gain is 2 words with an interval that includes zero. One point I only worked out after the run: with three seeds per side, if the two recipes were the same, all three of one side falling below all three of the other in a chosen direction has a 1 in 20 chance. That only counts seed luck, since every run is scored on the same 64 clips.

M01 shows no difference between the two adapters. Seeds 18 and 19 scored 85 and 83 against the adapter on stock's 84 to 87, the three-seed means are 86.0 and 85.7 words, and seed 17's 90 was the worst of the three. So the first run's -3.55 came from seed 17 being the worst draw, and it doesn't repeat. Against the F-1 model alone the adapter on F-1 is ahead on all three seeds and shown on none.

Keeping prompts together changed no verdict. Every interval that excluded zero still does, and every one that included zero still does. With prompts together, M04's seed-17 lead over the adapter on stock has a lower end of 0.45. The grouped intervals are the ones to quote in the report.

The adapter on F-1 has more seed-to-seed spread than the adapter on stock (7 and 8 words, against 3 and 6), so one seed per speaker isn't enough to compare recipes at the 4-point scale. The spread also leaves out the F-1 model's own training randomness, which would add to it. The run-to-run spread of the whole recipe (train the F-1 model, then the adapter) isn't measured here.

A shared base plus a personal adapter is supported on M04 and does no measurable harm on M01. It hasn't been tried on a third speaker. M04's three-seed mean (43.06%) is about 15 points above my goal in the high 20s, and M01's (50.89%) about 23 points above. No run so far gets there.

Scoring the predictions against the readings written before the run.

Answer 1 (M04's three seeds within 5 points of each other) held. The scores were 66, 74 and 68 wrong, a range of 8 words or 4.97 points. It passed by 0.03 points, so a ninth word would have failed it. The adapter on stock's range was 6 words (3.73 points).

Answer 2 (the adapter on F-1 near 84 on M01, read as seeds 18 and 19 each between 77 and 91) held. The seeds scored 85 and 83, one word either side of 84, with a mean of exactly 84.0. The band was wide, and seed 17's 90 was the outlier.

Answer 3 (M04's lead came from the F-1 base, read as both new seeds at 75 or fewer) held, at 74 and 68. Seed 18's margin is thin, 2 words under the adapter on stock's best seed (76). The test shows that the lead repeats but not why, and it doesn't vary the F-1 model's own training. So "came from the F-1 base" fits the data, but the test doesn't isolate it.

The answers were consistent with each other, as the pre-run notes said they could be. M04's new seeds landed in the 58 to 74 range that satisfies answers 1 and 3 together, and answer 2's "no effect of the F-1 base on M01" held alongside answer 3's effect on M04. Nothing in this test explains why the two speakers differ.

What this doesn't show is listed under Limits: mainly why M04 benefits and M01 doesn't, the F-1 model's own training spread, and other speakers.

Possible next tests, none chosen yet: the same adapter on F-1 for M02 (first check that its bottom-8 F-1 model has a score on M02's F-2 test clips), whole blocks retrained on top of an F-1 base, and F-3 once the choice between equal epochs and equal training steps is settled.

</details>

## Related

[F-2: M01 and M04, adapter after 8 blocks, on the F-1 model](../f2-m01-m04-adapter8-on-f1/notes.md), [F-2: M01 and M04, adapter after 8 blocks](../f2-m01-m04-adapter8/notes.md), [F-2: M02, adapter after 8 blocks, seeds 18 and 19](../f2-m02-adapter8-seeds/notes.md), [F-1: leave-one-speaker-out, bottom 8 blocks](../f1-loso-bottom8/notes.md), [Check: cleaned-up train.py](../check-train-py-cleanup/notes.md)
