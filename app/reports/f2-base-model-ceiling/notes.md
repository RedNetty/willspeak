# F-2: base-model ceiling. 1.1B, 0.6B v3 and a 110M hybrid with the adapter (M01, M02 and M04)

Proposal: F-2 (single-speaker fine-tuning, §5.2 / H3), as a check on the model choice. The 1.1B model only measures a ceiling and won't go on the Pi. The 110M model is the small-device test.

Status: set up 2 Oct 2026. The adapter runs have never trained. The `1016-followup` queue (16 October) marked B done without running it, because B's completion marker was `M04-test-stock-110m-nemo`, a stock decode lab test 4 had already written on 7 October. The marker now points at an adapter result. The three stock decodes do exist from lab test 4 and are scored below. Stage B of `lab.py` is three stock decodes per model, then 15 adapter runs, about 1.5 to 2 hours of GPU plus downloads.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

I won't take anything above 35% word error to the prototype expo, and the best pooled raw figure with the current model, Parakeet-TDT-0.6B-v2, is 48.5%. I picked that model because it fits on the Raspberry Pi. This test asks how much of the gap is the model and how much is the data. It trains the same adapter (fixed schedule, seed 17) on three other NeMo models and compares them with the 0.6B v2 adapter, which got 201 of 425 wrong on that seed. A much better model would mean model size is part of the limit. If none is better, it's the data.

On 2 Oct I also looked for a smaller model for a smaller device. None of the smaller ones beats 0.6B v2 on ordinary speech. Mean WERs from the model cards on the Open ASR leaderboard, read 2 Oct (the test sets may have changed between card dates, so these are rough):

| Model | Size | Mean WER on its card | Can `train.py` train it? |
|---|---|---|---|
| parakeet-tdt-0.6b-v2 (the current model) | 600 M | 6.05 | yes |
| parakeet-tdt-1.1b | 1.1 B | 7.02 | yes |
| canary-180m-flash | 182 M | 7.12 | no, it is a multi-task model and needs new code |
| parakeet-tdt_ctc-110m | about 110 M | 7.49 | probably; it is a hybrid with a second decoder (this test) |
| moonshine-base | 61 M | 9.99 | no, it is a Hugging Face Transformers model and needs a new script |

The 110M hybrid is the only small model the existing scripts can train, so it's in. That gives four sizes (110 M, 600 M twice, 1.1 B), and the summary shows what the adapter wins back on each. I haven't reviewed the literature on adapting small models to dysarthric speech, so I'm not claiming this is new. If there's a useful trend, the next step is a trainer for canary-180m-flash or Moonshine.

## Setup

The adapter is the 1,049,600-number layer `h + W h + b` after encoder block N, trained alone. Its width follows the encoder, so a narrower model gets a smaller adapter. Pooled is the sum of wrong words over M01 (169 test words), M02 (95) and M04 (161), 425 in all. Stock is the downloaded model untouched.

From the Hugging Face cards (read 2 Oct):

1. `nvidia/parakeet-tdt-1.1b`: FastConformer-TDT, about 1.1 billion parameters, 64,000 hours of English, CC-BY-4.0, mean WER 7.02. I'm assuming 42 blocks of width 1024 (not on the card, not checked). Adapters after block 8, like the 0.6B runs, and block 14, about the same share of depth as 8 of 24.
2. `nvidia/parakeet-tdt-0.6b-v3`: 600 million parameters, 25 European languages, CC-BY-4.0. It detects the language itself, so a dysarthric English clip might come out as another language. The stock decode will show that.
3. `nvidia/parakeet-tdt_ctc-110m`: about 110 million parameters ("around 114M" elsewhere on the card), TDT and CTC decoders, CC-BY-4.0, mean WER 7.49, 16 kHz mono, TDT decoding by default. I'm guessing 17 blocks of width 512. The run prints the trainable count: 262,656 means width 512, 1,049,600 means 1024. Adapters after block 6 (about a third of 17) and block 8.

The 1.1B is about twice the device model's size, so it only measures a ceiling and wouldn't replace anything. v3 is the same size as v2 and could replace it. The 110M is about a fifth. `train.py --base NAME` and `nemo_hyp.py --pretrained NAME` already take a model name, and the adapter file records its base model so it can't be decoded with the wrong one.

The script handles the obvious failures. If the 1.1B doesn't fit in the 4060's 8 GB at batch 2, the failed run is retried at batch 1 with 16 accumulation steps (same 16 clips per update) under a name ending `-b1`, and that model's other runs use batch 1. A failed stock decode skips that model's training and gets listed. If a model's first run fails at both batch sizes, its other runs are skipped, so read the first error. `train.py` also stops if a model has too few blocks. Its disk estimate assumes 24 blocks and is only a guide.

The 110M model has risks I can't remove. NeMo hybrids add a CTC loss to the TDT loss by default (I didn't check the weight), so even with the CTC decoder frozen the adapter trains on a slightly different objective. The dev-score patch in `train.py` changes NeMo's `WER` class, which both decoders use, and it's untested on a hybrid. `nemo_hyp.py` uses the `transcribe` default, which the card says is TDT. Any of these would show up as a missing or odd row. Downloads are about 4 GB for the 1.1B, which is also roughly twice as slow per step, so the run time is only an estimate.

## Plan

1. Check that the 0.6B v2 stock and seed-17 adapter results exist and no output does.
2. Per model, decode and score the three test sets with stock. If that works, train, decode and score the adapters (1.1B: blocks 8 and 14; v3: block 8; 110M: blocks 6 and 8) on all three speakers, seed 17, fixed schedule.
3. Print wrong words per speaker and pooled for every stock and adapter model, with the lowest new adapter count. Then a table, smallest model first, of stock, best adapter and the share of stock errors fixed. Then `ci.py` intervals per speaker against the 0.6B v2 adapter.

How I planned to read it (written before the run): 148 or fewer pooled (35.0% of 425 is 148.75) would mean a bigger model could reach the bar, and the device would need more hardware or a server. That would be a one-seed ceiling. More than 25 words better than 201 but above 148 means size helps but isn't enough. Within 25 words of 201 means the 0.6B isn't the limit and data quantity is next. For the 110M, an adapter result below 323 (v2 stock) would mean a model a fifth the size beats the big one untouched, useful for a small device. At or above 323, the adapter can't rescue it.

Positions are picked on the test words, which flatters models tried at two positions a little; the second table notes this. One seed per cell, so gaps under about 25 words aren't read as real.

## Why

The base model is the only lever that moves everything at once, and it's one afternoon of GPU. If the answer is "no better", I can stop looking at models and spend the next weeks on recordings. The small model tells me what a smaller device would cost with the adapter in place.

## Limits

This gives stock and adapter scores for three models on the three TORGO test sets, one seed and one schedule, and how the adapter's gain changes with size. It says nothing about running them on the Pi. Positions and learning rate aren't tuned (blocks 8 and 14 on the 1.1B, 6 and 8 on the 110M are guesses). It can't rank models within 25 words, it doesn't try Whisper, Canary, Moonshine or anything `train.py` can't train, and it says nothing about the client's speech.

## Reading and Video

None added.

## Results

No adapter result yet. Stage B hasn't trained anything. The `1016-followup` queue (`logs/lab-queue-1016-followup.status.json`) lists B as done only because `reports/finetune-torgo/M04-test-stock-110m-nemo` existed from lab test 4 on 7 October. No `f2-<speaker>-adapter*-1p1b|v3|110m-seed17` model or decode exists, so the earlier State remark that "B was already complete" was wrong.

Stock decodes from lab test 4 (7 October, NeMo, pooled M01 + M02 + M04, 425 reference words, lower is better):

| Model | M01 (169) | M02 (95) | M04 (161) | Pooled | Single words (122) | Sentence words (303) | Substitutions / deletions / insertions |
|---|---:|---:|---:|---:|---:|---:|---|
| 0.6B v2 (current) | 131 | 62 | 130 | 323 (76.0%) | 135 | 188 | 195 / 46 / 82 |
| 1.1B | 116 | 55 | 136 | 307 (72.2%) | 133 | 174 | 215 / 31 / 61 |
| 0.6B v3 | 122 | 57 | 140 | 319 (75.1%) | 123 | 196 | 197 / 53 / 69 |
| 110M hybrid | 121 | 65 | 129 | 315 (74.1%) | 119 | 196 | 237 / 25 / 53 |

Lab test 4's intervals couldn't separate any of them from v2: 1.1B was +3.8 points (CI -3.5 to +10.7), v3 +0.9 (CI -5.9 to +7.8).

Hypothesis 3 (v3 within 20 words of v2, 303 to 343) held, 319 against 323. Hypothesis 4 (110M worse, 335 to 395) didn't. The 110M made 315 errors, 8 fewer than v2 and inside the no-difference interval. It inserts fewer words (53 against 82), which helps on single words (119 against 135 wrong), but it's worse on sentences (196 against 188). Hypotheses 1, 2, 5 and 6 need the adapter runs.

The comparison point has moved since. The 0.6B v2 adapter on stock averages about 212 to 213 over three seeds (stages K, P, R, S, U), and severe pretraining under the adapter reaches 178.0 (stage P). A bigger model would now have to beat 178.

Sentences only (secondary score, added 21 October 2026): the same decodes scored on sentence clips alone, 303 of the 425 words, where 35% is 106 wrong or fewer. Means over the seeds listed, read from each run's `results.json` with `scripts/asr/sentence_rescore.py`. The full ranking is in `docs/measurements.md` under "Secondary score: sentences only".

| Recipe (run name) | Seeds | Sentence words wrong (of 303) | Sentence WER | M01 / M02 / M04 | All-clip WER |
| --- | ---: | ---: | ---: | --- | ---: |
| `stock 1p1b (nemo)` | 1 | 174.0 | 57.4% | 50.8 / 41.2 / 74.3 | 72.2% |
| `stock 110m (nemo)` | 1 | 196.0 | 64.7% | 57.4 / 61.8 / 74.3 | 74.1% |
| `stock v3 (nemo)` | 1 | 196.0 | 64.7% | 59.0 / 47.1 / 81.4 | 75.1% |
| `stock 0.6B v2 (reference)` | 1 | 188.0 | 62.0% | 61.5 / 47.1 / 71.7 | 76.0% |

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Pre-run Hypotheses

Written 2 Oct 2026 before any run, and scored as written. No separate prediction was recorded.

1. No new adapter model reaches 148 wrong pooled. A bigger model isn't known to be better on dysarthric speech, and the adapter only has 1 M numbers to correct it with.
2. The best 1.1B adapter is within 25 words of 201 (176 to 226).
3. v3 stock is within 20 words of v2 stock (303 to 343 pooled), since they're the same size and design.
4. 110M stock is worse than v2 stock, 335 to 395 pooled, because its card WER is higher (7.49 against 6.05).
5. The best 110M adapter lands between 227 and 322: more than 25 words worse than every 0.6B adapter, but better than v2 stock at 323. Least certain of the six.
6. The adapter fixes a smaller share of stock errors on the 110M than on v2 (under 38%, which is (323 - 201) / 323), because in a shallower model there are fewer blocks above it. Low confidence.

### Commands and Setup

Run from `app/` in the WSL tree on Yack after `git pull`, GPU free, laptop plugged in and set not to sleep. The script stops if a `train.py` run is going. The script is `app/lab.py`, a file in the repository, so the command is one line. The script's first lines print the commit, the `train.py` and `phrasebook.py` hashes (`train.py` was 032323d84c43 for the 21:46 sweep and changed on 2 Oct with the augmentation flags, so expect a different hash), GPU memory and free disk.

All five stages in order (W, B, A, N, F, about 6 hours, estimated), output copied to a file:

```bash
python lab.py run W B A N F
```

For stage B alone add `"B"` after the stamp, as in `python lab.py run B`. Any list works, like `"W B"` or `"A N"`. Stages are independent and stop at once if an output already exists. At the end the script writes `logs/big-<stamp>-summaries.txt`; paste that back. Each stage also writes `logs/big-<stamp>-<stage>.txt`.

I tested the script on a stub copy of the tree in an isolated test environment with fake manifests and fake training, decoding and scoring, including a simulated out-of-memory to check the batch-1 retry and a simulated 110M failure to check its other runs get skipped. That covers shell logic, file names and the summary code. It tests nothing about training, NeMo, downloads or real scores.

### Full Results

No adapter run exists. The stock rows are from `reports/finetune-torgo/<speaker>-test-stock-<1p1b|v3|110m>-nemo/results.json` (read 21 October). Lab test 4's note has the intervals.

### Conclusions

*Added 21 October 2026 from the stock decodes. The adapter question is still open.*

On stock decoding neither the 1.1B nor v3 is established as better than v2 on these speakers, so the base model isn't the obvious limit at the start. The 110M hybrid isn't worse than v2 untrained (315 against 323), which goes against pre-run hypothesis 4 and keeps it in play for a small device.

Whether the adapter gains more on a bigger model still needs the stage to run: `python lab.py run B` (about 2 hours). Its marker is now an adapter result, so the queue won't skip it again.

</details>

## Related

[F-2: augmentation variants](../f2-augmentation-variants/notes.md), [F-3: M02 data quantity](../f3-m02-data-quantity/notes.md), [Phrase matching](../phrasebook-matching/notes.md), [F-2: adapter position seeds](../f2-adapter-position-seeds/notes.md)
