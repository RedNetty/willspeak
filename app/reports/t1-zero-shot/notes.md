# T-1: TORGO zero-shot baseline

Proposal: §5.1 / H1

Status: recorded 20 Sep 2026, CI added 23 Sep.

See also the [Report Context](../Report%20Context.md) and the [Measurement guide](../../../docs/measurements.md).

## Idea

How does stock Parakeet handle severe dysarthric speech before any fine-tuning?

## Setup

TORGO sentences, one recording per take: 292 severe clips, and 755 clips across all eight dysarthric speakers. Stock int8 with greedy decoding and four threads, on Yack in WSL2.

## Plan

Score the severe sentences by speaker, then all sentences by severity, and compare the severe-group WER with H1's 60% threshold. The original prediction and error counts stay in the record.

## Why

This is the starting point, and it shows where recognition breaks down. Later fine-tune comparisons still need stock scored again on their own test clips.

## Limits

This shows stock-model errors on these TORGO sentences and how the severity groups differ. It says nothing about the intended user and doesn't replace the mixed word/sentence F-1 and F-2 baselines. WER counts word edits, so it isn't the share of sentences understood.

## Results

| Dataset | Clips | WER | CER |
|---|---:|---:|---:|
| Severe sentences | 292 | 59.59% | 41.90% |
| All dysarthric sentences | 755 | 32.05% | 21.69% |

The severe WER has a 95% CI of 55.21% to 63.77%. H1's 60% threshold is inside it, so H1 is still unresolved.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

I fixed the predictions below before either run, during the 20 Sep working session. The file was written up afterwards and no number was changed.

### Assumptions behind the predictions

1. Stock Parakeet has seen no TORGO and no dysarthric fine-tuning, but its training set is big enough that it has probably heard some atypical speech.
2. Human difficulty partly predicts machine difficulty, since the shared cause is imprecise articulation. But the model guesses where I stayed silent.
3. TORGO's severity labels are human judgements. Whether the machine reproduces that ordering is something this run tests. I'm not assuming it.

### What I expected before running

Calibration: I listened to 10 severe-speaker clips (seed 0) and transcribed them cold before seeing the references. My own WER was 72%, with 11 substitutions, 43 deletions and 0 insertions. Per speaker that was M02 55%, M04 81% and M01 86% (one clip). F01 wasn't in the sample.

My predictions per speaker:

- F01 65%. My by-ear sample didn't include this speaker, so this is just an assumption from the severity label.
- M01 80%. By ear I scored 86%, so I'm assuming the model does slightly better.
- M02 50%.
- M04 78%.
- All severe 69%, weighting F01 at about 7% of the words and the other three at about 31% each. The weighting barely changes my answer (68.3 as a straight average, 68.8 weighted) because F01 happens to sit near my average. It would matter if F01 were an outlier.

I'd be surprised by anything below 20% or above 150%, and I'd check the manifest before putting any faith in either. 20% would mean stock Parakeet is fairly usable on dysarthric speakers, which would be the most surprising thing in all of this. 150% would mean it's inventing whole sentences.

Against my own 72%, I think the zero-shot model will do slightly better than me. I wrote nothing for words I couldn't make out by ear, and the model almost always outputs something.

Error profile: mostly substitutions and almost no deletions, because the decoder puts a word in every slot where I left a blank. If it also inserts, WER can pass 100%, most likely on the short references. Clip 06's reference is two words, so three spurious words already put that clip over 100%.

Ordering: M01 worst and M02 best, which is how they landed for me by ear, though M01 rests on a single clip. I can't rank F01 against them at all. If the machine's order differs from mine, that's the finding. TORGO's severity labels were assigned by human listeners, so a machine that disagrees is telling me machine difficulty and human difficulty aren't the same thing.

H1 (severe zero-shot WER over 60%) holds if the all-severe row is above 60%. If it lands under 35%, I was wrong that stock recognition is the bottleneck. The argument would then have to move from 'the model cannot understand him' to 'the model cannot understand him specifically, and it has to run offline on a board he can carry'. That's a weaker headline, and I'd say so on the poster instead of burying it.

Plumbing check: `--limit 20` scored the first 20 manifest rows, which are all of F01's sentences, at WER 57.61%. F01's row in the severe-speakers run should return exactly that. If it differs, decoding isn't deterministic.

### Results

Machine: `Yack - 33, Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.43, Python 3.14.4`

Severe-speakers run (severe speakers, grouped by speaker):

```bash
python scripts/asr/evaluate.py --manifest data/public/torgo/manifests/severe-sentences.jsonl \
--audio-root data/public/torgo/TORGO --group-by speaker --out reports/torgo-severe-zero-shot
```

292 clips, 2,170 reference words. It ran in about 5 min, going by the log timestamps. Overall WER 59.59%, CER 41.90%, PER 44.51%.

| speaker | clips | ref words | WER | sub | del | ins | errors |
|---------|------:|----------:|-------:|----:|----:|----:|-------:|
| F01 | 20 | 184 | 57.61% | 76 | 21 | 9 | 106 |
| M01 | 90 | 650 | 56.46% | 293 | 31 | 43 | 367 |
| M02 | 93 | 661 | 45.99% | 211 | 46 | 47 | 304 |
| M04 | 89 | 675 | 76.44% | 287 | 126 | 103 | 516 |
| **all** | 292 | 2,170 | **59.59%** | 867 | 224 | 202 | 1,293 |

F01 returned 57.61% with the same counts as the plumbing check on those clips (184 ref, 76 sub, 21 del, 9 ins), so decoding is deterministic.

All-speakers run (all dysarthric speakers, grouped by severity): the same command with `--manifest data/public/torgo/manifests/all-sentences.jsonl --group-by severity --out reports/torgo-all-zero-shot`. 755 clips, 5,532 reference words, runtime not recorded. Overall WER 32.05%, CER 21.69%, PER 23.25%.

| severity | speakers | clips | ref words | WER | sub | del | ins |
|----------|----------|------:|----------:|-------:|----:|----:|----:|
| mild | F04, M03 | 198 | 1,479 | 3.99% | 24 | 22 | 13 |
| moderate | F03 | 140 | 1,000 | 13.00% | 85 | 11 | 34 |
| moderate-severe | M05 | 125 | 883 | 32.96% | 172 | 81 | 38 |
| severe | 4 above | 292 | 2,170 | 59.59% | 867 | 224 | 202 |

The 32.05% overall can't be compared with the severe number, because 463 of the 755 clips are mild and moderate speakers.

Prediction against actual:

| speaker | predicted | actual | off by | clips I had heard |
|---------|----------:|-------:|-------:|------------------:|
| F01 | 65% | 57.61% | +7.4 | 0 |
| M01 | 80% | 56.46% | +23.5 | 1 |
| M02 | 50% | 45.99% | +4.0 | 4 |
| M04 | 78% | 76.44% | +1.6 | 5 |
| all severe | 69% | 59.59% | +9.4 | - |

I predicted M01 worst and M02 best. The measured order was M04 worst (76.44%) and M02 best (45.99%).

Error profile, mine against the model:

| source | sub | del | ins |
|--------|------:|------:|------:|
| me, by ear (n=54) | 20.4% | 79.6% | 0.0% |
| Parakeet, severe | 67.1% | 17.3% | 15.6% |

Insertions per reference word were 9.3% for severe and 0.9% for mild.

### What I concluded

1. This run doesn't resolve H1. Severe zero-shot WER came out at 59.59%, 0.41 points short of the 60% threshold on 292 utterances. A gap that small on that many utterances is inside the noise, so the honest statement is that this run can't separate 59.59% from 60%. The bootstrap interval from ci.py decides it. Until then H1 is neither confirmed nor rejected, and I won't quote 59.59% as either.

2. My predicted ordering was wrong and my earlier by-ear ordering was right. M04 is worst at 76.44% and M02 best at 45.99%. The predictions I made from four and five heard clips were off by 4.0 and 1.6 points. The one from a single clip was off by 23.5. F01, where I'd heard nothing and guessed from the severity label alone, was off by 7.4. One unrepresentative clip was worse than no clip at all, because it gave me false confidence in a ranking the measurement reversed.

3. The error profile came out as predicted, and it's the reverse of mine. The model substitutes and inserts where I fell silent: 67% substitutions and 16% insertions against my 80% deletions. Severe speech gets ten times the insertion rate of mild speech (9.3% of reference words against 0.9%). For the device this matters directly. When the recogniser fails it produces a fluent wrong sentence instead of going quiet, and the box would speak it.

4. The severity ladder is monotonic and steep, at 3.99%, 13.00%, 32.96% and 59.59%, roughly doubling each step. At the group level the machine reproduces the ordering human listeners gave these speakers. Inside the severe label, though, the spread is 30 points (M02 45.99%, M04 76.44%). The label predicts the group but not the speaker.

5. What this means for the device: we already mostly knew that these technologies work for mild speakers. This run came out only slightly better than I expected, and ASR falls apart at the severe end. Severe speakers see a 59.59% WER, six in ten words wrong. That's the gap this project is meant to fill, or at the very least I want to build a protocol for the best ways to bring WER down for severe speakers. Technology already has a solution for mild speakers, but my friend who inspired this project is a severe speaker, and existing tech fails severe speakers.

6. After writing and running ci.py, which tells us how much to trust the WER, we can be about 95% sure the WER is in the range 55.21% to 63.77%. 60% is inside that range, so the result neither confirms nor denies H1. The 292 clips come from four speakers, so the real range is probably somewhat broader.

</details>

## Related

- [F-1: severe speakers, leave-one-speaker-out, top 8 blocks](../f1-loso-top8/notes.md)
- [F-2: severe speakers, bottom 8 blocks, fixed 5 epochs](../f2-severe-bottom8-5epochs/notes.md)
