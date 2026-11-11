# F-1: severe speakers, leave-one-speaker-out, top 8 blocks

Proposal §5.2 / H2.

Status: recorded 24 Sep 2026 with the first recipe.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

Does speech from other dysarthric speakers help the model recognise a severe speaker it has never heard?

## Setup

Four held-out speakers: F01, M01, M02 and M04. Each model trains on the other seven speakers (106–122 min), prompt-disjoint, with frozen batch norm, normalized transcripts, the top-eight encoder recipe and the old dev score.

## Plan

Compare each int8 model with stock on its own LOSO test set. Also score the M02-held-out model on M02's 37 F-2 test clips.

## Why

This is H2's test of whether public multi-speaker training helps before any personal recordings exist.

## Limits

It can show improvement on these held-out TORGO speakers. H2's under-20% target wasn't met. The LOSO test WER can't be compared directly with F-2's, which uses a different test set; H3 needs both models on the same clips with a comparable recipe.

## Results

| Held-out speaker | Test clips | Stock WER | F-1 WER | Gain, points |
|---|---:|---:|---:|---:|
| F01 | 23 | 73.91% | 67.39% | 6.52 |
| M01 | 80 | 70.05% | 62.32% | 7.73 |
| M02 | 84 | 59.52% | 51.43% | 8.10 |
| M04 | 87 | 91.63% | 74.38% | 17.24 |

The intervals clear zero for M01, M02 and M04. On the 37 F-2 clips, M02's F-1 model and the normalized-transcript F-2 model both score 52.63%.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Context

The first F-1 run asks H2's question: if the model trains on every other dysarthric speaker in TORGO, does it understand a severe speaker it has never heard? That's where the device is on day one, before anyone records the client.

The recipe is the one that worked for the M02 F-2 model: frozen batch norm, normalised transcripts, 16 encoder layers frozen, batch 2 x 8, default warm-up (100 steps; an epoch is now about 115 steps). Scoring is on the int8 export, which is what the Pi runs.

Training data (the other seven dysarthric speakers, mild to severe) and each test set:
  F01 held out: train 2012 clips, 122 min | test 23 clips (20 words, 3 sentences)
  M01 held out: train 1850 clips, 108 min | test 80 clips (59 words, 21 sentences)
  M02 held out: train 1839 clips, 109 min | test 84 clips (62 words, 22 sentences)
  M04 held out: train 1836 clips, 106 min | test 87 clips (68 words, 19 sentences)
That's about twelve times the audio the M02 F-2 model had (9 minutes of M02).

Numbers that already exist:
  T-1, stock, severe sentences only: F01 57.61%, M01 56.46%, M02 45.99%, M04 76.44%
  F01's 23 test clips are the same 23 as the export check: stock exported 73.91%
  M02 F-2 (37 clips, Session2, all inside the 84 here): stock exported 60.00%, the M02 F-2 model exported 52.63%
  Single words score far worse than sentences: M02 stock exported, words 107.4%, sentences 41.2%.
  These test sets are about three quarters single words.
  F01's test set is only 23 clips, so its interval will be the widest.

The extra run has the M02-held-out model also score M02's 37 F-2 test clips. That puts general dysarthric data (109 minutes, other people) against M02's own data (9 minutes, his voice) on the same clips.

### What I expect before running the first F-1 run

Stock WER, exported, per speaker (F01 is already known: 73.91%):
  M01: 69–72%. Words are 59 of about 206 reference words. Sentences use T-1's 56.46%
  M02: 61–64. Words are 62 of about 216 (29%). Sentences use 45.99%
  M04: 84–88%. Our most severe speaker. Words are 68 of about 201 (34%). Sentences use 76.44%
Fine-tuned WER, exported, per speaker:
  F01: 64–69%. Her test set is half single words, so she should gain the most (about 5–10 points). With only 23 clips, it's the noisiest number.
  M01: 63–67%. About 4–7 points below stock.
  M02: 56–60%. About 4–7 points. He has the least room, since he's already the easiest speaker.
  M04: 78–83%. About 4–8 points. He has the most room, but his speech is the least like the mild and moderate speakers who make up much of the training data.
Which speaker gains most, and which least: F01, because her test set has the most single words. Gains least: M02, because he has the least room.
How many of the four ci.py intervals clear zero: probably 1 of 4. With about 85 clips, the intervals are roughly ±6 points,
and gains of 5–7 sit right on that edge. F01's 23 clips make hers the least likely to clear.

The M02-held-out model on M02's 37 F-2 test clips, against the M02 F-2 model's 52.63%: about 55–58%. That's better than stock (60.00%) but worse than the M02 F-2 model (52.63%). I'd expect 9 minutes of his own voice to beat 109 minutes of other people's.
I would be surprised if any speaker ended up worse than stock, which would mean other people's speech misleads the model. Or if the held-out model beat the M02 F-2 model on M02's clips, which would mean general dysarthric data beats his own.

### Results

Machine: Yack, WSL2, 24 GB given to WSL

Per speaker (exported, stock against fine-tuned on the same test clips):
  F01: stock 73.91%   fine-tuned 67.39%   ci.py Delta 6.52 (95% CI -3.92 to 30.43)   best epoch <>
  M01: stock 70.05%   fine-tuned 62.32%   ci.py Delta 7.73 (95% CI 1.60 to 14.75)   best epoch <>
  M02: stock 59.52%   fine-tuned 51.43%   ci.py Delta 8.10 (95% CI 1.84 to 16.13)   best epoch <>
  M04: stock 91.63%   fine-tuned 74.38%   ci.py Delta 17.24 (95% CI 7.52 to 27.57)   best epoch 7 (dev WER 34.9%)
M02-held-out model on M02's 37 F-2 test clips: 52.63%   (stock exported 60.00%, the M02 F-2 model exported 52.63%)
  ci.py, stock exported against the held-out model: Delta 7.37 (95% CI 0.00 to 17.65), the interval reaches zero
  Both models make 50 word errors out of 95, but not the same ones:
  single words 22 of 27 (the M02 F-2 model: 24), sentences 28 of 68 (the M02 F-2 model: 26)

Word errors out of reference words, stock -> fine-tuned:
  F01: single words 22/20 -> 18/20   sentences 12/26 -> 13/26
  M01: single words 61/59 -> 53/59   sentences 84/148 -> 76/148
  M02: single words 64/62 -> 50/62   sentences 61/148 -> 58/148
  M04: single words 76/68 -> 63/68   sentences 110/135 -> 88/135
Inserted words (extra words the model added), stock -> fine-tuned:
  F01 4 -> 0   M01 24 -> 8   M02 24 -> 4   M04 45 -> 13

Prediction against actual:
  Stock M01: predicted 69–72%, measured 70.05% (inside the range)
  Stock M02: predicted 61–64%, measured 59.52% (below the range)
  Stock M04: predicted 84–88%, measured 91.63% (above the range)
  Reference words: predicted about 206, 216 and 201 for M01, M02 and M04; measured 207, 210 and 203
  Fine-tuned F01: predicted 64–69%, measured 67.39% (inside); gain predicted 5–10, measured 6.52
  Fine-tuned M01: predicted 63–67%, measured 62.32% (just below); gain predicted 4–7, measured 7.73
  Fine-tuned M02: predicted 56–60%, measured 51.43% (below); gain predicted 4–7, measured 8.10
  Fine-tuned M04: predicted 78–83%, measured 74.38% (below); gain predicted 4–8, measured 17.24
  Gains most: predicted F01, measured M04 (17.24). Gains least: predicted M02, measured F01 (6.52)
  Intervals that clear zero: predicted about 1 of 4, measured 3 of 4 (M01, M02, M04; F01 does not)
  M02-held-out model on M02's 37 F-2 test clips: predicted 55–58%, measured 52.63%, the same as the M02 F-2 model
  Surprise, a speaker worse than stock: did not happen, all four improved
  Surprise, the held-out model beating the M02 F-2 model: did not happen, it tied (50 errors each)

### What I concluded

1. Tonight's runs answer H2's question, "Does training on the other seven speakers, and then testing on one the model has never heard, give a lower WER than stock?", and the answer looks like yes. It isn't 100% proof, but it points strongly that way: three of the four intervals clear zero, and only F01's (23 clips) doesn't.
2. The surprise is how much they gained. I predicted about 1 of 4 intervals would clear zero, and 3 did. No speaker was worse off after training on the others and being tested as the holdout. Most of the gain came from single words, where the fine-tuned models stopped adding extra words (M04 went from 45 inserted words to 13). M04 gained the most (17.24 points), but he also started the worst (91.63%), so he had the most room.
3. The M02-held-out model did better on M02's 37 F-2 test clips than I predicted: 52.63% measured, against 55–58% predicted.
4. It didn't beat the M02 F-2 model; they tied at exactly 50 errors each out of 95. So 109 minutes of other people's speech did as well as 9 minutes of M02's own voice, though they missed different words (single words 22 against the F-2 model's 24, sentences 28 against its 26).

</details>

## Related

[F-1: severe speakers, leave-one-speaker-out, bottom 8 blocks](../f1-loso-bottom8/notes.md)
