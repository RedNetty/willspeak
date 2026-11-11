# Lab test 1: weight interpolation, stock to whole-block at 25, 50 and 75%

Proposal: F-2 (H3), an extension asking how much of the whole-block change is needed. Prototype bar: 35%.

Status: run 6 October 2026 on the lab computer, results read 14 October. A later stage 1 attempt stopped on a missing checkpoint (`logs/lab-1-1.txt`), which doesn't affect these decodes.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

Fine-tuning moves every weight in blocks 0 to 7. Interpolating between stock and fine-tuned weights (weight-space ensembling, as in WiSE-FT, Wortsman et al., CVPR 2022) sometimes keeps most of the gain and loses less of what stock did well. I decoded a quarter, half and three quarters of the way to see if any point matches or beats the full model.

## Setup

Pooled means wrong words summed over M01, M02 and M04, 425 in all. Stock is Parakeet-TDT-0.6B-v2. The best raw result so far is about 201 to 210 wrong (about 48.5%, adapter after block 8, three seeds), and the 35% bar is 148.

Whole-block is the F-2 model that trains encoder blocks 0 to 7 in full. M01 and M04 use the fixed five-epoch runs (`models/parakeet-ft-f2-<SPK>-bottom8-5epochs`). M02 uses `models/parakeet-ft-M02e-A3` (dev-selected, 42 of 95 wrong in NeMo), a different recipe, which carries into cross-speaker comparisons.

The blend is `stock + w * (whole-block - stock)` with each speaker's own model. Tensors identical to stock (blocks 8 to 23, the decoder, the preprocessor) are untouched and BatchNorm step counters come from stock. `nemo_hyp.py --stock --mix FILE=WEIGHT` builds it in memory, with no new `.nemo` written.

## Plan

1. Check each whole-block model and the stock and whole-block decodes exist, and no output does.
2. Decode each test set at w = 0.25, 0.50 and 0.75 and score with `evaluate.py`.
3. `pool_table.py`: one table with wrong words per speaker, pooled count, WER with a 95% interval and empty answers.

Decision rule, set before the run: a change counts only if at least 2 of the 3 speakers move with the pooled count and pooled WER moves more than 4.5 points, the noise seen across repeated seeds (4 to 5 points per speaker). Anything else is "unclear", which doesn't mean equal. `pool_table.py` prints the verdict. Three agreeing seeds can settle a smaller gap, but each cell here is one run.

A blend within 4.5 points of whole-block (and clearly ahead of stock) would be a candidate for a model that forgets less, though forgetting isn't measured. A blend beating whole-block on 2 of 3 speakers would mean full fine-tuning overshoots. A straight line from stock to whole-block would mean stopping early gains nothing.

## Why

One afternoon of decoding, no training, and the models already exist.

## Limits

One fine-tuned model per speaker. Nothing on typical speech, int8 export of a blend, or a model trained to be blended.

## Results

NeMo results on M01/M02/M04, 166 clips. A blend is stock plus w of the way to the whole-block weights.

| System | M01 wrong | M02 wrong | M04 wrong | Pooled wrong | WER | Gain over stock (95% CI, clips) |
|---|---:|---:|---:|---:|---:|---|
| stock | 131/169 | 62/95 | 130/161 | 323 | 76.0% | - |
| blend 25% | 131 | 55 | 131 | 317 | 74.6% | +1.4 (-2.9 to +6.3), unclear |
| blend 50% | 131 | 46 | 123 | 300 | 70.6% | +5.4 (+1.7 to +10.2), better |
| blend 75% | 130 | 44 | 129 | 303 | 71.3% | +4.7 (-2.3 to +12.2), printed verdict better, interval includes zero |
| whole-block | 130 | 40 | 121 | 291 | 68.5% | +7.5 (+1.9 to +14.2), better |

No blend beats whole-block. The best, 50%, is 9 words behind, inside noise. Nearly all the gain is M02's (62 wrong down to 44-46 at 50-75%), M01 doesn't move and M04 only gains at 50%. Blending keeps less of the gain, and what it was meant to protect wasn't measured. Nothing reaches the bar.

Sources: `table.json` (written 6 Oct on the lab computer) and the decodes in `reports/finetune-torgo/lab-01-w{25,50,75}-<speaker>-test-nemo/`. Runs on the RTX 4000 Ada lab computer, 6-8 Oct 2026, reviewed 14 October 2026.

Sentences only (secondary score, added 21 October 2026): the same decodes on sentence clips alone, 303 of the 425 words, where 35% is 106 wrong. From each run's `results.json` via `scripts/asr/sentence_rescore.py`. All recipes are ranked in `docs/measurements.md` under "Secondary score: sentences only".

| Recipe (run name) | Seeds | Sentence words wrong (of 303) | Sentence WER | M01 / M02 / M04 | All-clip WER |
| --- | ---: | ---: | ---: | --- | ---: |
| `lab-01-w75` | 1 | 169.0 | 55.8% | 60.7 / 35.3 / 62.8 | 71.3% |
| `lab-01-w50` | 1 | 172.0 | 56.8% | 60.7 / 36.8 / 64.6 | 70.6% |
| `lab-01-w25` | 1 | 183.0 | 60.4% | 61.5 / 42.6 / 69.9 | 74.6% |
| `stock 0.6B v2 (reference)` | 1 | 188.0 | 62.0% | 61.5 / 47.1 / 71.7 | 76.0% |

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

No prediction was written before this run (6 October 2026), and none is added after the fact.

### Commands and Setup

Run from `app/` in the WSL tree on Yack after `git pull`, with the GPU free and the PC set not to sleep. The script stops if a `train.py` run is going, and each stage stops if one of its outputs already exists.

```bash
python lab.py run 1
```

At the end the script writes `logs/lab-<stamp>-summaries.txt`, which is the file recorded in these notes. The stage's full output is `logs/lab-<stamp>-1.txt`.

The commands come from `lab.py`. A dry run (`python lab.py run --dry`, `tests/unit/test_lab_cli.py`) checks its flags against each script's options, and each script's logic has unit tests on synthetic data, which aren't recognition results. Loading a blend through `nemo_hyp.py --mix` on the real model wasn't tested beforehand. `weights.py` and the blend arithmetic are unit-tested on small checkpoints.

### Full Results

See Results above for the full comparison and sources.

### Conclusions

*Interpretation added 21 October 2026 from the recorded results above.*

On the pooled test set no blend beat the whole-block model. The best, 50%, had 300 wrong (70.6%) against 291 (68.5%). That 2.1-point gap isn't settled either way by the project rule (2 of 3 speakers agreeing and more than 4.5 points, about 19 words).

The blends don't sit on a straight line between stock (323) and whole-block (291). Almost all their gain is M02's (62 wrong falling to 46 and 44 at 50% and 75%). M01 stays at 130 to 131 at every point and M04 improves only at 50% (123). Blending didn't find a gain the full move lacks.

Only the 50% blend and whole-block have a gain over stock with an interval clear of zero: +5.4 points (+1.7 to +10.2) and +7.5 points (+1.9 to +14.2). The 75% blend prints "better", but its interval (-2.3 to +12.2) includes zero.

Later recipes went well past this five-epoch whole-block baseline. Whole-block training at 1.6e-4 for 40 epochs averaged 194.7 wrong over three seeds, and the adapter on the severe-pretrained model at 6e-4 averaged 178.0 ([stage P and G2](../f2-adapter-levers/notes.md)).

Next step: none for the expo model. Blending is only worth another look as a guard against losing typical speech, which this test didn't measure.

</details>

## Related

[F-2: severe speakers, bottom 8 blocks, fixed 5 epochs](../f2-severe-bottom8-5epochs/notes.md), [F-2: M02, bottom 8 blocks, which parts](../f2-m02-bottom8-parts/notes.md), [Lab test 2](../lab-02-weight-merging/notes.md)
