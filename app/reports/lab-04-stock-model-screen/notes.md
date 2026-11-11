# Lab test 4: stock model screen, other speech models on the severe speakers before any training

Proposal: F-2 (H3), an extension about the base model. Prototype bar: 35%.

Status: run 7 October 2026, results read 14 October. Cohere has no result. The Nemotron and CTC rows come from the logged table.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

The model so far is Parakeet-TDT-0.6B-v2, picked because it fits on the Raspberry Pi. Other models might already do better on severe dysarthric speech with no training. I decode the three severe test sets with each one and report pooled WER with a 95% interval and the share of empty answers.

## Setup

Pooled means wrong words summed over M01 (169 test words), M02 (95) and M04 (161), 425 in all. Stock v2 gets 323 wrong (76.0%), the best so far is about 201 to 210 (about 48.5%, adapter after block 8, three seeds), and the 35% bar is 148 or fewer.

The candidates are in `scripts/asr/candidates.py` (names checked against the Hugging Face cards on 2 Oct). NeMo, in the training environment: Parakeet-TDT-0.6B-v3, Parakeet-TDT-1.1B, Parakeet-TDT-CTC-110M (a small hybrid), Nemotron speech streaming 0.6B, Parakeet-RNNT-0.6B and Parakeet-CTC-0.6B. Hugging Face, in a second environment: Qwen3-ASR-0.6B, Cohere Transcribe (2B), Granite speech 4.1 (2B) and Granite 4.0 1B speech. Cohere and Granite are too big to fine-tune on the 8 GB card and wouldn't fit the Pi, so they are only there to show what a much larger model does. `candidates.py` says which ones `train.py` can train. Stock decodes of 1.1B, v3 and the 110M hybrid share file names with stage B of `lab.py` and get reused.

## Plan

1. Check the three test manifests and v2 stock decodes exist and no output does.
2. Decode each candidate per speaker and score with `evaluate.py`. Failed decodes are listed and skipped.
3. Run `pool_table.py` with v2 as baseline.

Rule, set before the run: a change counts only when at least 2 of 3 speakers move the same way as the pooled count and pooled WER moves by more than 4.5 points, the seed noise seen so far (4 to 5 points per speaker). Anything else is "unclear", which doesn't mean equal. `pool_table.py` prints the verdict. Three agreeing seeds could settle a smaller gap, but this is one run per cell.

If a candidate beats v2, it's worth fine-tuning (lab test 7 checks whether stock order predicts fine-tuned order). If nothing does, the 0.6B model isn't the bottleneck and effort goes to data and training. If a speech-LLM scores well with lots of empty or very long answers, read them before trusting the WER, since these models can rewrite what they hear.

## Why

It's a quick check on the project's biggest assumption, that v2 is the right base.

## Limits

Each model gets one stock decode on the last-session test clips. This says nothing about fine-tuning gains (that's lab test 7), the Pi or the intended user. All output goes through the project's normaliser, so a model isn't penalised for digits or punctuation.

## Results

Stock decodes on the three severe test sets, 425 reference words. Intervals resample clips.

| Model | Wrong | WER | Gain over v2 (95% CI) | Empty |
|---|---:|---:|---|---:|
| Parakeet v2 (shipped) | 323 | 76.0% | - | 34/166 |
| Parakeet v3 | 319 | 75.1% | +0.9 (-5.9 to +7.8), unclear | 33 |
| Parakeet 1.1B | 307 | 72.2% | +3.8 (-3.5 to +10.7), unclear | 4 |
| 110M TDT-CTC | 315 | 74.1% | +1.9 (-4.9 to +9.2), unclear | 11 |
| Nemotron | 331 | 77.9% | -1.9 (-10.1 to +7.4), unclear | 25 |
| RNNT 0.6B | 314 | 73.9% | +2.1 (-4.2 to +9.0), unclear | 6 |
| CTC 0.6B | 334 | 78.6% | -2.6 (-9.1 to +4.5), unclear | 1 |
| Qwen3 | 352 | 82.8% | -6.8 (-16.8 to +2.9), worse | 2 |
| Granite 4.0 / 4.1 | 793 / 1265 | 186.6% / 297.6% | worse | 26 / 2 |

No candidate showed an improvement over v2. The Granite rows are broken outputs in this setup (more wrong words than reference words), so they're not a fair verdict on those models. Qwen3 gets 168 wrong against 161 reference words on M04, so it's inserting words. Switching base won't get to the bar.

Source: pool_table log (`logs/pool_table-20261007-215706-910898.log`), RTX 4000 Ada lab computer, 6-8 Oct 2026, reviewed 14 October 2026.

Sentences only (secondary score, added 21 October 2026). Same decodes, single-word clips left out: 303 of the 425 words, so 35% is 106 wrong or fewer. Means over the seeds listed, read from each run's `results.json` with `scripts/asr/sentence_rescore.py`. The full ranking is in `docs/measurements.md` under "Secondary score: sentences only".

| Recipe (run name) | Seeds | Sentence words wrong (of 303) | Sentence WER | M01 / M02 / M04 | All-clip WER |
| --- | ---: | ---: | ---: | --- | ---: |
| `stock 1p1b (nemo)` | 1 | 174.0 | 57.4% | 50.8 / 41.2 / 74.3 | 72.2% |
| `stock rnnt06 (nemo)` | 1 | 183.0 | 60.4% | 59.0 / 47.1 / 69.9 | 73.9% |
| `stock 110m (nemo)` | 1 | 196.0 | 64.7% | 57.4 / 61.8 / 74.3 | 74.1% |
| `stock v3 (nemo)` | 1 | 196.0 | 64.7% | 59.0 / 47.1 / 81.4 | 75.1% |
| `stock ctc06 (nemo)` | 1 | 205.0 | 67.7% | 64.8 / 54.4 / 78.8 | 78.6% |
| `stock nemotron (nemo)` | 1 | 213.0 | 70.3% | 68.0 / 67.6 / 74.3 | 77.9% |
| `stock qwen3 (hf)` | 1 | 215.0 | 71.0% | 69.7 / 45.6 / 87.6 | 82.8% |
| `stock granite41 (hf)` | 1 | 541.0 | 178.5% | 309.0 / 76.5 / 99.1 | 297.6% |
| `stock granite4 (hf)` | 1 | 658.0 | 217.2% | 86.9 / 77.9 / 441.6 | 186.6% |
| `stock 0.6B v2 (reference)` | 1 | 188.0 | 62.0% | 61.5 / 47.1 / 71.7 | 76.0% |

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

No prediction was written before this run (7 October 2026), and none is added afterwards.

### Commands and Setup

Run from `app/` in the WSL tree on Yack after `git pull`, with the GPU free and the PC set not to sleep. The script stops if a `train.py` run is already going, and each stage stops straight away if one of its outputs already exists.

```bash
python lab.py run 4
```

At the end the script writes `logs/lab-<stamp>-summaries.txt`, which is the file these notes record. The stage's full output is `logs/lab-<stamp>-4.txt`. The Hugging Face models need their own environment (see the top of `scripts/asr/hf_hyp.py`). If `~/venvs/willspeak-hf` is missing, the script skips them and says so.

The commands come from `lab.py`. Its flags were checked against each script's options with a dry run (`python lab.py run --dry`, in `tests/unit/test_lab_cli.py`), and each script's logic has unit tests on synthetic data. Those tests aren't recognition results. Before this run nothing in the stage had touched the real models or data. The three Hugging Face loaders in `hf_hyp.py` were written from the model cards and never exercised, and `nemo_hyp.py --pretrained` hadn't been tried on the Nemotron and CTC models.

### Full Results

The Results section above has the full comparison and its sources.

### Conclusions

Written 21 October 2026 from the recorded results above.

None of the eight other stock models beat the shipped Parakeet-TDT-0.6B-v2 (323 wrong of 425, 76.0%) with an interval that excludes zero. The closest is Parakeet-TDT-1.1B at 307 wrong (72.2%), 3.8 points better with an interval of -3.5 to +10.7, so it's unclear.

The models fail in different ways. v2 returns nothing on 34 of 166 clips, while 1.1B does that on 4 and CTC 0.6B on 1, but fewer empty answers didn't turn into fewer wrong words. Qwen3 and both Granite models insert words. The Granite rows (186.6% and 297.6%) are broken outputs in this setup and not a fair verdict on those models.

Lab test 7 then fine-tuned four of these bases with the same whole-block recipe on M02 and M04, and v2 still came out lowest (66.4%). So the stock screen didn't point to a better starting model.

No base other than v2 has been fine-tuned with the adapter yet. Stage B (the adapter on 1.1B, v3 and the 110M model, see [base-model ceiling](../f2-base-model-ceiling/notes.md)) has no result.

Next step: keep v2 as the base for the expo model. Another base is only worth revisiting if stage B shows a lead bigger than the roughly 19-word noise band.

</details>

## Related

[F-2: base-model ceiling](../f2-base-model-ceiling/notes.md), [Lab test 7](../lab-07-stock-vs-fine-tuned/notes.md)
