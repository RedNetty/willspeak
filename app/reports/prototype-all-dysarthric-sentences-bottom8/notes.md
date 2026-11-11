# Prototype model, sentences only: all eight dysarthric speakers, bottom 8 blocks

Proposal: §2 deliverable (the tuned model for the Pi). A variant of the prototype model.

Status: Run 30 Sep 2026 and results recorded. Conclusions written 21 October 2026 from the recorded results.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

Train the prototype model again with every single-word recording removed, so it only learns from multi-word sentences, and compare it with the prototype model trained on everything.

## Setup

About three quarters of TORGO's dysarthric recordings are single words. On the device, Parakeet handles sentences and the exemplar matcher handles single words and the spoken control phrases, so sentence WER is what matters most for Parakeet. The earlier F-1 runs gained mostly on single words, which might mean the single words were taking most of the training's attention.

The only change from the prototype model is the training data. `--kinds sentence` keeps only sentence recordings in training and dev (dev picks the checkpoint, so it's scored on sentences too). Recipe, speakers and test prompts are the same.

One side effect: fewer recordings means about a quarter as many optimizer steps per epoch, so this model also gets less training overall. If it loses, that could be why, and not the missing words.

## Plan

1. Train on the sentence recordings of all eight dysarthric speakers (`export/all-dysarthric`, `--kinds sentence`).
2. Export to int8.
3. Build two test lists from the all-speakers test set: every test sentence, and the four severe speakers' test sentences.
4. Score stock, the prototype model (all recordings) and this model (sentences only) on the full test set (words and sentences reported separately), both sentence lists, and the control speakers.

## Why

If training on sentences alone gives better sentence WER, that's the better model for the Pi's sentence path, and single words can stay the matcher's job.

## Limits

This shows whether removing single words helps or hurts sentences, and what it costs on single words and typical speech, on test prompts no model trained on. It can't separate "no single words" from "fewer training steps". Both models trained on all eight speakers, so it says nothing about a speaker they never heard.

## Results

Removing the single words didn't measurably improve sentences: 41.14% against 41.79% on the severe speakers' test sentences and 21.01% against 22.15% on all eight, with both intervals including zero. It cost 5.54 points on the full test set and 3.21 on typical speech. The all-recordings prototype stays the better model. Full table in the Run Record.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

I genuinely believe we will see a much smaller WER with only sentence training, I think training on words as well has been poisoning our model. I believe we will see -10-20% WER from this test. Thats hopeful though.

### Commands

Run from app/ in WSL after the prototype model's run and scoring have finished, with the GPU free and the PC awake. Training is short, about a quarter of the prototype run, and scoring takes about 25 minutes. Each block stops at its first error.

```bash
( set -e
source ~/venvs/willspeak-train/bin/activate
python scripts/asr/train.py --export data/public/torgo/export/all-dysarthric --check --kinds sentence
python scripts/asr/train.py --export data/public/torgo/export/all-dysarthric --out models/parakeet-ft-prototype-all-dysarthric-sentences-bottom8 --kinds sentence --train-parts all --batch 2 --grad-accum 8 --freeze-batchnorm --normalize-targets --dev-scoring evaluate --keep-stock
python scripts/asr/export.py --nemo models/parakeet-ft-prototype-all-dysarthric-sentences-bottom8/parakeet-ft.nemo --out models/parakeet-ft-prototype-all-dysarthric-sentences-bottom8/onnx-int8 )

( set -e
source .venv/bin/activate
E=data/public/torgo/export/all-dysarthric
python - <<'PY'
import json
E = "data/public/torgo/export/all-dysarthric"
rows = [json.loads(line) for line in open(f"{E}/test.jsonl", encoding="utf-8")]
severe = {"F01", "M01", "M02", "M04"}
def write(name, keep):
    kept = [r for r in rows if keep(r)]
    with open(f"{E}/{name}", "w", encoding="utf-8") as f:
        f.write("".join(json.dumps(r) + "\n" for r in kept))
    print(f"{name}: {len(kept)} clips")
write("test-sentences.jsonl", lambda r: r["kind"] == "sentence")
write("test-severe-sentences.jsonl", lambda r: r["kind"] == "sentence" and r["speaker"] in severe)
PY
ALL=prototype-all-dysarthric-bottom8
SEN=prototype-all-dysarthric-sentences-bottom8
for T in test test-sentences test-severe-sentences; do
  G=speaker; if [ "$T" = test ]; then G=kind; fi
  python scripts/asr/evaluate.py --manifest $E/$T.jsonl --audio-root $E --group-by $G --out reports/finetune-torgo/all-dysarthric-$T-stock
  python scripts/asr/evaluate.py --manifest $E/$T.jsonl --audio-root $E --group-by $G --model-dir models/parakeet-ft-$ALL/onnx-int8 --out reports/finetune-torgo/$ALL-on-all-dysarthric-$T
  python scripts/asr/evaluate.py --manifest $E/$T.jsonl --audio-root $E --group-by $G --model-dir models/parakeet-ft-$SEN/onnx-int8 --out reports/finetune-torgo/$SEN-on-all-dysarthric-$T
  echo "== $T: stock against sentences-only"
  python scripts/asr/ci.py --manifest $E/$T.jsonl --stock reports/finetune-torgo/all-dysarthric-$T-stock --finetuned reports/finetune-torgo/$SEN-on-all-dysarthric-$T
  echo "== $T: all recordings against sentences-only"
  python scripts/asr/ci.py --manifest $E/$T.jsonl --stock reports/finetune-torgo/$ALL-on-all-dysarthric-$T --finetuned reports/finetune-torgo/$SEN-on-all-dysarthric-$T
done
python scripts/asr/evaluate.py --manifest data/public/torgo/export/controls/test.jsonl --audio-root data/public/torgo/export/controls --group-by kind --model-dir models/parakeet-ft-$SEN/onnx-int8 --out reports/finetune-torgo/controls-$SEN
echo "== controls: stock against sentences-only"
python scripts/asr/ci.py --manifest data/public/torgo/export/controls/test.jsonl --stock reports/finetune-torgo/controls-stock --finetuned reports/finetune-torgo/controls-$SEN
echo "== controls: all recordings against sentences-only"
python scripts/asr/ci.py --manifest data/public/torgo/export/controls/test.jsonl --stock reports/finetune-torgo/controls-$ALL --finetuned reports/finetune-torgo/controls-$SEN )
```

In the training log, look for the `--kinds sentence` line (train and dev counts before and after), "210461318 trainable", "stock dev WER" (a value, scored on dev sentences only, so it will differ from the prototype run's), the `curve:` lines, and "averaged ... checkpoints" or "releasing the stock model".

A positive Delta in the "all recordings against sentences-only" lines means the sentences-only model is better.

### Results

  Training: train 479 sentence clips (56.0 min; the all-recordings run had 2,095 clips, 119.2 min), dev 102;
  stock dev WER 40.20% (sentences only, so not comparable with the all-recordings run's 49.90%); selected
  fine-tuned, best epoch 8 (from 0), dev WER 28.8%, 4 checkpoints averaged; 300 optimizer steps in all
  (the all-recordings run: about 1,300)
  Training curve: models/parakeet-ft-prototype-all-dysarthric-sentences-bottom8/curve.csv (and run.json "curve")

| Test set | n clips | Stock | Prototype (all recordings) | Prototype (sentences only) | All against sentences-only: Delta (95% CI) |
| --- | ---: | ---: | ---: | ---: | --- |
| Severe speakers' test sentences | 65 | 58.42% | 41.79% | 41.14% | 0.66 (-3.87 to 5.10) |
| All eight speakers' test sentences | 174 | 28.58% | 22.15% | 21.01% | 1.14 (-0.95 to 3.17) |
| Full test set (words and sentences) | 696 | 44.06% | 31.37% | 36.91% | -5.54 (-7.81 to -3.47) |
| ... of which single words (from by_group) | 522 | 80.46% | 53.07% | 74.33% | - |
| Controls (typical speech) | 1179 | 12.40% | 9.86% | 13.07% | -3.21 (-4.26 to -2.18) |

Run 30 Sep 2026 on Yack, all scores Int8. I saved the prediction in this file at 17:07, during training and before any test score existed.

Sentences-only against stock: 17.29 better on the severe test sentences (11.04 to 23.87), 7.57 better on all test sentences (4.68 to 10.82) and 7.14 better on the full test (4.85 to 9.73). On the controls it was 0.68 worse (-1.61 to 0.33, includes zero).

What the numbers show, before interpreting them. On test sentences the two prototypes are within about 1 point, and both intervals include zero, so there's no measurable difference. On the full test set, which is mostly single words, sentences-only is 5.54 points worse with an interval clear of zero, and that comes from the single words it never trained on. On typical speech it's 3.21 points worse than the all-recordings prototype (interval clears zero) and no better than stock, while the all-recordings prototype keeps its 2.5-point gain. Both fine-tuned models beat stock clearly on sentences.

Prediction against actual: I predicted a much lower WER, 10-20 points, from sentence-only training. On test sentences the difference was 0.66 (severe) and 1.14 (all eight) points, neither measurable, and it was worse on the full test set and on typical speech.

### Conclusions

*Written 21 October 2026 from the recorded results above.*

On test sentences the sentences-only model and the all-recordings prototype are within about 1 point (65 severe test sentences, 174 for all eight speakers, Int8, lower is better). Both intervals include zero, so removing single words from training isn't shown to help sentences or to hurt them.

The cost shows up elsewhere. It's 5.54 points worse on the full 696-clip test set (interval clears zero), mostly from single words (74.33% against 53.07%). It's also 3.21 points worse on typical speech, where it lost the prototype's gain over stock (13.07% against stock's 12.40%, interval includes zero).

My prediction of a 10 to 20 point drop in WER didn't hold, and the idea that single words were "poisoning" the model isn't supported here.

Stage K on the pooled lab set (425 words, three seeds, adapter on stock, `reports/f2-adapter-levers`) later found the same pattern with a different recipe. Sentences-only training left the sentence count unchanged (117 of 303 wrong) and raised single-word errors from 95 to 129, while both kinds together were best (212.0 against 246.0).

Keep training on words and sentences together. The sentences-only model isn't an expo candidate.

</details>

## Related

[Prototype model: all eight dysarthric speakers, bottom 8 blocks](../prototype-all-dysarthric-bottom8/notes.md), [Typical-speech check](../typical-speech-check/notes.md)
