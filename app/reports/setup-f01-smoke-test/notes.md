# Setup: F01 smoke test and stock export check

Proposal: §4.2, export prerequisite for F-4

Status: Recorded 24 Sep 2026.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

Check that training and ONNX export run, and find out whether training or export hurts recognition.

## Setup

F01 has one session: 85 training clips (3.4 min), 10 dev and 23 test. One epoch gave six optimizer steps, and warm-up was ten. This is a setup check, so it isn't a session-disjoint F-2 result.

## Plan

Score prebuilt stock, stock through our own export, and the one-epoch fine-tune on the same 23 clips, plus one second of silence.

## Why

A broken export or a model that talks on silence needs explaining before any real fine-tuning.

## Limits

This separates export-only changes from changes made by the smoke run. It can't show useful fine-tuning, and it doesn't finish F-4 since the Pi benchmark was skipped.

## Results

| Model | Test WER | One second of silence |
|---|---:|---|
| Prebuilt stock | 73.91% | Empty output |
| Re-exported stock | 73.91% | Empty output |
| Smoke test | 93.48% | "I don't know." |

The smoke test came out 19.57 points worse than the re-exported stock (95% CI 4.65 to 30.56 points worse).

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

The smoke test was a quick trial fine-tune. Six small steps changed its weights, though only slightly. The silence check did turn something up: on one second of silence, stock returned "" and the smoke test model said "I don't know."

### What I expect before running the export check

Stock on the same 23 F01 clips: 50-60% - from existing testing this is typically the range we end in.

Our re-export of our stock model against the prebuilt stock: I would say there is likely a 1-2% point gap or margin for change, where the process of which we're exporting the model is different from
how the sherpa-onnx developers exported the prebuilt model.

Re-exported stock on 1 s of silence: nothing - From our testing I'm expecting to see blank, the weights are the same as stock and only the export is different.

The smoke test against stock: expecting a much higher error percentage, the smoke test is already at 93.48%.. so about 33-43 points worse than my stock guess. - The literature shows a large margin of gain comes from
the first 5-8 minutes of recordings, but we're only training on 85 clips (3.4 min), each seen once, and using very little time during the train portion.

Question 2, does it say anything during silence: I'm expecting the prebuilt stock and our re-export to both return silence or "", so the smoke test model's "I don't know." comes from the training steps.
I would be surprised if: 30-40% is the range stock gets, then we're getting better than expected results, and that means these 23 clips are easier than the T-1 clips.

### Results

Machine: Yack, WSL2, 24 GB given to WSL after the 24 Sep RAM upgrade

The smoke test, one epoch on F01

```bash
python scripts/asr/train.py --export data/public/torgo/export/F01 --out models/b4-f01 --epochs 1 --freeze-encoder-layers 16 --batch 2 --grad-accum 8 --warmup-steps 10
```

  85 train clips, 3.4 min, 10 dev; 6 optimizer steps, warmup of 10 never reached
  F01 test, 23 clips: WER 93.48%   CER 68.78%   PER 73.55%
  1 s of silence: "I don't know."   Prebuilt stock on the same silence: ""

The export check, same 23 F01 test clips
  Prebuilt stock:            WER 73.91%   CER 51.85%   PER 51.61%   (ci.py 95% CI 57.14% to 113.04%, n = 23)
  Stock through export.py:   WER 73.91%   CER 51.85%   PER 51.61%
  Stock through export.py on silence: ""
  ci.py, prebuilt stock against stock through export.py: Delta 0.00 points (95% CI 0.00 to 0.00)
  ci.py, stock through export.py against the smoke test: Delta -19.57 points (95% CI -30.56 to -4.65), the smoke test is worse

Prediction against actual
  Stock on the 23 clips: predicted 50-60%, measured 73.91%
  Re-export against prebuilt: predicted a 1-2 point gap, measured 0.00
  Re-export on silence: predicted nothing, measured nothing
  the smoke test against stock: predicted 33-43 points worse, measured 19.57 worse
  Silence: predicted stock and re-export silent, so the smoke test's answer comes from training; measured the same
  Surprise range, stock at 30-40%: not reached

### What I concluded

1. The pre-built stock and our own export stock scored exactly the same, and that means when the first F-2 run compares a trained model to stock, the conversion
will not explain a difference.
2. The smoke test's training, 19.57 points worse than stock. The interval runs from 30.56 to 4.65 points worse and never reaches zero, so the drop is
there and not just chance. It started answering silence, and that's a bit of an issue. The first F-2 run can't use this format until I do the batch-norm diagnosis to see what is causing this.
3. My prediction of having no export gap is real, and shows both staying silent on silence.

</details>

## Related

[Setup: F01 batch-norm diagnosis](../setup-f01-batch-norm/notes.md)
