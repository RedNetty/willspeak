# Lab test 15: int8 screen, which checkpoints will survive export to int8

Proposal: F-4 (H4), export and the Pi. Prototype bar: 35%.

Status: completed 8 October 2026, reviewed 12 October. The screen saved `screen.json` on the lab computer. No new int8 WER was measured.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

`export.py` quantises the encoder to int8 with ONNX Runtime's dynamic quantizer, which uses one scale and zero point per weight tensor. A channel whose weights are all under half a step lands on the zero point and goes dead. Earlier CPU-only experiments (design work, `docs/research/`, round 5) found such channels in the depthwise convolutions of blocks 1 and 22 of the stock model. Fine-tuning moves weights, so each model needs checking before its int8 score is trusted.

## Setup

`quant_screen.py` applies the quantizer's formula to every encoder weight tensor and counts, per block, depthwise channels whose integer weights all equal the zero point. Its formula matched ONNX Runtime's output integer for integer on a small depthwise convolution. The exporter probably folds BatchNorm into the depthwise weights, which changes which channels die, so it reports both `dead_raw` and `dead_folded`. For an exported file it reads the stored ConvInteger and MatMulInteger weights directly, which is the real answer.

Screened: stock, every candidate in the Hugging Face cache, each speaker's whole-block and F-1 models, other lab models (`models/parakeet-ft-lab-*`), and any `encoder.int8.onnx` under `models/`.

## Plan

1. Run `quant_screen.py` over what exists and write `screen.json`.
2. For any flagged model, measure the real int8 cost: `export.py --nemo <model> --out <folder>/onnx-int8`, then `evaluate.py --manifest <test manifest> --audio-root <export> --model-dir <folder>/onnx-int8 --group-by kind --out <new folder>`, and compare with its NeMo decode.

Rule set before the run: dead channels only hint at an accuracy cost. The measurement is int8 WER against NeMo WER on the same clips, within the noise of the F-4 export check. A model with far more dead channels than stock gets exported and measured first, and if its int8 WER is worse than its NeMo WER, this is the likely cause. If counts match stock, int8 is probably fine but still gets measured once. If raw and folded counts differ, the exported file's count is the one that matters, and exporting one model shows which one the folding matches.

## Why

A model that scores well in NeMo but badly after export is useless on the Pi, and this catches the likely cases in minutes.

## Limits

It counts dead weight channels only. It says nothing about the accuracy cost or about quantising anything other than weights. Adapters have no int8 export, so `--adapter-after-blocks` models aren't covered.

## Results

The screen found 2,022 depthwise channels with all quantized weights at their
zero point in the installed stock int8 encoder:

| Encoder block | Channels with zero-point-only weights |
|---|---:|
| 1 | 1,002 / 1,024 |
| 22 | 1,020 / 1,024 |

The simulated folded counts from the checkpoint match the installed export, and the raw weights have no dead channels. The v2 whole-block and F-1 models also have 2,022. Other folded totals: v3 2,458, 1.1B 3,618, RNNT 3,031, 110M 0. Nemotron and CTC were skipped because their caches weren't available.

Source: completed screen log (`logs/quant_screen-20261008-211316-095722.log`). It records writing `reports/lab-15-int8-screen/screen.json`, which is now in this folder (checked 21 October).

So there's a real weight-quantization issue to test with per-channel or MatMul-only export. It doesn't give a WER cost, doesn't mean whole blocks do nothing, and doesn't explain the poor float NeMo results. Alternative exports need comparing on identical clips.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

None written before the run (8 October 2026), and none added since.

### Commands and Setup

Run from `app/` in the WSL tree on Yack after `git pull`, GPU free, PC set not to sleep. The script stops if a `train.py` run is already going, and each stage stops if one of its outputs already exists.

```bash
python lab.py run 15
```

The script writes `logs/lab-<stamp>-summaries.txt` at the end, which is what these notes record. Full output is in `logs/lab-<stamp>-15.txt`. Exported files need the `onnx` package, which the training environment has.

Commands come from `lab.py`. The flags were checked against each script with a dry run (`python lab.py run --dry`, `tests/unit/test_lab_cli.py`) and the logic has unit tests on synthetic data, which aren't recognition results. Nothing in this stage had run on the real checkpoints beforehand, and it measures no int8 WER.

### Full Results

The linked log has the checkpoint and export counts.

### Conclusions

The installed export shows the folded-weight problem the design work predicted. Next is measuring WER across export recipes. Channel counts alone can't tell us about accuracy.

</details>

## Related

[F-4: export and the Pi](../../../docs/measurements.md), [Lab test 7](../lab-07-stock-vs-fine-tuned/notes.md)
