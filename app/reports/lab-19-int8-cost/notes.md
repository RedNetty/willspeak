# Lab test 19: what int8 costs, three export recipes on the two best model types

Proposal: device behaviour (`docs/design.md`, Runtime model contract), not a proposal item. Prototype bar is 35%.

Status: run 16 October 2026 (`1016-retry`, lab computer, no prediction written first). Rerun 27 October 2026 (queue `1027-1530`) with all 18 recipe scores, see Results.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

Every lab number is a NeMo fp32 decode, but the Pi 5 runs int8 files from `scripts/asr/export.py`. Lab test 15 found almost all depthwise convolution channels in stock encoder blocks 1 and 22 have zero int8 weights (1,002 and 1,020 of 1,024), but not what that costs in words. So here the same model is decoded both ways.

## Setup

Models: each speaker's whole-block model and adapter on F-1. Recipes: `default` (all matrix products and convolutions in int8, as every earlier export), `matmul` (`--int8-ops matmul`, convolutions stay fp32 so the dead channels aren't quantised) and `perchannel` (`--per-channel`, one int8 scale per output channel). The int8 files are decoded with sherpa-onnx through `evaluate.py --model-dir`, a different decoder program from NeMo.

## Plan

1. Export 18 files (2 model types x 3 speakers x 3 recipes) and decode each speaker's test clips with each.
2. Delete each export after its decode when `trim_models` is on (the report folder keeps the scores).
3. Pool against stock and each model's own NeMo fp32 decode.

Decision rule, set before the run: nothing beyond the 4.5-point noise rule, applied to the difference between a model's int8 row and its own NeMo fp32 row.

Outcomes, written before the run. Default within noise of fp32: int8 costs nothing measurable and the dead channels don't matter. Default worse but matmul or perchannel close to fp32: the cost is real and the device uses that recipe. All recipes worse by a similar amount: the gap is the decoder program or the matrix-product quantisation, and a sherpa-onnx fp32 decode would separate them (not part of this test). Int8 better than fp32 by more than noise: the decoders differ, it's not a gain.

## Why

The 35% bar is judged on the device. If int8 adds a few points, every other lab test understates the gap.

## Limits

This shows the change in wrong words from NeMo fp32 to each int8 recipe, for two model types and three speakers. It doesn't show speed or memory on the Pi, accuracy in a real room, or what causes a gap (decoder program or quantisation).

## Results

Lab run 16 October 2026, queue `1016-retry` on the RTX 4000 Ada (code a9e7242 or later), from `logs/lab-1016-retry-19.txt` and `reports/lab-19-int8-cost/table.json`. Pooled wrong words of 425 (M01 169, M02 95, M04 161), stock 323. Int8 goes through sherpa-onnx and fp32 through NeMo, so a small gap could be the decoder.

| Model | NeMo fp32 | int8 default | int8 `--int8-ops matmul` | int8 `--per-channel` |
| --- | --- | --- | --- | --- |
| Whole-block | 291 (68.5%) | 303 (71.3%) | 285 (67.1%) | export written, not scored |
| Adapter on F-1 | 197 (46.4%) | 205 (48.2%) | 193 (45.4%) | export written, not scored |

Default costs 8 to 12 words (2 to 3 points). Keeping convolutions in fp32 (`--int8-ops matmul`) costs nothing measurable: 285 and 193 are 6 and 4 below fp32, inside clip noise. Empty answers: whole-block 21 to 24 of 166 clips in every form, adapter on F-1 3 to 6. That settles the round-5 dead-channel question. Export with `--int8-ops matmul`, then time it on the Pi, since fp32 convolutions are slower.

All six `--per-channel` exports crashed after writing their ONNX files, in `export.py`'s int8 weight check. It broadcast per-channel zero points along axis 0, but a `MatMulInteger` weight is (in, out), so the channel axis is 1 (`shapes (4096,1024) (1024,1)`). Fixed 20 October. The files on the lab computer can be scored without re-exporting (`evaluate.py --model-dir models/lab-19-export-<model>-<speaker>-perchannel`). That's the only reason the stage was marked failed.

Rerun 27 October 2026 (queue `1027-1530`, `logs/lab-1027-1530-19.txt`). With the fix, per-channel exports finished and were scored, so every recipe has a result (pooled wrong of 425, stock 323):

| Model | NeMo fp32 | int8 default | int8 matmul | int8 per-channel |
|---|---:|---:|---:|---:|
| Whole-block | 291 | 303 | 285 | 290 |
| Adapter on F-1 | 197 | 205 | 193 | 206 |

Per-channel is no better than default for the adapter (206 vs 205) and sits in between for whole-block, so it doesn't win the cost back. `--int8-ops matmul` stays the recommended device recipe. Sentences only, the adapter on F-1 gets 109 in NeMo (36.0%); the other sentence scores are in the log. The per-channel arm no longer needs scoring.

An earlier attempt on 15 October stopped at once with `MISSING models/lab-baselines/whole-block-M01/parakeet-ft.nemo`. Its pre-step rebuilt the three F-1 models (about 1 hour) but skipped the whole-block and adapter-on-F-1 models because their test decodes existed, and this stage needs the weights. Fixed in `lab.py` (`prepare_baseline`): stages that load other baselines' weights now retrain a model when only its decodes are on disk. Run `python lab.py run 19` again.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

No prediction was written before this run (16 October 2026, queue `1016-retry`), and none is added after.

### Commands and Setup

Run from `app/` in the WSL tree on Yack after `git pull`, GPU free, PC set not to sleep. It exports 18 ONNX folders of a few hundred MB each, removed after decoding when `trim_models` is on.

```bash
python lab.py run 19
```

At the end the script writes `logs/lab-<stamp>-summaries.txt`, which is the file these notes record. The stage's full output is `logs/lab-<stamp>-<stage>.txt`.

The commands come from `lab.py`, with flags checked against each script by a dry run (`python lab.py run --dry`; `tests/unit/test_lab_cli.py`, `tests/unit/test_lab_hardening.py`) and logic covered by unit tests on synthetic data, which aren't recognition results. At setup on 14 October 2026 nothing had run on real models or data and no GPU was free for validation.

### Full Results

See Results. The per-model gains against stock with clip-bootstrap intervals are in `table.json`, and every row's interval against stock excludes zero.

### Conclusions

Use `--int8-ops matmul` for the device export. The default recipe's 8 to 12 lost words are real but small. Per-channel is unmeasured.

</details>

## Related

[Lab test 15](../lab-15-int8-screen/notes.md), [Adapter on F-1](../f2-m01-m04-adapter8-on-f1/notes.md)
