# Lab test 28: the deployable export

Proposal: F-4 and device behaviour (`docs/design.md`, Runtime model contract). Follows lab test 19. Prototype bar: 35%, which is 148 wrong of 425 pooled test words.

Status: written 20 October 2026, run in the `1020-new` queue on the lab computer (done 02:09 on 21 October, lab clock). Results read from the printed tables in `logs/` on 21 October 2026; the queue's own tables and reports stayed on the lab computer.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

Lab test 19 found the default int8 recipe costs 8 to 12 words and `--int8-ops matmul` costs nothing measurable, on whole-block and the adapter on F-1. The model I'd actually ship now is the pretrained adapter (stage P, 178 of 425), which had never been exported. So I export it both ways, decode with sherpa-onnx, and compare with the same weights in NeMo and lab 19's exports.

## Setup

Per speaker, the pretrained adapter (seed 17, 6e-4) is trained fresh as `lab-pre-ad8-<speaker>` so the weights on disk and the NeMo answers come from the same run. Then `export.py --nemo <pretrained .nemo> --adapter adapter.pt` runs with the default recipe and with `--int8-ops matmul`, each scored by `evaluate.py --model-dir`. Per-channel waits until lab 19's six per-channel files are scored.

## Plan

1. Rebuild the pretrained models where their weights were released; train and decode the fresh adapter.
2. Two exports and two sherpa-onnx decodes per speaker.
3. One pooled table with stock, NeMo, both int8 forms and lab 19's rows.

Decision rule, set before the run: an int8 form within 10 words of NeMo is deployable, and matmul wins if it's at least as good as the default. If matmul is within noise, it goes to the Pi for F-4 and a speed test. If both forms are 10 or more words worse, the adapter in the graph quantises badly and I'd try per-channel or keep the adapter block in fp32.

## Why

Expo is 3 November. The Pi needs one model folder, and I want its WER known before copying it.

## Limits

This measures the int8 cost of this adapter on these clips under two recipes. It doesn't cover Pi speed or memory, which F-4 measures on the Pi, or the per-channel recipe.

## Results

Both int8 forms are inside the 10-word rule, and matmul is the one to ship. Fresh pretrained adapter, decoded by sherpa-onnx as on the Pi:

| System | M01 | M02 | M04 | Pooled (of 425) | WER | Empty answers |
| --- | --- | --- | --- | --- | --- | --- |
| stock (NeMo) | 131 | 62 | 130 | 323 | 76.0% | 34/166 |
| pretrained adapter, NeMo fp32 | 68 | 39 | 69 | 176 | 41.4% | 0/166 |
| int8, default recipe | 70 | 36 | 74 | 180 | 42.4% | 6/166 |
| int8, `--int8-ops matmul` | 67 | 39 | 71 | 177 | 41.6% | 1/166 |
| adapter on F-1, int8 default (lab 19) | 83 | 45 | 77 | 205 | 48.2% | 6/166 |
| adapter on F-1, int8 matmul (lab 19) | 80 | 44 | 69 | 193 | 45.4% | 5/166 |

Default costs 4 words over fp32 (176 to 180), matmul costs 1 (176 to 177). Both are within seed noise and matmul is at least as good, so matmul is the deployable form. Its gain over stock (34.4 points, 95% interval 27.1 to 42.1) matches fp32's (34.6). Neither meets the bar; the best int8 row is 29 words above 148. It's also the first adapter-inside-the-graph export decoded by sherpa-onnx, and it loads, returns nothing for silence and transcribes a real recording correctly.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

No prediction was written before the `1020-new` queue started (the command `python lab.py queue 25 29 28 30 27 26 --stamp 1020-new --detach` is recorded in `app/scripts/asr/jack_notes.md`, 20 October), and none is added after the fact.

Numbers from earlier runs for comparison, noted on 21 October before any result came in. Stage P's three seeds of this recipe scored 183, 176 and 175 (seed 17 was 183); GPU runs aren't bit-for-bit repeatable, so the fresh model's own NeMo row is the reference. In lab 19 the default cost 12 words on whole-block (291 to 303) and 8 on the adapter on F-1 (197 to 205), and matmul nothing measurable (285 and 193). Lab 19's per-channel exports crashed in the weight check (fixed 20 October) and were never scored.

### Commands and Setup

```bash
python lab.py run 28
```


About 3 adapter trainings, 6 exports and 6 CPU decodes, roughly 30 minutes. Trimming deletes the matmul export folders after scoring; to keep one for the Pi, run `python lab.py --set trim_models=off run 28 --rerun`.

### Full Results

Sources: `logs/pool_table-20261020-230709-329639.log` and `logs/lab-1020-new-28.txt`. Intervals resample clips.

| System | Pooled | WER | 95% CI | Gain over stock, points (95% CI) |
| --- | --- | --- | --- | --- |
| stock | 323 | 76.0% | 66.7 to 86.3 | |
| pretrained adapter, NeMo fp32 | 176 | 41.4% | 35.7 to 48.2 | +34.6 (27.1 to 42.4) |
| int8 default | 180 | 42.4% | 36.3 to 49.6 | +33.6 (25.9 to 41.4) |
| int8 matmul | 177 | 41.6% | 35.6 to 48.7 | +34.4 (27.1 to 42.1) |
| adapter on F-1, int8 default (lab 19) | 205 | 48.2% | 42.0 to 55.6 | +27.8 (20.0 to 36.1) |
| adapter on F-1, int8 matmul (lab 19) | 193 | 45.4% | 38.6 to 53.5 | +30.6 (23.5 to 37.9) |

The int8 weight check that the export prints after quantising reproduces the dead-channel finding from the earlier CPU-only experiments. In the default export, 5 of 295 quantised weight tensors have more than half their values at the zero point: the depthwise convolutions in blocks 22 (99.8% zeros), 1 (99.2%), 2 (67.4%) and 0 (55.6%), and the second feed-forward matrix of block 23 (61.0%). The matmul export has 1 of 218, the block-23 matrix. All three speakers' exports show the same counts. Those weights do nothing on the device, which fits the 4-word cost of the default recipe.

The fresh model's 176 against stage P's seed-17 183 is a 7-word gap between two trainings of one recipe, and that's the noise to keep in mind for every row above. The silence check gave an empty answer for both recipes. The 73 to 93 ms decode times printed on the lab computer include model load and aren't Pi timings.

Not measured: speed, memory or WER on the Pi (F-4), sentence-only int8 scores, per-channel. Trimming deleted the matmul folders, so nothing is on the Pi yet.

### Conclusions

The device file exists as a recipe: the pretrained adapter exported with `export.py --nemo <pretrained .nemo> --adapter adapter.pt --int8-ops matmul`. Its int8 cost is within noise, so the NeMo WER carries over to sherpa-onnx. What's left is the Pi: regenerate the export (about 30 minutes) and run F-4 and the speed benchmark.

</details>

## Related

[Lab test 19](../lab-19-int8-cost/notes.md), [F-4](../../../docs/measurements.md#f-4--export-and-the-pi-h4)
