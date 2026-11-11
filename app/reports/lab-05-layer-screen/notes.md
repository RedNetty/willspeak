# Lab test 5: encoder layer screen, which layer of Parakeet matches his words best across days

Proposal: T-3 (matcher accuracy across sessions) and F-2 (H3), an extension. Prototype bar: 35%.

Status: completed 28 October 2026. The 8 October run saved the embeddings and last-session table but its causal summary crashed. The fixed summary ran in queue `1028-rest` and produced the all-session and causal tables.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

The exemplar matcher uses WavLM layer 9, a projection fitted on other speakers, and cosine similarity to his recorded words. Earlier CPU-only experiments (`docs/research/`, round 5, background only) found layer 10 of the int8 Parakeet encoder matched words across days better. Here I score every layer of the real encoder, stock and fine-tuned, with the matcher's recipe on TORGO.

## Setup

Layer 0 is the subsampling output and layer k is the output of block k - 1, so 24 blocks give layers 0 to 24. Each word clip is embedded alone as the mean over its frames.

Scoring copies the shipped matcher: subtract the speaker's session mean, fit a shrinkage LDA (128 dimensions, alpha 0.1) on every other speaker, enrol one take of each word from one session, test on the other sessions, and repeat over enrolment sessions. Intervals resample words since the same words recur every session. A second table tests on the last session only, like the F-2 split.

Three archives: stock, the M02 whole-block model (trained on M02's earlier sessions) and the F-1 model for M02 (never heard M02). Fine-tuned rows are only compared with stock on the last-session table.

The main table's session mean includes the test clips, so absolute numbers are optimistic, but the layer ranking is fair. `--causal` adds an accuracy whose session mean only uses earlier clips.

## Plan

1. Make the WavLM word embeddings (`data/emb/torgo-words.npz`) if missing, then embed every clip at all 25 layers for the three Parakeet models.
2. Run `layer_screen.py` with WavLM as baseline: accuracy pooled over severe speakers and over all speakers with two or more sessions, with intervals and the severe gain over WavLM.
3. Same with `--last-session`.

Decision rule, set before the run: a layer beats WavLM only if the printed word-level bootstrap interval on the severe gain excludes zero. A winner would be a candidate front end that saves loading WavLM, though the device would need its activations from the int8 model in sherpa-onnx, which hasn't been checked. No winner means the shipped front end stays. Fine-tuned layers beating stock on the last-session table would mean fine-tuning organises speech in a way a matcher can use, once you account for which model heard which speaker.

## Why

The matcher handles the closed-vocabulary words, so a better front end helps them with no new training.

## Limits

Single TORGO words only. Nothing about the intended user or int8 activations (these come from float NeMo). Fitting takes a few seconds per layer per held-out speaker; `LAYERS` limits the layers if it's slow.

## Results

The last-session screen finished on 2,385 shared clips, six speakers with at least two sessions (F03, F04, M01, M02, M04, M05). Severe pool is M01/M02/M04. Top-1 means picking the right word among the enrolled ones, not ASR WER.

| Feature row | Severe top-1 accuracy | Gain over WavLM |
|---|---:|---:|
| WavLM baseline | 46.6% | — |
| stock Parakeet layer 9 | 61.6% | +15.1 points, printed 95% CI +5.0 to +25.7 |
| stock Parakeet layer 10 | 61.0% | +14.4 points |
| M02 whole-block layer 10 | 66.4% | +19.9 points |
| M02 F-1 layer 10 | 73.3% | +26.7 points |

Source: completed last-session log (`logs/layer_screen-20261008-205355-495046.log`). `results-last-session.json` was written on the lab computer and is now in this folder (checked 21 October: WavLM severe 0.4658, matching 46.6%). The three regenerated Parakeet archives each kept 2,385 clips after dropping zero-frame clips.

The higher rows look promising, but session means use complete test sessions and the intervals don't correct for searching many layers. The F-1 model for M02 was trained on M01/M04 among others, so its 73.3% isn't unseen-speaker performance. No live, causal or int8 gain is established here.

### Second run, 28 October 2026

Queue `1028-rest`, started 15:54 lab clock; sources `logs/lab-1028-rest-5.txt` and `logs/lab-1028-rest-summaries.txt`. The 27 October queue had failed stage 5 by trimming the stored `whole-block-M02` weights too early (stage 5 wasn't on the keep-weights list; fixed). This run rebuilt whole-block M02 and F-1 M02, kept the 8 October embeddings, and rescored with the fixed summary.

The last-session table matches 8 October exactly (WavLM 46.6, stock layer 9 61.6, M02 whole-block layer 10 66.4, M02 F-1 layer 10 73.3). The all-session table (enrol on one session, test on the others) is new. The causal column counts accuracy query by query in arrival order.

| Feature row | Severe-speaker top-1 [95% CI] | Gain over WavLM | Causal, severe |
|---|---:|---:|---:|
| WavLM baseline | 42.7 [34.5, 50.2] | | 43.1 |
| stock Parakeet layer 9 | 56.9 [50.4, 63.6] | +14.2 [+6.3, +22.4] | 58.4 |
| stock Parakeet layer 10 | 58.0 [50.6, 66.3] | +15.3 [+6.6, +24.5] | 63.5 |
| stock Parakeet layer 24 | 59.5 [51.2, 67.9] | +16.8 [+9.0, +25.1] | 59.9 |
| M02 whole-block layer 10 | 62.4 [54.5, 69.8] | +19.7 [+11.8, +28.0] | 64.6 |
| M02 F-1 layer 10 | 70.8 [63.1, 77.9] | +28.1 [+20.8, +36.3] | 71.9 |

All models have the same shape. Layers 0 to 6 are far below WavLM (stock layer 0 is 10.9). Stock reaches WavLM at layer 7 (+2.2, interval -6.2 to +9.7), is clearly above from layer 9, and sits around 55 to 59 through layer 24 with nothing beating layer 10 beyond the intervals. F-1 peaks at layer 10 (70.8) and stays at 62 to 69 after.

For stock layers the causal column is within 1 to 5 points of the session-mean numbers, so that optimism was small. Same limits as before: no correction for searching 25 layers, the whole-block and F-1 rows saw audio from some scored speakers (M02's own sessions, and M01 and M04 for F-1), and these are enrolled-word accuracies.

The first all-session `--causal` run crashed pooling `(accuracy, query_count)` pairs as scalars. The fix uses the causal query counts and skips speakers with zero scored queries. Saved embeddings can be rescored on CPU, no GPU pass needed.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

None written before this run (8 October 2026), and none added afterwards.

### Commands and Setup

Run from `app/` in the WSL tree on Yack after `git pull`, GPU free, PC set not to sleep. The script stops if a `train.py` run is going, and each stage stops if one of its outputs exists.

```bash
python lab.py run 5
```

The script ends by writing `logs/lab-<stamp>-summaries.txt`, the file these notes record. Full output is `logs/lab-<stamp>-5.txt`. `LAYERS="0,4,8-16,24"` in front of `bash` screens a subset.

The commands come from `lab.py`. Flags were dry-run checked (`python lab.py run --dry`) (`tests/unit/test_lab_cli.py`), and script logic has unit tests on synthetic data, which aren't recognition results. Not run on real models or data at setup: `embed_layers.py` on real NeMo (hooks tested on a stand-in encoder) and `layer_screen.py` at full size.

### Full Results

The linked last-session log has the completed table. When first written, the causal table was waiting on a summary rerun.

### Conclusions

Parakeet's intermediate features are worth a causal, closed-vocabulary comparison with WavLM before anything in the matcher is replaced. These accuracies don't measure open speech recognition or intended-user performance.

</details>

## Related

[T-3: closed-vocabulary and cross-session accuracy](../../../docs/measurements.md), [Lab test 14](../lab-14-route-scoring/notes.md)
