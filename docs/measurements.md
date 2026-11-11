# Measurements

Paths and commands below are relative to `app/` unless they refer to documentation.


Every project number belongs to a run made on the project's hardware. Public
experiment notes are tracked in Git; generated outputs remain ignored except
the selected T-1 baseline evidence. [Report Context](../app/reports/Report%20Context.md)
maps every note to the proposal, its purpose, and its existing output paths.
Numbers from the CPU pilot references are design context.

## `notes.md` template

Keep the overview short; expand the run record for reproducibility.

~~~markdown
# <proposal code · subject · what changed>

**Proposal:** <section / hypothesis>
**Status:** <planned, running, or recorded; actual date; missing information>

[Report Context](../app/reports/Report%20Context.md)

## The Idea
<The question this run answers.>

## Context
<Prior result, data, speakers, sessions, and what changes.>

## Plan
<Comparison, fixed conditions, scoring, and decision rule.>

## Why
<How this answers the proposal or changes the next design choice.>

## What This May or May Not Show
<What a gain, loss, or inconclusive interval means; limits on the conclusion.>

## Results
<Small table: sample size, metric, comparator, units, and CI.
For a planned run, say 'Not run'; an empty result is not zero.>

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)
<Original prediction, dated before running; 'Not recorded' if absent.>

### Commands and Setup
<Exact commands; machine, OS and runtime; model and decoder; manifest,
audio root, split sessions and sizes; output path; release rule.>

### Full Results
<Measurements, checks, and prediction versus actual.>

### Conclusions
<My original conclusion, or 'Not recorded'.>

</details>

## Related Reports
<Links to prerequisites, comparators, or follow-up notes.>
~~~

Write the prediction **before** running. Do not invent a missing prediction,
fill a blank result from an expectation, or change a missed prediction after
seeing the score. Existing notes retain their original run records.

---

## Run names

Name a run by its proposal code, subject, and change:
F-2 · M02 · bottom 8 blocks · all parts; new folders use lowercase,
f2-m02-bottom8-all. Notes now use this scheme even for earlier runs.
Setup diagnostics use setup-…; the forgetting check uses typical-speech-check.

**Existing generated-output and model folders keep their actual names.**
The lookup below preserves those paths and earlier codes for reproducibility;
[Report Context](../app/reports/Report%20Context.md#notes-map) lists current
note folders. H5, T-4 and P-G are later repository extensions, outside the
supplied proposal's H1–H4 / T-1–T-3 / F-1–F-4 / P-A/P-B/P-D/P-E/P-F.

| Name | Earlier code | Generated/model paths or current notes |
|---|---|---|
| Data · fetch TORGO, build the manifests | B1 | `data/public/torgo/` |
| T-1 · zero-shot baseline, severe speakers / all dysarthric speakers | B2 (Run A / Run B) | `reports/torgo-severe-zero-shot`, `reports/torgo-all-zero-shot` |
| Setup · training machine | B3 | `models/smoke-test` |
| Smoke test · F01 · 1 epoch | B4 | `models/b4-f01`, `reports/b4-f01` |
| Export check · stock model on F01 | B5 | `models/stock-export`, `reports/b5-stock`, `reports/b5-stock-export` |
| Batch-norm diagnosis · F01 | B6 | `models/b6-f01`, `reports/b6-f01` |
| F-2 · M02 · top 8 blocks · batch norm updating | C1 | `models/parakeet-ft-M02`, `reports/finetune-torgo/M02` |
| NeMo decode check · first F-2 run | C1 check | in `reports/f2-m02-top8-bn-updating/notes.md` |
| Export-parts check · first F-2 run | C1 check 2 | in `reports/f2-m02-top8-bn-updating/notes.md` |
| F-2 · M02 · top 8 blocks · batch norm frozen | C1b | `models/parakeet-ft-M02b`, `reports/finetune-torgo/M02b` |
| F-2 · M02 · top 8 blocks · batch norm frozen · int8 export | C1c, part 1 | `reports/finetune-torgo/M02b-test-int8` |
| F-2 · M02 · top 8 blocks · normalised transcripts | C1c, part 2 | `models/parakeet-ft-M02c`, `reports/finetune-torgo/M02c` |
| F-2 · M02 · top 8 blocks · normalised transcripts · int8 export | C1d | `reports/finetune-torgo/M02c-test-int8` |
| F-2 · M02 · bottom 8 blocks · all parts | C1e, Model A | `models/parakeet-ft-M02e-A` |
| F-2 · M02 · bottom 8 blocks · attention + conv only | C1e, Model B | `models/parakeet-ft-M02e-B` |
| F-2 · M02 · bottom 8 blocks · both, failed dev-score fix (not scored) | A2, B2 | `models/parakeet-ft-M02e-A2`, `models/parakeet-ft-M02e-B2` |
| F-2 · M02 · bottom 8 blocks · all parts · fixed dev score | C1e rerun, A3 | `models/parakeet-ft-M02e-A3`, `reports/finetune-torgo/M02e-A3-test-*` |
| F-2 · M02 · bottom 8 blocks · attention + conv only · fixed dev score | C1e rerun, B3 | `models/parakeet-ft-M02e-B3`, `reports/finetune-torgo/M02e-B3-test-*` |
| F-2 · M02 · top 8 blocks · fixed dev score | C1f | `models/parakeet-ft-M02f`, `reports/finetune-torgo/M02f-test-*` |
| F-1 · leave-one-speaker-out · top 8 blocks (first recipe) | C2 | `models/parakeet-ft-loso-<SPK>`, `reports/finetune-torgo/loso-<SPK>` |
| F-2 · M01, M02, M04 (F01 as context) · bottom 8 blocks · fixed 5 epochs | C3 | `models/parakeet-ft-f2-<SPK>-bottom8-5epochs`, `reports/finetune-torgo/f2-<SPK>-bottom8-5epochs-test-*`, notes `reports/finetune-torgo/f2-severe-bottom8-5epochs/`. Created as `<SPK>-c3`; the H3 block renames them |
| F-1 · leave-one-speaker-out · bottom 8 blocks (current recipe) | C2b | new: `models/parakeet-ft-f1-loso-<SPK>-bottom8`, `reports/finetune-torgo/f1-loso-<SPK>-bottom8`. If already started with the earlier command: `loso-<SPK>-b` |
| F-2 · M02 · adapter after 8 blocks | none | new: `models/parakeet-ft-f2-M02-adapter8` (holds `adapter.pt`), `reports/finetune-torgo/f2-M02-adapter8-test-nemo`, notes `reports/f2-m02-adapter8/` |
| F-2 · M02 · adapter after 8 blocks · seeds 18 and 19 | none | new: `models/parakeet-ft-f2-M02-adapter8-seed<N>`, `reports/finetune-torgo/f2-M02-adapter8-seed<N>-test-nemo`, notes `reports/f2-m02-adapter8-seeds/` |
| Check · cleaned-up train.py · reruns of recorded results | none | new: `models/parakeet-ft-f2-<SPK>-adapter8-recheck-seed<N>` (M01, M04), `models/parakeet-ft-f2-M02-adapter8-recheck`, `models/parakeet-ft-f2-M02-bottom8-recheck`, `models/parakeet-ft-f2-M02-adapter8b` (the recorded 15-epoch run, scored in the check), `reports/finetune-torgo/*-recheck*-test-nemo`, `reports/finetune-torgo/f2-M02-adapter8b-test-nemo`, notes `reports/check-train-py-cleanup/` |
| F-2 · M01 and M04 · adapter after 8 blocks | none | new: `models/parakeet-ft-f2-<SPK>-adapter8`, `reports/finetune-torgo/f2-<SPK>-adapter8-test-nemo`, `reports/finetune-torgo/f1-loso-<SPK>-bottom8-on-F2-test-nemo` (the F-1 models decoded in NeMo), notes `reports/f2-m01-m04-adapter8/` |
| F-2 · M01 and M04 · adapter after 8 blocks, on the F-1 model | none | new: `models/parakeet-ft-f2-<SPK>-adapter8-on-f1`, `reports/finetune-torgo/f2-<SPK>-adapter8-on-f1-test-nemo`, notes `reports/f2-m01-m04-adapter8-on-f1/` |
| F-2 · M01 and M04 · adapter on the F-1 model · seeds 18 and 19 | none | new: `models/parakeet-ft-f2-<SPK>-adapter8-on-f1-seed<N>`, `reports/finetune-torgo/f2-<SPK>-adapter8-on-f1-seed<N>-test-nemo`, notes `reports/f2-m01-m04-adapter8-on-f1-seeds/` |
| F-2 · M02 · adapter on the F-1 model · seeds 17, 18 and 19 | none | new: `models/parakeet-ft-f2-M02-adapter8-on-f1-seed<N>`, `reports/finetune-torgo/f2-M02-adapter8-on-f1-seed<N>-test-nemo`, `reports/finetune-torgo/f1-loso-M02-bottom8-on-F2-test-nemo` (the F-1 model decoded in NeMo, made only if missing), notes `reports/f2-m02-adapter8-on-f1-seeds/` |
| F-2 · adapter position sweep · M01, M02, M04 | none | new: `models/parakeet-ft-f2-<SPK>-adapter<N>-pos` (N = 4, 12, 16, 24; also 8 for M02), `reports/finetune-torgo/f2-<SPK>-adapter<N>-pos-test-nemo`, notes `reports/f2-adapter-position-sweep/` |
| Phrase matching · saved hypotheses · M01, M02, M04 | none | new: `reports/phrasebook-matching/run/results.json`, notes `reports/phrasebook-matching/` (no models; reads existing `hypotheses.jsonl` files) |
| F-2 · adapter block 8 and its neighbours · seeds, blocks 6 and 10 | none | new: `models/parakeet-ft-f2-<SPK>-adapter<N>-pos` (N = 6, 10), `models/parakeet-ft-f2-<SPK>-adapter<N>-pos-seed<S>` (S = 18, 19), `models/parakeet-ft-f2-M02-adapter8-fixed-seed<S>`, and the matching `reports/finetune-torgo/*-test-nemo`, notes `reports/f2-adapter-position-seeds/` |
| Phrase matching · rejection rule | none | new: `reports/phrasebook-matching/rejection/results.json` |
| Phrase matching · wider rejection grid | none | new: `reports/phrasebook-matching/rejection-wide/results.json` |
| Typical-speech check · every fine-tuned model | C4 | `reports/finetune-torgo/controls-<model>` |
| Phrase matching · single words and the whole test set | none | new: `reports/phrasebook-matching/words/results.json`, `reports/phrasebook-matching/device-both/results.json` (stage W of the retired `scripts/asr/big_runs.sh`, now `python lab.py run W`; set up 2 Oct) |
| F-2 · base-model ceiling · M01, M02, M04 | none | new: `reports/finetune-torgo/<SPK>-test-stock-<tag>-nemo` (tag 1p1b, v3 or 110m), `models/parakeet-ft-f2-<SPK>-adapter<N>-<tag>-seed17` (a name ending `-b1` if batch 1 was needed), `reports/finetune-torgo/f2-<SPK>-adapter<N>-<tag>-seed17-test-nemo`, notes `reports/f2-base-model-ceiling/` (stage B) |
| F-2 · augmentation variants · M01, M02, M04 | none | new: `models/parakeet-ft-f2-<SPK>-adapter8-aug-<speedwide, nospeed, nospec, tempo, noise or tempo-noise>-seed<S>`, `reports/finetune-torgo/f2-<SPK>-adapter8-aug-<variant>-seed<S>-test-nemo`, notes `reports/f2-augmentation-variants/` (stage A for the first three, stage N for the last three; N needs the `train.py` with `--time-stretch`) |
| F-3 · data-quantity curve · M02 | C5 | new: `data/public/torgo/export/M02-min<1, 3, 5, 7>`, `models/parakeet-ft-f3-M02-min<M>-<epochs or updates>-seed<S>`, `reports/finetune-torgo/f3-M02-min<M>-<arm>-seed<S>-test-nemo`, notes `reports/f3-m02-data-quantity/` (stage F; set up 2 Oct) |
| Lab test 1 · weight interpolation · stock to whole-block at 25, 50, 75% | none | new: `reports/finetune-torgo/lab-01-w<25, 50, 75>-<SPK>-test-nemo` (no new model; the blend is built in memory), notes `reports/lab-01-weight-interpolation/` (stage 1 of the retired `scripts/asr/lab_runs.sh`, now `python lab.py run 1`; the lab tests were set up 2 Oct) |
| Lab test 2 · weight merging · others' changes, and all three | none | new: `reports/finetune-torgo/lab-02-<others, all>-<SPK>-test-nemo`, `reports/lab-02-weight-merging/geometry.json`, notes `reports/lab-02-weight-merging/` (stage 2) |
| Lab test 3 · hypothesis bounds · best-of-N and vote | none | new: `reports/lab-03-hypothesis-bounds/bounds/`, notes in the same folder (stage 3; reads existing `hypotheses.jsonl` files) |
| Lab test 4 · stock model screen · other models on the severe speakers | none | new: `reports/finetune-torgo/<SPK>-test-stock-<tag>-<nemo or hf>` (the same names stages B and 7 reuse), notes `reports/lab-04-stock-model-screen/` (stage 4) |
| Lab test 5 · encoder layer screen · stock, whole-block, F-1 | none | new: `data/emb/lab-05-<v2-stock, v2-whole-block-m02, v2-f1-loso-m02>-layers.npz`, `reports/lab-05-layer-screen/results.json` and `results-last-session.json`, notes in the same folder (stage 5) |
| Lab test 6 · reference score · true words and silence | none | new: `reports/lab-06-reference-score/<SPK>-<stock, whole-block>.json`, notes in the same folder (stage 6) |
| Lab test 7 · stock against fine-tuned · four models on M02 and M04 | none | new: `models/parakeet-ft-lab-07-<tag>-<SPK>` (a name ending `-b1` if batch 1 was needed), `reports/finetune-torgo/lab-07-<tag>-<SPK>-test-nemo`, `reports/lab-07-stock-vs-fine-tuned/rank.json`, notes in the same folder (stage 7) |
| Lab test 8 · feed-forward modules only · blocks 0–7 | none | new: `models/parakeet-ft-lab-08-ff-<SPK>`, `reports/finetune-torgo/lab-08-ff-<SPK>-test-nemo`, notes `reports/lab-08-feed-forward-only/` (stage 8) |
| Lab test 8 · feed-forward modules, bisected | none | new: `models/parakeet-ft-lab-08-<ff1, ff2>-<SPK>`, `reports/finetune-torgo/lab-08-<ff1, ff2>-<SPK>-test-nemo`, `reports/lab-08-feed-forward-only/table-bisect.json` (stage 8b, after reading stage 8) |
| Lab test 9 · depth sweep | none | new: `models/parakeet-ft-lab-09-<b2, b4, b12, b16, mid8>-<SPK>`, `reports/finetune-torgo/lab-09-<arm>-<SPK>-test-nemo`, notes `reports/lab-09-depth-sweep/` (stage 9) |
| Lab test 10 · silence margin · margin loss, non-speech clips, both | none | new: `models/parakeet-ft-lab-10-<margin, noise, both>-<SPK>`, `reports/finetune-torgo/lab-10-<arm>-<SPK>-test-nemo`, `reports/finetune-torgo/lab-13-battery-<label>.jsonl` (shared with test 13), `reports/lab-10-silence-margin/battery.json`, notes in the same folder (stage 10) |
| Lab test 11 · severe pre-training, then his own sessions | none | new: `data/public/torgo/export/loso-severe-<SPK>`, `models/parakeet-ft-lab-11-<pre, then>-<SPK>`, `reports/finetune-torgo/lab-11-<pre, then>-<SPK>-test-nemo`, notes `reports/lab-11-severe-pretraining/` (stage 11) |
| Lab test 12 · pseudo-label loop · three rounds | none | new: `data/public/torgo/export/lab-12-<SPK>-r<R>`, `models/parakeet-ft-lab-12-<SPK>-r<R>`, `reports/finetune-torgo/lab-12-<SPK>-r<R>-<teacher.jsonl, test-nemo>`, notes `reports/lab-12-pseudo-label-loop/` (stage 12) |
| Lab test 13 · silence false inserts | none | new: `data/public/torgo/export/silence-battery`, `reports/finetune-torgo/lab-13-battery-<stock, whole-block-<SPK>, f1-<SPK>>.jsonl`, `reports/lab-13-silence-false-inserts/score.json`, notes in the same folder (stage 13) |
| Lab test 14 · route scoring · words to the matcher | none | new: `reports/lab-14-route-scoring/route-<stock, whole-block, adapter-on-stock>/`, notes `reports/lab-14-route-scoring/` (stage 14; no models) |
| Lab test 15 · int8 screen | none | new: `reports/lab-15-int8-screen/screen.json`, notes in the same folder (stage 15; no models) |
| Lab test 16 · synthetic dysarthric speech · stock model on degraded control recordings | none | new: `data/public/torgo/export/controls`, `data/public/torgo/export/synth-validation`, `reports/lab-16-synthetic-realism/summary.json` (WER per severity and the calibrated severity), notes in the same folder (stage 16; `scripts/asr/synth_dysarthria.py`, written 14 Oct) |
| Lab test 17 · training on synthetic speech · mix, synthetic then own, severe plus synthetic then own | none | new: `data/public/torgo/export/synth-<mix, pre, sev>-<SPK>-s<severity>-x<scale>`, `models/parakeet-ft-lab-17-<mix, pre, then, sevpre, sevthen>-<SPK>`, `reports/finetune-torgo/lab-17-<arm>-<SPK>-test-nemo`, `reports/lab-17-synthetic-training/table.json`, notes in the same folder (stage 17) |
| Lab test 18 · best system hardened · silence margin and synthetic speech on the adapter on F-1 | none | new: `models/parakeet-ft-lab-18-<sil, synth, synthsil>-<SPK>`, `reports/finetune-torgo/lab-18-<arm>-<SPK>-test-nemo`, `reports/finetune-torgo/lab-13-battery-<arm>-<SPK>.jsonl`, `reports/lab-18-best-system-hardened/table.json` and `battery.json`, notes in the same folder (stage 18) |
| Lab test 19 · int8 cost · default, matmul-only and per-channel exports | none | new: `models/lab-19-export-<model>-<SPK>-<default, matmul, perchannel>` (removed after scoring when trimming is on), `reports/lab-19-int8-cost/<model>-<SPK>-<recipe>/`, `reports/lab-19-int8-cost/table.json`, notes in the same folder (stage 19) |
| Lab test 20 · grouped cross-validation · adapter recipes on held-out prompts | none | new: `data/public/torgo/export/cv-<SPK>-<all, f0..f3>` (hard links, no extra disk), `models/parakeet-ft-lab-20-<recipe>-<SPK>-f<k>`, `models/parakeet-ft-lab-20-final-<recipe>-<SPK>` (only when a recipe other than the control is chosen), `reports/finetune-torgo/lab-20-<recipe>-<SPK>-f<k>-test-nemo.jsonl`, `reports/lab-20-grouped-cv/oof/<recipe>-<SPK>.jsonl`, `selection.json`, `table.json`, notes in the same folder (stage 20) |
| Lab test 21 · text-to-speech · synthesiser speech degraded and mixed into the adapter on F-1 | none | new: `data/public/torgo/export/lab-21-tts-<SPK>`, `lab-21-<tts, ttslow>-mix-<SPK>`, `models/parakeet-ft-lab-21-<tts, ttslow>-<SPK>`, `reports/finetune-torgo/lab-21-<arm>-<SPK>-test-nemo`, `reports/lab-21-tts-synthetic/table.json`, `battery.json`, notes in the same folder (stage 21) |
| Lab test 22 · test-time adaptation · pseudo-label self-training in chunks | none | new: `data/public/torgo/export/lab-22-<SPK>-r<k>`, `models/parakeet-ft-lab-22-<SPK>-r<k>`, `reports/finetune-torgo/lab-22-<SPK>-chunk<k>-{manifest,answers}.jsonl`, `lab-22-<SPK>-tta-test-nemo`, `reports/lab-22-test-time-adaptation/by-chunk-<SPK>.json`, `table.json`, notes in the same folder (stage 22) |
| Lab test 23 · hypothesis selector · vote, weighted vote, medoid and learned selection | none | new: `reports/lab-23-hypothesis-selector/{systems.json, labelled/, hyps/, results.json, table.json}`, notes in the same folder (stage 23; decodes his labelled clips with stock and F-1) |
| Script · `ci.py` | C6 | `scripts/asr/ci.py` |
| T-2 and T-3 · the matcher on public speakers | C7 | |
| P-A loaded · rerun clean | D1 | |
| P-B end-to-end latency | D2 | |
| P-D cold start | D3 | |
| P-E stop phrase | D4 | |
| P-F battery | D5 | |
| F-4 on the Pi | D6 | |
| Matcher on the Pi | D7 | |
| The self-tracking gate | D8 | |
| P-G other voices in the room | D9 | |
| H5 · the identity gate | D10 | |
| Storyboard (submission, table, remote view, prototype preparation reports) | E1, E1b, E1c, E1d, E1e | |
| Failure list and demo script | E2 | |
| Prototype Expo | E3 | |
| Poster | E4 | |

---

## TORGO (public speakers)

Prerequisite, once: `python scripts/corpus/fetch_torgo.py` (add `--controls`
for the typical-speech check) then `python scripts/corpus/torgo_manifest.py --export`
([training.md](training.md#public-corpora-torgo)). Read
`data/public/torgo/manifests/summary.md` first: it says how many usable
utterances each speaker has and how they fell into train / dev / test.

### T-1 · Zero-shot sentence WER on the severe speakers

```bash
python scripts/asr/evaluate.py \
  --manifest data/public/torgo/manifests/severe-sentences.jsonl \
  --audio-root data/public/torgo/TORGO --group-by speaker \
  --out reports/torgo-severe-zero-shot
```

Then the same on `all-sentences.jsonl` with `--group-by severity` into
`reports/torgo-all-zero-shot`. About 290 severe clips; a few minutes on a
laptop, longer on the Pi (run it there too if time allows, it is the same
model and the same code, and the report records the machine).

| speaker | severity | n clips | WER | CER |
|---|---|---:|---:|---:|
| F01 | severe | 20 | 57.61% | - |
| M01 | severe | 90 | 56.46% | - |
| M02 | severe | 93 | 45.99% | - |
| M04 | severe | 89 | 76.44% | - |
| **all severe** | | 292 | **59.59%** | 41.90% |
| mild (F04, M03) | | 198 | 3.99% | - |
| moderate (F03) | | 140 | 13.00% | - |
| moderate-severe (M05) | | 125 | 32.96% | - |

Severe-speakers run `reports/torgo-severe-zero-shot`, all-speakers run `reports/torgo-all-zero-shot`,
20 Sep 2026, stock Parakeet-TDT-0.6B-v2 int8, greedy, 4 threads.
Per-group CER is not produced, `by_group` in `results.json` carries WER only,
so the CER column is corpus-level per run: severe 41.90%, all dysarthric 21.69%.

Read `results.json` → `by_group`; `report.html` has the worst utterances and
the phoneme confusions, pick two for the storyboard's "what goes wrong"
scene. Context for the reader, not a comparison row: a 2024 study that removed
TORGO's prompt overlap reports that state-of-the-art recognisers produce
"extremely high" WER on severe speakers (Hui, Zhang & Mohan, arXiv 2411.00980).

### F-1 · Multi-speaker fine-tuning, one speaker held out (H2)

Design, traps and recipe: [research/finetune-torgo.md](research/finetune-torgo.md).
Per speaker, on the training machine (`export/loso-<SPK>/`: train and dev
are the other dysarthric speakers, test is this speaker's test prompts):

```bash
python scripts/asr/train.py --export data/public/torgo/export/loso-M01 \
  --out models/parakeet-ft-loso-M01 --freeze-encoder-layers 16 --batch 2 --grad-accum 8     # 8 GB card; 5 on a 16 GB card
python scripts/asr/export.py --nemo models/parakeet-ft-loso-M01/parakeet-ft.nemo
python scripts/asr/check.py --model-dir models/parakeet-ft-loso-M01/onnx-int8
python scripts/asr/evaluate.py --manifest data/public/torgo/export/loso-M01/test.jsonl \
  --audio-root data/public/torgo/export/loso-M01 --group-by kind \
  --out reports/finetune-torgo/loso-M01-stock                        # stock int8, same clips
python scripts/asr/evaluate.py --manifest data/public/torgo/export/loso-M01/test.jsonl \
  --audio-root data/public/torgo/export/loso-M01 --group-by kind \
  --model-dir models/parakeet-ft-loso-M01/onnx-int8 --out reports/finetune-torgo/loso-M01
```

Write the prediction in `notes.md` **before** the first `train.py` run.

| held-out speaker | severity | n test (sent / word) | stock WER | F-1 WER | Δ points | 95% CI |
|---|---|---:|---:|---:|---:|---:|
| F01 | severe | | | | | |
| M01 | severe | | | | | |
| M02 | severe | | | | | |
| M04 | severe | | | | | |
| M05 / F03 / F04 / M03 | | | | | | |

`results.json` → `wer` and `by_group` (`torgo-sentence`, `torgo-word`). The
CI comes from the per-utterance error counts (a bootstrap over utterances;
say which). Both rows are the **int8 export**, the thing that runs on the Pi.
`ci.py` keeps each comparison under `ci/<comparison_id>.json` and a latest copy
at `ci.json`. Record the baseline and sampling unit with the interval. For
repeated prompts, `--cluster-by text` keeps their takes together; for recording
visits, use `--cluster-by speaker session`. These intervals describe fitted
models on the observed data, not seed variation or performance on a new day.

### F-2 · Single-speaker fine-tuning, later session held out (H3)

Same commands with `export/<SPK>/` plus `--batch 2 --grad-accum 8
--warmup-steps 10` (a single speaker gives about 10 optimizer steps per epoch;
the full command block for the first F-2 run is in `TODO.md`). Train and dev come from the earlier
sessions, test from the last session, still prompt-disjoint -
`torgo_manifest.py` prints which sessions went where). F01 has one session
and gets the prompt-disjoint split only: context, outside F-2 and H3.

| speaker | train sessions → test session | n test | stock | F-1 | F-2 |
|---|---|---:|---:|---:|---:|
| M01 | | | | | |
| M02 | | | | | |
| M04 | | | | | |
| F01 (context only) | one session (prompt-disjoint only) | | | | |

### F-3 · Data-quantity curve (H3)

`python scripts/corpus/torgo_manifest.py --export --curve M01` writes
`export/M01-min1|5|15/` with the training set cut to the first 1 / 5 / 15
minutes of audio (same dev and test as F-2). Train each; the "all" row is
F-2 itself.

| minutes of training audio | n train | F-2 WER |
|---|---:|---:|
| 1 | | |
| 5 | | |
| 15 | | |
| all | | |

M02's run (2 Oct, stage F of the retired `scripts/asr/big_runs.sh`, now `python lab.py run F`) uses 1, 3, 5 and 7 minutes and the whole 9.0-minute set, each with equal epochs and equal updates and three seeds, scored as wrong words on its 95 test words; see `reports/f3-m02-data-quantity/notes.md`.

### F-4 · Export and the Pi (H4)

The desktop export check compares NeMo output with int8 on the same clips.
Record any decoder/export difference; NeMo and sherpa-onnx already give
different stock scores, so their equality is not the definition of F-4.

**F-4 is the exported int8 model on the Pi, scored on the same F-2 test
manifest as desktop int8** (proposal §5.2). Record both WERs, sample sizes,
decoder/runtime settings, and provenance. They should match within rounding.
The speed and memory benchmark then belongs to P-A:

```bash
WILLSPEAK_PARAKEET_DIR=models/parakeet-ft-M01/onnx-int8 \
python scripts/asr/bench.py --threads 4 --durations 3 --out reports/bench-pi5-fine-tuned
```

| model | 3 s clip decode s | RTF | peak RSS MB |
|---|---:|---:|---:|
| stock int8 | 0.365 | 0.122 | ~1275 |
| fine-tuned int8 | | | |

The F01 smoke run checked the desktop chain; its Pi step was skipped.
It does not complete F-4.

### Typical-speech check (the forgetting pitfall)

```bash
python scripts/asr/evaluate.py --manifest data/public/torgo/export/controls/test.jsonl \
  --audio-root data/public/torgo/export/controls --group-by speaker \
  --model-dir models/parakeet-ft-loso-M01/onnx-int8 --out reports/finetune-torgo/loso-M01-controls
```

| model | control-speaker WER | Δ vs stock |
|---|---:|---:|
| stock | | -|
| F-1 (each) | | |
| F-2 (each) | | |

A model that gains on its speaker and loses more than a few points here is
not the model that goes on the chair; say so on the poster if it happens.

### T-2 · What the matcher's projection buys (the R1 table)

```bash
python scripts/matcher/embed_manifest.py --manifest data/public/torgo/manifests/words.csv --out data/emb/torgo-words.npz
python scripts/matcher/train_projection.py --emb data/emb/torgo-words.npz --eval
```

The first command embeds the single-word clips with WavLM (≈ 3 clips/s on
a laptop CPU; faster on the 4060). The second holds out one speaker at a
time, fits the projection on the others, and prints cross-session top-1 per
speaker and macro-averaged, for raw / LDA / LDA+WCCN / LDA+NAP / the
per-session-mean control.

| variant | cross-session top-1 (macro) | session probe |
|---|---:|---:|
| raw cosine | | |
| LDA (ships) | | |
| LDA + WCCN | | |
| LDA + complete session mean (offline control) | | |

The sentence for the poster is the raw → LDA gap, in points, and the fact
that WCCN/NAP do not add to it. If the direction differs from the
CPU pilot table (round-3.md §4.1), that is a finding, write down why (a
different subset? the microphone folded into the session id?).

### T-3 · Closed-vocabulary and cross-session accuracy per speaker

```bash
python scripts/matcher/evaluate.py --emb data/emb/torgo-words.npz --k 1
```

Per speaker (projection fitted on the other seven): one exemplar per word,
every other word a distractor, top-1 and top-3; and enrol-one-session,
test-the-others. Macro average and the severe group.
The older centered rows use complete test-session means. Record the separate
causal-mean result printed by the evaluator before describing live behavior:
it uses all enrollment takes and only current/earlier queries for each test
session. It still tests a closed vocabulary, without ASR voting or rejection.

| space | top-1 (k=1) | top-3 | cross-session | severe top-1 | severe cross |
|---|---:|---:|---:|---:|---:|
| raw cosine | | | | | |
| session-normalised | | | | | |
| projected (full-session centering) | | | | | |

On the poster: "speakers held out; the vocabulary is shared across TORGO, so this
is transfer across speakers, not across words."

### T-4 · Speaker gate on public speakers (later extension; H5 undecided)

The identity gate of [design.md](design.md#other-voices-in-the-room):
enrol each TORGO speaker on one session, score that speaker's other
sessions (should pass) and the other seven speakers (should fail) with the
speaker-embedding model chosen for the Pi. Command and script to come with
the plumbing if the decision is yes.

| speaker | enrol session | own-session pass rate | other-speaker reject rate | threshold |
|---|---|---:|---:|---:|
| (one row per speaker, severe group marked) | | | | |

The poster sentence is the two rates at one threshold, and the sentence about
why a false reject costs more than a false accept for a person who cannot
press anything.

### Lab tests

**Everything still unfinished, one command (21 October).** On the lab computer, from the checkout's `app/`:

```bash
./willspeak.sh lab-remaining              # add --quick to skip the GPU smoke runs, --until 07:00 to stop starting new work
```

It does four things in order and stops before step 4 if the check finds a problem (`--force` starts anyway):

1. **Code.** `git pull --ff-only`, unless a tracked file has local changes.
2. **Doctor.** `lab.py doctor --fix --allow-download --controls` repairs environments, exports and manifests, and
   downloads the TORGO control speakers (about 2.5 GB) that lab tests 16 to 18 need. It never deletes.
3. **Check.** `lab.py check remaining`: the machine, a dry plan of every unfinished experiment with missing baselines
   counted as buildable, the data files, a two-step GPU smoke run per distinct recipe, and the lab unit tests.
4. **Queue.** `lab.py queue remaining --detach`. It decides on the lab computer what is not finished, so nothing done
   there is repeated, and works in this order: 5 24 28 30 19 20 23 25 27 26 B 16 17 18 29, then the sweeps L T E S C
   (L, T, E and S have every run decoded and only rebuild their summaries; C trains its 18 missing whole-block runs),
   then the long studies `acoustic-only` and `speaker-replay`, then anything else not done. For this shortcut it
   rebuilds missing baselines, lets the doctor download data, waits for a lab run that is already going (for example
   `1020-new`) instead of stopping, moves half-finished outputs aside instead of refusing, and pushes the
   `lab-results-*` branches to GitHub after every experiment. A setting you choose yourself (`--set push_results=0`,
   `python lab.py set wait_busy 0`) wins.

Watch it with `tail -f logs/lab-queue-<stamp>.out` (the path is printed); `kill <pid>` stops it and running the same
command again resumes, skipping finished work. If the push fails (no GitHub login on the lab computer), the results stay
on the local `lab-results-*` branches and the end of the log prints the push command. Fixed for this on 21 October:
stage B no longer counts as done from lab test 4's stock decodes and no longer refuses to start when they exist; lab
test 19 counts as done only when all 18 recipe scores exist and rebuilds an old table to score the missing per-channel
exports; a sweep is done when every run is decoded and its summary exists; a long study finished under any stamp counts
as done; a sweep whose runs are all decoded no longer retrains its pretrained phase-1 models. No predictions are
recorded for these runs; the notes say so.

**Pending training queue (12 October).** `lab.py` can run the remaining
experiments from the pending-run inventory together, excluding base-model
stage B and the already completed numbered lab tests. From the lab checkout's
`app/`, after transferring the updated code:

```bash
.venv/bin/python lab.py --set speakers="M01 M02 M04" --set seeds= \
  --set xflags=--offload-optimizer --set long_flags=--offload-optimizer \
  queue pending --stamp 1012-pending --detach
```

This runs L/T/E/S/C, A/N/F/W, then `long-adapter`, `acoustic-only`, and
`speaker-replay`, sequentially. The long experiments retain their documented
recipes and fixed three-speaker, three-seed selection procedure. `long_flags`
accepts resource settings (batch, accumulation, optimizer offload, disk limit
and keeping models); `xflags` applies to the older stages only.
The queue uses its existing disk checks, cleanup and result-branch snapshots;
it continues independent stages after a failure and reports failures at the end.
Watch `logs/lab-queue-1012-pending.out`. Replace `--detach` with `--dry` to
print the commands without training. Reuse the same stamp and settings to
reuse finished work; long-run resume verifies code, environments and inputs
and preserves incompatible study outputs in `-previous` folders before starting
a fresh study with the current inputs. The pending queue creates missing stock,
whole-block, F-1 and adapter baselines in `lab-baselines/`, including adapter
seeds 17/18/19 for A/N/F. It reuses completed prerequisite reports and preserves
incomplete prerequisite model folders before retrying. A long experiment
counts as completed only when both its test summary and battery are present.
With about 30 GB free, default cleanup keeps one long-study training model at
a time: search saves dev scores and releases weights, then the locked recipe
and control are trained again for the final three-seed comparison. Final dev
scores are stored separately as `confirmation-dev`; the search scores remain
the selection evidence. Each final model is tested and decoded on the silence
battery before its weights are released. This adds seed-17 retraining time.
Failed trials keep their logs and records, release weights, and allow remaining
trials to run. An incomplete search never locks a winner or claims completion.
`--keep-all-models` disables this storage limit and requires more disk space.

No new run or prediction is implied by preparing this command.

Fifteen tests set up on 2 October 2026 (the four below them, 16 to 19, were added on 14 October and have not been run). They go beyond the proposal's
H1–H4 and ask what could bring the pooled severe-speaker score toward the 35%
prototype bar, or show that nothing here can. Each is a numbered stage of `app/lab.py`
and has its own notes folder with the question, what each outcome would mean,
and an empty personal-prediction field. The times are estimates, not
measurements.

| Test | Stage | What it asks | Cost | Notes folder |
|---|---|---|---|---|
| 1 · weight interpolation | 1 | Does a quarter, half or three quarters of the whole-block change keep the gain? | decode, about 20 min GPU | `lab-01-weight-interpolation` |
| 2 · weight merging | 2 | Do the other speakers' weight changes help a speaker who gave no data? Does averaging all three cost his own gain? | decode, about 20 min GPU | `lab-02-weight-merging` |
| 3 · hypothesis bounds | 3 | If an oracle picks the best of the systems that exist on each clip, or they vote, how low can the pooled count go? | seconds, CPU | `lab-03-hypothesis-bounds` |
| 4 · stock model screen | 4 | Do other speech models start lower than Parakeet-TDT-0.6B-v2 on severe speech before any training? | decode, about 1 h GPU | `lab-04-stock-model-screen` |
| 5 · encoder layer screen | 5 | Which encoder layer, of which model, matches his words best across days? | about 1 h, GPU then CPU | `lab-05-layer-screen` |
| 6 · reference score | 6 | Are the wrong answers a modelling failure (true words unlikely) or a search failure (true words more likely than the answer)? How likely is silence? | about 15 min GPU | `lab-06-reference-score` |
| 7 · stock against fine-tuned | 7 | Does a model's stock WER order predict its fine-tuned WER order? | 8 training runs, about 4 h | `lab-07-stock-vs-fine-tuned` |
| 8 · feed-forward only | 8, 8b | Is the whole-block gain in the two feed-forward modules? If so, which one? | 3 + 6 runs, about 1.5 h + 3 h | `lab-08-feed-forward-only` |
| 9 · depth sweep | 9 | Is 8 blocks from the bottom the right amount and place? | 15 runs, about 7 h | `lab-09-depth-sweep` |
| 10 · silence margin | 10 | Do a loss against empty answers and synthetic non-speech clips stop the model staying silent on speech or writing words for noise? | 9 runs, about 5 h | `lab-10-silence-margin` |
| 11 · severe pre-training | 11 | Do the other severe speakers make a better starting point than all dysarthric speakers? | 6 runs, about 4 h | `lab-11-severe-pretraining` |
| 12 · pseudo-label loop | 12 | Can the stock model improve on his recordings without anyone reading what he said, using only a phrase list? | 9 runs, about 5 h | `lab-12-pseudo-label-loop` |
| 13 · silence false inserts | 13 | How many words does each existing model write for clips with no speech? | decode, about 15 min GPU | `lab-13-silence-false-inserts` |
| 14 · route scoring | 14 | What does the pooled score look like if single words go to the matcher and sentences go to Parakeet? | about 5 min, CPU | `lab-14-route-scoring` |
| 15 · int8 screen | 15 | Which checkpoints and exported files lose depthwise channels to int8? | about 5 min, CPU | `lab-15-int8-screen` |
| 16 · synthetic speech realism | 16 | How hard is healthy speech, made to sound dysarthric by signal processing, for the stock model, and at what severity is it as hard as each real speaker? | decode, about 1 h GPU | `lab-16-synthetic-realism` |
| 17 · training on synthetic speech | 17 | Does that synthetic speech lower wrong words when mixed into his data, used as pre-training, or added to the other severe speakers' data? | 9 runs, about 8 h | `lab-17-synthetic-training` |
| 18 · best system hardened | 18 | Do silence protection and synthetic speech, alone or together, change the adapter on F-1 (wrong words and false inserts)? | 9 runs and battery decodes, about 4 h | `lab-18-best-system-hardened` |
| 19 · int8 cost | 19 | How many words does the device's int8 file cost compared with NeMo fp32, for three export recipes? | 18 exports and decodes, about 3 h | `lab-19-int8-cost` |
| 20 · grouped cross-validation | 20 | Which adapter recipe has the fewest wrong words on prompts the model has not trained on, using every labelled clip instead of the 112-word dev set? | 48 trainings and decodes, about 6 h | `lab-20-grouped-cv` |
| 21 · text-to-speech | 21 | Does degraded synthesiser speech, or barely degraded synthesiser speech, help the adapter on F-1? | 6 runs and battery decodes, about 3 h | `lab-21-tts-synthetic` |
| 22 · test-time adaptation | 22 | Does the system improve on later clips by training on its own earlier answers, with no transcripts? | 9 trainings and decodes, about 3 h | `lab-22-test-time-adaptation` |
| 23 · hypothesis selector | 23 | How many wrong words can a selector that uses only what a device knows get out of the systems' existing answers? | 6 decodes and CPU, about 30 min | `lab-23-hypothesis-selector` |
| 24 · acoustic phrase scoring | 24 | Which listed phrase does the audio support, with thresholds chosen on the other speakers, and how much does list coverage matter? | 9 scoring passes, CPU and GPU, about 30 min with the baselines in place | `lab-24-phrase-scoring` |
| 25 · test-time adaptation on the pretrained adapter | 25 | Do severe pretraining (178 of 425) and self-training during use (179) add when combined, at three seeds? | 27 trainings and decodes, about 3 h | `lab-25-tta-on-pretrained` |
| 26 · pretraining recipe | 26 | Does how the severe-speaker pretrained model is trained (bottom 8 blocks, rate, schedule, all dysarthric first) change what the adapter reaches? | 12 phase-1 and 36 adapter trainings, about 3 h | `lab-26-pretraining-recipe` |
| 27 · test-time adaptation settings | 27 | Which of lab 22's guessed settings (chunks, order, gate, cap) matter? | 66 adapter trainings, about 2 h | `lab-27-tta-settings` |
| 28 · deployable export | 28 | What does the pretrained adapter cost as the device's int8 file, by two recipes? | 3 trainings, 6 exports, 6 CPU decodes, about 30 min | `lab-28-deployable-export` |
| 29 · hard core | 29 | Which clips and words does every saved system get wrong, and of what kind? | CPU, seconds | `lab-29-hard-core` |
| 30 · phrase scoring on the pretrained adapter | 30 | The device reading of the bar for the best raw system | 3 trainings and 3 scoring passes, about 30 min | `lab-30-phrase-scoring-pretrained` |
| 31 · adapter on the acoustic-only model | 31 | Does stage P's adapter, trained on top of each speaker's own acoustic-only model (191.7), add anything? | 3 full-encoder and 9 adapter trainings, about 2 h | `lab-31-adapter-on-acoustic-only` |
| 32 · acoustic recipe as severe pretraining | 32 | Does the acoustic recipe (all 24 blocks, head frozen, 1e-5) beat stage P's pretraining under the same adapter (178.0)? | 3 full-encoder and 9 adapter trainings, about 2.5 h | `lab-32-acoustic-severe-pretraining` |
| 33 · pooled acoustic model, adapter positions | 33 | One acoustic model on every dysarthric speaker, then a per-speaker adapter after blocks 4, 8, 12, 16 or 24: which position, and does it beat stage P? | 1 full-encoder and 45 adapter trainings, about 4.5 h | `lab-33-pooled-acoustic-adapters` |
| 34 · adapter position on stage P's model | 34 | Do adapters after blocks 12, 16 or 24 on stage P's pretrained model keep its accuracy and cut false words on the battery, as they did on lab 33's base? | 36 adapter trainings (plus up to 3 pretraining rebuilds), about 2.5 h | `lab-34-pretrained-adapter-positions` |
| 35 · silence preference, fixed | 35 | Does the model prefer saying nothing over the true words on severe speech but not on typical speech, with the true words in the model's own form, and does stage P's adapter remove that? | 14 scoring passes (plus rebuilds if weights were cleaned), about 45 min to 2 h | `lab-35-silence-preference` |
| 36 · language-model rescoring | 36 | Parakeet decodes each clip several ways; does a small text model, added to the recogniser's own score, pick the right whole sentence more often, on stock and on stage P's adapter, and which Pi-sized model does it? | 3 pretrained models and 15 adapters (reused if their weights are here), about 1,300 clip pools, 22 language models (27 with the `large` group); an estimated 5 to 7 h on the 4060, 7 to 10 h on the lab computer with `large`, about 20 min for stock with the named model alone | `lab-36-lm-rescoring` |
| 37 · kind routing | 37 | Does a block-8 adapter trained only on a speaker's single-word clips, used where the device's own first decode has one word, beat stage P's adapter on single words and on the pooled 425 words without hurting sentences? Same question for a sentence-only adapter, a duration detector and the true kind | 27 adapter trainings (plus up to 3 pretraining rebuilds), about 54 offline routed decodes (72 with the whole-block reference), about 1.5 to 2 h | `lab-37-kind-routing` |
| 38 · self-training on language-model-chosen pseudo-labels | 38 | Does lab 25's loop on stage P's adapter get further when a sentence's pseudo-label is the answer the recogniser and Qwen2.5-0.5B choose together, with or without a margin rule, than when it is the plain decode? | 81 adapter trainings (3 arms x 3 speakers x 3 seeds x 3 rounds), 63 small pools and their language-model scores; an estimated 4 h with the pretrained models on hand, 6 h on a fresh checkout (scaled from lab 25's logged 2.1 to 2.7 minutes a round) | `lab-38-lm-pseudo-labels` |
| 39 · wider candidate pools | 39 | Lab 36's pools hold nine decodes of one model. Does adding the greedy answers of other models of the same base (stage P's seeds 18 and 19, the block-12 and block-16 adapters) raise the number of clips whose pool holds the right sentence, and does the same chooser find the new ones, for what cost per clip? | up to 60 adapter trainings (plus 15 primaries and up to 3 pretraining rebuilds), 30 pools, one language model; an estimated 4 to 5 h, about 1 h for the test clips alone | `lab-39-wider-pools` |

**Lab 36 (set up and run 6 November; results below).** Three scripts do the work: `scripts/asr/lm_pool.py` decodes each
clip several ways in NeMo (`lm_views`: the greedy decode, every answer of two beam searches, the clip heard 10, 20 and 30% faster, pitch
moved down 10%, down 5% and up 5%), merges answers that differ only in capitals or punctuation, and scores each against the audio with the model's own
likelihood summed over alignments; `scripts/asr/lm_score.py` scores the same answers as English with each language model
(`list` shows the models and their groups, `fetch` downloads, `check` loads a model and shows that it reads ordinary sentences, `score` and
`complete` add to their files with `--extend`, `release` deletes a model file); `scripts/asr/lm_report.py`
adds the two scores with a weight and a per-word bonus chosen on the other speakers, and prints whole sentences right, the
paired interval of the change against the first decode, clips gained and lost, sentence WER, the effect on single words, the
ceiling (the right answer is somewhere in the pool) and each model's file size, memory and speed on that computer.
`lm_models` takes tags or groups: `pi` (the default: 22 models whose 4-bit or 8-bit file is under 1.7 GB, up to Spark-X2.6-1.7B of October 2026),
`large` (five models of 2 to 3 GB) and `ceiling` (Qwen3.8-27B, not for the Pi). Each model is tried with its start token, its end token and a new
line before the first word on three fixed sentences, the simplest one that reads well is kept, and a model the runtime misreads is refused.
The test builds `~/venvs/willspeak-lm` when it is missing (`./willspeak.sh lm-setup` does the same ahead of time), handles one language model at a
time, and on a computer with under 100 GB free deletes each model file after its scores are saved.
On Yack: `python lab.py --set lm_systems=stock --set lm_models=qwen25-05b-q8 run 36` is a first look with no training, then
`python lab.py run 36 --rerun`. On the lab computer: `./willspeak.sh lab-lm` (pull, environments, a download check, `lab.py check`, then the queue in the background with
`lm_models="pi large"` and `push_results=on`; `--quick` for the stock-only look, `--dry` for the plan).
The measure is whole sentences right; the decision rules and the model named in advance are in the note. Memory and speed on the Pi come
from running `lm_score.py score` there on the same pools ([pi5.md](pi5.md#10-language-model-rescoring-on-the-pi)).

**Lab 36 results (queue `1106-1346`, 27 models; read 9 November; full tables in
[the notes](../app/reports/lab-36-lm-rescoring/notes.md)).** On stage P's recipe, 120 held-out sentences (513 clips with
the single words), the first decode gets 29 right (24.2%, sentence WER 27.1%) and the right sentence is in the pool for 45 (37.5%).
The language model lifts whole sentences to 38 to 42 (31.7 to 35.0%) with 0 to 3 lost, intervals above zero for 24 of 27 models, and
sentence WER to about 23%; audio-score re-ranking alone gains none. On stock (292 sentences, pool ceiling 16.4% against 9.6%)
the same models gain 0 to 3 points, so the gain grows with fine-tuning. The model named in advance (`qwen25-05b-q8`) is above zero on
stage P and not on stock; two models pass the stock-and-stage-P screen (`qwen25-05b-q4`, `spark26-17b-q4`). The plan's harm rule fails: sentence
weights applied to single-word clips lose 10 to 31 of 393 single words where sentences gain 7 to 13. A device-visible detector (first decode
of one word) separates the kinds on 384 of 393 single words and 118 of 120 sentences; with single words routed around the
sentence weights, all 1,304 words go from 427 wrong (32.7%) to 376 to 396 (28.8 to 30.4%), and on the 425-word test count from 176 to
164 to 171 (inside the 19-word rule; the bar is 148). Single words are still where the gap is: the pool holds the right word for 302 of 393
clips and no feature found yet (language model, word frequency, decode votes) picks it better than +4 to +10 clips. Nothing was measured on the Pi.

**Lab 37 (set up 9 November; not run).** On the common lab set the whole gap to the 35% bar is single words (stage P: 85.7 of 303 sentence words, 92.3 of 122 single-word words). The test trains a block-8 adapter at 6e-4 on stage P's base on a speaker's single-word clips only (`train.py --kinds word`, which keeps train and dev clips of one kind and so needs no new export), a second on sentence clips only, and a control on every clip retrained on the same base, three seeds each. `scripts/asr/kind_route.py` then builds routed decodes from the saved ones: a clip goes to the word adapter if the control's first decode has exactly one word (lab 36's pools: 384 of 393 single-word clips and 2 of 120 sentence clips, about 97%), or if it is short (a cut chosen on the speaker's own train and dev clips), or if it truly is a single word (the oracle, not a device system); the rest go to the control or the sentence adapter. Scored by `pool_table.py` (all clips with the sentences-only table, and `--kind word`) and `scripts/asr/kind_route_report.py` (three-seed means by kind, the rules, the detector's mistakes). The decision rules, the lab K result that points against the idea (a single-word-only adapter on the stock base made single words worse, 101 against 95) and the arithmetic of the bar are in the note. `python lab.py queue routing --dry` shows the plan.

**Lab 38 (set up 9 November; not run).** Lab 22's chunked self-training loop, on stage P's adapter as lab 25 ran it, with the pseudo-labels of sentences
chosen by language model. After each chunk is decoded, `scripts/asr/lm_pool.py --no-reference` pools the chunk's clips with no transcript (it reads id,
audio and speaker only), `lm_score.py score` scores the answers with the one model `lm_label_model` (default `qwen25-05b-q4`; one shared score file,
so a sentence is scored once), and `scripts/asr/lm_pick.py pick` chooses with the fixed weights `lm_weight` 0.7 and `lm_bonus` 3 and records the
margin over the runner-up in nats. A clip whose plain decode is one word keeps it. `lm_pick.py filter` applies the arm's rule over the chunks so far
(`lm_label_arms`: `all`, `top` = the `lm_top_share` of sentence clips with the widest margin, `margin` = at least `lm_min_margin` nats), `tta.py build
--withhold` keeps the dropped clips out, and lab 22's agreement gate, dev-prompt check and cap apply as before. Lab 25's plain-label runs are the control
and are reused. `lm_pick.py audit` reads the true text once, after all training, and reports the word error of each arm's pseudo-labels (before and
after the gate, split into sentences and single words, with the plain decodes of the same clips and the clips the filter dropped).
`python lab.py queue selftrain-lm --detach` runs it; `--set lm_label_arms=all` runs one arm. The decision rules, including harm and the 19-word
rule, are in the note.

**Lab 39 (set up 9 November; not run).** Builds on lab 36's pools and chooser without changing them. `scripts/asr/lm_pool.py` gained `--extra-model NAME=BASE[,ADAPTER]` (repeatable; the
extra recogniser's greedy answer, and with `--extra-views greedy,beam4` its beam answers, join the pool tagged with NAME in each candidate's `views`), `--base-pool` (start from an earlier pool of
the same primary model instead of decoding its nine views again; refused unless the model files match by SHA-256) and `--subset-of ... --keep-extras ...` (cut a narrower arm from a wider pool with no
decoding). Every extra answer is scored against the audio by the primary model, so scores stay comparable; the 15-nat rule is held against the best of the primary model's own answers, so every arm's pool
contains the whole baseline pool and the ceiling can only rise or stay. `scripts/asr/lm_report.py` gained `--compare NEW=OLD` (paired change in sentences right, WER and the pool ceiling between two
systems on the same clips, with a clip-resampled and a prompt-resampled interval), decodes and encoder passes per clip, and counts of gained sentences and ceiling clips that came only from an extra model's
answers. `lab.py` stage 39 (`wide_extras` = `seeds`, `positions`, `both`; `wide_systems`; `wide_extra_views`; `wide_lm_models`, default `qwen25-05b-q4`; queue `widepool`) trains the missing primaries and
extras (the extras for a held-out fold on that fold's cross-validation export, the primary's own; checked by name and by manifest, and with a recipe that never lets dev clips choose), reuses lab 36's
pools when the primary adapter on disk still matches them and rebuilds the baseline pool under `reports/lab-39-wider-pools/` when it does not, and writes one report. The decision rules (an arm counts only if its
pool ceiling and its chooser both rise against the baseline arm with the interval above zero, with the cost per clip stated) are in the note, set before any run. First look:
`python lab.py --set wide_systems=prototype run 39`; everything: `python lab.py queue widepool --detach`.

**Labs 31 to 33 (set up 28 October; run 28-29 October in queue `1029-acoustic2`, no predictions written first).** `python lab.py queue acoustic --detach` runs them in order; each note holds
the reference numbers and the decision rule. Lab 33 trains on `export/pooled-M01-M02-M04`,
which `torgo_manifest.py --pooled` builds: every dysarthric speaker's training-split prompts, the three targets' clips from
their training sessions only, their own dev clips as dev, and no test clips; it refuses to write the export if any clip
shares a prompt with a test clip and records the composition in `pooled.json`. The pretraining length is a setting
(`acoustic_severe_epochs`, `acoustic_pooled_epochs`, 30 each; the one-speaker study used 100 epochs on 5 to 9 minutes of
audio), and so are the adapter positions (`pooled_positions`). For failures and disk problems on the lab computer, every
stage of `lab.py` now logs a storage snapshot at its start and end and after a failure (free disk, temp folder and RAM,
folder sizes, the largest model folders and whether they still hold `ckpt/`), also appended as JSON lines to
`logs/lab-<stamp>-storage.jsonl`; labs 31 to 33 add one after each full-encoder training, delete that run's checkpoints
once its averaged `.nemo` reads back intact, keep a base model until every result that needs it is saved, check for
14.5 GB free before each full-encoder training, and keep a per-run record (`runs.json`: status, minutes, wrong words, free
disk, the base checkpoint each adapter used, and every attempt's code hashes and settings).

**Results of labs 31 to 33 (`1029-acoustic2`, run 28 October 20:19 to 29 October 02:06, read from the logs on 29 October).** Wrong words of 425, three-seed means; stock 323, bar 148, stage P 178.0.

- **31 · adapter on the acoustic-only model:** 196.3 (189, 194, 206) against 191.7; within the rule, no gain. Sentences 112.0 of 303 against 101.0; the rebuilt base repeated the study's seed 17 to the word (192: 84, 36, 72). The adapter brought back breath inserts (4, 8, 1 of 20).
- **32 · acoustic recipe as severe pretraining:** 228.3 (231, 220, 234) against stage P's 178.0; a loss on all three speakers. The pretrained model alone, never hearing the speaker, scored 234, lower than F-1 alone (248) and lab 11's whole-block pretraining (275).
- **33 · pooled acoustic model, adapter positions:** shared base alone 222; block 4 218.7, 8 216.0, 12 210.0, 16 215.3, 24 223.3. Every position a loss against stage P; no position beats another by the rule. Adapters after block 12, 16 or 24 wrote words on 0 to 8 of 200 battery clips (base 56, stock 70).

**Results of lab 34 and the speaker-replay study (`1029-night`, run 29 October 14:24 to 30 October 10:08, read from the logs on 30 October; no predictions were written first).** Wrong words of 425, three-seed means.

- **34 · adapter position on stage P's model:** block 8 174.3 (176, 172, 175) on freshly rebuilt pretrained models, within noise of stage P's 178.0; block 12 224.7, 16 234.7, 24 250.7, each a loss on all three speakers. The pretrained model alone scored 244. Battery clips with words at seed 17 (of 600): base 98, block 8 98, 12 82, 16 57, 24 133; almost all on the 60 edge clips per model. No position halves block 8's false words, so block 8 stays.
- **Speaker replay:** dev at seed 17 (of 112): control 36, `personal75` 34, `personal50` 34, `severe50` 32, so the rule chose `severe50`. Test: control 186.7 (183, 189, 188; sentences 98.3 of 303), `severe50` 202.0 (202, 199, 205), worse at every seed and on every speaker by 15.3 words; under the rule, but the pooled interval excludes zero at seeds 17 and 19. Replay rejected; the dev lead reversed on test, as in the long-adapter study.

**Results of labs 25 to 30 (`1020-new`, run 20-21 October, read from the logs on 21 October; no predictions were written first).** Wrong words of 425 unless stated; stock 323, bar 148. The tables and reports the queue wrote stayed on the lab computer, so the numbers come from the printed tables in `logs/`.

- **25 · loop on the pretrained adapter:** three-seed mean 178.0 to 172.3 (41.9% to 40.5%; seeds 183 to 179, 176 to 173, 175 to 165). Under the 4.5-point rule; M02 flat. The best looped run is 165 (38.8%).
- **26 · pretraining recipe:** no arm beats stage P's control (178.0): bottom 8 blocks 200.3, rate 4e-5 181.7, longer 180.3, dysarthric first 185.3. The pretraining line is closed.
- **27 · loop settings (adapter on F-1, seed 17):** no loop 197, lab 22's settings 179, half cap 178, gate 0.3 / none / 0.7 at 187 / 191 / 194, eight chunks 197, recorded order 204. The gain does not survive a change of order or chunk count.
- **28 · int8 file (pretrained adapter, seed 17):** NeMo fp32 176, int8 default 180, int8 matmul 177. Both inside the 10-word rule; ship the matmul form. The default export has 5 mostly-zero weight tensors (depthwise convolutions in blocks 0, 1, 2 and 22, a matrix in block 23), matmul has 1. Not yet measured on the Pi.
- **29 · hard core:** over eight saved systems the per-clip oracle is 124 (29.2%); 56 clips are wrong in every system and all 56 are single words; no sentence clip is wrong in every system.
- **30 · phrase scoring on the pretrained adapter:** raw 41.4%; with the full list 30.8% (text) and 31.5% (acoustic); 35.7% at 80% coverage and 41.9% at 50%. The adapter on F-1 (lab 24) is 26.4% and 31.8% at the same points. On sentences both end near 17 to 19% at 80% coverage; the difference is in single words, which the list does not help for the pretrained adapter.

**Scoring.** Pooled means the wrong words added over the three severe test sets:
M01 169 + M02 95 + M04 161 = 425 reference words. Stock gets 323 wrong, the best
results so far are 178 (the pretrained adapter at 6e-4, three seeds, 15 Oct),
172.3 (the same with test-time adaptation, three seeds, 20 Oct; not established)
and 179 (the adapter on F-1 with test-time adaptation, one seed, 16 Oct), and
35% is 148 or fewer. A change
counts as better or worse only when at least 2 of the 3 speakers move the same
way and the pooled WER moves by more than 4.5 points, the run-to-run noise that
repeated seeds have shown. `pool_table.py` prints the verdict. "Unclear" does
not mean equal.

**Before running.** Record my prediction for each stage in its `notes.md`
before preparing the command. A discussion in chat can be copied into the
note in my words. The runner does not check for predictions.

**Running.** In WSL on Yack, from `~/willspeak/app`, after pulling the updated
repository:

```bash
git pull
python lab.py run 14 3 15
```

The stage arguments list the tests. Add `--dry` to print commands without
running them. With no stage arguments, the runner does `14 3 15 1 2 6
13 4 5`, the stages that decode or score what already exists and train nothing
(about 3 hours in all). The training stages `7 8 8b 9 10 11 12` run only when
named, and 8b only after 8 has been read. Every stage refuses to overwrite its
own outputs, lists a failed run and carries on, and retries a failed training run
once at batch 1 with 16 accumulation (the same 16 clips per update, under a name
ending `-b1`). When it finishes, paste the summary path printed by `lab.py`
(under `logs/`). `python lab.py vars` lists the current runner settings. The old
`scripts/asr/lab_runs.sh` command has been retired by the lab-tool migration.
Stage 4's models outside NeMo need the Hugging Face environment
`~/venvs/willspeak-hf` (`scripts/asr/hf_hyp.py` says how to make it); without it
they are skipped with a message.

**Disk space and moving results.** Training writes intermediate checkpoints
and large final model files. `lab.py space` reports, transfers and cleans them.
Run it from `~/willspeak/app` on the lab computer:

```bash
python lab.py space report
python lab.py space git --stage
python lab.py space git-results --commit
python lab.py space pack
python lab.py space clean --apply
python lab.py space clean --long-models --apply
python lab.py space archive-models --to /big/disk --apply --remove-after-verify
```

**Checks and repairs (added 14 October).** `python lab.py doctor` checks the Python environments, the free disk,
the TORGO audio, every export folder (manifests, WAV files that exist, are 16 kHz mono and are not silent or cut
short), the saved manifest fingerprints and the finished outputs under `reports/`. It never deletes: a damaged output
is renamed with `-corrupt`, an export with missing audio is rebuilt from the TORGO download, a decode left without its
score is scored again from the saved hypotheses. `python lab.py doctor --fix` carries out the repairs it lists;
`--controls` also checks the control speakers, `--deep` reads every WAV, `--pin` records fingerprints of the three
speakers' manifests in `.lab/manifest-pins.json` so a later change to them is caught. `python lab.py check` runs it as
part of the pre-flight, and the queue runs it before each experiment (the `auto_fix` variable turns the repair step off).
The queue also retries an experiment once (`stage_retries`) when its log shows a cause a retry can fix: out of GPU
memory, out of RAM, a full disk or a network error. A retry reuses everything already finished. While it runs, the
queue keeps `logs/lab-queue-<stamp>.status.json` up to date (done, running, failed, blocked and why, free disk).

**Freeing disk by exporting models.** Finished models are the large files. `python lab.py export-models` zips each
model folder, uploads it, checks it, and only then removes the big weight files from this machine (the decodes, scores
and `run.json` stay). Baseline models that later stages read are never freed unless asked for. Setup for Google Drive:

```bash
sudo apt install rclone
rclone config                     # n, name gdrive, type drive, accept the defaults, sign in in the browser
rclone lsd gdrive:                # lists the Drive top folder if it worked
python lab.py set export_to rclone:gdrive:willspeak-models
python lab.py export-models list              # models, sizes, which are finished and which are protected
python lab.py export-models export            # dry run
python lab.py export-models export --apply --free-after-verify
python lab.py export-models check             # compare what is stored with what the manifests say
python lab.py export-models import parakeet-ft-lab-09-b2-M01   # bring one back, checksum verified
```

A plain folder (a USB disk or a share) works the same way: `python lab.py set export_to /mnt/big/willspeak-models`.
When `export_to` is set, the queue sends finished models away by itself when free disk falls below its minimum. On a
computer with no browser, `rclone authorize "drive"` on another machine gives the token to paste during `rclone config`.
Google Drive accounts have their own storage limit and upload limit per day; a large export may need several nights.

**Grouped cross-validation, the selector, text-to-speech and test-time adaptation (lab tests 20 to 23, added 14 October).**
These four are queued after 16 to 19 (`NEW_QUEUE` in `lab.py` runs 20 before 23, because the selector learns from lab 20's
held-out answers). Their scripts work without a GPU and have their own commands:

- `scripts/asr/cv_folds.py` (`make`, `check`, `merge`, `select`): cuts a speaker's train and dev clips into folds that hold out
  whole prompts, merges each recipe's held-out decodes, and picks a recipe by pooled held-out wrong words with a paired
  bootstrap over prompts. It keeps the control unless another recipe beats it by `cv_min_gain` points with an interval below zero
  and a majority of speakers.
- `scripts/asr/hyp_selector.py`: the vote, weighted vote, medoid and a ridge selector over several systems' answers, fitted
  on out-of-fold answers and scored leave-prompt-out. `--systems-file` takes per-speaker file names.
- `scripts/asr/tts_synth.py` (`backends`, `speak`, `make`, `merge`): speech from espeak-ng or piper, 16 kHz mono, never for a
  prompt in `--exclude-from`. Install one engine: `sudo apt install espeak-ng`, or `pip install piper-tts` with `.onnx`
  voices in `tts_voices`. `tts_extra_text` takes a file of extra phrases, for example a caregiver's list.
- `scripts/asr/tta.py` (`chunks`, `manifest`, `build`, `merge`, `score`): the chunked test-time protocol. It reads id, audio path,
  duration and speaker of the test manifest and never its text. Per-utterance entropy adaptation is not built.

Settings (`python lab.py set NAME VALUE`): `cv_folds`, `cv_arms`, `cv_min_gain`, `selector_folds`, `tts_backend`, `tts_voices`,
`tts_variants`, `tts_max_clips`, `tts_extra_text`, `tta_chunks`, `tta_order`, `tta_agree_min`, `tta_max_pseudo`. Stage W (matching
single words) now uses whichever models have answers for all three severe speakers and needs only the stock model's.
All four were unit-tested on synthetic files and dry-planned; none has run on the real models or data.

**Synthetic dysarthric speech (lab tests 16 and 17).** `scripts/asr/synth_dysarthria.py` makes healthy control
recordings sound dysarthric with signal processing (slower units with the pitch kept, pauses, smeared consonants,
flattened vowels, hypernasal colouring, breathiness, tremor). `profile` and `preview` show what it does to one speaker's
pace and to one clip; `selftest` checks the processing on a toy signal; `build` writes an export folder that
`train.py` accepts, with the speaker's own data plus the synthetic clips and a refusal for any clip whose prompt is
in that speaker's dev or test set. The controls export comes from `python scripts/corpus/torgo_manifest.py --export
--only controls` (the control speakers are fetched with `python scripts/corpus/fetch_torgo.py --controls`, about 2.5 GB).
Whether the audio is close enough to real dysarthric speech is not known; lab test 16 measures how hard it is for the
stock model and lab test 17 measures whether training on it helps.

`clean` and `archive-models` print what they would do until `--apply` is given.
`clean --apply --lab-models` also deletes finished lab-test models, but only those whose decodes and
`results.json` are already inside a local `lab-results-*` branch; the weights can be retrained, so this frees
space and is not an archive. Models still needed by test 15 are kept until its `screen.json` exists.
`clean --long-models` lists completed long-study model folders; with `--apply`
it removes them only after `summary.json`, `battery.json`, and every report file
are saved in a local `lab-results-*` branch. Partial studies and unsaved reports
stay in place.

Before leaving the lab computer, `python lab.py check` checks the machine and
dry-plans the selected stages. Queue execution now repeats that prerequisite
check before each stage, records missing-input stages as blocked, and continues
with independent stages. Before another experiment starts, the queue saves a
result branch, cleans safe model files, and stops if free space is still below
its configured minimum.

`clean` only removes `ckpt/` when the run's `run.json` says complete and the final
`.nemo` is whole (a cut-off file from a full disk is detected and kept). Models the
runner reads as inputs, the prototype candidates and `models/lab-baselines/` are never
offered for removal. Lab models wait until stage 15 has run, since it scans them.
`archive-models` copies each finished, scored model with a SHA-256 check and removes
the original only with `--remove-after-verify`. Decodes, tables and logs are ignored
by Git, so they never go on the normal branch. When Git is the only way off the lab
computer, `git-results --commit` builds a side branch (`lab-results-<stamp>`) from
HEAD plus the force-added results, without touching your branch, index or files, using
your own Git identity; push it with `git push -u origin lab-results-<stamp>`. Here, run
`git fetch origin lab-results-<stamp>` and then
`python lab.py space fetch-results origin/lab-results-<stamp>` from `app/`, which writes
the files without staging them and refuses to overwrite a different one. `pack` and
`unpack` do the same through a file. Only `notes.md` and code go on the normal branch.

**What has been checked.** Unit tests on synthetic data cover every new script,
and `tests/unit/test_lab_cli.py` checks that every flag in every command the
runner would run is accepted by the script it names. These are software checks,
not recognition results. Not yet run on real models or TORGO audio: weight
blending (`nemo_hyp.py --mix`), `reference_score.py`, the margin loss inside a
training run, `hf_hyp.py`, `embed_layers.py` on NeMo, `train.py --base` with
another model, `--train-parts feed_forward*`, `--train-blocks` and
`--train-first-block`, the severe-speaker exports, the battery decodes, and the
runner itself. A first run can fail on loading, memory or file layout in ways
the tests cannot see.

**The 35% bar.** Whether it means the raw model output or what the device says
with its phrase list was asked and is not settled. Every test above scores raw
output except test 14, which scores the matcher's answers for single words. The
phrase-list result so far is in `reports/phrasebook-matching/`.

---

### Sweeps L, T, E, S, C and the 13 to 14 October queue

My lab run (stamp `1012-pending`, RTX 4000 Ada, rebuilt baselines at seed 17), NeMo decoding, M01 + M02 + M04, 425 test words, stock 323 wrong. Mean wrong words over seeds 17 to 19 unless stated; lower is better. Counts come from per-speaker WER times reference words, so each can be off by one word. The queue's own summary tables were skipped (Python 3.10 had no `datetime.UTC`; fixed 14 Oct), so these were rebuilt from the evaluate logs. The 35% bar is 148 wrong.

| Sweep | Arm | Wrong (mean) | WER |
| --- | --- | --- | --- |
| Adapter, default | 15 epochs, 3e-4 | 213.0 | 50.1% |
| C | severe pretraining, then adapter | **182.3** (184 / 182 / 181) | **42.9%** |
| E | adapter 8 / 30 epochs | 233.3 / 200.3 | 54.9% / 47.1% |
| L | adapter 1e-4 / 6e-4 / 1e-3 | 238.3 / 201.7 / 200.7 | 56.1% / 47.5% / 47.2% |
| T | adapter, stock-form targets | 206.3 | 48.5% |
| S | adapter on stock, 5 more seeds (20 to 24) | 213.2 | 50.2% |
| S | adapter on F-1, 5 more seeds | 203.6 (M02 worse: 47.6 against 42.4) | 47.9% |
| Whole-block, default | 5 epochs, 2e-5 | 304.7 | 71.7% |
| E / L | whole-block 20 epochs / 8e-5 | 256.7 / 267.0 | 60.4% / 62.8% |

Stage C's whole-block arms (including pretrained whole-block) failed: 18 runs stopped at "3.9 GB free" and the queue went on to fail every remaining run. The grid runner now frees finished weights and retries once, then stops the sweep if disk is still short, and the failure classifier recognises this message as a disk failure. Stages A, N, F: see `reports/f2-augmentation-variants/notes.md` and `reports/f3-m02-data-quantity/notes.md`.

**The 14 to 15 October queue (`python lab.py queue ideas --stamp 1015-ideas`, my run).** Stage U (average five seeds' adapters): 209 wrong against 212.4 for the mean of the five seeds, no gain. Stage R: weight decay, warmup and averaging width all within 4 words of the default 212.0; trainable batch-norm 324.7 (stock is 323). Stage K: adapter on both kinds 212.0, sentences only 246.0, single words only 280.7. Stage G (two seeds): whole-block 5 epochs at 2e-5 303.5; 20 epochs at 8e-5 216.0; 20 epochs at 1.6e-4 207.0; 40 epochs at 8e-5 **201.5** (47.4%). Stage P did not train anything: it tried to retrain a pretrained model into a folder left behind by stage C and the trainer refused it; `lab.py` now moves the empty folder aside, so P is ready to run. Details and the per-speaker tables: `reports/f2-adapter-levers/notes.md` and `reports/f2-adapter-soup/notes.md`. No prediction was written before this queue.

### Every recorded system on the common lab set (cross-experiment comparison, 21 October)

This table puts every recorded recipe family side by side on the same 166 test clips (M01 + M02 + M04, 425 reference words: 122 single-word words and 303 sentence words). It was read on 21 October from the 723 `results.json` files of test evaluations in `reports/`, so the counts are exact rather than rebuilt from logs. Wrong words are means over the seeds shown; lower is better; the 35% bar is 148 wrong. Families that reuse the same runs (for example the default adapter in K, P and R) appear once. No new run was made for this table.

| System | Seeds | Wrong (WER) | M01 / M02 / M04 | Single words wrong (WER) | Sentence words wrong (WER) | Sub / del / ins |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| P · severe pretraining, then adapter at 6e-4 | 3 | **178.0 (41.9%)** | 74.3 / 37.0 / 66.7 | 92.3 (75.7%) | **85.7 (28.3%)** | 147.7 / 10.7 / 19.7 |
| P · severe pretraining, then adapter, 30 epochs | 3 | 180.0 (42.4%) | 78.3 / 34.3 / 67.3 | 90.0 (73.8%) | 90.0 (29.7%) | 145.0 / 11.3 / 23.7 |
| C · severe pretraining, then adapter | 3 | 182.3 (42.9%) | 76.0 / 38.0 / 68.3 | 90.3 (74.0%) | 92.0 (30.4%) | 149.3 / 10.0 / 23.0 |
| P · severe pretraining, then whole-block 8e-5, 20 epochs | 3 | 192.3 (45.3%) | 84.0 / 36.7 / 71.7 | **85.0 (69.7%)** | 107.3 (35.4%) | 154.3 / 16.3 / 21.7 |
| G2 · whole-block 1.6e-4, 40 epochs | 3 | 194.7 (45.8%) | 83.3 / 39.3 / 72.0 | 88.0 (72.1%) | 106.7 (35.2%) | 148.7 / 17.0 / 29.0 |
| P · adapter 30 epochs at 6e-4, no pretraining | 3 | 195.3 (46.0%) | 84.0 / 44.7 / 66.7 | 87.3 (71.6%) | 108.0 (35.6%) | 157.3 / 10.7 / 27.3 |
| Adapter on F-1 (7 October rebuild) | 1 | 197.0 (46.4%) | 78 / 47 / 72 | 88 (72.1%) | 109 (36.0%) | 150 / 22 / 25 |
| G/G2 · whole-block 8e-5, 40 epochs | 3 | 198.7 (46.7%) | 87.0 / 41.0 / 70.7 | 87.3 (71.6%) | 111.3 (36.7%) | 150.3 / 21.3 / 27.0 |
| S · adapter on F-1 | 5 | 203.6 (47.9%) | 84.6 / 47.6 / 71.4 | 91.4 (74.9%) | 112.2 (37.0%) | 161.4 / 16.4 / 25.8 |
| Adapter on stock (K/P/R default) | 3 | 212.0 (49.9%) | 89.0 / 41.0 / 82.0 | 94.7 (77.6%) | 117.3 (38.7%) | 163.3 / 18.3 / 30.3 |
| S · adapter on stock | 5 | 213.2 (50.2%) | 89.0 / 42.0 / 82.2 | 96.2 (78.9%) | 117.0 (38.6%) | 166.0 / 17.2 / 30.0 |
| F-1 alone | 1 | 248 (58.4%) | 91 / 56 / 101 | 106 (86.9%) | 142 (46.9%) | 181 / 31 / 36 |
| Whole-block, 5 epochs at 2e-5 | 3 | 304.7 (71.7%) | 128.3 / 47.7 / 128.7 | 133.7 (109.6%) | 171.0 (56.4%) | 190.3 / 42.0 / 72.3 |
| Parakeet-TDT-1.1B, stock | 1 | 307 (72.2%) | 116 / 55 / 136 | 133 (109.0%) | 174 (57.4%) | 215 / 31 / 61 |
| Stock 0.6B v2 | 1 | 323 (76.0%) | 131 / 62 / 130 | 135 (110.7%) | 188 (62.0%) | 195 / 46 / 82 |

A single-word WER above 100% means the model writes more wrong or extra words than there are reference words.

What the comparison shows:

- **Sentences are already in the goal range; single words are the whole gap.** The best system gets 28.3% WER on sentence words, and each of its three seeds is under 30% (28.7%, 27.7%, 28.4%). Stock gets 62.0% on the same words. On single words the same system is at 75.7%, and no recipe in the project gets single words below about 70%. With sentences at 85.7 wrong, the bar needs single words at 62 wrong or fewer (51% WER).
- **Severe pretraining's gain is almost all on sentences.** From the adapter on stock (212.0) to the pretrained adapter at 6e-4 (178.0), sentence errors fall 31.6 words and single-word errors 2.4.
- **Different systems are best on different kinds of clip.** Whole-block from the pretrained model is best on single words (85.0) and the pretrained adapter best on sentences (85.7). Using each on its own kind would give about 170.7 wrong (40.2%), still 23 above the bar, and only with the kind of clip known in advance, which the device does not know.
- **Different speakers prefer different recipes.** M02's lowest counts come from whole-block training and pretraining (whole-block 20 epochs 35.3, pretrained adapter 30 epochs 34.3), while M01 and M04 do best with the pretrained adapter. Picking the best family per speaker on the test clips would give about 175, so per-speaker selection would gain little and would be tuned on test.
- **Fine-tuning mainly removes extra and missing words.** Stock makes 82 insertions and 46 deletions; the best adapters make about 20 and 11. Substitutions fall much less (195 to 148). Whole-block training at the old default keeps most insertions (72.3), which is why it reaches more than 100% on single words. Trainable batch-norm (stage R) fails the other way: 175 deletions on average, about 110 of them on M04, so the model goes quiet.
- **Seed noise.** Across the 51 distinct recipe families with three or more seeds, the median spread between the best and worst seed is 10 words (the largest 19) and the median standard deviation is about 4.3 words. Two three-seed means less than about 8 words apart are not distinguishable; this is consistent with the project's 19-word rule being conservative.
- **Reproducibility.** The 15-epoch adapter on stock at seed 17 scored exactly 208 in three separate studies (the 7 October rebuild, the long-adapter study and the acoustic-only study), and the acoustic-only study's seed-17 confirmation dev repeated its search dev word for word. The same recipe in the 14 October queues scored 205 / 216 / 215 against 208 / 219 / 212, so a code change can move a seed by about 3 words.
- **The 112-word dev set ranks recipes coarsely but cannot pick among the best ones.** Over 58 families that have both dev and test scores, the dev and test rankings agree well (Spearman 0.90; 0.82 over 176 single runs). Among the 16 families under 205 wrong on test, the agreement drops to 0.37 and is not established. Choosing by dev mean would have picked whole-block from the pretrained model at 1.6e-4 (dev 35.7, test 192.7), 14.7 words worse on test than the pretrained adapter at 6e-4 (dev 37.7, test 178.0). An expo model chosen on dev is honest but should be expected to land near 190, not 178.
- **Two corrections found in the data.** The acoustic-only study has seed-17 test results from a second attempt that its note had missed (control 208; the head-frozen 24-block model 84 on M01 and 36 on M02, and 0 of 140 battery clips with words); see `reports/f2-acoustic-only/notes.md`. Stage B (larger base models with the adapter) never trained: the `1016-followup` queue counted it as done because its completion marker was a stock decode from lab test 4; see `reports/f2-base-model-ceiling/notes.md`.

### Secondary score: sentences only (21 October)

Every result now has a second WER: the sentence clips alone, with the single-word clips left out. On the common lab
set that is 303 of the 425 reference words (M01 122, M02 68, M04 113); the 122 words from single-word clips do not
count. Lower is better. The 35% line on this score is **106 wrong of 303**. The main pooled score and its 148-word bar
are unchanged; this sits beside them.

How it is produced:

- `scripts/asr/sentence_rescore.py` rescored every saved test decode in `reports/` on 21 October without decoding
  anything again. In this checkout it reads the sentence counts that `evaluate.py` saved in each `results.json`
  (exact; 723 evaluations). On a computer with the test manifests (the lab computer) it scores the decodes
  themselves, which also covers decodes that were never scored by clip kind.
- `pool_table.py`, which ends every lab test, now prints and saves a sentences-only table after the main one, with
  its own intervals and verdicts (`"sentences"` in `table.json`); `--kind sentence` makes it the main table.
- Sweep summaries (`reports/lab-big-*/summary.json`) carry `sentence_by_seed` and `sentence_mean`.
- Every `lab.py queue` ends by writing `reports/sentence-only/table-<stamp>.json`, and `lab_space.py git-results` now
  carries that folder and the `lab-big-*` summaries to GitHub (it skipped `lab-big-*` before).
- Not covered yet: labs 19, 20 and 22 to 30, whose decodes are only on the lab computer, and the September F-2 runs
  scored without the clip-kind split. The next queue's table covers them.

What it shows (means over the seeds listed):

- **The leader does not change.** Severe pretraining then the adapter at 6e-4 is first on both scores: 85.7 wrong of
  303 sentence words (28.3%; seeds 87, 84, 86). By speaker its sentence WER is M01 31.1%, M02 23.0%, M04 28.3%.
- **Six recipes are under the 35% line on sentences**, all of them severe pretraining with an adapter or with
  whole-block training (85.7 to 103.7 wrong). No recipe without severe pretraining is under it; the best of those is
  the adapter with stock-form targets at 106.3, just over the line.
- **The ranking barely moves** (Spearman 0.98 between the two scores over 91 recipes). The top five are the same five.
  The largest move is the adapter with stock-form targets, 18th on all clips and 7th on sentences: its single words
  are the worst of any adapter (82.0%), so the main score hid a sentence result as good as the best plain adapters.
- **Seed noise on sentences** is smaller in words than on all clips: across 51 recipes with three or more seeds the
  median spread is 8 words (largest 16), standard deviation about 3.3. The leader's 4.3-word lead over the next recipe
  (30 epochs instead of 6e-4, 90.0) is inside that noise.
- **Stock** is at 62.0% on sentences (188 wrong). M01's sentences are the hardest for every good recipe and M02's
  the easiest.

<details>
<summary>All 91 recipes run on M01, M02 and M04, ranked by sentence words wrong</summary>

| # | Recipe (run name) | Seeds | Sentence words wrong (of 303) | Sentence WER | M01 / M02 / M04 sentence WER | All-clip WER | Single-word WER |
| ---: | --- | ---: | ---: | ---: | --- | ---: | ---: |
| 1 | `big-p-pre-ad8-lr6e-4` | 3 | 85.7 | 28.3% | 31.1 / 23.0 / 28.3 | 41.9% | 75.7% |
| 2 | `big-p-pre-ad8-ep30` | 3 | 90.0 | 29.7% | 34.7 / 20.1 / 30.1 | 42.4% | 73.8% |
| 3 | `big-c-pre-ad8` | 3 | 92.0 | 30.4% | 32.0 / 24.5 / 32.2 | 42.9% | 74.0% |
| 4 | `big-p-pre-ad8-ep30-lr6e-4` | 3 | 94.3 | 31.1% | 35.8 / 25.0 / 29.8 | 44.7% | 78.4% |
| 5 | `big-p-pre-ad8-default` | 3 | 95.0 | 31.4% | 32.8 / 26.0 / 33.0 | 43.5% | 73.8% |
| 6 | `big-p-pre-wb-lr1.6e-4-ep20` | 3 | 103.7 | 34.2% | 39.9 / 27.9 / 31.9 | 45.3% | 73.0% |
| 7 | `big-t-ad8-stockform` | 3 | 106.3 | 35.1% | 37.7 / 36.8 / 31.3 | 48.5% | 82.0% |
| 8 | `big-g-wb-lr1.6e-4-ep40` | 3 | 106.7 | 35.2% | 37.4 / 32.8 / 34.2 | 45.8% | 72.1% |
| 9 | `big-p-pre-wb-lr8e-5-ep20` | 3 | 107.3 | 35.4% | 40.7 / 30.9 / 32.4 | 45.3% | 69.7% |
| 10 | `big-e-ad8-ep30` | 3 | 107.3 | 35.4% | 39.3 / 32.8 / 32.7 | 47.1% | 76.2% |
| 11 | `big-p-ad8-ep30-lr6e-4` | 3 | 108.0 | 35.6% | 40.7 / 38.7 / 28.3 | 46.0% | 71.6% |
| 12 | `big-l-ad8-lr6e-4` | 3 | 108.3 | 35.8% | 40.2 / 32.4 / 33.0 | 47.5% | 76.5% |
| 13 | `lab-baselines/adapter-f1` | 1 | 109.0 | 36.0% | 36.1 / 38.2 / 34.5 | 46.4% | 72.1% |
| 14 | `big-g-wb-lr8e-5-ep60` | 3 | 110.0 | 36.3% | 41.5 / 34.3 / 31.9 | 46.4% | 71.6% |
| 15 | `big-g-wb-lr8e-5-ep40` | 3 | 111.3 | 36.7% | 41.0 / 33.3 / 34.2 | 46.7% | 71.6% |
| 16 | `big-l-ad8-lr1e-3` | 3 | 111.7 | 36.9% | 40.7 / 34.8 / 33.9 | 47.2% | 73.0% |
| 17 | `f2-acoustic-only/1012-pending/control/` | 1 | 112.0 | 37.0% | 37.7 / 33.8 / 38.1 | 48.9% | 78.7% |
| 18 | `lab-baselines/adapter` | 1 | 112.0 | 37.0% | 37.7 / 33.8 / 38.1 | 48.9% | 78.7% |
| 19 | `big-s-ad8-on-f1` | 5 | 112.2 | 37.0% | 38.4 / 40.6 / 33.5 | 47.9% | 74.9% |
| 20 | `big-g-wb-lr1.6e-4-ep20` | 3 | 113.3 | 37.4% | 39.9 / 36.8 / 35.1 | 48.6% | 76.5% |
| 21 | `lab-21-tts` | 1 | 114.0 | 37.6% | 40.2 / 36.8 / 35.4 | 48.0% | 73.8% |
| 22 | `big-r-ad8-avg8` | 3 | 116.7 | 38.5% | 39.3 / 35.3 / 39.5 | 49.7% | 77.6% |
| 23 | `big-u-soup` | 1 | 117.0 | 38.6% | 37.7 / 33.8 / 42.5 | 49.2% | 75.4% |
| 24 | `big-s-ad8-default` | 5 | 117.0 | 38.6% | 38.5 / 35.3 / 40.7 | 50.2% | 78.9% |
| 25 | `big-k-ad8-sentences-only` | 3 | 117.0 | 38.6% | 35.0 / 39.7 / 41.9 | 57.9% | 105.7% |
| 26 | `big-r-ad8-avg1` | 3 | 117.3 | 38.7% | 39.9 / 33.8 / 40.4 | 49.7% | 77.0% |
| 27 | `big-k-ad8-default (same runs as big-p-ad8-default, big-r-ad8-default)` | 3 | 117.3 | 38.7% | 39.6 / 34.3 / 40.4 | 49.9% | 77.6% |
| 28 | `f2-long-adapter/1012-pending/control/ (same runs as big-c-ad8-default, big-e-ad8-default, big-l-ad8-default, big-t-ad8-default)` | 3 | 117.7 | 38.8% | 39.9 / 35.3 / 39.8 | 50.1% | 78.1% |
| 29 | `big-u-ad8` | 5 | 117.8 | 38.9% | 39.3 / 34.7 / 40.9 | 50.0% | 77.5% |
| 30 | `f2-adapter8-aug-nospeed` | 3 | 118.0 | 38.9% | 38.8 / 34.3 / 41.9 | 49.6% | 76.0% |
| 31 | `big-r-ad8-wd0` | 3 | 118.3 | 39.1% | 39.9 / 34.3 / 41.0 | 50.1% | 77.6% |
| 32 | `big-r-ad8-wd1e-2` | 3 | 118.7 | 39.2% | 38.8 / 35.3 / 41.9 | 50.3% | 77.9% |
| 33 | `f2-adapter8-aug-noise` | 3 | 119.7 | 39.5% | 38.8 / 38.7 / 40.7 | 50.5% | 77.9% |
| 34 | `big-r-ad8-wd1e-1` | 3 | 120.0 | 39.6% | 39.6 / 35.8 / 41.9 | 50.7% | 78.1% |
| 35 | `big-r-ad8-warmup30` | 3 | 120.0 | 39.6% | 40.2 / 36.3 / 41.0 | 50.7% | 78.4% |
| 36 | `lab-baselines/adapter` | 2 | 120.5 | 39.8% | 41.0 / 36.0 / 40.7 | 50.7% | 77.9% |
| 37 | `big-g-wb-lr8e-5-ep20` | 3 | 120.7 | 39.8% | 40.4 / 37.7 / 40.4 | 50.9% | 78.4% |
| 38 | `f2-long-adapter/1012-pending/stock-200/` | 3 | 121.0 | 39.9% | 47.0 / 39.7 / 32.4 | 52.4% | 83.3% |
| 39 | `f2-adapter8-aug-speedwide` | 3 | 121.7 | 40.2% | 39.3 / 37.7 / 42.5 | 51.5% | 79.8% |
| 40 | `f2-adapter8-aug-nospec` | 3 | 123.0 | 40.6% | 43.4 / 38.2 / 38.9 | 51.1% | 77.0% |
| 41 | `f2-adapter8-aug-tempo-noise` | 3 | 124.0 | 40.9% | 39.9 / 39.7 / 42.8 | 52.1% | 79.8% |
| 42 | `f2-adapter8-aug-tempo` | 3 | 124.3 | 41.0% | 38.8 / 37.7 / 45.4 | 51.5% | 77.6% |
| 43 | `lab-21-ttslow` | 1 | 128.0 | 42.2% | 36.1 / 50.0 / 44.2 | 50.4% | 70.5% |
| 44 | `big-e-ad8-ep8` | 3 | 129.3 | 42.7% | 41.3 / 38.7 / 46.6 | 54.9% | 85.2% |
| 45 | `big-l-ad8-lr1e-4` | 3 | 133.0 | 43.9% | 39.3 / 44.1 / 48.7 | 56.1% | 86.3% |
| 46 | `lab-11-then` | 1 | 142.0 | 46.9% | 44.3 / 38.2 / 54.9 | 58.4% | 86.9% |
| 47 | `lab-baselines/f1` | 1 | 142.0 | 46.9% | 39.3 / 54.4 / 50.4 | 58.4% | 86.9% |
| 48 | `big-e-wb-ep20` | 3 | 146.3 | 48.3% | 51.6 / 27.9 / 56.9 | 60.4% | 90.4% |
| 49 | `big-l-wb-lr8e-5` | 3 | 150.0 | 49.5% | 54.4 / 30.9 / 55.5 | 62.8% | 95.9% |
| 50 | `big-e-wb-ep12` | 3 | 156.0 | 51.5% | 53.3 / 33.3 / 60.5 | 63.4% | 92.9% |
| 51 | `big-l-wb-lr4e-5` | 3 | 160.7 | 53.0% | 55.5 / 34.8 / 61.4 | 66.5% | 100.0% |
| 52 | `lab-09-b16` | 1 | 161.0 | 53.1% | 54.9 / 35.3 / 61.9 | 67.1% | 101.6% |
| 53 | `big-e-wb-ep8` | 3 | 161.3 | 53.2% | 54.9 / 35.8 / 61.9 | 66.4% | 98.9% |
| 54 | `lab-11-pre` | 1 | 162.0 | 53.5% | 50.0 / 50.0 / 59.3 | 64.7% | 92.6% |
| 55 | `lab-baselines/whole-block` | 1 | 167.0 | 55.1% | 60.7 / 32.4 / 62.8 | 68.5% | 101.6% |
| 56 | `lab-09-b12` | 1 | 168.0 | 55.4% | 57.4 / 38.2 / 63.7 | 68.7% | 101.6% |
| 57 | `big-t-wb-stockform` | 3 | 168.0 | 55.4% | 58.7 / 34.8 / 64.3 | 72.7% | 115.6% |
| 58 | `lab-10-noise` | 1 | 169.0 | 55.8% | 57.4 / 36.8 / 65.5 | 68.5% | 100.0% |
| 59 | `lab-01-w75` | 1 | 169.0 | 55.8% | 60.7 / 35.3 / 62.8 | 71.3% | 109.8% |
| 60 | `big-s-wb-default` | 5 | 169.8 | 56.0% | 58.2 / 39.1 / 63.9 | 72.3% | 112.8% |
| 61 | `big-g-wb-default` | 3 | 170.3 | 56.2% | 59.6 / 38.2 / 63.4 | 72.4% | 112.6% |
| 62 | `lab-10-both` | 1 | 171.0 | 56.4% | 57.4 / 39.7 / 65.5 | 68.7% | 99.2% |
| 63 | `lab-02-all` | 1 | 171.0 | 56.4% | 58.2 / 39.7 / 64.6 | 69.6% | 102.5% |
| 64 | `lab-09-mid8` | 1 | 171.0 | 56.4% | 54.1 / 45.6 / 65.5 | 69.6% | 102.5% |
| 65 | `big-e-wb-default (same runs as big-l-wb-default, big-t-wb-default)` | 3 | 171.0 | 56.4% | 59.6 / 38.2 / 64.0 | 71.7% | 109.6% |
| 66 | `lab-01-w50` | 1 | 172.0 | 56.8% | 60.7 / 36.8 / 64.6 | 70.6% | 104.9% |
| 67 | `stock 1p1b (nemo)` | 1 | 174.0 | 57.4% | 50.8 / 41.2 / 74.3 | 72.2% | 109.0% |
| 68 | `lab-02-others` | 1 | 175.0 | 57.8% | 54.9 / 47.1 / 67.3 | 72.0% | 107.4% |
| 69 | `lab-08-ff` | 1 | 175.0 | 57.8% | 60.7 / 41.2 / 64.6 | 73.4% | 112.3% |
| 70 | `lab-10-margin (same runs as lab-10-margin-first-run)` | 1 | 175.0 | 57.8% | 60.7 / 42.6 / 63.7 | 74.1% | 114.8% |
| 71 | `big-l-wb-lr1e-5` | 3 | 175.3 | 57.9% | 60.9 / 41.7 / 64.3 | 73.3% | 111.5% |
| 72 | `lab-08-ff1` | 1 | 176.0 | 58.1% | 60.7 / 42.6 / 64.6 | 72.9% | 109.8% |
| 73 | `big-k-ad8-words-only` | 3 | 180.0 | 59.4% | 59.8 / 42.2 / 69.3 | 66.0% | 82.5% |
| 74 | `lab-08-ff2` | 1 | 180.0 | 59.4% | 59.8 / 45.6 / 67.3 | 73.9% | 109.8% |
| 75 | `lab-12-r3` | 1 | 181.0 | 59.7% | 60.7 / 41.2 / 69.9 | 73.6% | 108.2% |
| 76 | `lab-12-r2` | 1 | 182.0 | 60.1% | 60.7 / 41.2 / 70.8 | 74.1% | 109.0% |
| 77 | `stock rnnt06 (nemo)` | 1 | 183.0 | 60.4% | 59.0 / 47.1 / 69.9 | 73.9% | 107.4% |
| 78 | `lab-12-r1` | 1 | 183.0 | 60.4% | 60.7 / 42.6 / 70.8 | 74.1% | 108.2% |
| 79 | `lab-01-w25` | 1 | 183.0 | 60.4% | 61.5 / 42.6 / 69.9 | 74.6% | 109.8% |
| 80 | `big-l-wb-lr5e-6` | 3 | 184.7 | 60.9% | 60.9 / 47.1 / 69.3 | 75.8% | 112.6% |
| 81 | `lab-09-b4` | 1 | 186.0 | 61.4% | 62.3 / 45.6 / 69.9 | 75.8% | 111.5% |
| 82 | `lab-09-b2` | 1 | 187.0 | 61.7% | 61.5 / 48.5 / 69.9 | 77.2% | 115.6% |
| 83 | `stock 0.6B v2` (the same decodes saved in three places) | 1 | 188.0 | 62.0% | 61.5 / 47.1 / 71.7 | 76.0% | 110.7% |
| 84 | `stock 110m (nemo)` | 1 | 196.0 | 64.7% | 57.4 / 61.8 / 74.3 | 74.1% | 97.5% |
| 85 | `stock v3 (nemo)` | 1 | 196.0 | 64.7% | 59.0 / 47.1 / 81.4 | 75.1% | 100.8% |
| 86 | `stock ctc06 (nemo)` | 1 | 205.0 | 67.7% | 64.8 / 54.4 / 78.8 | 78.6% | 105.7% |
| 87 | `big-r-ad8-bn-trainable` | 3 | 212.7 | 70.2% | 63.4 / 58.8 / 84.4 | 76.4% | 91.8% |
| 88 | `stock nemotron (nemo)` | 1 | 213.0 | 70.3% | 68.0 / 67.6 / 74.3 | 77.9% | 96.7% |
| 89 | `stock qwen3 (hf)` | 1 | 215.0 | 71.0% | 69.7 / 45.6 / 87.6 | 82.8% | 112.3% |
| 90 | `stock granite41 (hf)` | 1 | 541.0 | 178.5% | 309.0 / 76.5 / 99.1 | 297.6% | 593.4% |
| 91 | `stock granite4 (hf)` | 1 | 658.0 | 217.2% | 86.9 / 77.9 / 441.6 | 186.6% | 110.7% |

Recipes run on only some speakers (scored on their own sentence words, so not comparable with the table above):

| Recipe (run name) | Speakers | Seeds | Sentence words wrong | Sentence WER | All-clip WER |
| --- | --- | ---: | ---: | ---: | ---: |
| `lab-09-b2-first-run` | M01 | 1 | 75.0 of 122 | 61.5% | 78.1% |
| `f2-acoustic-only/1012-pending/b24-frozen-high-lr/` | M01 + M02 | 1 | 65.0 of 190 | 34.2% | 45.5% |
| `f3-min7-updates` | M02 | 3 | 25.0 of 68 | 36.8% | 44.9% |
| `f3-min5-updates` | M02 | 3 | 25.7 of 68 | 37.7% | 46.0% |
| `f3-min5-epochs` | M02 | 3 | 25.7 of 68 | 37.7% | 48.1% |
| `lab-07-v2-first-run` | M02 | 1 | 26.0 of 68 | 38.2% | 51.6% |
| `f3-min7-epochs` | M02 | 3 | 28.3 of 68 | 41.7% | 50.2% |
| `f3-min1-epochs` | M02 | 3 | 29.7 of 68 | 43.6% | 58.2% |
| `f3-min3-epochs` | M02 | 3 | 31.3 of 68 | 46.1% | 53.7% |
| `f3-min3-updates` | M02 | 3 | 32.0 of 68 | 47.1% | 56.1% |
| `f3-min1-updates` | M02 | 3 | 33.0 of 68 | 48.5% | 60.0% |
| `lab-07-v2` | M02 + M04 | 1 | 97.0 of 181 | 53.6% | 66.4% |
| `lab-07-rnnt06` | M02 + M04 | 1 | 99.0 of 181 | 54.7% | 68.8% |
| `lab-07-1p1b` | M02 + M04 | 1 | 106.0 of 181 | 58.6% | 70.3% |
| `lab-07-v3` | M02 + M04 | 1 | 122.0 of 181 | 67.4% | 76.2% |

</details>

**Pushing results (`git-results`).** The 80 MB cap now counts only the result files that are new since the newest `lab-results-*` branch, because every branch holds the whole result set and Git sends only objects the remote lacks. When the new data is over the cap it is split over branches named `<name>-p1of3` and so on, each adding to the one before; push them all, and fetch the last one on this computer. With nothing new it builds no branch. Until 15 October it counted every result file, and at 103.7 MB it refused to build anything.

## Raspberry Pi 5 (prototype)

Prerequisite: [pi5.md](pi5.md#the-deploy-folder) bring-up.

### P-A · Recogniser speed (done 8 Sep 2026: quote label `raspberrypi-20260908-165319`)

```bash
python scripts/asr/bench.py --threads 1 2 4 --durations 1 3 10 --out reports/bench-pi5-idle
# then with the service running and the Speak page open in the kiosk, Chromium closed elsewhere:
python scripts/asr/bench.py --threads 2 --durations 3 --out reports/bench-pi5-loaded --with-app
```

| condition | threads | load s | 3 s clip decode s | RTF | peak RSS MB |
|---|---:|---:|---:|---:|---:|
| idle | 1 | 3.2 | 0.616 | 0.205 | ~1275 |
| idle | 2 | 3.2 | 0.396 | 0.132 | ~957 |
| idle | 4 | 3.2 | 0.365 | 0.122 | ~1275 |
| server + page open | 4 | | 0.455 end to end (0.407 decode) | | 1248 |

The loaded row needs a clean rerun (the 8 Sep run had Chromium open and its
idle-vs-loaded comparison came out inverted). Decision recorded in
`notes.md`: the thread count written into `deploy/pi5/willspeak.service`, and
that the Pi is the demo box (RTF far under 0.7). For comparison the Sky-2's
Celeron measured 0.51× real time idle at 2 threads (my run, 1 Sep 2026).

### P-B · End-to-end latency

Ten short (1–2 s) and ten long (4–6 s) utterances on the kiosk; time from
the end of speech to the first sound from the speaker (phone stopwatch, or
the page's status line and `latency_ms` in the server log). `latency_ms`
is the decode alone; the model now loads at server start, so the first
utterance no longer pays the ~3 s load, if the log line shows `load=`,
the server was started lazily (`WILLSPEAK_ASR_EAGER=0`) and that utterance
does not count.

| length | median s | worst s | of which recogniser (from the log) |
|---|---:|---:|---:|
| 1–2 s | | | |
| 4–6 s | | | |

One sentence naming the stage you would attack first and why.

### P-D · Cold start

Three power cycles: seconds from power on to the Speak page's "Listening"
state, with nobody touching anything. Note anything that needed a hand.

### P-E · Stop phrase

Twenty attempts at the stop phrase after a spoken sentence (does the box
retract?), and twenty minutes of ordinary conversation near the box (how
many false stops?). Miss rate, false-alarm rate, the `stop_text_sim`
threshold you settled on, and the sentence about why a miss and a false
alarm cost different amounts.

### P-F · Battery (done in part, 9 Sep 2026)

Run 1 on the 65 W 20,000 mAh bank: 38 min, about 3 percentage points, no
throttling; decoding was 0.4% of the time, so battery life is set by idle
draw, not by recognition. Not quotable yet, rerun until the bank has
dropped 20 points or more, then report the bracket.

### P-G · Other voices in the room (later extension; started 11 Sep 2026)

Background and the design: [design.md](design.md#other-voices-in-the-room).

**11 Sep 2026 (my living-room test, a second person talking):** "Hello, how
are you" → *Hearing you…* 5–10 s late, recogniser 3741 ms (that figure
included the ~3.25 s model load if it was the first utterance after the
server started, see the design note's correction; the load now happens
at start-up). Turning the sensitivity slider down made it fast and it
transcribed only me. That is the first row of the table; it is a
manual-slider result and it does not survive a quiet day, which is what the self-tracking gate
is for.

Then, with the self-tracking gate in place, the same test on two days,
three deliberate loudness levels each (quiet, normal, loud), ten short
utterances per level, a second person talking at about 1 m; the lavalier
at the collar, and once at the chair mount if there is one:

| day | level | his utterances that opened the gate | other-voice utterances that opened it | threshold the tracker settled on (dBFS) | median `latency_ms` |
|---|---|---:|---:|---:|---:|
| 11 Sep | manual slider | | | (slider value) | 3741 → after the change: |
| | quiet | | | | |
| | normal | | | | |
| | loud | | | | |

Read the server log for each utterance (`[willspeak] live clip …: dur=… decode=…`):
a long `dur` with few `words` is the gate staying open on the room. One
sentence on whether the collar or the mount was used and why it matters
(the 6 dB-per-doubling arithmetic in the design note).

---

## What goes on the poster

- T-1's severe-speaker WER, with n, next to the literature's framing.
- **F-1 and F-2: stock → fine-tuned WER per speaker with n and CI**, the
  numbers that say the architecture is worth building, with the
  typical-speech check beside them, and F-3's curve as "how many minutes".
- T-3's raw → projected gap, per the severe group, as "what the matcher adds".
- P-A's RTF and RSS as "runs on an $80 board, no cloud, no GPU", and F-4
  as "including the fine-tuned model".
- P-B's median latency and P-E's two error rates as "what it is like to use".
- The limitation panel: eight public speakers; the person it is for comes
  next, after ethics approval.
- A description of completed work and its evidence.

### Findings for the poster

Results from recorded runs that go beyond what was planned, ranked by how new they are. "Not found in the
literature" means not found in the literature sweeps done for this project, which were about one search per topic;
it is not a claim that nobody has published it. Every count below is wrong words of 425 pooled test words on the
three two-session severe speakers (M01, M02, M04; stock 323, 76.0%), unless the line says otherwise. Three speakers
from one corpus, read speech, so none of this is yet known to hold for the intended user.

1. **Where a single adapter belongs: about one third of the way up the encoder.** One 1,049,600-parameter layer
   after block 8 of 24 beat every other position on all three speakers, three seeds each: block 8 206.0, block 6
   215.0, blocks 4 / 10 / 12 / 16 / 24 254.0 / 251.3 / 263.3 / 254.3 / 293.7
   ([position seeds](../app/reports/f2-adapter-position-seeds/notes.md)). It held on a different base: on stage P's
   pretrained model block 8 scored 174.3 against 224.7 to 250.7 deeper
   ([lab 34](../app/reports/lab-34-pretrained-adapter-positions/notes.md)). No position comparison on a transducer
   for dysarthric speech was found in the literature.
2. **The lower encoder is what needs training, on this model too.** On M02's 37 test clips, training blocks 0 to 7
   gave 44.21% int8 WER against 62.11% for the top 8 blocks and 60.00% for stock (bottom beats top by 17.89 points,
   CI 8.49 to 30.56). The disordered-speech papers train lower layers of older LSTM models; this repeats it on
   Parakeet's FastConformer.
3. **The standard int8 export damages Parakeet's depthwise convolutions.** In the stock int8 encoder, 1,002 of 1,024
   channels in block 1 and 1,020 of 1,024 in block 22 sit at the zero point ([lab 15](../app/reports/lab-15-int8-screen/notes.md)).
   The default export costs 8 to 12 words; leaving the convolutions in fp32 (`--int8-ops matmul`) costs nothing
   measurable on the lab computer's x86 CPU (whole-block 291 fp32, 303 default, 285 matmul;
   [lab 19](../app/reports/lab-19-int8-cost/notes.md); the expo model 176 / 180 / 177,
   [lab 28](../app/reports/lab-28-deployable-export/notes.md)). Not yet measured on the Pi's ARM CPU, whose int8
   kernels differ: single clips there have gone both ways (one 8-word sentence 5 wrong matmul against 2 default;
   M02's "The little schoolhouse stood empty." 3 matmul against 6 default). F-4 on the Pi with both exports decides
   it; until then the device half of this finding stays off the poster.
4. **The best recipe: other severe speakers first, then a per-speaker adapter.** 178.0 (41.9%) against stock's 323
   (76.0%), rebuilt at 174.3; on sentences 85.7 of 303 (28.3%), on single words 75.7%
   ([stage P](../app/reports/f2-adapter-levers/notes.md)).
5. **Single words are the hard core.** All 56 clips that every one of eight systems gets wrong are single words; no
   sentence is missed by all of them ([lab 29](../app/reports/lab-29-hard-core/notes.md)). Context is what the model
   uses to recover severe speech.
6. **Picking among good recipes on a TORGO-sized dev set is mostly noise.** Seeds of one recipe spread by a median
   of 10 words; dev and test rankings agree overall (Spearman 0.90) but not among the 16 best families (0.37), and a
   dev-chosen model would have scored 192.7 instead of 178.0 ([every recorded system](#every-recorded-system-on-the-common-lab-set-cross-experiment-comparison-21-october)).
   Most published TORGO comparisons are one run each.
7. **The ASR encoder is a better matcher than WavLM.** Severe closed-vocabulary top-1: stock Parakeet layer 9
   61.6% against WavLM 46.6% (gain CI +5.0 to +25.7 points), M02's F-1 model at layer 10 73.3%
   ([lab 5](../app/reports/lab-05-layer-screen/notes.md)). One encoder could serve both paths on the Pi.
8. **Open, would be the most new: does the model prefer silence?** Lab test 6 found that for stock and whole-block,
   saying nothing scores higher than the true words on 81 to 100% of severe test clips, which would explain the
   missing words; nothing in the literature explains that failure. It had a formatting confound and no typical-speech
   comparison. [Lab test 35](../app/reports/lab-35-silence-preference/notes.md) fixes both and scores stage P;
   until it runs, this finding is not on the poster.
