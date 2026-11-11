# Lab test 38: does a language model choose better pseudo-labels for self-training than Parakeet's own first answer?

Proposal: F-2 and device design (extension), joining lab test 25 (self-training on stage P's adapter) and lab test 36 (language-model rescoring). Measure: pooled wrong words of 425 on the test clips, as in lab test 25, plus how good the pseudo-labels were against the true words.

Status: written 9 November 2026. Prediction not yet written. Code and unit tests only. Nothing has run on a GPU, no language-model file was downloaded and nothing was decoded.

[Report Context](../Report%20Context.md), [Measurement guide](../../../docs/measurements.md)

## Idea

Lab test 22's loop learns from his unlabelled recordings. Test clips arrive in chunks, each chunk gets decoded, and the answers become pseudo-labels for retraining the adapter on top of his labelled clips. Lab test 25 ran it on stage P's pretrained adapter and went from 178.0 to 172.3 wrong words of 425, a 5.7-word gain against a rule that needs about 19.

The teacher there is the adapter's first answer. Lab test 36 found a better one for sentences: of Parakeet's nine decodes per clip, the one with the highest `audio score + 0.7 × language-model score + 3 × words` (Qwen2.5-0.5B, file `qwen25-05b-q4`, weights fixed, no labels needed) is right more often than the first. Here those picks become the pseudo-labels, to see if better labels take the loop further.

Three arms against one control:

| Arm | Pseudo-label of a sentence | Pseudo-label of a single word |
| --- | --- | --- |
| Control (lab test 25, reused) | the first decode | the first decode |
| `all` | the language-model choice, every sentence | the first decode |
| `top` | the language-model choice, only the half of the sentences with the widest margin over the runner-up | the first decode |
| `margin` | the language-model choice, only sentences whose margin is at least 3 nats | the first decode |

A sentence an arm drops isn't trained on at all (it doesn't fall back to its first decode), so the arms differ in which clips teach too.

## Setup

The loop is lab tests 22 and 25's code (`tta_loop` in `lab.py`, `scripts/asr/tta.py`) for M01, M02 and M04 at seeds 17, 18 and 19, starting from stage P's adapter on the severe-pretrained model (adapter after block 8, 6e-4, 15 epochs). Test clips go into four chunks by hash of clip id. The static adapter decodes chunk 0. Round k trains the same recipe from the same start on his labelled clips plus the pseudo-labelled clips of chunks 0 to k-1, then decodes chunk k. Scoring uses those ordinary decodes merged over the four chunks, exactly as in lab test 25, so the control compares to the digit.

After each chunk is decoded, `lm_pool.py --no-reference` builds the nine-decode pool, `lm_score.py score` adds the language-model scores to one shared score file, and `lm_pick.py pick` chooses with fixed weights 0.7 and 3 (lab test 36's preset, chosen on no speaker). It also records the margin, the chosen answer's lead in total score over the runner-up. Before the next round `lm_pick.py filter` applies the arm's rule to all picks so far and `tta.py build --withhold` keeps dropped clips out of the export and counts them. Lab test 22's agreement gate, dev-prompt check and cap apply as usual. The last chunk is never pooled since no later round uses it.

Single words are left alone, because lab test 36's weights lost clips on them. A clip is a single word if its plain decode (what the control trains on) is one word. Those keep the plain decode, get no margin and are never dropped.

`margin = best total − runner-up total`, in nats. A pool with no real competitor (one answer, or the others impossible by the audio) gets a null margin, counted as widest. `top` is causal: each round it's recomputed over the sentence-like clips of chunks 0 to k-1 only, ties broken by clip id. The 3-nat cutoff was chosen without any data, not tuned on lab test 36's labelled pools or anything else. Every `pick` prints its chunk's margin deciles, so a filter keeping almost nothing or almost everything shows in the log. If at round 3 it keeps under a fifth or over nine tenths of a speaker's sentence-like clips, I'll call it uninformative and not rerun with another value here.

No training step sees the test text, as in lab tests 22 and 25. The text still sits in some files: `tta.py manifest` copies whole test rows into the chunk manifests, `nemo_hyp.py` loads them through a loader that needs the field but only uses the audio path, and `evaluate.py` and `tta.py score` read it to score answers (that's evaluation). For the new parts, `lm_pool.py --no-reference` reads only id, audio path and speaker (`read_unlabelled`; its row builder `pool_row` has no path to the text) and writes an empty reference with kind `unknown`, which `lm_report.py` refuses. `lm_pick.py pick` and `filter` never open a manifest. `tta.py build` reads the test manifest through `pseudo_label.read_audio_rows` (id, audio, duration, speaker), as in lab test 22. The only new reader of true words is `lm_pick.py audit`, run once after the last training step, and its numbers never reach a training file.

`tests/unit/test_lm_pseudo_labels.py` runs the chain (pool row, scores, pick, filter, round export) from a manifest with a marker sentence in its text column and from one with no text column, and checks the marker ends up in no file the loop writes. From the dry plan it checks every pool command has `--no-reference`, no pick or filter command gets a test manifest, and the audit is the only command with `--truth` and comes after the last `train.py`.

The control is lab test 25's loop, not rerun. Its chunk manifests, per-round decodes (`lab-25-<speaker>-s<seed>-chunk<k>-answers.jsonl`), static decodes and by-chunk tables are in the repository and reused under the same names. Nothing is written under lab test 25's folder except the merged answers if missing. Its round exports and models aren't in the repository, so the label-quality table shows control labels before the gate from saved decodes, and after the gate only on a computer that still has `data/.../export/lab-25-*-r3`. If decodes are missing the stage retrains the loop with the same code.

Numbers from earlier runs, for comparison:

Stage P's static adapter: 183, 176 and 175 wrong of 425 at seeds 17, 18 and 19, mean 178.0 (41.9%). Stock 323, the 35% bar is 148. Lab test 25's loop: 179, 173 and 165, mean 172.3 (40.5%). Summed over seeds, M01 went 223 to 209, M02 111 to 113, M04 200 to 195.

From its by-chunk tables (`reports/lab-25-tta-on-pretrained/by-chunk-<speaker>-s<seed>.json`), summed over speakers and seeds, chunks 0 to 3 hold 345, 288, 261 and 381 words. Static gets 117, 142, 119 and 156 wrong, the loop 117, 141, 111 and 148. That's 1 word in chunk 1 (nothing) and 8 each in chunks 2 and 3.

Of the static adapter's 178 wrong words, 92.3 on average are on the 122 single-word words and 85.7 on the 303 sentence words. Only sentence labels change and only chunks 1 to 3 (about three quarters of clips) can be affected, so this test can move well under 86 words.

Lab test 36's printed report (lab computer, 6 November, 13:47 to 19:18 lab clock; not yet in its own note): on stage P's recipe with 120 sentences held out by cross-validation, the first decode is right on 29 (24.2%). `qwen25-05b-q4` with weights from the other speakers gets 38 (31.7%), +7.5 points (interval +2.5 to +12.5, 10 gained, 1 lost), sentence WER 27.1% to 22.5%. `qwen25-05b-q8` gets 39 (32.5%), WER 22.9%. A perfect chooser gets 45 (37.5%, WER 17.3%). On stage P's 44 test sentences alone it went 8 to 10, inside noise. On single words the same weights went from 221 to 195 right of 393.

Seed noise: across 51 recipe families the median spread of three or more seeds is 10 words (standard deviation about 4.3).

## Plan

1. Language-model environment (`./willspeak.sh lm-setup`, or the stage builds it) and the one model file (`qwen25-05b-q4`, 398 MB), downloaded once.
2. The three pretrained models and each seed's static adapter as in lab test 25, reused where possible. Stage P's adapter is kept until chunk 0's pool exists. On a computer that trims weights it's retrained for that (same recipe and seed), without re-decoding.
3. The control, reused or rebuilt.
4. Per speaker, seed and arm, three rounds of pick, filter, build, train, decode, pool, score, pick.
5. A pooled table (stock, F-1, static, control, each arm per seed, three-seed means) and `label-quality.json`: word error of the final round's pseudo-labels for control and each arm, split sentence-like vs single-word, plus per arm the labels before the gate (control's from saved decodes), what the margin filter dropped, and the plain decodes of the same clips.

Decision rules, set before any run:

Each arm is compared with the control on the three-seed mean, by the project's rule: at least 2 of 3 speakers move the same way (summed over seeds, as in lab test 25) and pooled WER moves by more than about 4.5 points, which is 19 words. With the control at 172.3 an arm needs 153 or fewer. The comparison with the static 178.0 comes second.

`all` and `top` are named in advance and `margin` is a declared second filter. Best of three looks better than it is, so an arm meeting the rule while the other two stay within 10 words of the control gets reported as one of three, not established.

An arm harms if its mean is 10 or more words above the control's (the median seed spread) and at least 2 of 3 speakers are worse. If the control is rebuilt and lands more than 10 words from 172.3, arms are read against the rebuilt control and the difference reported. The by-chunk tables (`by-chunk-<arm>-<speaker>-s<seed>.json`) must show a gain that holds or grows in chunks 2 and 3, as lab test 22 asked, not one only in chunk 1. The label-quality table doesn't decide anything alone. It says whether the language model's labels were better (lower word error on the same clips, more exactly right) and what the filters threw out.

What outcomes would mean, written before the run:

1. An arm beats the control with better labels in the audit: a better teacher gives a better loop. The device should keep a language model beside Parakeet for the overnight queue, and lab test 36's rescoring then also helps the personal model.
2. Better labels, no better accuracy: label quality isn't the limit. The loop learns from clips it already gets right, or the gains hit the same few clips.
3. A filtered arm beats `all` and the control: fewer, surer labels teach more, and the margin is worth taking to the device.
4. `all` beats the filtered arms: quantity beats surety at this size and the filters throw away teaching clips.
5. Nothing moves, labels about as good as plain: the language model changes few sentences that survive the gate, or the gate removes the corrections (below).
6. An arm is worse with worse labels: the language model is confidently wrong on his sentences and self-training amplifies it. Don't use it as a teacher.
7. Later chunks worse than chunk 1: errors compound.

Known confound: lab test 22's agreement gate is the same in every arm. A pseudo-label is only used if it agrees (similarity at least 0.5) with F-1's and the static adapter's answers. A language-model correction that disagrees with both wrong answers can be rejected, so the gate may remove exactly the labels being tested. Rejections are counted in each export's `tta.json` (`gate`, next to `margin filter`) and in the log per round and arm. The gate stays as is. `--set tta_agree_min=0` would change it for every arm including the control, which is a different test.

## Why

Lab test 25 showed the loop is worth little on the better start, and lab test 36 showed a language model fixes errors fine-tuning leaves. So far the loop has learned from its own mistakes. This is a cheap check of whether a transcript-free chooser can feed it better ones, with no new training recipe.

## Limits

It can show whether language-model-chosen sentence labels, with or without a margin rule, beat plain decodes in the self-training loop at three seeds, and how good each kind of label was.

It can't speak to a real deployment's clip order or a longer run, other language models or weights (fixed here), single-word labels (unchanged), or the intended user. The chunks are quarters of one session's test set, the same 166 clips as lab tests 22 and 25. TORGO's prompts are public and repeat across speakers, so a text model may know some. Lab test 36 has a memorisation check; this test doesn't.

The pool's first decode comes from the pool builder (batch of one) and the plain answer from the ordinary decode (batch of four), and they can differ. Arms use the plain answer for single words and unchanged sentences, and `lm_pick.py pick` prints how many clips differ per chunk.

## Run Record

<details>
<summary>Predictions, commands, full results, and conclusions</summary>

### Prediction (Before the Run)

Not yet written.

### Commands and Setup

All times below are estimates. Nothing here has run on a GPU. At setup on 9 November 2026 the code was checked only by unit tests (`tests/unit/test_lm_pseudo_labels.py`, plus labs 22, 25 and 36's tests) and a dry plan.

From `app/`, with nothing else on the GPU:

```bash
python lab.py run 38 --dry          # the plan, with every command
python lab.py run 38                # the test
python lab.py queue selftrain-lm --detach
```

The stage builds the environment (`~/venvs/willspeak-lm`) if missing. The model file goes in `models/lm/` and is deleted at the end on a computer with under 100 GB free (scores are kept). `--set lm_label_arms=all` runs one arm and cuts training to a third. `--rerun` moves an old `table.json` aside.

Settings: `lm_label_model` (default `qwen25-05b-q4`), `lm_label_arms` (`all top margin`), `lm_weight` (0.7), `lm_bonus` (3), `lm_top_share` (0.5), `lm_min_margin` (3). From lab test 36: `lm_views`, `lm_threads`, `lmpy`. From lab test 22: `tta_chunks`, `tta_order`, `tta_agree_min`, `tta_max_pseudo`. Also `trim_models`.

Time, scaled from a measurement: lab test 25's logged decodes (`created_at` in `lab-25-*-chunk*-answers.provenance.json`, lab computer, 20 October) put its 27 rounds (build, adapter training, decode) between 20:43 and 21:44, 2.1 to 2.7 minutes each (M01 2.1, M02 2.7, M04 2.2). This test has 81 rounds, so about 3.2 hours of training. Pools are small (lab test 36 ran about 0.2 seconds a clip, so about 14 clips take seconds), but each round reloads the recogniser and language model, and 63 of those I'd guess at half an hour, not measured. So about 4 hours if the three pretrained models and nine static adapters are still on disk.

On a fresh checkout the control and static answers come with the repository but the weights don't, so the pretrained models and static adapters get retrained (chunk 0's pool needs the adapter weights). That's an estimated 1.5 to 2 hours more, not measured, about 6 hours total. On a computer that trims weights the nine static adapters get trained a second time the same way. These are for the lab computer's RTX 4000 Ada. Yack's 4060 hasn't run this loop.

Outputs in `reports/lab-38-lm-pseudo-labels/`: `table.json` (pooled table), `label-quality.json`, `by-chunk-<arm>-<speaker>-s<seed>.json` and `control-by-chunk-<speaker>-s<seed>.json`, `pools/`, `scores/`, `picks/` (each chunk's choice with margin, before filtering), and `labels/` (each round's picks after the arm's rule, dropped clips marked).

### Full Results

Not run.

### Conclusions

Not run.

</details>

## Related

[Lab test 22](../lab-22-test-time-adaptation/notes.md), [Lab test 25](../lab-25-tta-on-pretrained/notes.md), [Lab test 36](../lab-36-lm-rescoring/notes.md), [Stage P and the adapter levers](../f2-adapter-levers/notes.md)
