# TODO: my project work

Paths and commands below are relative to `app/` unless they refer to documentation.

*The only task list in this repository. Every item is something you create -
a run and its notes, a script, a document, a demo, and each one says where
its commands and tables are. Cleanup, deletions and doc maintenance are
not your job and never appear here.*

Codes used everywhere (proposal, `measurements.md`, `reports/`):
**H1–H4** hypotheses · **T-1–T-3** public-speaker tests · **F-1–F-4** fine-tune
experiments · **P-A/P-B/P-D/P-E/P-F** Pi measurements.
H5, T-4 and P-G are later extensions outside the supplied proposal, as are
**Lab tests 1–15** (extension tests, [measurements](measurements.md#lab-tests)).
The [report map](../app/reports/Report%20Context.md) gives current note folders and goals.
Commands and empty tables: [measurements.md](measurements.md).
Every item is named by its proposal code and what changed, for example
`F-2 · M02 · bottom 8 blocks · all parts`. The old B, C, D and E codes are
looked up in [measurements.md](measurements.md#run-names).
Dates: [plan.md](plan.md#fall-2026-plan-the-public-corpus-semester).

The one habit that runs through everything: **the prediction goes in
`notes.md` before the run.** This preserves what was expected before seeing the
result. Measurements remain useful when a prediction was missed; label the
missing prediction instead of inventing one afterward. Template at the top of
`measurements.md`.

---

## 1 · Now: final steps to the poster (file ~Mon Nov 16, presented Thu Nov 19)

In order. Each is something you create; nothing here is cleanup.

- [ ] **Lab test 35 · silence preference, fixed.** The one run left that can add a new finding: does the model
      prefer saying nothing over the true words on severe speech but not on typical speech, and does stage P's
      adapter remove that? Lab test 6 found 81 to 100% of severe clips prefer silence, but scored the true words in
      the wrong form and never scored the controls or the best system. Context, reference points and what each
      outcome means: [notes](../app/reports/lab-35-silence-preference/notes.md). Your prediction in chat first,
      then on the lab computer `python lab.py run 35` (about 45 minutes to 2 hours) and push the results.
- [ ] **Lab test 36 · language-model rescoring: the Pi, and what comes next.** Ran 6 November on the lab computer (27
      models); a text model lifts stage P's whole sentences from 24.2% to 31.7 to 35.0% on 120 held-out sentences and does
      much less on stock, and it hurts single words unless they are routed around it. Results, decision-rule verdicts and the
      replays: [notes](../app/reports/lab-36-lm-rescoring/notes.md). Still to do: the Pi measurement for the two models that
      pass the screen (`qwen25-05b-q4`, `spark26-17b-q4`; memory, seconds per clip, and the nine decodes per clip,
      [pi5.md](pi5.md#10-language-model-rescoring-on-the-pi)); a wider bonus grid and several models' scores averaged (both can be
      replayed from the saved pools with `lm_report.py`); and the follow-up tests ideas 5 to 7 in
      [research §9](research/finetune-torgo.md#9-after-lab-test-36-language-model-rescoring-on-the-labs-own-recipe).
- [ ] **Lab test 37 · routing single words and sentences to different adapters.** The whole gap to the 35% bar is
      single words. A block-8 adapter trained only on single-word clips (and one on sentences only) sits beside the
      ordinary adapter on stage P's base, three seeds; each test clip then gets the word adapter's answer if the
      ordinary adapter's first decode has one word (the device can see that), or if the clip is short, and the
      ordinary answer otherwise. The true kind is shown beside it as a bound. Measure: wrong words of 425, split into
      single words and sentences, and the clips routed the wrong way. Context, the lab K result that points against
      it, and the decision rules: [notes](../app/reports/lab-37-kind-routing/notes.md). Your prediction in chat first,
      then on the lab computer `python lab.py run 37` (about 1.5 to 2 hours, plus about 25 minutes if the pretrained
      models are rebuilt; `python lab.py queue routing --detach` in the background) and push the results.
- [ ] **Lab test 38 · self-training on language-model-chosen pseudo-labels.** Lab test 25's loop on stage P's
      adapter, but a sentence's pseudo-label is the answer that Parakeet's score plus Qwen2.5-0.5B's choose together
      (single words keep the plain decode), for every sentence, for the widest half of the margins and for margins of
      3 nats or more; three speakers, three seeds, against lab test 25's plain labels, with the word error of each
      arm's pseudo-labels against the true words beside the table. Context, the decision rules and what each outcome
      means: [notes](../app/reports/lab-38-lm-pseudo-labels/notes.md). Your prediction in chat first, then on the lab
      computer `python lab.py queue selftrain-lm --detach` (an estimated 6 hours on a fresh checkout, 4 with the
      pretrained models on hand; `--set lm_label_arms=all` runs one arm in a third of that) and push the results.
- [ ] **Lab test 39 · wider candidate pools.** Lab test 36's pools hold the right sentence for 45 of 120 sentences and
      the chooser finds fewer; this adds greedy answers from other models of the same base (stage P's seeds 18 and 19,
      the block-12 and block-16 adapters, or all four), every answer still scored by the primary model's audio score
      and the same text model, and reports pool ceiling, sentences right and decodes per clip for each arm. The extra
      models for a held-out fold are trained without that fold. Context, the decision rules set before the run:
      [notes](../app/reports/lab-39-wider-pools/notes.md). Your prediction in chat first. Then a first look,
      `python lab.py --set wide_systems=prototype run 39` (about 15 adapters, an estimated 1 hour), and the whole
      test, `python lab.py queue widepool --detach` (an estimated 4 to 5 hours with the folds, more if the pretrained
      models have to be rebuilt; best after test 36 has run, whose pools it reuses when its adapters are still on disk).
- [ ] **F-4 on the Pi · the expo models, matmul against default.** Both `lab-28-export-{M01,M02,M04}-matmul`
      and `-default` are on the Pi. Score all six on their speaker's test clips (166 clips, 425 words) on the Pi,
      against lab 28's desktop rows (matmul 67 / 39 / 71, default 70 / 36 / 74 wrong). On the desktop matmul cost
      nothing; on the Pi one 8-word sentence came out 5 wrong with matmul and 2 with default, and another clip the
      other way round, so only the full set says whether the Pi's ARM int8 kernels change the answer. Commands in
      [pi5.md](pi5.md#comparison-demo) and [measurements §F-4](measurements.md). Prediction first.
- [ ] **P-A loaded · speed with the expo model.** `bench.py --with-app --model-dir models/lab-28-export-M02-matmul`
      on the Pi with only the kiosk open; the matmul export keeps convolutions in fp32, so its speed has never been
      measured against stock's 0.396 s. Use a 5 V / 5 A supply (`vcgencmd get_throttled` reported under-voltage).
- [ ] **Personal device model · 5 Nov · stage P recipe, no scoring.** The IRB confirmed
      that a model built for him, with no measurements and nothing published, needs no review; file their
      email with the project. Record him into his own corpus, build the shared base once, train and export
      his model, put it on the Pi and demo it with Try your voice. Every step and what stays private:
      [training.md](training.md#8-a-personal-model-for-the-device-no-scoring). Nothing about his speech goes
      in `reports/`, on the poster or into a score. Record *different* sentences rather than repeats; the
      reason is in [finetune-torgo.md](research/finetune-torgo.md#8-whole-sentences-right-what-helps-what-does-not-and-where-more-data-comes-from).
- [ ] **SAP data access · not needed for the poster.** The Speech Accessibility Project is the only public
      English corpus big enough to train a disordered-speech base (idea 2 in
      [finetune-torgo.md §8](research/finetune-torgo.md#8-whole-sentences-right-what-helps-what-does-not-and-where-more-data-comes-from)).
      Find a faculty sponsor and the university's authorised signing official, write the one-page proposal,
      and send it with the signed data use agreement from university email addresses to
      speechaccessibility@beckman.illinois.edu (term 3 years or less).
- [ ] **Recording quality gate · 5 Nov · real rooms, my voice.** The Record page now checks every take for
      background noise, echo and a second voice ([design.md](design.md#recording-quality-gate)); its thresholds
      were set on mixtures only. With the lavalier and my own voice, record the same 20 prompts in five setups:
      quiet room, TV on, fan on, someone talking nearby, and the microphone across the room. Write my prediction
      first: which setups the gate will send back, and how often. Then count the takes it flagged in each setup
      (Review shows **QUALITY FLAG**; status `retake` is a first failure). The gate runs on his recordings
      automatically when he enrolls; nothing about it is scored on his speech.
- [ ] **Poster.** 36 × 36 in, the course template, PDF. The findings worth a panel are listed in
      [measurements.md](measurements.md#findings-for-the-poster), each with its report. Every number comes from a
      note under `reports/`; include the limitation panel.

The rest of this file is the record of finished work and the items that wait for spring.

## The expo prototype (done 3 Nov)

- [x] **Prototype for the expo · per-speaker models against stock, on the Pi.** Stage P's pretrained adapter
      after block 8, exported with `--int8-ops matmul`, for M01, M02 and M04, beside stock on the Compare page
      (`compare.py`, `/compare`; `docs/design.md`, **Expo comparison demo**). Pi set up 4 Nov (kiosk opens
      `/compare`, 30 demo clips, model picker). Shown at the expo.

---

## Week 2 (Sep 14–20): TORGO and the baseline

- [x] **Data · fetch TORGO, build the manifests.** [training.md](training.md#public-corpora-torgo).
      On the 4060 desktop: `fetch_torgo.py`, then `--controls`, then
      `torgo_manifest.py --export`. Read `manifests/summary.md`; two lines in
      `reports/t1-zero-shot/notes.md`: usable utterances per severe speaker, and which
      of them have more than one     session (that decides who F-2 runs on).
- [x] **T-1 · the zero-shot baseline.** [measurements §T-1](measurements.md).
      Prediction first (H1 says over 60%; write what you expect per speaker).
      Severe-sentences manifest, then all sentences grouped by severity. Pick two
      transcript pairs from `report.html` for the storyboard's "what goes wrong"
      panel. Fill the table.
- [x] **Setup · the training machine.** [training.md](training.md#parakeet-training-and-evaluation):
      WSL2, driver, venv, `torch.cuda.is_available()`, `train.py --check`, then
      `--max-steps 2` into a throwaway folder. Done when step 2 runs on the GPU.
      If it fights you for more than an evening, the Kaggle fallback is in the
      same section. Always pass `--export data/public/torgo/export/<SPK>`; without
      it `train.py` looks in `data/export` (the recorder's own corpus) and fails.
      *Done 23 Sep:* `--check` passed on F01 (85 train clips, 3.4 min, 10 dev);
      the 2-step smoke run trained on the 4060 and saved
      `models/smoke-test/parakeet-ft.nemo` (training 20 s). On the way: Python
      3.12 env, libNVVM, numba-cuda, numpy<2.3, WSL2 swap, /tmp is RAM-backed,
      checkpoints now keep only the trainable part, all in training.md §2.
      Rebuild or repair the env with `./willspeak.sh train-setup`.
- [x] **Smoke test · F01 · 1 epoch.** A throwaway F-4 chain: one-epoch fine-tune of F01: `train.py --epochs 1`
      → `export.py` → `check.py` → `evaluate.py` on the export → copy `onnx-int8`
      to the Pi → `bench.py` with `WILLSPEAK_PARAKEET_DIR` pointed at it.
      [measurements §F-4](measurements.md). The number is meaningless; that
      it ran is the point, the export is the proposal's "sharp edge".
      *Done 24 Sep* (after the RAM upgrade): `models/b4-f01`, 6 optimizer steps
      (warmup 10 was never reached, so the weights barely moved); export validated;
      F01 test (23 clips) WER 93.48%, CER 68.78%. The Pi bench is skipped by choice:
      the first real model gets it in F-4 on the Pi. Seen on the way: on one second of silence
      (`check.py`) the fine-tuned export says "I don't know." and the stock model says
      nothing, the stock export check finds out why.
- [x] **Export check · stock model on F01.** Stock through our own export. Two questions before the first F-2 run: what does the
      stock model score on the same 23 F01 clips (the T-1 F01 figure, 57.6%, is a
      different set of clips), and does `export.py` itself change the model? Prediction
      first, in `reports/setup-f01-smoke-test/notes.md`: your answer to each of the two questions below.
      ```bash
      python scripts/asr/evaluate.py --manifest data/public/torgo/export/F01/test.jsonl --audio-root data/public/torgo/export/F01 --out reports/b5-stock
      python scripts/asr/export.py --stock --out models/stock-export
      python scripts/asr/evaluate.py --manifest data/public/torgo/export/F01/test.jsonl --audio-root data/public/torgo/export/F01 --model-dir models/stock-export --out reports/b5-stock-export
      python scripts/asr/check.py --model-dir models/stock-export
      ```
      Three models, each differing from the next in one thing: prebuilt stock,
      stock through `export.py` (same weights, our conversion), the smoke test (our
      conversion, six training steps). Prebuilt against re-exported isolates the export; re-exported against smoke test
      isolates the training.
      *Question 1, accuracy.* Is the re-export worse than prebuilt stock (the export costs accuracy: fixed
      before the first F-2 run)? Is the smoke test worse than the re-export (six small steps did damage: the
      recipe gets a look before the first F-2 run)? "Worse" means your `ci.py` interval on the
      difference stays clear of zero:
      `python scripts/asr/ci.py --manifest data/public/torgo/export/F01/test.jsonl --stock reports/b5-stock --finetuned reports/b5-stock-export`,
      then again with `--stock reports/b5-stock-export --finetuned reports/b4-f01`.
      *Question 2, silence.* Does the re-export say "I don't know." on silence? Yes: the export
      causes it. No: the six training steps did.
      *Done 24 Sep.* Prebuilt stock 73.91% WER on the 23 clips (95% CI 57.14–113.04%);
      stock through `export.py` identical (Δ 0.00, CI 0.00 to 0.00) and silent on
      silence, so the export is clean. The smoke test against the re-export: Δ −19.57 points
      (CI −30.56 to −4.65), so the training run did real damage and also made the
      model answer silence. Notes in `reports/setup-f01-smoke-test/notes.md`.
- [x] **Batch-norm diagnosis · F01.** What in six small steps did the damage? The weights hardly moved in
      the smoke test, but something else did: every batch-norm layer keeps a stored average and
      spread of what it normally sees, and in training mode it rewrites them from each
      batch (new = 0.9 × old + 0.1 × batch), frozen layers included. After the smoke test's 43
      batches of two F01 clips, about 1% of the base model's values was left
      (0.9 multiplied by itself 43 times is 0.011). `--freeze-batchnorm` keeps them at
      the base model's values. The batch-norm diagnosis is the smoke test again with that one change, so the two runs
      differ in one thing only. Prediction first, in `reports/setup-f01-batch-norm/notes.md`: its WER
      against the smoke test's 93.48% and stock's 73.91%, and what it says on silence.
      ```bash
      python scripts/asr/train.py --export data/public/torgo/export/F01 --out models/b6-f01 --epochs 1 --freeze-encoder-layers 16 --batch 2 --grad-accum 8 --warmup-steps 10 --freeze-batchnorm
      python scripts/asr/export.py --nemo models/b6-f01/parakeet-ft.nemo --out models/b6-f01/onnx-int8
      python scripts/asr/check.py --model-dir models/b6-f01/onnx-int8
      python scripts/asr/evaluate.py --manifest data/public/torgo/export/F01/test.jsonl --audio-root data/public/torgo/export/F01 --model-dir models/b6-f01/onnx-int8 --out reports/b6-f01
      python scripts/asr/ci.py --manifest data/public/torgo/export/F01/test.jsonl --stock reports/b5-stock-export --finetuned reports/b6-f01
      ```
      Readings: close to stock and silent on silence, so the batch-norm statistics
      were the damage and `--freeze-batchnorm` becomes the default for the first F-2 run.
      Still near 93%: the statistics were not it, and the learning rate gets the next look.
      *Done 24 Sep.* The batch-norm diagnosis 73.91% WER, identical to stock on every clip (ci.py Δ 0.00,
      CI 0.00 to 0.00), and silent on silence. The batch-norm statistics were the whole
      of the smoke test's damage. Frozen batch norm is now `train.py`'s default
      (`--no-freeze-batchnorm` reproduces the smoke test). Notes in `reports/setup-f01-batch-norm/notes.md`.

## Weeks 3–7: the experiments (prediction → command → `notes.md` → table)

The reasoning the examiner will ask about, why prompt-disjoint, why one
microphone, why freeze the bottom of the encoder, why a typical-speech check -
is [research/finetune-torgo.md](research/finetune-torgo.md). Read it once.

- [x] **F-2 · M02 · top 8 blocks · batch norm updating** (week 3, for the storyboard). One severe speaker: earlier
      sessions in, last session out; the two-session severe speaker with the most
      data (the TORGO fetch told you: M02, 156 training clips, 37 test clips). Stock and fine-tuned
      on the same `test.jsonl`. Prediction first, in `reports/f2-m02-top8-bn-updating/notes.md`:
      M02's stock WER on these 37 clips (T-1 had M02 at 45.99% on its sentences),
      the fine-tuned WER, and whether ci.py's interval on the difference clears zero.
      This is the first run long enough to learn anything: about 10 epochs of
      about 10 steps, a few minutes on the 4060.
      ```bash
      python scripts/asr/train.py --export data/public/torgo/export/M02 --out models/parakeet-ft-M02 --freeze-encoder-layers 16 --batch 2 --grad-accum 8 --warmup-steps 10
      python scripts/asr/export.py --nemo models/parakeet-ft-M02/parakeet-ft.nemo --out models/parakeet-ft-M02/onnx-int8
      python scripts/asr/check.py --model-dir models/parakeet-ft-M02/onnx-int8
      python scripts/asr/evaluate.py --manifest data/public/torgo/export/M02/test.jsonl --audio-root data/public/torgo/export/M02 --group-by kind --out reports/finetune-torgo/M02-stock
      python scripts/asr/evaluate.py --manifest data/public/torgo/export/M02/test.jsonl --audio-root data/public/torgo/export/M02 --group-by kind --model-dir models/parakeet-ft-M02/onnx-int8 --out reports/finetune-torgo/M02
      python scripts/asr/ci.py --manifest data/public/torgo/export/M02/test.jsonl --stock reports/finetune-torgo/M02-stock --finetuned reports/finetune-torgo/M02
      ```
      *Run 24 Sep:* stock 60.00%, fine-tuned 105.26% (ci.py Δ −45.26, CI −66.33 to −22.92);
      on dev (same session as training) stock 62.32%, fine-tuned 89.86%; all splits head
      mic, so neither a microphone change nor the new session explains it. The fine-tuned
      model drops whole utterances (empty output) and garbles others.
      *Found later:* The first F-2 run ran with batch norm updating. The frozen default was committed at
      17:48, after it started (17:16), and its log has no "batch-norm statistics stay"
      line, so it repeated the smoke test's setting at full length. The checks below describe that model.
      **NeMo decode check · is it the checkpoint or its export?** NeMo decodes the checkpoint
      itself; if that is also near 90% on dev, the training made the model worse (the
      recipe needs a review); if it is near stock, the export or sherpa-onnx
      changed the fine-tuned model. Stock is decoded the same way as a
      control, and should land near its sherpa-onnx 62.32%.
      ```bash
      python scripts/asr/nemo_hyp.py --nemo models/parakeet-ft-M02/parakeet-ft.nemo --manifest data/public/torgo/export/M02/dev.jsonl --audio-root data/public/torgo/export/M02 --out reports/finetune-torgo/M02-dev-nemo.jsonl
      python scripts/asr/evaluate.py --manifest data/public/torgo/export/M02/dev.jsonl --audio-root data/public/torgo/export/M02 --hyp reports/finetune-torgo/M02-dev-nemo.jsonl --group-by kind --out reports/finetune-torgo/M02-dev-nemo
      python scripts/asr/nemo_hyp.py --stock --manifest data/public/torgo/export/M02/dev.jsonl --audio-root data/public/torgo/export/M02 --out reports/finetune-torgo/M02-dev-stock-nemo.jsonl
      python scripts/asr/evaluate.py --manifest data/public/torgo/export/M02/dev.jsonl --audio-root data/public/torgo/export/M02 --hyp reports/finetune-torgo/M02-dev-stock-nemo.jsonl --group-by kind --out reports/finetune-torgo/M02-dev-stock-nemo
      ```
      *Done 24 Sep.* Dev WER, NeMo decoding: stock 57.97%, fine-tuned 75.36%; exported:
      stock 62.32%, fine-tuned 89.86%. Two problems: the training itself lost about 17
      points, and the export lost another 14.5 on the fine-tuned model (30 deletions
      against NeMo's 12) where stock loses about 4.
      **Export-parts check · which part of the export?** An fp32 export (no int8 at all) and
      one with only the encoder in int8, both scored on dev. fp32 near 75%: int8 is the
      cause; the encoder-only row then says whether it is the encoder or the small
      decoder and joiner. fp32 near 90%: the cause is elsewhere in the export.
      ```bash
      python scripts/asr/export.py --nemo models/parakeet-ft-M02/parakeet-ft.nemo --out models/parakeet-ft-M02/onnx-fp32 --no-int8
      python scripts/asr/evaluate.py --manifest data/public/torgo/export/M02/dev.jsonl --audio-root data/public/torgo/export/M02 --group-by kind --model-dir models/parakeet-ft-M02/onnx-fp32 --out reports/finetune-torgo/M02-dev-fp32
      python scripts/asr/export.py --nemo models/parakeet-ft-M02/parakeet-ft.nemo --out models/parakeet-ft-M02/onnx-int8-enc --int8-parts encoder
      python scripts/asr/evaluate.py --manifest data/public/torgo/export/M02/dev.jsonl --audio-root data/public/torgo/export/M02 --group-by kind --model-dir models/parakeet-ft-M02/onnx-int8-enc --out reports/finetune-torgo/M02-dev-int8-enc
      ```
      *Done 24 Sep.* fp32 export 81.16% with 12 deletions, the same as NeMo; int8
      encoder only 89.86% with 28. The int8 encoder is what empties the words; the int8
      decoder and joiner cost nothing. The export fix (per-channel int8, or the eight
      fine-tuned layers left fp32) waits for a model that beats stock.
- [x] **F-2 · M02 · top 8 blocks · batch norm frozen.** F-2 on M02 again, transcripts in scoring form. One change from the first F-2 run:
      `train.py` now trains on lowercase, unpunctuated transcripts (`--normalize-targets`,
      the default), which removes the "⁇" characters and makes the dev WER that picks
      checkpoints count words instead of capitals. Scored as the checkpoint in NeMo, so
      the int8 problem is out of the comparison; the first F-2 run's checkpoint and stock are decoded
      the same way on the same 37 test clips. Prediction first, in
      `reports/f2-m02-top8-bn-frozen/notes.md`: the batch-norm-frozen run's test WER in NeMo, against stock and
      against the first F-2 run in NeMo. Turn NVIDIA Instant Replay off first.
      ```bash
      python scripts/asr/train.py --export data/public/torgo/export/M02 --out models/parakeet-ft-M02b --freeze-encoder-layers 16 --batch 2 --grad-accum 8 --warmup-steps 10
      python scripts/asr/nemo_hyp.py --nemo models/parakeet-ft-M02b/parakeet-ft.nemo --manifest data/public/torgo/export/M02/test.jsonl --audio-root data/public/torgo/export/M02 --out reports/finetune-torgo/M02b-test-nemo.jsonl
      python scripts/asr/evaluate.py --manifest data/public/torgo/export/M02/test.jsonl --audio-root data/public/torgo/export/M02 --hyp reports/finetune-torgo/M02b-test-nemo.jsonl --group-by kind --out reports/finetune-torgo/M02b-test-nemo
      python scripts/asr/nemo_hyp.py --nemo models/parakeet-ft-M02/parakeet-ft.nemo --manifest data/public/torgo/export/M02/test.jsonl --audio-root data/public/torgo/export/M02 --out reports/finetune-torgo/M02-test-nemo.jsonl
      python scripts/asr/evaluate.py --manifest data/public/torgo/export/M02/test.jsonl --audio-root data/public/torgo/export/M02 --hyp reports/finetune-torgo/M02-test-nemo.jsonl --group-by kind --out reports/finetune-torgo/M02-test-nemo
      python scripts/asr/nemo_hyp.py --stock --manifest data/public/torgo/export/M02/test.jsonl --audio-root data/public/torgo/export/M02 --out reports/finetune-torgo/M02-test-stock-nemo.jsonl
      python scripts/asr/evaluate.py --manifest data/public/torgo/export/M02/test.jsonl --audio-root data/public/torgo/export/M02 --hyp reports/finetune-torgo/M02-test-stock-nemo.jsonl --group-by kind --out reports/finetune-torgo/M02-test-stock-nemo
      python scripts/asr/ci.py --manifest data/public/torgo/export/M02/test.jsonl --stock reports/finetune-torgo/M02-test-stock-nemo --finetuned reports/finetune-torgo/M02b-test-nemo
      ```
      If the batch-norm-frozen run is still worse than stock, that is consistent with the prompt-overlap trap
      (research/finetune-torgo.md, trap 1): nine minutes of one speaker, mostly single
      words, with not one test word in training. The next test is then the first F-1 run, which trains on
      the other speakers' far larger data.
      *Done 24 Sep, but not as planned:* the transcript change was still uncommitted, so
      it ran with raw transcripts and, unlike the first F-2 run, with batch norm frozen. It is therefore
      "the first F-2 run with batch norm frozen". Test WER in NeMo: stock 65.26%, the first F-2 run 103.16%, the batch-norm-frozen run 63.16%;
      ci.py stock against the batch-norm-frozen run Δ 2.11 points (CI −3.57 to 8.96), so no measurable difference
      from stock. Notes: `reports/f2-m02-top8-bn-frozen/notes.md` (no prediction was written).
- [x] **F-2 · M02 · top 8 blocks · normalised transcripts.** Two small runs, prediction
      first for each in `reports/f2-m02-top8-normalized-transcripts/notes.md`. (1) The batch-norm-frozen run's checkpoint through
      the int8 export, scored export against export (stock exported scored 60.00%): does the
      int8 encoder still empty a healthy fine-tune's words (trap 5)? (2) The new run: the batch-norm-frozen run plus
      normalised transcripts, the one change that run was meant to test. The flags are spelled out
      so an out-of-date `train.py` stops with an error instead of running the old recipe.
      ```bash
      python scripts/asr/export.py --nemo models/parakeet-ft-M02b/parakeet-ft.nemo --out models/parakeet-ft-M02b/onnx-int8
      python scripts/asr/evaluate.py --manifest data/public/torgo/export/M02/test.jsonl --audio-root data/public/torgo/export/M02 --group-by kind --model-dir models/parakeet-ft-M02b/onnx-int8 --out reports/finetune-torgo/M02b-test-int8
      python scripts/asr/ci.py --manifest data/public/torgo/export/M02/test.jsonl --stock reports/finetune-torgo/M02-stock --finetuned reports/finetune-torgo/M02b-test-int8
      python scripts/asr/train.py --export data/public/torgo/export/M02 --out models/parakeet-ft-M02c --freeze-encoder-layers 16 --batch 2 --grad-accum 8 --warmup-steps 10 --freeze-batchnorm --normalize-targets
      python scripts/asr/nemo_hyp.py --nemo models/parakeet-ft-M02c/parakeet-ft.nemo --manifest data/public/torgo/export/M02/test.jsonl --audio-root data/public/torgo/export/M02 --out reports/finetune-torgo/M02c-test-nemo.jsonl
      python scripts/asr/evaluate.py --manifest data/public/torgo/export/M02/test.jsonl --audio-root data/public/torgo/export/M02 --hyp reports/finetune-torgo/M02c-test-nemo.jsonl --group-by kind --out reports/finetune-torgo/M02c-test-nemo
      python scripts/asr/ci.py --manifest data/public/torgo/export/M02/test.jsonl --stock reports/finetune-torgo/M02-test-stock-nemo --finetuned reports/finetune-torgo/M02c-test-nemo
      ```
      Compare like with like: exported against exported, NeMo against NeMo. On these 37
      clips the same stock model scores 60.00% exported and 65.26% in NeMo.
      *Done 24 Sep.* (1) The batch-norm-frozen run exported 56.84% against stock exported 60.00% (Δ 3.16, CI −4.26
      to 11.39): the int8 export does not hurt a healthy fine-tune, so no export fix is
      needed. (2) The normalised-transcripts run in NeMo 57.89% against stock 65.26% (Δ 7.37, CI −1.11 to 17.50) and
      the batch-norm-frozen run 63.16%: normalised transcripts helped, and the gain over stock now nearly clears
      zero. Recipe for the first F-1 run onwards: frozen batch norm, normalised transcripts.
- [x] **F-2 · M02 · top 8 blocks · normalised transcripts · int8 export.** The number the Pi would give: the normalised-transcripts run exported
      against stock exported (60.00%). Prediction first, in
      `reports/f2-m02-top8-normalized-transcripts/notes.md`.
      ```bash
      python scripts/asr/export.py --nemo models/parakeet-ft-M02c/parakeet-ft.nemo --out models/parakeet-ft-M02c/onnx-int8
      python scripts/asr/evaluate.py --manifest data/public/torgo/export/M02/test.jsonl --audio-root data/public/torgo/export/M02 --group-by kind --model-dir models/parakeet-ft-M02c/onnx-int8 --out reports/finetune-torgo/M02c-test-int8
      python scripts/asr/ci.py --manifest data/public/torgo/export/M02/test.jsonl --stock reports/finetune-torgo/M02-stock --finetuned reports/finetune-torgo/M02c-test-int8
      ```
      *Done 24 Sep.* The normalised-transcripts run exported 52.63% against stock exported 60.00% (Δ 7.37, CI −1.12
      to 17.65): the same gain as in NeMo, just short of measurable on 37 clips. Deleted
      words 8 of 95 (stock 11; the batch-norm-frozen run exported 0). This is the F-2 number for M02 on the Pi's model.
- [x] **F-1 · leave-one-speaker-out · top 8 blocks (first recipe)** (weeks 4–5), the four severe speakers.
      `export/loso-<SPK>/`, four runs, one `notes.md` (`reports/f1-loso-top8/notes.md`,
      prediction first), one table. Tests H2. About 30 minutes per speaker; keep the PC from
      sleeping and the GPU free (NVIDIA Instant Replay, Wallpaper Engine off). The loop stops
      at the first error; the last two lines score the M02-held-out model on M02's 37 F-2 test clips.
      ```bash
      ( for S in F01 M01 M02 M04; do
        python scripts/asr/train.py --export data/public/torgo/export/loso-$S --out models/parakeet-ft-loso-$S --freeze-encoder-layers 16 --batch 2 --grad-accum 8 --freeze-batchnorm --normalize-targets &&
        python scripts/asr/export.py --nemo models/parakeet-ft-loso-$S/parakeet-ft.nemo --out models/parakeet-ft-loso-$S/onnx-int8 &&
        python scripts/asr/evaluate.py --manifest data/public/torgo/export/loso-$S/test.jsonl --audio-root data/public/torgo/export/loso-$S --group-by kind --out reports/finetune-torgo/loso-$S-stock &&
        python scripts/asr/evaluate.py --manifest data/public/torgo/export/loso-$S/test.jsonl --audio-root data/public/torgo/export/loso-$S --group-by kind --model-dir models/parakeet-ft-loso-$S/onnx-int8 --out reports/finetune-torgo/loso-$S &&
        python scripts/asr/ci.py --manifest data/public/torgo/export/loso-$S/test.jsonl --stock reports/finetune-torgo/loso-$S-stock --finetuned reports/finetune-torgo/loso-$S || exit 1
      done ) &&
      python scripts/asr/evaluate.py --manifest data/public/torgo/export/M02/test.jsonl --audio-root data/public/torgo/export/M02 --group-by kind --model-dir models/parakeet-ft-loso-M02/onnx-int8 --out reports/finetune-torgo/loso-M02-on-F2-test &&
      python scripts/asr/ci.py --manifest data/public/torgo/export/M02/test.jsonl --stock reports/finetune-torgo/M02-stock --finetuned reports/finetune-torgo/loso-M02-on-F2-test
      ```
      *Done 24 Sep.* Stock exported against fine-tuned exported: F01 73.91% → 67.39% (Δ 6.52,
      CI −3.92 to 30.43), M01 70.05% → 62.32% (Δ 7.73, CI 1.60 to 14.75), M02 59.52% → 51.43%
      (Δ 8.10, CI 1.84 to 16.13), M04 91.63% → 74.38% (Δ 17.24, CI 7.52 to 27.57). All four
      improve and three intervals clear zero; F01 (23 clips) does not. Most of the gain is on
      single words, and inserted words fall (M04 45 → 13). The M02-held-out model on M02's 37 F-2 test
      clips scores 52.63% (Δ 7.37, CI 0.00 to 17.65): 50 of 95 word errors, the same count as
      the normalised-transcripts run. Numbers and prediction against actual in `reports/f1-loso-top8/notes.md`;
      the conclusions there are still to write.
- [x] **F-2 · M02 · bottom 8 blocks · which parts to train (two arms).** The earlier top-8 runs trained the
      top eight encoder blocks; the disordered-speech papers train the lower encoder. The whole-blocks model
      trains all of blocks 0-7; the attention-and-conv model trains only `self_attn` and `conv` inside them. Same data,
      seed and recipe as the normalised-transcripts run otherwise (`--train-parts`, trainable counts 210461318 and 76161670,
      exact). Predictions, results and conclusions in `reports/f2-m02-bottom8-parts/notes.md`.
      *Done 27 Sep.* The whole-blocks model Int8 47.37% against stock 60.00% (Delta 12.63, CI 4.88 to 23.81);
      the attention-and-conv model 57.89% (Delta 2.11, CI -9.18 to 12.00). Training's dev score counted capitals and
      punctuation, so checkpoints were picked partly on formatting; fixed in train.py
      (`--dev-scoring evaluate` and `--keep-stock`, both default).
- [x] **F-2 · M02 · bottom 8 blocks · which parts, fixed dev score.** Both models rerun (the failed-fix repeat ran with the
      first, non-working fix). Stock dev now scores 57.97%, matching evaluate.py. *Done 27 Sep.*
      The bottom-8 model Int8 44.21% against stock 60.00% (Delta 15.79, CI 6.45 to 28.09); the bottom-8 attention-and-conv model 62.11%
      (Delta -2.11, CI -14.29 to 6.02); the bottom-8 model beats the bottom-8 attention-and-conv model by 17.89 (CI 7.04 to 32.57). Notes:
      `reports/f2-m02-bottom8-parts/notes.md`.
- [x] **F-2 · M02 · top 8 blocks · fixed dev score.** The bottom-8 model beat the normalised-transcripts run in int8 by 8.42 points
      (CI 1.05 to 17.05), but the normalised-transcripts run was picked with the old score. One run of that recipe
      (`--freeze-encoder-layers 16`) with `--dev-scoring evaluate --keep-stock` makes it like for
      like. Original prediction, commands, and recorded 29 Sep result:
      `reports/f2-m02-top8-fixed-dev/notes.md`.
- [x] **F-2 · M01, M02, M04 · bottom 8 blocks · fixed 5 epochs.** F-2 on the two-session severe speakers, with the bottom-8 recipe. M01 and M04
      (earlier sessions in, last session out), and M02 again. F01 has one session, so by the
      proposal (4.1) it is not an F-2 speaker; the command block runs it, and its row is context
      only. M01's dev set is too small to pick a checkpoint on, so every run here trains a fixed
      5 epochs and averages the last 4 (`--select last`); the rule and why, the commands and the
      result table are in `reports/f2-severe-bottom8-5epochs/notes.md`. M02 (fixed 5 epochs) against the bottom-8 model shows what the
      fixed rule costs.
- [x] **F-1 · leave-one-speaker-out · bottom 8 blocks (current recipe).** H3 is F-2 against F-1, and the first F-1 run trained the top
      eight blocks with the old dev score (the top-8 rerun: top eight gains about nothing on M02). This reruns
      the four leave-one-speaker-out models with the bottom-8 recipe, about 2 hours on Yack, after the F-2 speaker runs.
      Commands and recorded 29 Sep table: `reports/f1-loso-bottom8/notes.md`.
- [x] **H3 · F-2 against F-1 on the same clips.** After the F-1 rerun: each of its models scored on its
      speaker's F-2 test clips, against that speaker's model from the F-2 speaker runs. Section "F-2 against F-1" in
      `reports/f2-severe-bottom8-5epochs/notes.md`. Recorded 30 Sep: F-1 beats F-2 on M01/M04; M02's interval includes zero. This tests H3 at the available data and training budget.
- [x] **Typical-speech check.** Every fine-tuned model on
      `export/controls/test.jsonl`. One column. A regression of more than a few
      points is that model's headline, not a footnote. After the F-2 speaker runs and the F-1 rerun; commands and table in
      `reports/typical-speech-check/notes.md`. Recorded 30 Sep: no model worse than stock (12.40%);
      F-1 models 2.3-2.7 points better, F-2 models 0-1 point. Conclusions still to write.
- [x] **Prototype model · all eight dysarthric speakers · bottom 8 blocks.** The model the
      prototype runs (and F-4 and the P-tests use): the F-1 bottom-8 recipe with nobody held out.
      Prediction first, then one block of commands; scored on the controls and every dysarthric
      speaker's test clips, against stock and F-1. `reports/prototype-all-dysarthric-bottom8/notes.md`.
- [x] **Prototype model, sentences only · all eight dysarthric speakers · bottom 8 blocks.** Your
      idea: the same run with every single-word recording removed (`train.py --kinds sentence`),
      compared with the prototype model on sentence WER, single words and typical speech. After
      the prototype run. Prediction first. `reports/prototype-all-dysarthric-sentences-bottom8/notes.md`.
- [x] **Prototype model, severe focus · all eight dysarthric speakers · bottom 8 blocks.** Aimed at
      F01, M01, M02, M04: checkpoints picked on the severe speakers' dev recordings, their training
      recordings used twice per epoch, learning rate 5e-5. Goal: the severe speakers' test sentences
      (prototype: 41.79%); guard: mild speakers and typical speech. Prediction first.
      `reports/prototype-severe-focus-bottom8/notes.md`.
- [x] **Prototype model, severe focus, repeated with the optimizer in RAM.** The same run with
      `--offload-optimizer` added (1.66 GB less GPU memory in the 30-update check). Answers two
      things: whether every epoch stays fast, and how far two runs of one recipe land apart.
      Prediction first. `reports/prototype-severe-focus-bottom8-offload/notes.md`.
- [x] **F-2 · M02 · adapter after 8 blocks.** Instead of retraining blocks 0-7, train one small
      layer (1,049,600 numbers) placed after block 7 (the last of blocks 0-7) and leave every other weight alone. Scored on
      M02's 37 test clips against stock (65.26% in NeMo) and the bottom-8 model (44.21%). Write
      your four prediction sentences first, then run the commands in the notes, then fill in
      Results and Conclusions. In the first minute of the training log look for `1049600 trainable`
      and `stock dev WER 0.5797`; if either is missing, stop and check the setup before going on.
      `reports/f2-m02-adapter8/notes.md`.
- [x] **F-2 · M02 · adapter after 8 blocks · seeds 18 and 19.** The first adapter run's recipe twice
      more with only the seed changed, to see how far the adapter's score moves between runs before
      the 4.21-point gap to the bottom-8 model is read as real. Your prediction is already in the
      notes; run the commands there, then fill in Results and Conclusions.
      `reports/f2-m02-adapter8-seeds/notes.md`.
- [x] **F-2 · M01 and M04 · adapter after 8 blocks.** The M02 adapter on the two other severe
      speakers, 15 epochs on a fixed schedule (about the same number of training steps as M02's
      adapter, not the same number of epochs), scored in NeMo against stock, the F-2 bottom-8
      models and the F-1 models decoded in NeMo on the same clips. Your prediction is already in
      the notes; run the commands there, check the three log lines they list, then fill in
      Results and Conclusions. `reports/f2-m01-m04-adapter8/notes.md`.
- [x] **Check · cleaned-up train.py · reruns of recorded results.** Rerun three recorded results
      on the commit you pull: the M01 and M04 adapters (seeds 17, 18, 19), the M02 adapter with the
      default recipe, and the M02 whole-block bottom-8 model, each against a pass band written
      before the run. The prediction is already in the notes (entered for you, because these are
      reruns). Pull once, run the commands there without pulling again, paste the output, then
      fill in Results and Conclusions. `reports/check-train-py-cleanup/notes.md`.
- [x] **F-2 · M01 and M04 · adapter after 8 blocks, on the F-1 model.** The same adapter and
      recipe as the M01 and M04 adapter runs, trained on top of each speaker's F-1 model instead
      of the stock model, scored in NeMo against stock, the adapter on stock and the F-1 model
      alone. Your prediction is already in the notes; run the commands there, check the log
      lines they list (including that the log's base model is the F-1 file), then fill in
      Results and Conclusions. `reports/f2-m01-m04-adapter8-on-f1/notes.md`.
- [x] **F-2 · M01 and M04 · adapter on the F-1 model · seeds 18 and 19.** Repeat the adapter on
      F-1 run with two more seeds per speaker, to measure its spread and see whether M04's lead
      over the adapter on stock holds, and re-score the seed-17 comparisons with each prompt's
      takes kept together. Your prediction is already in the notes; run the one script there,
      check the log lines it lists (including that each log's base model is the F-1 file),
      paste the output or attach the file it saves, then fill in Results and Conclusions.
      `reports/f2-m01-m04-adapter8-on-f1-seeds/notes.md`.
- [x] **F-2 · M02 · adapter on the F-1 model · seeds 17, 18 and 19.** The adapter trained on
      top of M02's F-1 model with M02's own recipe (the dev set picks the checkpoints), three
      seeds, scored in NeMo against stock, the adapter on stock (46 wrong words on every run on
      record), the F-1 model alone and the whole-block model. My pre-run predictions are
      recorded in the notes. *Completed 1 Oct:* 49, 45 and 42 word errors
      out of 95; mean 47.72% against 48.42% for the adapter on stock. No same-seed
      interval establishes a difference. Results and predictions are recorded in
      `reports/f2-m02-adapter8-on-f1-seeds/notes.md`.
- [x] **F-2 · adapter position sweep · M01, M02 and M04.** The adapter after 4, 8 (M02 only; M01
      and M04 reuse their results), 12, 16 and 24 blocks on stock, fixed 15-epoch schedule,
      seed 17, 13 new runs, about 40 minutes of GPU. The whole curve is reported and no position
      is picked by its test score. My pre-run predictions are recorded in the notes. *Completed 1 Oct:*
      block 8 was lowest on all three speakers (pooled 201 wrong of 425; the other positions
      245 to 291). M02's block 8 under the fixed schedule scored 38, against 46 under its
      default recipe. Size (a narrow or multi-position adapter) needs new code and is not in this test.
      `reports/f2-adapter-position-sweep/notes.md`.
- [x] **Phrase matching on the saved hypotheses · M01, M02 and M04.** Replace each saved sentence
      hypothesis (stock, F-1, adapter on stock, adapter on F-1, whole-block) with the nearest phrase
      from a random phrasebook of 10, 25, 50, 100 or all prompts, with the true phrase kept
      (closed) and removed (open). CPU only, a few minutes. My pre-run predictions are
      recorded in the notes. *Completed 1 Oct:* adapter on stock 3.0% spoken WER with 342 phrases and the true phrase
      on the list, stock 14.9%; with the true phrase removed 79 to 89%. `reports/phrasebook-matching/notes.md`,
      `scripts/asr/phrasebook.py`.
- [x] **F-2 · adapter block 8 and its neighbours · seeds and blocks 6 and 10.** M02's block-8
      adapter on seeds 18 and 19 (fixed schedule), blocks 6 and 10 on all three speakers, then
      seeds 18 and 19 for blocks 4, 6, 8, 10, 12, 16 and 24 on all three speakers: 48 runs,
      about 2 hours, run overnight on 1 Oct. Three seeds per position, the best position decided
      by the three-seed mean. My pre-run predictions are recorded in the notes. *Completed 2 Oct:* block 8
      has the lowest mean (206 of 425) but block 6 is 9 words behind (not shown); the other
      positions are 45 to 88 words behind. M02's block-8 adapter: 38, 44, 42 against 46. M02's
      seeds are not shown below the default recipe by clip (intervals include zero).
      `reports/f2-adapter-position-seeds/notes.md`.
- [x] **Phrase matching · a rejection rule.** Keep the nearest phrase only if its distance is at
      most t, otherwise keep the raw text, for t from 0.05 to 0.50, with the true phrase on and
      off the list and at 100, 90, 80 and 50% coverage. CPU, seconds, in the same overnight run.
      *Completed 2 Oct:* with the true phrase off the list the rule keeps the spoken WER at the raw score,
      but up to t = 0.50 it also gives away most of the closed gain (adapter on stock 12.5% against
      3.0%); the grid ended at its best cell. `reports/phrasebook-matching/notes.md` (follow-up section).
- [x] **Phrase matching · a wider rejection grid.** The same script, t from 0.40 to 0.80, seconds on
      the CPU. *Completed 2 Oct:* the rule pays when the list covers under about 90 to 95% of what is
      said (adapter on stock, 80% coverage: 12.1% at t = 0.65 against 18.1% with no rule and 37.0% raw;
      50%: 21.8% at t = 0.55 against 40.8%), and costs when the list is complete (3.0% with no rule).
      The best cell is within noise of its neighbours; no threshold is recommended.
      `reports/phrasebook-matching/notes.md`.
- [x] **Prototype target · the 2 Oct run set.** My bar: nothing above 35% goes to the prototype
      expo. Five tests, stages of `lab.py`, set up 2 Oct and not run. The notes
      contain the pre-run hypotheses. W: the matched output on single words
      and on the whole test set (CPU, a minute; `reports/phrasebook-matching/notes.md`). B: other
      base models with the adapter, now including a 110M hybrid as the small-device test
      (`reports/f2-base-model-ceiling/notes.md`). A: augmentation variants that change what every
      run already does, and N: tempo stretch and white noise added on top
      (`reports/f2-augmentation-variants/notes.md`). F: the data curve (next item). About 6 hours
      in all; any list of stages can be given instead (`"A N"`). `train.py` changed (new flags), so
      commit and push it with the script, then on Yack:
      `python lab.py run W B A N F`;
      paste `logs/big-$S-summaries.txt` back. Whether the 35% applies to the raw model output or to
      what the device outputs with its list was asked and not settled; W measures the second, B, A and F the first.
      *Update 14 Oct:* W, A, N and F ran in the 13 to 14 Oct queue (A and N: no variant beats the
      default; F: see the data-quantity item). B has not run. W (read 15 Oct): with the true phrase on a 462-phrase list
      the adapter on F-1 says 18.8% wrong (sentences 2.6%, single words 59.0%) against 46.4% raw; see
      `reports/phrasebook-matching/notes.md`.
      *Update 21 Oct:* B has still never trained. The 16 Oct queue counted it as done because its
      completion marker was a stock decode that lab test 4 had already written, and B also refused to
      start when those stock decodes existed. Both are fixed in `lab.py`; run `python lab.py run B`
      (about 2 h). The stock rows already show 110M at 315 and 1.1B at 307 against v2's 323.
      *Done 27 Oct:* B trained in queue `1027-1530`: no new base beats v2's adapter (1.1B with the adapter after
      block 14 203, v3 251, 110M 249 to 251, against 208); see `reports/f2-base-model-ceiling/notes.md`.
- [x] **Follow-up queue · 15 Oct · "do all of them just to see".** One command, `python lab.py queue followup`:
      lab test 24 (acoustic phrase scoring against the phrase list, thresholds picked on the other speakers,
      coverage of the test prompts: `reports/lab-24-phrase-scoring/notes.md`), P again (with whole-block from
      the pretrained models), G2 (G's runs plus a third seed and two arms further out), the unrun lab tests
      16 to 23 and B (B is skipped if its marker already exists). No predictions were written for any of
      them; each note says so. L, T, E, S, C (261 runs) are unrun and not in this queue. Lab test 21 needs
      `sudo apt install espeak-ng`.
      *Update 20 Oct:* the `1016-retry` queue (16 Oct) finished 24, 20, 23 and 22; 19 ran but its
      per-channel arm crashed in the weight check (fixed; score the six exports with `evaluate.py
      --model-dir`). Results are in each note: TTA 179 of 425 (lab 22), selector worse than the
      single system (23), F-1 base established on held-out prompts (20), `--int8-ops matmul` costs
      nothing (19), 26 to 32% with a list under cross-speaker thresholds (24). Labs 16 to 18 still
      need the TORGO control speakers (`python lab.py doctor --fix --allow-download`).
      *Done 27 Oct:* labs 16 to 18 and lab 19's per-channel arm ran in queue `1027-1530`; results are in each note.
- [x] **Every unfinished lab · one command (21 Oct).** After committing and pushing this session's changes from
      the Windows checkout, on the lab computer in `app/`: `./willspeak.sh lab-remaining` (pull, doctor with the control
      speakers' download, check with GPU smoke runs, then `lab.py queue remaining --detach`). It waits for `1020-new` if
      that is still running, skips everything already finished there, and pushes results after each experiment.
      Order and fixes: `docs/measurements.md`, **Lab tests**. No predictions are recorded for these runs.
      *Done 27 to 30 Oct:* queues `1027-1530`, `1028-rest` and `1029-night` finished every stage in the list
      (stage 5 in `1028-rest`, speaker-replay in `1029-night`); every note has its results.
- [x] **Lab test 25 · test-time adaptation on the pretrained adapter · three seeds.** New 20 Oct:
      lab 22's loop (four chunks, agreement gate, no transcripts) on stage P's `pre-ad8-lr6e-4`
      runs, seeds 17 to 19, so the two best results (178 and 179) are finally combined.
      Discuss the context and my prediction in chat, record it in
      `reports/lab-25-tta-on-pretrained/notes.md`, then `python lab.py run 25` (about 3 h; the
      pretrained models are rebuilt if their weights were cleaned).
      *Update 21 Oct:* queued on 20 Oct as `python lab.py queue 25 29 28 30 27 26 --stamp 1020-new --detach`
      without a prediction. No result has reached GitHub yet. On the lab computer, push the queue's local
      result branches (`git push origin 'refs/heads/lab-results-*:refs/heads/lab-results-*'`) or commit
      the reports and logs, then the notes can be filled. Each note now lists the numbers that would count.
      *Result 21 Oct:* mean 178.0 to 172.3 over three seeds (5.7 words, under the rule); see the note.
- [x] **Lab tests 26 to 30 · 20 Oct · combine the wins and look at the errors.** 26: the pretraining
      recipe under the adapter (bottom 8 blocks, 4e-5, longer, all dysarthric then severe; about 3 h).
      27: lab 22's loop with one setting changed at a time (8 chunks, recorded order, gate 0.3 / 0.7 /
      none, half cap; about 2 h). 28: the pretrained adapter exported to int8 both ways and decoded by
      sherpa-onnx, the file the Pi would get (30 min). 29: the hard core of errors across every saved
      system, CPU (seconds; it reads the test references after the fact, so tune nothing on it).
      30: lab 24's phrase scoring on the pretrained adapter (30 min). Each note has blank prediction
      fields except 29, which is description. Cheapest first: `python lab.py run 29 28 30`, then
      `python lab.py run 27 26`; or everything new in one queue, `python lab.py queue new --stamp
      1021-new --detach` (25 to 30 and labs 16 to 23 that still lack results).
      *Update 21 Oct:* in the same `1020-new` queue as lab test 25; results are in each note (no pretraining
      variant beats stage P; the loop's gain depends on its settings; the matmul int8 file costs 1 word;
      every clip all systems miss is a single word; behind a phrase list the pretrained adapter is no better
      than the adapter on F-1).
- [x] **F-2 · training ideas, 14 Oct · stacked levers, adapter soup, regularisation, word and sentence
      training, bigger whole-block steps.** Five new stages of `lab.py`, about 15 hours of GPU in all:
      P (pretraining plus 30 epochs and 6e-4, which each helped alone), U (average five seeds'
      adapters), R (weight decay, warmup, checkpoint averaging, trainable batch-norm), K (words only,
      sentences only) and G (whole-block at higher rate and 40 epochs). Context and my prediction first, in chat;
      record it in `reports/f2-adapter-levers/notes.md` and `reports/f2-adapter-soup/notes.md`
      before the command: `python lab.py queue ideas --stamp 1015-ideas --detach`.
      *Update 15 Oct:* U, R, K and G ran on 14 Oct (no prediction was written first); results are in the
      two notes. P stopped before training because of a leftover folder and is fixed; run it again with
      `python lab.py run P` after your prediction is in the notes.
      *Update 21 Oct:* P ran in the 15 Oct follow-up queue: severe pretraining then the adapter at 6e-4,
      178.0 of 425 wrong (41.9%), the best raw system so far. Its sentence words are at 28.3% and its
      single words at 75.7%, so single words are the gap to the bar (`reports/f2-adapter-levers/notes.md`).
- [x] **F-2 · adapter after block 8 · targets in the model's own form.** *Run 13 Oct (stage T, my
      lab run):* adapter 206.3 wrong of 425 against 213.0 for scoring form (seed ranges touch);
      whole-block 309.0 against 304.7. No established gain. Original item: Every recorded run trained
      on lowercase, unpunctuated targets; the stock model writes "Tomato." and tokenises it on a
      different path from "tomato", and in adapter mode the frozen decoder and joint have to be
      pushed into the new form through the 1,049,600-number adapter before anything about the
      speaker is learned. `train.py --target-form stock` (new) trains on the model's own form;
      dev and test scoring are unchanged. The one earlier format comparison (24 Sep, 63.16% against
      57.89% NeMo on M02) was one run each and was picked with the old dev score, so it decides
      nothing. Run: block-8 adapter, fixed 15 epochs, M01/M02/M04, seeds 17/18/19, against the
      three-seed block-8 mean of 206 wrong words of 425. Context and your prediction first, in
      chat, then the commands ([training.md, Target form](training.md#4-fine-tune)).
- [x] **F-2 · adapter after block 8 · learning rate.** *Run 13 Oct (stage L):* adapter 1e-4 238.3,
      3e-4 default 213.0, 6e-4 201.7, 1e-3 200.7 wrong; whole-block 5e-6 322.0, 1e-5 311.3, 2e-5 304.7,
      4e-5 282.7, 8e-5 267.0 (not levelled off). Original item: No learning rate has ever been swept, for
      blocks or for the adapter; 3e-4 is the rate the CPU pilot used. Adapter at 1e-4, 3e-4 and 1e-3,
      three speakers, three seeds, the choice made on the pooled severe dev clips (M01 4, M02 24,
      M04 9; `lab.py` now scores dev beside test) and test scored once for the chosen rate.
      Context and prediction first.
- [x] **F-3 · the data-quantity curve** (week 7). *Run 14 Oct (stage F):* one minute 55 to 57 wrong of
      95 (stock 62), about five minutes 44 to 46, seven minutes 43 to 48; results in
      `reports/f3-m02-data-quantity/notes.md`. Original item: M02, adapter after block 8: subsets of 1, 3, 5 and 7
      minutes, the whole 9.0-minute set and stock, two schedules (equal epochs and equal updates, which
      settles the decision this item used to wait on), three seeds, 24 runs, about 1.5 hours. Stage F
      of `lab.py`; set up 2 Oct, not run. It runs the manifest step itself (it
      rewrites M02's export, and the script stops if M02's own manifests change).
      `reports/f3-m02-data-quantity/notes.md`.
- [x] **F-2 · long adapter training.** New 9 Oct: 15/75/200 epochs, stock versus
      scoring-form targets, and learning rates 1e-4/3e-4/1e-3. Run
      `bash scripts/asr/long_adapter.sh <stamp>` on the lab computer.
      [Plan and expectations](../app/reports/f2-long-adapter/notes.md).
      Seed-17 pooled dev selects one recipe; it and the control are confirmed
      at seeds 18/19 before the test report. *Run 12 Oct:* the dev-picked stock-200 recipe averaged
      222.7 wrong against 213 for the control; no seed separates them.
- [x] **F-2 · acoustic adaptation with the text head frozen.** New 9 Oct:
      bottom 8/all 24 blocks, head frozen/trainable, two rates, 100 epochs,
      against a fresh adapter control. Run
      `bash scripts/asr/acoustic_only.sh <stamp> --offload-optimizer`.
      [Plan](../app/reports/f2-acoustic-only/notes.md). Omit offload if VRAM is
      sufficient; short GPU smoke runs precede the long experiment.
      *Update 21 Oct:* a second attempt (13-14 Oct) picked all 24 blocks with the head frozen at 1e-5
      and reached seed-17 test on M01 and M02 (84 and 36 wrong against the control's 87 and 42, and
      no words on any of the 140 non-speech clips) before a code change stopped it. To finish: train
      that recipe for M04 at seed 17 and for all three speakers at seeds 18 and 19, with the flags in
      the note, then score against the long-adapter control seeds (208, 219, 212).
      *Result 28 Oct (queue `1027-1530`, no prediction written first):* finished. All 24 blocks, head
      frozen, 1e-5: 192, 191, 192 wrong of 425 (mean 191.7) against the control's 208, 219, 212 (213.0),
      better on all three speakers at every seed; sentences only 101.0 of 303 (33.3%); no words on any
      of the 140 non-speech clips for all nine models. Still behind stage P (178.0). The note's Results
      section is not written yet; the tables are in `logs/lab-long-acoustic-1027-1530.txt`.
- [x] **Lab tests 31 to 33 · 28 Oct · the acoustic recipe stacked with adapters.** 31: the block-8
      adapter (6e-4) on each speaker's own acoustic-only model. 32: the acoustic recipe as the severe
      pretraining instead of stage P's, then stage P's adapter. 33 (my idea): one acoustic model on every
      dysarthric speaker at once (`export/pooled-M01-M02-M04`, built by `torgo_manifest.py --pooled`),
      then a per-speaker adapter after blocks 4, 8, 12, 16 and 24. Three adapter seeds each; every note
      has the reference numbers, the decision rule and a blank prediction. Discuss the context and my
      predictions in chat and record them in the three notes, then on the lab computer
      `python lab.py queue acoustic --detach` (about 8.5 hours estimated; or `python lab.py run 31`
      alone, about 2 hours). Settings: `acoustic_severe_epochs`, `acoustic_pooled_epochs` (30 each)
      and `pooled_positions`. Each test writes `runs.json` (per-run status, minutes, wrong words, free
      disk) and storage snapshots to `logs/lab-<stamp>-storage.jsonl`; `finished.json` only when
      nothing failed, so a rerun resumes.
      *Done 28-29 Oct* (queue `1029-acoustic2`, 20:19 to 02:06, no predictions written first; notes filled
      29 Oct). Wrong of 425, three-seed means: **31** 196.3 against the acoustic-only model's 191.7, no
      gain, sentences worse (112.0 against 101.0); **32** 228.3 against stage P's 178.0, a clear loss (the
      acoustic pretrained model alone, never hearing the speaker, is 234, the best such model recorded);
      **33** 210.0 (block 12) to 223.3 (block 24), every position a loss against stage P, the shared base
      alone 222; adapters after block 12 or later nearly stopped false words on the battery. Stage P stays
      the leader.
- [x] **Lab test 34 and the speaker-replay rerun · 29 Oct.** 34: stage P's adapter after blocks 8, 12, 16
      and 24 on its severe-pretrained model, three seeds, with the battery: does a deeper adapter keep
      stage P's accuracy and cut false words, as on lab 33's base? Then speaker-replay again under a new
      stamp (the budget fix is in). Write both predictions in the notes
      ([34](../app/reports/lab-34-pretrained-adapter-positions/notes.md),
      [replay](../app/reports/f2-speaker-replay/notes.md)), pull on the lab computer, then
      `python lab.py queue positions --stamp 1029-night --detach` (about 2.5 h, then about 17 h). Do not
      pull while the replay study runs.
      *Result 30 Oct (no predictions written first):* **34** block 8 on rebuilt stage P bases 174.3 (176, 172,
      175); blocks 12, 16, 24 224.7, 234.7, 250.7, all losses; battery clips with words (of 600) 98, 82, 57, 133,
      so no position halves false words and block 8 stays. **Replay** dev chose `severe50` (32 of 112), which
      scored 202.0 on test against the personal-only control's 186.7, worse at every seed and speaker;
      replay rejected. Both notes are filled.
- [x] **F-2 · continuous other-speaker replay.** New 9 Oct: personal-only,
      75/25 and 50/50 personal/other speech, and severe-only replay, matched
      100-epoch clip/update budgets. Run
      `bash scripts/asr/speaker_replay.sh <stamp>`.
      [Plan](../app/reports/f2-speaker-replay/notes.md). Preserves original
      exports and excludes held-out prompts/audio. No recognition result yet.
      *Update 28 Oct:* started in queue `1027-1530`; the logs pushed at 09:02 on 28 Oct end in the control's
      M02 seed-17 run (epoch 12 of 100), after its four smoke runs and M01. If it stopped, resume it with
      the same stamp (`python lab.py queue speaker-replay --stamp 1027-1530 --detach`).
      *Update 29 Oct:* both seed-17 searches are incomplete. `1027-1530` trained all 12 trials and four
      failed the old clip-budget check; `1028-rest` repeated five trials to the word, failed `personal75`
      M02 again, and stopped the last six because PR 8 was pulled mid-study. Dev wrong of 112 from the
      first search: control 36, `personal75` 36, `personal50` 35, `severe50` 32. A rerun needs a new stamp
      (about 7 h search, 10 h confirmation); do not pull a change to `long_runs.py` while it runs.
      *Done 30 Oct (`1029-night`):* see the lab 34 item above; replay rejected.
- [x] **Lab tests 1–15 · extension tests toward the 35% bar.** *Run 6-8 Oct; every note has its
      results and an interpretation (21 Oct). Only lab 5's causal table still needs a CPU rescore.*
      Fifteen tests, stages of `lab.py`, set up 2 Oct. Each note has the question, what each
      outcome would mean, and an empty prediction: record my prediction for each stage
      in its note before preparing the command. The nine stages that train
      nothing (`14 3 15 1 2 6 13 4 5`, about 3 hours, an estimate) are the default; the
      training stages (`7 8 8b 9 10 11 12`, about 30 hours in all) run only when named, and 8b only
      after you have read 8. The test table, scoring rule and what is still unverified are in
      [measurements §Lab tests](measurements.md#lab-tests). Commit and push, then on Yack:
      `python lab.py run 14 3 15`
      (put the stage numbers you chose inside the quotes); paste `logs/lab-$S-summaries.txt` back.
      The 35% question (raw model output or the device's output with its list) applies here too;
      only test 14 measures the second.
- [x] **Augmentation code · noise and tempo-only stretch in `train.py`.** *Completed 2 Oct:* added
      the requested augmentation controls. `--time-stretch MIN MAX`,
      `--white-noise MIN_DB MAX_DB` and `--extra-aug-prob` (default 0.5), all off by default, so
      every earlier command trains exactly as before (`build_augmentor()` returns the old speed-only
      dictionary). The run tries each added augmentation on made-up audio before the model loads.
      Also `scripts/asr/clip_levels.py`, which prints how loud the training clips are against a
      noise range. Unit tests pass, with NeMo's perturbation classes too; no training run has used
      the flags yet (stage N). Still open: `--max-steps 2` on Yack with each flag on is a cheap
      check before the long run. `docs/training.md`, `reports/f2-augmentation-variants/notes.md`.
- [x] **Script · `ci.py` in `scripts/asr/`, a script you write.** 95% bootstrap CI on ΔWER
      for the F-1/F-2 tables. Input: two report folders (stock, fine-tuned) that scored
      the same manifest; per utterance, error count from `willspeak.evaluate.align`
      and the reference length; resample utterances with replacement 2,000 times
      (`numpy.random.default_rng(0).integers(0, n, n)`), each time corpus WER for
      both (sum errors / sum reference words, never a mean of per-utterance rates)
      and the difference; the 2.5th and 97.5th percentiles are the interval. Print
      stock, fine-tuned, Δ, interval, n; write `ci.json` into the fine-tuned folder.
      Resampling utterances, not words, keeps each sentence's errors together -
      that is what makes the interval honest.
      *Done 23 Sep.* Takes `--manifest` too (references live there, not in the
      report folder); `--finetuned` is optional, so it also gives a single model's
      interval. T-1 severe: 59.59%, 95% CI 55.21% to 63.77% -60% is inside, so H1
      is neither confirmed nor rejected. Checked by matching `results.json` (0.5959)
      and by passing the same folder as stock and fine-tuned (Δ 0.00, CI 0.00 to 0.00).
      `test_ci.py` skipped by choice.
- [ ] **T-2 and T-3 · the matcher on public speakers** (week 6).
      [measurements §T-2, §T-3](measurements.md) and
      [device.md](device.md#exemplar-matcher-operation). Embedding ~4,000 clips is minutes on the
      4060, long on a CPU. Note where your T-2 table differs from the CPU pilot
      table ([research/round-3.md](research/round-3.md)) and why it might.

## The Pi (weeks 4–8)

Bring-up: [pi5.md](pi5.md#running-willspeak-on-the-raspberry-pi-5-and-measuring-it) and [pi5.md](pi5.md#the-deploy-folder).
Tables: [measurements, Raspberry Pi 5](measurements.md).

- [x] **P-A · Recogniser speed, idle**, done 8 Sep (`raspberrypi-20260908-165319`).
- [ ] **P-A loaded, rerun clean.** Now the third item of **Now** above, with the expo model.
- [ ] **P-B end-to-end latency.** Ten short, ten long; stopwatch or the page's
      status line; one sentence on the stage you would attack first.
- [ ] **P-D cold start.** Three power cycles timed to "Listening";
      `setup.sh --service` first.
- [ ] **P-E stop phrase.** Twenty stops after a sentence, twenty minutes of
      nearby conversation; miss rate, false-alarm rate, the threshold, and why the
      two errors cost different amounts.
- [ ] **P-F battery.** Rerun until the bank has dropped 20 points or more;
      report the bracket, not a point.
- [ ] **F-4 on the Pi.** Now the second item of **Now** above. Score a fine-tuned int8 model on the same F-2 test clips as desktop int8 and compare WER within rounding. Record provenance; the speed/memory benchmark is P-A.
      Adapters now export (`export.py --stock --adapter <adapter.pt>`); read the int8 weight check the
      export prints before trusting the folder, and score the export on the F-2 clips on Yack first
      ([training.md, Export](training.md#5-export-for-deployment)). Install it as the boot service's
      model with `setup.sh --service --model-dir <folder>`.
- [ ] **Matcher on the Pi, a decision.** `setup.sh --with-matcher`, time one
      bank rebuild and one live match. Either it fits the latency budget and the
      demo runs both paths, or the demo runs Parakeet plus your text stop-phrase
      detector and the poster says so. `reports/bench-pi5-matcher/notes.md`.
- [ ] **The self-tracking gate, code you write.** Background §1–§3 and the
      spec §5 of [design.md](design.md#other-voices-in-the-room): three rules (open
      threshold, learn from what was spoken, the leak) and one plain sentence for
      the status line. `vadTick`, `beginUtterance` and the confirm path in
      `static/live.html`, about sixty lines; `qc.speech_dbfs` / `qc.noise_dbfs` are
      already in the transcribe response. Prediction first: the margin below his
      level you expect to need, and why 12 dB is the starting guess. Ask for a
      review when it is green; the code is yours.
      *Built 27 Oct:* the three rules, the status sentence, Not him and New session, and the threshold in the
      server's `live clip` line (`gate=`), with tests; [design.md §5](design.md#5-the-self-tracking-gate-implemented-27-oct-not-yet-measured)
      lists the choices made. Not yet run with a real microphone. Still open: my prediction for the margin,
      then P-G below measures it.
- [ ] **P-G other voices in the room.** Your 11 Sep row first, then the
      two-day, three-level table with the self-tracking gate's tracker, a second person talking at a
      metre, the lavalier at the collar. [measurements §P-G](measurements.md).
      One sentence on collar versus mount.
- [x] **H5, the identity gate, a decision.** Read
      [design.md §6–§7](design.md#other-voices-in-the-room). This semester (implement the
      model plumbing and decision rule, then run T-4; F-3's middle points are the cut) or
      spring (the poster's limitation panel says so).
      *Decided 27 Oct: spring. Too much for this semester.* The poster's limitation panel says
      that someone talking close to the microphone can still be repeated in his voice; T-4 and the
      identity gate move to the spring.

## Storyboard and expo (done)

- [x] **Storyboard** presented Mon Sep 28; PDF submitted on Gradescope.
- [x] **Prototype Expo** Tue Nov 3, with the Compare demo on the Pi; Canvas write-up of other stations done.

## Client phase (after IRB approval: nothing here runs this semester)

- [ ] CITI human-subjects training; a protocol with the professor as PI; consent
      stored before the first recording.
- [ ] Recording sessions per [plan.md](plan.md#recording-protocol); family voices
      for the projection; the personalised fine-tune with the TORGO recipe; his
      take and session curves.
- [ ] Decide whether open-set rejection is needed at all with `vocab_override`
      off; the "how is he today" policy, design, measure, then enable.
