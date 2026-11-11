# Parakeet model decision and technical review

Paths and commands below are relative to `app/` unless they refer to documentation.

Last reviewed: 1 September 2026.

Status: accepted architecture decision. This is the up-to-date model review
for sentence recognition. It separates published results from ones that still
need measuring on this project's corpus.

## Decision

WillSpeak standardizes on **NVIDIA Parakeet-TDT-0.6B v2** for English sentence
recognition. Deployment uses the sherpa-onnx int8 transducer export on CPU.
There is no alternate sentence-recognition backend in the repository.

The codebase reflects the decision directly:

- `src/willspeak/asr/parakeet.py` is the only sentence-ASR adapter;
- `scripts/fetch_models.py` installs the supported runtime graph;
- `scripts/asr/check.py` loads and decodes with that graph;
- `scripts/asr/train.py` is the only sentence-model training path;
- `scripts/asr/export.py` is the only deployment export path;
- `scripts/asr/evaluate.py` evaluates Parakeet or imported hypotheses.

This decision does not remove the exemplar matcher. Parakeet answers an
open-vocabulary question: "what sentence was spoken?" The matcher answers a
closed-vocabulary question: "which enrolled word or phrase is closest?" Those
are different tasks with different failure modes.

## Model anatomy

The official v2 model card describes a 600-million-parameter English model
built from a FastConformer encoder and a Token-and-Duration Transducer (TDT)
decoder. It consumes mono 16 kHz audio and emits text with punctuation and
capitalization. NVIDIA reports that the original NeMo model can also produce
word, segment, and character timestamps. The current sherpa-onnx adapter in
WillSpeak uses text output only.

FastConformer combines convolutional local modeling with self-attention over
the acoustic sequence. The transducer then joins an acoustic encoder state
with a prediction-network state to score the next token. TDT extends a regular
RNN-T-style transducer with token-duration predictions, allowing decoding to
advance by more than one encoder step. The practical consequences for WillSpeak
are:

1. Recognition is not conditioned on a free-form text-generation prompt.
2. Short utterances do not require a fixed 30-second input window.
3. The model can be exported as separate encoder, decoder, and joiner graphs.
4. Greedy transducer decoding is available in a small CPU runtime.
5. Decoder-level contextual biasing is possible in principle without treating
   a word list as preceding transcript text (sherpa-onnx needs a BPE
   vocabulary file for it, which the stock int8 release does not ship; the
   project does not use biasing).

The upstream model was trained on about 120,000 hours of English speech,
including human-transcribed and pseudo-labeled material, then fine-tuned on a
smaller high-quality subset. That scale explains the useful general acoustic
prior; it does not establish accessibility performance for a particular
speaker. See the
[NVIDIA v2 model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
for the architecture, training-data summary, license, and typical-speech
benchmarks.

## Why this model family is relevant to dysarthric speech

The strongest direct evidence is the Interspeech 2025 Speech Accessibility
Project Challenge. The challenge used more than 400 hours from more than 500
people with diverse speech disabilities and evaluated systems remotely with
WER and a semantic score. The winning submission fine-tuned a larger
Parakeet-TDT model and reported 8.11% WER, compared with the organizers'
17.82% baseline. The challenge report says 12 of 22 valid teams beat that WER
baseline.

The winning result is evidence for the architecture and training recipe, not a
performance promise for willspeak. Important differences include:

- the winning system used a 1.1B model, while WillSpeak standardizes on 0.6B v2;
- it trained on a large multi-speaker accessibility corpus;
- it used data expansion, augmentation, checkpoint merging, and tuned
  decoding;
- the challenge aggregate does not predict one speaker's WER;
- WillSpeak deploys a quantized CPU graph and initially has far less personal
  data.

Primary sources:

- [Takahashi et al., Fine-tuning Parakeet-TDT for Dysarthric Speech Recognition](https://www.isca-archive.org/interspeech_2025/takahashi25_interspeech.html)
- [Zheng et al., The Interspeech 2025 Speech Accessibility Project Challenge](https://arxiv.org/abs/2507.22047)

The broader personalization literature points the same way but also limits
expectations. Personalized models can make large relative gains, yet
severity, domain, microphone, and recording volume remain dominant. Google
reported strong short-phrase results across hundreds of personalized models,
while also identifying cerebral-palsy dysarthria as particularly difficult.
Conversational speech remains harder than prompted phrases. Those findings
support a personalized sentence model plus a constrained phrase path rather
than one universal recognition mode.

## Why v2 instead of v3 today

NVIDIA now publishes the multilingual Parakeet-TDT-0.6B v3 model. It is a
credible successor, not a second supported backend. WillSpeak remains on v2
because the product is currently English-only and these artifacts are aligned
to v2:

- NeMo fine-tuning base and tokenizer assumptions;
- sherpa-onnx archive URL and file layout;
- packaged runtime directory name;
- model-check and evaluation scripts;
- any stored local baselines.

Changing the base model without a same-split comparison would reset the
evidence chain. A v3 migration requires:

1. Pin the exact v3 checkpoint and sherpa-onnx archive.
2. Review its license, tokenizer, supported languages, and output
   normalization.
3. Load and decode the graph on every target CPU.
4. Run stock v2 and v3 on the identical sticky test split.
5. Repeat with equivalent personalization data and stopping rules.
6. Compare WER, critical-phrase errors, insertions, latency and peak memory.
7. Replace v2 in one explicit change and update every path and document.

References for that future decision:

- [NVIDIA Parakeet-TDT-0.6B v3 model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- [Parakeet multilingual technical report](https://arxiv.org/abs/2509.14128)

## Runtime representation

The deployed model directory is a single coherent export:

```text
models/parakeet-tdt-0.6b-v2-int8/
  encoder.int8.onnx
  decoder.int8.onnx
  joiner.int8.onnx
  tokens.txt
```

`ParakeetASR` prefers int8 graphs and falls back to matching fp32 filenames if
they exist. It passes the following contract to sherpa-onnx:

| Setting | WillSpeak value | Reason |
|---|---|---|
| sample rate | 16,000 Hz | upstream model input |
| feature dimension | 128 | v2 upstream exporter/preprocessor contract |
| model type | `nemo_transducer` | sherpa-onnx loader selection |
| provider | CPU | deployment baseline |
| decoding | greedy search | deterministic, low-overhead baseline |

The runtime is lazy and thread-safe. Importing the server does not load a
600-million-parameter graph; the first transcription does. The same recognizer
is then reused for every request.

The supported sherpa-onnx file layout and `nemo_transducer` configuration are
documented in the
[sherpa-onnx NeMo transducer guide](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/nemo-transducer-models.html).

## Export pipeline

`scripts/asr/export.py` performs four jobs:

1. Restore a local `.nemo` checkpoint or download the stock v2 checkpoint.
2. Export encoder, decoder, and joiner modules to ONNX.
3. Write token and SentencePiece vocabulary files.
4. Apply dynamic int8 quantization and attach metadata expected by
   sherpa-onnx.

The encoder uses unsigned int8 weights in the current script; decoder and
joiner use signed int8. The intermediate fp32 graphs are removed after a
successful quantization because keeping both silently doubles model
storage, including external fp32 weight files. Export requires sherpa-onnx and
ends with a real adapter smoke test. If that decode fails, the destination
directory is not published, and an existing destination is refused.

Never mix files between checkpoints: token order and joiner output dimensions
have to match. Four correctly named files do not make a valid
directory, which is why `scripts/asr/check.py` loads
the graph and performs a decode.

## Fine-tuning design

Status (1 Oct 2026): TORGO whole-block and adapter experiments have run on
the 4060. [Report Context](../../app/reports/Report%20Context.md) maps their
measured results; the prototype model is still undecided. The current
whole-block recipe trains encoder blocks 0–7, keeps batch-norm statistics
fixed and normalizes transcripts. Adapter runs freeze the base and currently
have no Pi export. [finetune-torgo.md](finetune-torgo.md) preserves the original
rationale; [training.md](../training.md#4-fine-tune) describes current controls.

`scripts/asr/train.py` deliberately keeps the training policy small and
auditable:

- input is only `train.jsonl` and `dev.jsonl` from a corpus export;
- `test.jsonl` is never opened by the training script;
- the default is full-model fine-tuning from the v2 checkpoint;
- AdamW, cosine annealing, gradient clipping, and development WER checkpoint
  selection are explicit;
- a fixed seed is recorded;
- a two-phase curriculum can expose good or unknown-condition sessions first
  and all accepted sessions second;
- the final `.nemo` file and `run.json` are saved together, including configuration,
  train/dev input hashes, runtime versions, and the completed checkpoint hash.

The curriculum only changes the order of difficult days; it does not exclude
them. A system that works only on good recording days is not a successful
personalized system.

NeMo's current ASR documentation describes manifest-based train/validation
configuration and pretrained-model fine-tuning:
[NeMo ASR configuration guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/configs.html).

### Training hypotheses to test

These are hypotheses, not established project results:

| Question | Controlled comparison | Primary metric |
|---|---|---|
| Does personalization help? | stock v2 vs personalized v2 | sticky-test WER |
| Does curriculum help? | same seed/data/epochs, curriculum off vs on | condition-stratified WER |
| Is full fine-tuning justified? | full vs a documented parameter-efficient run | test WER and stability |
| Does more data still help? | nested 20%, 40%, 70%, 100% train subsets | data-quantity curve |
| Do corrections transfer? | read-only vs read-plus-corrections training | read and correction test sets |
| Does quantization regress accuracy? | NeMo checkpoint vs int8 export | paired utterance errors |

All comparisons must hold the prompt-level split, normalizer, decode settings,
and aggregation code fixed. The test split should be scored only for a named
comparison, not after every training tweak.

## Contextual biasing (not used)

sherpa-onnx can bias a transducer toward a word list with modified beam
search, but only with a BPE vocabulary file the stock int8 release does not
include. WillSpeak previously carried the switch; it could never activate,
so it was removed during code cleanup. If a future
export ships the vocabulary, biasing is not harmless configuration: a higher
score raises recall for personal names while pulling unrelated audio toward
those terms, so it would need functional recall, false-substitution and
critical-phrase false-activation measurements before use.

## Confidence, timestamps, and output honesty

The current sherpa-onnx adapter does not expose calibrated token or word
posterior probabilities. WillSpeak therefore does not display a model
confidence and does not synthesize one from similarity, transcript length, or
the constant token values returned by the adapter.

The adapter's `words` list is a compatibility shape, not confidence evidence.
Any future confidence UI requires a calibration study with a held-out set,
reliability plots, and an explicit mapping from model score to empirical error
probability.

The upstream NeMo model supports timestamps, but the current deployment path
does not promise them. Documentation and API consumers must describe only the
fields the deployed adapter actually supplies.

## Performance evidence and non-evidence

The upstream model card reports an average WER of 6.05% across a collection of
typical-speech English benchmarks under greedy transducer decoding. It also
reports degradation under added noise. Those figures are useful as a sanity
check for the checkpoint, but they are not a deployment target for dysarthric
speech.

The 8.11% SAP challenge result is closer to the target domain but used a
different model size, training corpus, and training system. It supports
testing and personalizing Parakeet; it does not justify claiming 8.11% for
willspeak.

The only release evidence that counts is produced locally from:

1. a frozen export and split assignment file;
2. reviewed reference transcripts;
3. the exact deployment ONNX graph;
4. the exact decoder configuration;
5. a report stratified by prompt type and session condition;
6. target-device latency measurements.

## Known constraints and risks

- The stock model was trained mainly for general English, not one person's
  dysarthric speech.
- Full fine-tuning a 0.6B model requires a CUDA training machine and is prone
  to overfitting on a tiny or repetitive corpus.
- The deployment graph is offline utterance-batch recognition, not a token
  streaming contract.
- Dynamic int8 quantization may change accuracy and must be compared with the
  source checkpoint.
- Greedy search is fast but may leave accuracy on the table; a more complex
  decoder must earn its latency and error tradeoff.
- A model load on one machine does not prove compatibility or latency on the
  deployment machine.
- Punctuation and capitalization complicate exact-text comparisons unless the
  normalizer is fixed.
- Published population averages conceal large differences by severity,
  session, environment, and phrase domain.

## Readiness and release gate

A Parakeet build is ready for evaluation only when all of these pass:

```bash
python scripts/fetch_models.py --check --parakeet-only
python scripts/asr/check.py
python scripts/asr/check.py --audio path/to/reviewed.wav
python -m pytest
python scripts/asr/evaluate.py --out reports/parakeet-stock
```

A personalized build is ready for deployment only when it additionally:

- improves sticky-test WER or deliberately holds it level;
- does not regress critical phrases beyond the agreed tolerance;
- has no unexplained insertion increase;
- is tested across recorded session conditions;
- meets the target CPU latency and memory budget;
- passes stop, sleep, and wake tests in the deployment room;
- records model directory, graph hashes, decoder settings, and corpus export.

## Review triggers

Reopen this decision when the target language changes, v2 fails the measured
latency budget, v3 materially improves the same local test split, a newer
Parakeet release has direct accessibility evidence, or sherpa-onnx drops the
v2 graph format. A review produces a replacement decision; it does not add a
permanent second backend.
