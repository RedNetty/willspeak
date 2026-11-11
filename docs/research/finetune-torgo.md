# Fine-tuning Parakeet on TORGO: the plan, the traps, and what it can prove

Paths and commands below are relative to `app/` unless they refer to documentation.

*Written 10 September 2026, when the decision was made to add a Parakeet
fine-tune to the semester ("how can we prove this system has any advantage
if we don't show numbers?"). This replaces the 6 September plan's "no
fine-tuning this semester". Every choice below has a reason an examiner
can push on.*

**Current reading (1 Oct 2026).** This is the original experimental rationale.
The planned top-layer recipe below has been superseded by recorded bottom-eight
and adapter comparisons. [Report Context](../../app/reports/Report%20Context.md)
maps those results, and [training.md](../training.md#4-fine-tune) describes current
commands. A predicted effect here is not a measured result.

## 1. What the experiment claims, and what it cannot

The device rests on the personalisation finding: a general
recogniser fails on severe dysarthric speech, and a small amount of the
speaker's own data recovers most of the loss (Shor et al. 2019: 62% relative
WER reduction on ALS speech, 71% of it from five minutes of data; Green et al.
2021: 31% → 4.6% WER across 432 speakers). The client phase would show that on
the person the device is for. This semester cannot record him, so the
experiment tests whether training helps **public speakers with dysarthria**,
in the setting that mirrors his: **one speaker, their own earlier
recordings, tested on prompts they were never trained on, on a later day.**

What it proves, if it works: *fine-tuning Parakeet on one severe speaker's own
speech lowers WER on that speaker's held-out prompts by X points*, measured
on the exact int8 graph that then runs on the Pi. That number is the
case for building the architecture.

What it does not prove: anything about the client (different person, room,
microphone, and a wheelchair), or about conversational speech (TORGO is read
prompts). Both go on the limitations panel.

## 2. The traps in the literature, and the design that avoids each

**Trap 1: prompt overlap (Hui, Zhang & Mohan 2024, arXiv 2411.00980).**
Every TORGO speaker reads the same prompt list, and each speaker repeats
prompts across sessions. A random split puts the same sentence in train and
test, and the model learns the sentence, not the speaker. Their re-evaluation
with the overlap removed turned "impressive" TORGO results into "extremely
high" WER. **Design:** the split is by *normalised prompt text*, globally -
a prompt is train, dev or test for **every** speaker at once
(`willspeak.torgo.assign_split`, applied by `scripts/corpus/torgo_manifest.py`). No test prompt is ever in any
training set, for any speaker. This is the same sticky-split rule the
recorder's export already uses for the client corpus.

**Trap 2: the same recording twice.** TORGO recorded most utterances on two
microphones at once (head-worn and array). The two files are the same
speech. **Design:** one microphone per utterance, head mic where it exists,
array mic otherwise, chosen *before* splitting, so the two copies can never
land on opposite sides.

**Trap 3: repetition is not diversity (this project, August 2026).** Seven
minutes of one paragraph read over and over made a Whisper fine-tune
*worse* (12.5% → 37.5% test WER). **Design:** train on every prompt the
speaker has (words and sentences, all sessions), never on repeats of a few;
early-stop on dev WER; report the data-quantity curve (1 / 5 / 15 minutes /
all) so the poster can say how much data the gain needs.

**Trap 4: catastrophic forgetting.** Fine-tuning a 0.6-billion-parameter
model on a few hundred utterances of one voice can wreck it for everyone
else, and the client will say things his training data never contained.
**Design:** a small learning rate, few epochs, the lower encoder layers
frozen, and a *regression check* on typical speech (the TORGO control
speakers' held-out prompts, same split) reported next to the gain. A model
that gains 15 points on the speaker and loses 10 on typical speech is a
different, weaker result, and the table must show both.

**Trap 5: evaluating the wrong artifact.** The model that runs on the Pi
is a dynamic-int8 ONNX export, not the training checkpoint. **Design:** the
reported WER comes from `scripts/asr/evaluate.py` on the exported int8
graph, and the same graph is copied to the Pi and benchmarked there
(`scripts/asr/bench.py`). The checkpoint-versus-export difference is
reported once (quantisation parity, [F-4 measurements](../measurements.md#f-4--export-and-the-pi-h4)).

**Trap 6: tuning on the test set.** Everything that gets chosen, epochs,
learning rate, which encoder layers to freeze, is chosen on the dev split.
The test split is scored once per named comparison, and the prediction is
written in `notes.md` before that run.

**Trap 7: numbers from the wrong source.** The CPU pilot machine used to develop
this recipe did not run GPU training. Every project result must come from a
recorded run on the stated hardware, saved under `reports/`. Tiny CPU checks
validate the tooling; they do not measure recognition accuracy.

## 3. Data

TORGO (Rudzicz, Namasivayam & Wolff 2012): eight speakers with dysarthria
from cerebral palsy or ALS and seven controls, ~19,000 utterances,
free for academic use with citation. Severity from the corpus paper:

| speaker | severity | used for |
|---|---|---|
| F01, M01, M02, M04 | severe | the speakers the claims are about (F-1, F-2) |
| M05, F03, F04, M03 | moderate–severe to mild | training material for F-1; their own rows for context |
| FC01–03, MC01–04 | control (typical speech) | the forgetting check |

Per speaker the manifest builder keeps one file per utterance, drops the
non-speech prompts (`[relax your mouth…]`, picture descriptions, `xxx`), and
tags each item with `speaker`, `severity`, `session`, `kind` (word/sentence)
and the global `split`. Expect roughly 100–700 usable utterances per severe
speaker (F01, with one session, has 118; the male severe speakers have more), of which ~70% are single words, so the sentence WER rows will have
small n, and the tables carry n and a confidence interval.

Splits: 70 / 10 / 20 percent of prompts by a stable hash of the normalised
text, the same assignment for every speaker. Words and sentences are hashed
separately so both kinds land in every split.

## 4. Experiments (the labels are the proposal's: H1–H4, T-1, F-1–F-4)

### T-1 · Zero-shot baseline (H1)

Stock Parakeet int8 on the severe speakers' sentences (all of them, no
training involved, so no split needed) and, for the comparisons below, on
each export's `test.jsonl`, the same graph, the same clips the fine-tuned
models are scored on.

### F-1 · Multi-speaker fine-tuning, one speaker held out (H2)

For each dysarthric speaker *s*: fine-tune on the **other** dysarthric
speakers' train prompts, early-stop on their dev prompts, evaluate on
*s*'s **test** prompts (`export/loso-<s>/`). Speaker-disjoint and
prompt-disjoint at once: the held-out speaker's test prompts were in
nobody's training set. Compare with stock on the same clips.

Commands and the result table: [F-1 measurements](../measurements.md#f-1--multi-speaker-fine-tuning-one-speaker-held-out-h2).

### F-2 · Single-speaker fine-tuning, later session held out (H3)

For a speaker with two or more sessions: fine-tune on the earlier
sessions' train prompts, evaluate on the **last session's test prompts**
(`export/<s>/`). Session-disjoint *and* prompt-disjoint: a prompt the model
never saw, on a day it never heard, the setting the device is actually
used in. F01 has one session and gets the prompt-disjoint split only; the
table says so. Compare with stock and with F-1 on the same clips.

Commands and the result table: [F-2 measurements](../measurements.md#f-2--single-speaker-fine-tuning-later-session-held-out-h3).

### F-3 · Data-quantity curve (H3)

F-2 repeated with the training set cut to the first 1, 5 and 15 minutes of
audio and then everything (`torgo_manifest.py --export --curve M01`;
`export/M01-min1|5|15/`, same dev and test). This decides how
long a recording session to ask for in the client phase. Prediction to
write first: steep, flattening inside the first five to fifteen minutes.

Commands and the result table: [F-3 measurements](../measurements.md#f-3--data-quantity-curve-h3).

### F-4 · Export and the Pi (H2, H4)

The fine-tuned checkpoint → `export.py` → int8 ONNX → `check.py` → the same
`test.jsonl` scored with both NeMo and the export (record any difference)
→ the same export and clips copied to the Pi → `evaluate.py` on the Pi,
compared with desktop int8. A matching desktop NeMo/int8 score alone does not
complete F-4. `bench.py` with `WILLSPEAK_PARAKEET_DIR` pointed at the export
measures P-A speed and memory. Do the export check on a throwaway one-epoch fine-tune in week 2, long
before the real runs, because a model that will not export never reaches
the device. The deployment model is currently undecided. A model trained on
M01 does not establish a personal-training benefit for my voice.

### Typical-speech check (the forgetting pitfall)

Each fine-tuned model, and stock, on the control speakers' test prompts
(`export/controls/`, needs `fetch_torgo.py --controls`). One column next to
F-1/F-2. A model that gains on its speaker and loses more than a few
points here is a different, weaker result.

## 5. Original recipe and later diagnostics

Model: `nvidia/parakeet-tdt-0.6b-v2` from NeMo, the same checkpoint the
stock int8 export came from. Script: `scripts/asr/train.py` (NeMo
manifests, AdamW, cosine schedule, dev-WER checkpoint selection), then
`scripts/asr/export.py`, `scripts/asr/check.py`, `scripts/asr/evaluate.py`.

**What the literature says to freeze: as little as possible.** The SAP-2025
winner (Parakeet-TDT, Toyohashi) found that freezing the decoder and joint
*hurt* and full unfreezing won; the EMNLP-2025 idiosyncratic-speech study
found encoder fine-tuning matters more than decoder fine-tuning. So the decoder and
joint are always trained, and the encoder is frozen from the bottom only as
far as the card forces.

**Memory, from the real model** (loaded in the CPU pilot on 10 Sep 2026):
618 M parameters -24 conformer layers of ~25 M each plus an ~8 M front end
in the encoder, ~9 M in decoder + joint. With fp32 master weights, bf16
autocast and AdamW, a run needs about 2.5 GB for the weights plus 12 bytes
per *trainable* parameter (gradient + two optimizer moments), before
activations:

| `--freeze-encoder-layers N` | trainable | weights + grads + Adam | fits |
|---:|---:|---:|---|
| 0 (full) | 618 M | ~9.9 GB | 16 GB card only, small batch |
| 5 (front end + layers 0–4: the Round 3 recipe) | ~484 M | ~8.3 GB | 16 GB card (Kaggle/Colab T4) |
| 12 | ~309 M | ~6.2 GB | 4060 8 GB, batch 1–2 + accumulation, borderline |
| 16 | ~209 M | ~5.0 GB | 4060 8 GB, comfortably |
| 24 (encoder frozen) | 9 M | ~2.6 GB | any GPU, a smoke test, not a recipe |

So: on the 4060 start at **N = 16** (batch 2, `--grad-accum 8`); if it
trains cleanly and dev WER is still falling at the end, try N = 12. On a
16 GB card use **N = 5**. Report N with every result, it is part of the
recipe. Freezing the lower layers is also the forgetting guard (trap 4):
the acoustic front end that works for everyone stays as it was.

**Optimisation.** lr 2e-5 (frozen-lower-layer runs tolerate 5e-5), cosine
schedule with 100 warm-up steps (never a reloaded Noam schedule), tokenizer
unchanged, 15 epochs maximum, early stop when dev WER has not improved for 3
epochs, gradient clipping 1.0, NeMo's SpecAugment on, **speed perturbation
0.85–1.15 on every training clip** (the single strongest augmentation for
low-intelligibility speech in the CUHK/UASpeech studies; `train.py` does it
by default), the best four dev-WER checkpoints averaged before saving
(`--average-top 4`, the SAP-2025 recipe), no LM, greedy decoding at
evaluation (the deployment decoder). Seed fixed and recorded in `run.json`.
Batch-norm statistics stay frozen at the base model's values (`train.py`'s
default since 24 Sep 2026): re-estimated from batches of two clips, they cost
19.6 WER points on F01 in a six-step run and made the model answer silence,
while the same run with them frozen matched stock exactly (the recorded smoke test and batch-norm diagnosis).
Single-speaker runs (F-2, F-3) take about 10 optimizer steps per epoch, so they
use `--warmup-steps 10`; 100 warm-up steps would cover most of the run.
Transcripts are trained in the scoring form (`willspeak.normalize`): TORGO's prompts
mix lowercase words with capitalised, punctuated sentences, a few hold a character
the tokenizer cannot spell, and NeMo's in-training dev WER counts capitals, so raw
prompts let formatting pick the checkpoint. Measured on its own (the batch-norm-frozen run against the normalised-transcripts run,
24 Sep 2026): M02 test WER in NeMo 63.16% raw, 57.89% normalised.
**Trap 5 observed (the first F-2 run, 24 Sep 2026).** The dynamic-int8 encoder, which costs stock
about 4 points on M02 dev, turned a fine-tuned checkpoint's words into empty output:
deletions 12 in NeMo and in an fp32 export, 28 with only the encoder in int8; int8
decoder and joiner cost nothing. Every fine-tuned result is therefore reported for
the checkpoint (NeMo, `scripts/asr/nemo_hyp.py`) and for the int8 export. Caveat:
The first F-2 run trained with batch-norm statistics updating (the frozen default had not reached
the training machine). On the healthy batch-norm-frozen run the int8 export did not hurt: 63.16% in
NeMo, 56.84% exported, the same direction and size as stock (65.26%, 60.00%). The
silenced words were a symptom of the damaged model, not of the export.
**Batch norm, confirmed at full length (the first F-2 run against the batch-norm-frozen run, 24 Sep 2026).** The same
M02 run with the statistics updating scored 103.16% test WER in NeMo; with them
frozen, 63.16%, against stock's 65.26% (ci.py Δ 2.11 points, 95% CI −3.57 to 8.96).

**What was verified without a GPU (10 Sep 2026).** The whole chain up to
the first gradient runs on CPU: TORGO export → `train.py --check` →
model load → freezing (parameter counts above are measured) → dataset with
speed perturbation → first forward pass. It stops there because NeMo's TDT
loss is CUDA-only ("TDT is not yet implemented for non CUDA computation"),
so no CPU smoke-train is possible; the first GPU run is the first
training run. `export.py --stock` exported all three ONNX graphs from the
real checkpoint before the int8 quantisation of the encoder ran out of the
pilot machine's 7 GB of RAM, the training machine needs ≥ 16 GB of system RAM
for the export step.

**Where it runs.** Preferred: the 4060 desktop under WSL2 (Ubuntu), because
the machine is mine and the client-phase data must never leave it. NeMo
does not run on native Windows. Fallback if WSL2 + CUDA + NeMo fights back
for more than an evening: a free Kaggle or Colab notebook (16 GB T4/P100,
enough for the same recipe or even a fuller unfreeze). TORGO is public, so
the fallback is fine *for this semester only*, it is not an option for the
client's recordings, and the plan says so.

**Time per run.** ~500 training utterances × 15 epochs on a T4-class GPU is
on the order of an hour. Four speakers plus a data-quantity curve is an
afternoon of GPU time, spread over two weeks of evenings with the
export/evaluate/copy-to-Pi steps in between.

## 6. Recording the work

Record predictions, commands and results using the
[measurement template](../measurements.md).

## 7. Where it fits in the semester

Follow the [semester schedule](../plan.md#fall-2026-plan-the-public-corpus-semester)
and the [task list](../TODO.md). This reference owns the experimental rationale;
scheduling changes belong in those documents.

## 8. Whole sentences right: what helps, what does not, and where more data comes from

*November 2026 CPU pilot round. Everything in the tables below ran in a 2-CPU pilot machine with no GPU, on the float NeMo model, so it is **design context, not
a project measurement**. Nothing here ran on the Pi, and none of it replaces
the recorded runs in `reports/`.*

The goal changed on 5 November: what counts is the whole sentence coming out
right, for any sentence he says, not only sentences on a list. WER can fall
while the number of correct sentences stays flat, so this round counts
**exact sentences** (every word right) first and WER second.

### Setup

- **Speakers and test clips.** M02 (92 sentence clips) and M01 (89) as a
  replication, split into two folds by prompt text: each fold trains on one
  half of the prompts and tests on the other, so no test sentence is ever
  trained on. Single-word clips are left out of the test.
- **Fine-tuning recipe.** Per-speaker fine-tuning of the LayerNorm and bias
  parameters in encoder blocks 0–7, learning rate 3e-4, early stopping on 8–9
  dev clips. This is a lighter recipe than the lab's stage P (block-8
  adapter), so the counts are not comparable with `reports/` numbers.
- **Two-stage version.** The same recipe trained first on the other severe
  speakers, then on the target speaker. TORGO's severe speakers read nearly
  the same prompts (172 of the other speakers' 174 sentence clips share a
  prompt with M02), so for each fold the first stage drops every clip whose
  sentence is in that fold's test half.
- **Language-model rescoring.** Qwen2.5-0.5B scores each candidate sentence;
  the pick maximises acoustic score + 0.7 × LM score + 3 × word count, with
  the weights chosen on stock F01 and M01, never on the test speaker.
- **Statistics.** Paired bootstrap over test clips, 2,000 resamples, seed 0.
  With 92 clips, one sentence is 1.1 points, so differences of one or two
  sentences are noise.

The repository's own stage P exports are prompt-disjoint too:
`torgo_manifest.py` splits by normalised prompt text, the same split for
every speaker, and `loso-severe-<SPK>` takes only the other speakers'
non-test clips. So the leakage the CPU pilot had to remove by hand cannot
happen in the lab runs.

### What helped

| M02, exact of 92 (M01, of 89) | M02 | M01 |
| --- | --- | --- |
| Stock, greedy | 15 | 8 |
| Per-speaker fine-tuning | 24 | 14 |
| Two-stage (other severe speakers first) | 27 | 16 (18 at last epoch) |
| Two-stage + one extra decode + LM rescoring | **30** | **19** |
| 16 decodes per clip + LM rescoring (not Pi-sized) | 31 | 23 |

- **Per-speaker fine-tuning** is the largest single step: 15 → 24 on M02,
  8 → 14 on M01, WER 46.2% → 32.5% and 57.7% → 37.4%.
- **The two-stage version** added 3 sentences on M02 and 2 on M01. Pooled
  over both speakers it was 43 against 38 (+2.8 points, interval −1.1 to
  +7.2): same direction on both, not established.
- **One extra decode plus rescoring** is the Pi-sized version. The extra
  decode re-runs the encoder with an 11.5 ms hop instead of 10 ms. The view
  was chosen on one speaker and tested on the other, and both directions
  chose the same one. It gave +3.3 points on M02 (interval 0.0 to +7.6) and
  +3.4 on M01 (0.0 to +7.9), roughly double the stock exact rate on both.
  Estimated cost on the Pi from the measured 0.40 s per 3 s decode: about
  0.8 s for two decodes plus about 0.3 s of LM scoring. This is an estimate,
  and severe speakers' real sentences run longer than 3 s.
- **Sixteen decodes per clip** gave the best counts (31 and 23) but cost about
  6 s per utterance on the Pi, so they are a ceiling, not a device option.

### How many different sentences the fine-tune needs

| Distinct training prompts | M02 exact / WER | M01 exact / WER |
| --- | --- | --- |
| 0 (stock) | 15 / 46.2% | 8 / 57.7% |
| 10 | 19 / 39.2% | 16 / 42.0% |
| 20 | 22 / 35.4% | 18 / 35.0% |
| All (about 37 and 35) | 24 / 32.5% | 14 / 37.4% |

Most of the gain arrives with the first 10 to 20 *different* sentences.
TORGO cannot show what happens past about 40, because each speaker has only
that many training prompts. Published work keeps improving far beyond that:
Euphonia's severe speakers kept gaining up to roughly 1,800 utterances, and
the share of severe home-automation speakers reaching the target WER rose
from 27% at 50 utterances to 69% with all their data (Green et al. 2021;
arXiv 2110.04612).

### What did not help

Every one of these landed within two sentences of the plain per-speaker
fine-tune, and none was established:

- **More audio of the same sentences.** His single-word clips (22 of 92),
  TORGO's second microphone recording of each training clip (25), 0.9× and
  1.1× speed copies (24). They add takes, not new word sequences.
- **Cloned speech of new sentences.** F5-TTS spoke 96 new everyday sentences
  in M02's voice from his training-half recordings. The clones kept his
  impairment (stock WER 58.6% on them against 46% on his real speech), but
  training on them gave 24, 22 and 20 for impaired-only, all, and
  babble-removed clones. Published successes used at least about 195 real
  utterances per speaker; zero-shot clones of severe TORGO speakers have
  helped mild speakers and hurt severe ones (arXiv 2508.05102).
- **Rewriting the output.** A word-edit search checked against the audio
  (26 to 27), a language model proposing whole sentences, a Whisper-small
  second opinion (27 to 28), and discriminative training on n-best lists
  (fold 1: 12 to 16 against 17).

The diagnostic behind those failures: of 64 wrong M02 sentences, 50 are
**model errors** (the acoustic score itself prefers the wrong text) and only
14 are **search errors** (the right sentence scores higher but nothing
proposed it). The TDT score also barely charges for missing words: a
one-word text such as "a" beats the true sentence on 40 of 92 clips. Any step
that rewrites or extends the decoder's output has to check length, or it
shortens sentences. More gain has to come from how the acoustic model hears
him, which means more distinct sentences from him or a better base model.

### Where this sits against published prompt-disjoint results

Most published TORGO numbers are on utterance splits where the test
sentences also appear in training (98.2% of M02's test prompts appear
elsewhere in the corpus), so they measure recall of seen sentences. With the
text held out, published systems sit between about 51% and 94% WER on severe
speakers: per-speaker wav2vec2-XLSR 85.07% on M01 and 90.43% on M02,
Whisper-Large-V2 zero-shot 56.3% (arXiv 2411.00980), SAP-trained
Whisper-medium 57.26% on F01, M01 and M02 (arXiv 2501.14994), and commercial
systems and multimodal LLMs above 51% (arXiv 2512.17474). Stock Parakeet in
the CPU pilot (57.7% on M01) sits where that literature predicts. No paper found
reports exact-sentence rates for severe speakers.

### Ideas, ranked by what they could do for whole sentences

1. **Hundreds of distinct sentences from him.** The strongest outside
   evidence, and the device can collect them: the Record page's coverage
   scheduler gives new prompts, and every sentence he or a caregiver confirms
   is a correctly labelled new sentence. Under the personal-model clearance
   these train his model; they are never scored. Collect *different*
   sentences, not repeats.
2. **A base trained on disordered speech at scale, then the per-speaker
   fine-tune.** The two-stage result above is a small version of this. The
   Interspeech 2025 SAP Challenge was won by a fine-tuned Parakeet-TDT-1.1B
   at 8.11% WER on text-disjoint test data against 17.82% for the Whisper
   baseline (arXiv 2507.22047); Google cut severe WER from 48.1% to 31.3% by
   adding about 1,000 h of disordered speech to USM (arXiv 2412.19315). SAP
   is about 6% CP speakers, so a SAP base is a starting point, not an answer
   on its own.
3. **One extra decode plus LM rescoring on the device.** The only
   decoding-side change that held across both speakers, and the only one
   priced for the Pi. It needs timing on the Pi with real utterance lengths
   before it is worth shipping.
4. **Checkpoint averaging and several seeds per recipe.** Small on its own
   (the SAP winner used it), but it makes one- and two-sentence differences
   readable.

Speech LLMs, LLM error correction and confirmation dialogues are poor fits:
zero-shot audio LLMs stay above 51% WER on severe TORGO speech, LLM
correction made Whisper worse (56.3% to 62.9–79.1%), and a spoken
"did you mean" step breaks the rule that the device speaks what it hears.

### Getting more data

| Source | Size and content | Access |
| --- | --- | --- |
| His own recordings | As many distinct sentences as he records | Cleared for the personal model; never scored or published |
| Speech Accessibility Project (SAP) | About 1,500 h from about 999 people, including CP speakers | Data use agreement signed by the data user and the university's authorised official, from organisational email addresses, term 3 years or less, with a one-page proposal sent to speechaccessibility@beckman.illinois.edu |
| Euphonia (Google) | Large, disordered speech | Not available outside Google |
| MotorSpeechBank (TalkBank) | Motor speech recordings | Consortium members; needs a faculty applicant |
| UASpeech | Dysarthric speech | By request; single words only, so it cannot test sentences |

SAP is the only public English source big enough for idea 2, and it needs a
faculty sponsor and the university's signing official. His own recordings
are the source for idea 1 and need nothing further.

## 9. After lab test 36: language-model rescoring on the lab's own recipe

*9 November 2026. The first part is a project measurement (the lab computer, queue `1106-1346`; tables in
`reports/lab-36-lm-rescoring/notes.md`). The ideas are design context, not results.*

### What the lab found

- Choosing among Parakeet's nine decodes with the audio score plus a small text model raises whole sentences right on stage P's recipe from 24.2% to
  31.7 to 35.0% (120 held-out sentences), with almost none lost; on stock the same models gain 0 to 3 points. The right sentence is in the pool
  for 37.5% of stage P's clips and 16.4% of stock's, which is why the gain grows after fine-tuning (section 8 saw the same direction in the CPU pilot).
- A 0.5B model at 4 bits (398 MB) does as well as the 4B models; quantisation, release date and size do not rank the models (the stock and
  stage P rankings agree at Spearman 0.22).
- Applied to single-word clips with the sentence weights, rescoring loses 10 to 31 of 393 single words. A one-word first decode tells the two kinds
  apart on about 97% of clips, so the device can route them; their own weights then gain a few clips.
- The 425-word count moves 176 to between 164 and 171, inside the 19-word rule; the bar is 148. Single words are the gap, and no language-model
  feature closes it.

### Ideas, in the order I would try them

Replayable now from the saved pools (no GPU; `scripts/asr/lm_report.py`):

1. Average several language models' scores, and widen the bonus grid (at least 22 of 81 chosen bonuses sat on the maximum, 8).
2. Give the chooser more features: how many decodes proposed an answer (votes, used for single words already), and a words-per-second prior from the clip duration (the TDT score barely charges for missing words, section 8).
3. Count how often the right answer is among the top three, as a measure for a screen that shows alternatives.

One lab queue each (set up, not run):

4. **Route by clip kind** (lab test 37): a small adapter trained on single-word clips only for clips the detector calls single words, stage P's adapter for the rest.
5. **Self-train on language-model-chosen labels** (lab test 38): lab 22's loop with rescored sentences as the pseudo-labels, keeping clips whose best answer wins by a clear margin.
6. **Wider pools** (lab test 39): answers from stage P's other seeds and from the block-12 and block-16 adapters added to the pool, every answer still scored by the one primary model.

Needs more data or new data:

7. **Scale the pretraining** from three severe speakers to hundreds (the Speech Accessibility Project, below), with the pretraining clips picked by embedding similarity to the target speaker (`data/emb`).
8. **An adapter library by severity cluster**, chosen from a few seconds of speech, so a new user starts from a good adapter before recording anything.
9. **Context for the language model**: the conversation partner's previous sentence as a prefix. TORGO has none, so it needs new data; in use it may be the largest sentence-level gain.
10. **A personal language model** biased toward the user's own phrase bank, tested on his own recorded sentences.

The rescoring gain depends on the pool: a stronger acoustic model puts the right sentence in the pool more often, so more data should help the language
model too, until the acoustic model gets most sentences right on its own. Nothing here shows the gains add up past that point.

## References

- Shor, J. et al. (2019). Personalizing ASR for Dysarthric and Accented
  Speech with Limited Data. Interspeech 2019. arXiv:1907.13511.
- Green, J. R. et al. (2021). Automatic Speech Recognition of Disordered
  Speech: Personalized Models Outperforming Human Listeners on Short
  Phrases. Interspeech 2021.
- Hui, Zhang & Mohan (2024). Enhancing AAC Software for Dysarthric Speakers
  in e-Health Settings: An Evaluation Using TORGO. arXiv:2411.00980.
- Rudzicz, F., Namasivayam, A. K. & Wolff, T. (2012). The TORGO database of
  acoustic and articulatory speech from speakers with dysarthria. Language
  Resources and Evaluation 46(4).
- Speech Accessibility Project Challenge, Interspeech 2025: arXiv:2507.22047.
- Whisper-medium fine-tuned on SAP, tested on TORGO: arXiv:2501.14994.
- Commercial ASR and multimodal LLMs on severe dysarthric speech:
  arXiv:2512.17474.
- Disordered speech added to USM pretraining: arXiv:2412.19315.
- Personalised models for severe speakers by amount of data: arXiv:2110.04612.
- Zero-shot voice cloning of TORGO speakers for training data: arXiv:2508.05102.
- Speech Accessibility Project, data access:
  https://speechaccessibilityproject.beckman.illinois.edu/
