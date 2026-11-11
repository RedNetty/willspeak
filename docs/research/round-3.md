# Research Round 3: the severe end, and what the CPU-only experiments found

Paths and commands below are relative to `app/` unless they refer to documentation.

> **Scope and evidence.** Everything in this document was produced
> on 2 September 2026: a literature review in five areas and a day of
> experiments on TORGO and EasyCall, run on a 2-CPU machine with no GPU. It
> is **context for design decisions, not a project measurement**. Nothing
> here used project hardware; the scripts, result files and
> logs stayed on that machine and are not in this repository. The recorded
> re-measurement of the shipped matcher recipe on TORGO (`reports/torgo-matcher/`)
> and fine-tune results (`reports/finetune-torgo/`) are the numbers the
> project quotes. Three things this round recommended were later **removed**
> from the code (10 Sep 2026) because they were never built or measured for
> any real speaker: cohort/AS-norm open-set rejection (§5, §7.3), the
> nearest-session bonus (§7.2 (3)), and sherpa-onnx hotword biasing (§2.4).
> The write-up is left as written so the reasoning can still be checked.

*2 September 2026. A literature review in five areas (Asia-Pacific groups, the very-severe / near-anarthric end, the speaker-verification toolkit, Parakeet fine-tuning and data recipes, and the commercial products the client is likely to have tried) plus a day of original experiments on TORGO (8 dysarthric speakers, 4,204 single-word recordings) and EasyCall (Italian, 31 dysarthric speakers, up to 6 repetition sessions each), run on a 2-CPU machine with no GPU. Claims marked **V** were read at the source; **S** = abstract or secondary. Every number in §4–§8 was produced that day.*

*Context for this round: I rate the client's speech on his worst days at the high end of severity, among the least intelligible. That is the regime every table below is read against.*

---

## 1. Bottom line

Seven things change. Everything else in Round 2 stands.

1. **The matcher's projection was under-fitted, and fixing the estimator is the largest gain in three rounds.** The August projection used scikit-learn's shrinkage LDA, which shrinks each word's covariance separately, with 2–10 takes per word that over-regularises to near-identity. Estimating one pooled within-class covariance from all ~3,600 residuals and shrinking it a fixed 10% toward identity (α = 0.1) lifts layer-9 1-shot top-1 .452 → .502 and **cross-session (enrol one day, test another) .384 → .454**. Adding the layer-9 frame *standard deviation* to the mean (statistics pooling, 1,536-d) adds again: **.517 / .746 / .810 top-1/3/5 one-shot and .497 / .721 / .796 cross-session** -+6.5 / +11 points over August; seven of eight speakers up one-shot (F01, who has 16 test items, moves .51 → .44) and all six multi-session speakers up cross-session; the four severe speakers .348 → .362 one-shot and **.338 → .389 cross-session** (top-3 .534 → .671). Both changes are free on a small CPU box (a standard deviation over frames already computed, and a 1,536×128 matrix instead of 768×128). The unseen-word control (§6.3) shows the gains are not memorised TORGO vocabulary (+3.6 cross-session on words the projection never saw). Concatenating layers 8–10 looked like a further +2 on the full vocabulary but is **zero on unseen words**, it memorises TORGO's word list, so it is not adopted. The R1 hypothesis (cross-session ≥ .45) is met by this route, not by the one Round 2 predicted.
2. **WCCN, NAP, CORAL: all out. The hypothesis that a projection learned from *other* speakers can remove *his* session drift is refuted.** With the August estimator WCCN looked like +2–3 points; with the corrected estimator it *costs* 2–3 (it was compensating for over-shrinkage), NAP never helped, unsupervised CORAL gave nothing, and WCCN hurts cross-lingually (§7). The "which session?" probe explains why: a linear classifier can still tell his sessions apart at 81% after LDA and 79–81% after WCCN or NAP, session variability is speaker-specific, and only the speaker's *own* session statistics remove it (51% after per-session mean subtraction, below the 61% majority-class chance). Consequence for the design: `T` in `projection.npz` stays identity; the session vector is the mechanism, and §6.5 says most of that mechanism's value comes from centring the *enrolment bank* on its own session means (always available offline), with the live running mean worth about +1 point once shrunk with n₀ ≈ 20 and *harmful* in the first ten utterances if not shrunk.
3. **Open-set rejection: the cohort must be other *words*, not other people saying *his* words, and multi-day enrolment is what makes rejection work.** AS-norm with a cohort of other speakers' recordings of the same vocabulary makes rejection *worse* (AUROC .69 → .65); a cohort restricted to words outside his vocabulary makes it better (.69 → .76, TPR at 10% false-accept .33 → .42); query-side z-norm alone is as good as the two-sided formula. With one exemplar per word the ≥ .85 hypothesis is not met and the severe speakers sit at .68. On EasyCall the same z-norm goes **.81 → .87 → .90 AUROC as enrolment grows from 1 to 3 to 5 sessions** (TPR at 10% FAR .57 → .74), and .94–.96 with an in-language projection, so the .85 bar is reachable with several days of enrolment. *(Later decision: the rejection path was removed from the runtime rather than shipped unmeasured; the matcher keeps its narrow role, stop phrase, and a confident override that is off by default.)*
4. **Never-average was wrong, multi-day enrolment is right, and two plausible ideas failed.** On EasyCall with the projection in place, a *mean prototype* across sessions beats max-over-takes at every enrolment size (5 sessions: .786 vs .760; severe .554 vs .517), exactly as the speaker-verification literature said and Round 2 doubted; each additional enrolment session adds ~5 points with no plateau at five (1 → 5 sessions: .547 → .786; severe .333 → .554), and the *most recent* session alone beats the oldest alone by 8 points. Order-preserving segmental pooling (3 or 5 segments) is far worse than a single mean (.326 vs .452), the lesson DTW taught in August. And using the fixed 10-word probe list to *calibrate* the feature space (anchor shift or affine map) hurts at every anchor count (5/10/20): a 5–20-take estimate of the session shift is noisier than the shift itself. The probe list stays a measurement of within-session drift, never a calibration input.
5. **Parakeet is confirmed on the CPU pilot's severe test, and the fine-tune recipe is now specific.** Zero-shot on the four severe TORGO speakers' sentences (287 clips, primary microphone, CPU), Parakeet-TDT-0.6B int8 beats Whisper-small int8 on every speaker, WER .618 vs .753 overall (F01 .56 vs .67, M01 .59 vs .75, M02 .46 vs .63, M04 .82 vs .91), at 3.2× the speed, and its only failure mode is an empty output (8/287), never a runaway. The remaining question is what fine-tuning buys. From the SAP-2025 winner (Toyohashi, Parakeet-TDT, **V**), NeMo's own docs (**V**), the KIT/CMU 100-hour case (**V**) and the EMNLP-2025 idiosyncratic-vs-normative study (**V**): full fine-tune with nothing frozen (freezing decoder/joint *hurt*), LR 1e-5–3e-5 with a constant/cosine schedule, never the reloaded Noam schedule, tokenizer unchanged, batch 2–8 with gradient accumulation to 16–32, bf16, SpecAugment plus *wide* speed perturbation ({0.85…1.15}, the single best augmentation for very-low-intelligibility speakers in two CUHK/UASpeech studies), early stopping on a fixed ~50-sentence never-trained held-out set (ValSub), and averaging the 3–4 best checkpoints. Prime on a severity-*mixed* pool of other dysarthric speakers before his data: two independent groups found that halves the personal data needed and that adding *more severe* speakers helps everyone. *(The recipe as changed for an 8 GB card is in [finetune-torgo.md](finetune-torgo.md).)*
6. **Nobody has closed the "how bad is today" loop, and it is cheap to close.** Audio-derived severity/intelligibility estimation is now good (listener-effort R² .92 in ALS; spastic-dysarthria blind intelligibility r .98 from spectral/energy features; 4-way severity 60–80%), but no published system changes recogniser behaviour from it, the only state-switching AAC system uses interaction telemetry. The session vector is already a state estimate; §9 sketched a policy for the client phase.
7. **What the company he tried could not do, I can, and one thing they do, I copy.** Voiceitt's own FAQ scopes it to mild-to-moderate; Google's Project Relate admits severe voices may not work and is no longer accepting new users; the 2025 Euphonia evaluation *excludes* profound speakers from its test set; no AAC vendor (Tobii Dynavox, PRC-Saltillo, Jabbla) recognises dysarthric speech at all. The one documented system that works for a single severe speaker used ~100 hours of that person's speech and in-app corrections, a volume no appointment-based service can collect and I can. What to copy: get a usable v0.1 out of the first 5–10 minutes of clean labelled speech (Euphonia: 71–85% of the eventual gain), stage enrolment in levels, and treat every real conversation as correction data.

Realistic expectations did not move: leakage-free severe-speaker open-vocabulary WER is 80–94% zero-shot (NP-TORGO, **V**) and 10–30% after heavy personalisation on good days; phrase/exemplar matching is what carries bad days (Apple LPM: severe .70 vs .02 for ASR on the same utterances). That was already the August design, and the new numbers support it.

---

## 2. What the literature added (beyond Round 2)

### 2.1 Asia-Pacific (China, Hong Kong, Singapore, Japan, Korea, India)

| Finding | Source | Why it matters here |
|---|---|---|
| Fine-tuning Parakeet-TDT-1.1B on SAP: **freezing decoder + joint hurt**; full unfreezing won; 4-checkpoint averaging; speed perturb 0.9–1.1 | Toyohashi UT, Interspeech 2025 (takahashi25) **V** | My exact model family; settles "what to freeze": as little as the card allows, never the decoder/joint |
| Fine-tuning first on other dysarthric speakers (normative), then on the one speaker (idiosyncratic), beats either alone and needs **half the personal data** (128 vs 256 utterances); encoder fine-tuning > decoder; models trained on *more severe* speakers generalise better | Stony Brook/Vanderbilt, EMNLP 2025, arXiv 2509.16718 **V** | Prime Parakeet on TORGO/UASpeech-class data before his; explains why the matcher's LDA-from-other-speakers works |
| Dysarthric phoneme identity concentrates in encoder layers 5–8; single-layer LoRA at layer 7 within 3.5% of full FT | SIT Singapore, arXiv 2608.01865 **V** | Motivated the layer sweep in §6 (9 still wins for *word identity*) |
| **Mixed-severity training beats severity-matched**; adding higher-severity data improves lower-severity recognition | NIT Sikkim + USC, CSSP 2026 **V** | Do not curate the donor pool to match him, use the full range |
| Personalised federated learning: very-low-intelligibility speakers gain most from speaker-dependent components | CUHK + NRC, arXiv 2606.13253 **V** | Consistent with everything else: the severe end needs *his* data most |
| Zero-shot commercial ASR and MLLMs (GPT-4o, Gemini 2.5, Deepgram, AssemblyAI, Whisper-v3) all ≥ 49% WER on severe TORGO speakers | Strathclyde preprint **V** (and arXiv 2512.17474 **V**) | No cloud fallback will save a bad day; drop the idea |
| Dual-stage user-defined KWS: CTC phoneme pre-filter → embedding verification → 187k-parameter LoRA continual adaptation (typical speech only, CC BY-NC-ND) | Shanghai U/XJTLU/NYU, arXiv 2605.22120 **V** | A template for "learn new phrases over time" once Parakeet is fine-tuned |
| SSL features (HuBERT best) beat MFCC/PLP for dysarthric KWS on TORGO/UASpeech (LOSO ATWV .33–.40) | NIT Sikkim, SIVP 2025 **V** | Speaker-independent floor; my enrolled-exemplar numbers are far above it |
| Speaker-dependent CP recognition with WFST + TTS augmentation, ASSETS 2025 (Hojo et al.) | ACM DL, paywalled **S** | Closest analogue to this case; worth a library read |
| Korean real-time aid on the AI-Hub corpus (Conformer → JETS TTS) reports "94.4% CER", internally inconsistent | MDPI Appl. Sci. 2025 **V** | Do not cite; the corpus stays restricted |
| Lie-group spectrogram normalisation: WER stays > 70–120%, GPU-only | arXiv 2504.12279 **V** | Skip |

### 2.2 The very severe / near-anarthric end

- **Non-verbal vocalisation as a channel.** ReCANVo (Nature Sci. Data 2023, **V**) -8 minimally-speaking individuals *including CP*, 7,077 caregiver-labelled vocalisations, often only **2–3 exemplars per label**: the realistic regime for a personal vocalisation vocabulary. voc2vec (arXiv 2502.16298, **V**) is a wav2vec2-style model pre-trained on ~125 h of non-verbal audio (ReCANVo included), ~90 M parameters, a candidate feature extractor for a "vocalisation mode" if a day comes when words are gone. Lea et al. (CMU/Euphonia, **V**): explicit "speech" and "background" *negative classes* cut false positives 98.4%, and 5-shot personalisation recovered 84.5% of the generic model's misses in a motor-speech cohort, the strongest evidence that **hard negatives beat a score threshold** for rejection.
- **Talkitt / Voiceitt's published trial** (Frontiers Psychol. 2023, **V**): speaker-dependent clustering of each person's idiosyncratic sub-word patterns, a cousin of my architecture; ~20 repetitions per word over 6 months; **60–95% accuracy depending on the participant**. That band, rather than the high-90s, is a realistic target for the phrase path on bad days.
- **Audio-visual.** The one large gain (AV-HuBERT reconstruction, ICASSP 2024: −42 WER points for the more-severe UASpeech group, **S/V**) was obtained with studio frontal video on isolated words; **TORGO's video was recorded but never released** (**V**), no corpus pairs wheelchair-camera video with severity labels, no CPU latency figures exist, and Hustad & Cahill found no listener benefit from lip video for 4/5 CP speakers. Verdict unchanged: not for now.
- **Other sensing** (sEMG, throat wearables, EMA): small-n engineering studies, none in CP, all needing electrode placement he cannot manage. Not pursued.
- **Severity-aware modelling, an open niche.** Audio-derived severity is measurable (SALR 70.5% 4-way LOSO; DSSCNet 68–79% after cross-corpus adaptation; ALS listener-effort R² .92 from a Whisper encoder; spastic-dysarthria blind intelligibility r .98 from spectral/energy features, all **V**), yet **no 2024–26 system changes ASR or matcher behaviour from it**; the only state-switching AAC system (MDPI Computers 2024, **V**) uses interaction telemetry.
- **Calibration warning.** NP-TORGO (arXiv 2411.00980, **V**): standard TORGO splits have 95–100% prompt overlap between train and test; on a non-overlapping split, severe-speaker WER is **80–94%** (words 93.9%) and mild speakers move from 3–5% to 45–73%. Every cross-paper WER comparison in this field needs its split checked first; relative gains within a paper are safer than absolute numbers across papers.
- **Layer selection.** wav2vec2 pathological-feature probing (arXiv 2402.01796, **V**) finds early-to-middle layers best and a learned weighted sum matching the best single layer while generalising better out-of-distribution.

### 2.3 Session-variability and open-set toolkit (what to borrow, with the math)

- **WCCN** (Hatch 2006, **V**): `W = Σ p(i)·C_i` pooled within-class scatter; `A Aᵀ = W⁻¹` by Cholesky; regularise `(1−λ)W + λI`. 11.7–28.5% relative EER reduction in the original. My runs in §4: +1.7 / +2.8 points at λ = 0.5 with the August estimator; negative after the estimator fix.
- **NAP** (Campbell 2010, **S**): `P = I − VVᵀ` over the top-k nuisance eigenvectors. My runs: nothing at k = 1…16 on either scatter estimate.
- **CORAL / CORAL+** (Sun 2016 **S**; Rajan, Kinnunen & Hautamäki 2019 **V**): recolour to the target session's covariance without labels. My run (plain CORAL on LDA features, oracle session covariances): no change (.384 → .384).
- **Multi-enrolment scoring** (Rajan et al. 2014, **V**): in clean speaker verification, *averaging* enrolment vectors beat multi-session PLDA, score averaging and max. §7 settles it on EasyCall with real multi-session data: prototype vs max vs top-2 mean.
- **AS-norm** (Matejka 2017, **S**): `z = ½[(s−μ_q)/σ_q + (s−μ_e)/σ_e]` over each side's top-k cohort scores. Gains are stable from K = 50 to 500 cohort items (LIE-TAS-norm, arXiv 2504.04512, **S**); **cohort quality beats cohort size** (Zigel & Cohen 2003, **S**). §5 adds the decisive detail: the cohort must not contain his vocabulary words.
- **Pooling.** MMFA (Electronics 2026, **V**): masking the 70% least-attended frames *improves* WavLM-Base+ speaker EER by ~28% and multi-layer aggregation beats any single layer. Graph-attentive pooling (Interspeech 2024, **S**) halves QbE-KWS false rejects over attentive-statistics pooling, but needs training. Segmental pooling (Settle & Livescu 2016, **S**) is the cheap order-preserving option, tested in §6 and rejected.
- **Few-shot / QbE state of the art usable on CPU**: GeMCL (arXiv 2605.13075, **V**), CLAP-IPA (NAACL 2024, **V**, too heavy), multi-axis WavLM-base-plus heads (arXiv 2605.02804, **V**), few-shot KWS in any language (arXiv 2104.01454, **V**: 5 examples per keyword → F1 .75). The recipe stays: frozen encoder, linear projection, cosine, with the changes in §1.

### 2.4 Fine-tuning and data recipes (Parakeet, unlimited access)

| Recipe element | Evidence |
|---|---|
| Full FT, nothing frozen; freezing decoder/joint hurt | Toyohashi SAP-2025 winner **V**; KIT/CMU: full FT beats LoRA 15–39% rel. **V** |
| LR 1e-5–3e-5, constant/cosine schedule (a reloaded Noam schedule resets LR high), tokenizer unchanged | NeMo fine-tuning docs **V**; NeMo #14140 (TDT head stagnates on vocab change) **V** |
| Batch 2–8 + gradient accumulation to 16–32, bf16, gradient checkpointing on 8 GB | Toyohashi fallback config (batch 2, epochs 10 → 20) **V** |
| Early-stop on a fixed ~50-sentence held-out set that predicts forgetting | ValSub, arXiv 2503.09906 **V** |
| Average the 3–4 best checkpoints | Toyohashi **V** |
| Speed perturbation, wide range {0.85 … 1.15}, is the single best augmentation for very-low intelligibility; two-stage (tempo down to 0.4–0.5 + dysarthric SpecAugment) VL 78.9 → 61.5 | CUHK augmentation study **V**; Bhat et al. Interspeech 2022 **V** |
| TTS cannot control severity or identity reliably; cloning adds little when real data exists | Parler-TTS reconstruction (sanchez25) **V**; arXiv 2606.19823 **V**; EUSIPCO 2025 Hungarian study **V** |
| Prime on other dysarthric speakers (severity-mixed) before his data | EMNLP 2025 **V**; CSSP 2026 **V**; CDSD cross-learning +13% rel. **V** |
| No CPU-feasible test-time adaptation exists for RNN-T/TDT (all SUTA/SGEM/DSUTA evidence is CTC + GPU) → periodic offline re-fine-tune on the 4060 | DSUTA EMNLP 2024 **V** |
| Biasing: TurboBias is native in NeMo for TDT (2–5% RTFx overhead, F-score +7–48 pts); sherpa-onnx hotwords are transducer-only and need `modified_beam_search` *(and a BPE vocabulary file the stock int8 release does not ship, the WillSpeak switch was removed 10 Sep 2026)* | arXiv 2508.07014 **V**; NeMo word-boosting docs **V**; sherpa-onnx docs **V** |
| Int8: static QDQ (per-channel, MinMax calibration) ≈ 2× RTF over dynamic int8 on AVX-512; expect less on small CPUs | heyneo case study **V** |
| Data: phoneme/biphone-coverage prompt list (~500 items) + fast correction UI took a non-normative speaker 70% → 25% WER in 90 minutes | Adapt4Me, arXiv 2603.20112 **V** |
| Corrections from real use are worth more per minute than more read speech once a baseline exists (38% → 7.8%) | KIT/CMU **V** |
| Euphonia collects 300 *scripted* phrases per person from a 3,598-phrase pool, diversity, not repetition | Frontiers 2025 **V** |

### 2.5 The commercial products (what the client probably tried)

| Product | Enrolment | Severe-speaker evidence | Status / limits |
|---|---|---|---|
| **Voiceitt** (ex-Talkitt) | 50 → 1,000+ self-recorded phrases, cloud, $49.99/mo | FAQ: "most effective for mild to moderate"; severe-dysarthria pilots only started 2025, no results | Consumer app needs internet; on-device only in an Intel enterprise edition |
| **Google Project Relate / Euphonia** | ≥ 500 phrases; model in < 24 h | 2019 research: severe ALS 42.7 → 20.9% WER; 2025 evaluation *excludes* profound speakers; help pages warn severe voices "may not" work | **Not accepting new users** (Play Store, 2026); no re-fine-tuning loop |
| **Apple Personal Voice / "Listen for Atypical Speech"** | 150 sentences (TTS banking, on-device); Siri toggle (no enrolment) | None published for recognition | Personal Voice is output only, complementary to this device, not a competitor |
| **Microsoft Azure + SAP data** | none (general model) | 18–60% relative gain by disability type (**S**) | No personalisation path |
| **Tobii Dynavox / PRC-Saltillo / Jabbla** | -| -| **No dysarthric speech input at all**; output devices only |
| **Single-speaker Whisper/Parakeet fine-tune (research)** | ~100 h of one person + 8.8 h corrections | 128% → 9.7% WER (read), 7.8% (corrections) | The only documented approach that works for one severe speaker, the one this project follows |

Five things unlimited home access buys that none of them have: hours instead of minutes; retraining as he changes; real spontaneous home speech and real noise; a vocabulary and prompt set built around *his* errors; no enrolment gate, queue or shutdown risk. Three things to copy: a usable v0.1 from the first 5–10 minutes of clean labelled speech, levelled enrolment stages, and corrections harvested from every real conversation.

---

## 3. Experimental setup (identical to August unless stated)

**Data.** TORGO, the 8 dysarthric speakers (F01, M01, M02, M04 severe; M05 moderate–severe; F03 moderate; F04, M03 mild), single-word prompts only: 4,204 recordings over the head-mounted and array microphones, 338 distinct words, 1–3 sessions per speaker (sessions are separate visits, i.e. different days). EasyCall (Italian; Turrisi et al. 2021; CC BY-NC 2.0): 31 dysarthric speakers rated 1–5 on the Therapy Outcome Measure (5 = severe: F10, M11, M14; 4: M05, M06, M07), each repeating the same 67-command list in 2–8 "sessions"; recorded on a phone at 8 kHz. The paper does not say whether EasyCall sessions were on different days, so it is treated as a repetition-block corpus, not a day-variability corpus; TORGO is the day-variability test.

**Features.** Frozen WavLM-base-plus after silence trimming and peak normalisation; mean-pooled hidden states of layers 6–12, plus the layer-9 frame standard deviation and 3-/5-segment means. Nothing is fine-tuned anywhere. The only learned object is a linear projection fitted by shrinkage LDA on *other* speakers (leave-one-speaker-out, so the test speaker never contributes), 128 dimensions.

**Protocols.** *A -1-shot closed vocabulary*: per speaker and primary microphone, the vocabulary is every word that speaker recorded (78–296 words, all acting as distractors), one random take per word is the exemplar, the test items are the remaining takes of words with ≥ 2 takes; 5 random draws; top-1/3/5, macro-averaged over speakers. *B, cross-session*: enrol one take per word from session A, test on session B's takes of those words, every ordered (microphone, A → B) pair, n = 646 test items over 6 speakers, n-weighted. *Open-set*: enrol half the vocabulary, the other half's recordings are impostors, score = top-1 similarity, AUROC and true-accept rate at 10% false-accept, 5 draws. "Session norm" = subtracting the mean of all utterances of the (speaker, session, microphone), the oracle version of the running session vector the live system keeps.

**Compute.** 2 CPU cores, no GPU; feature extraction ~3 clips/s; every experiment below runs in minutes.

## 4. R1: session-scatter compensation (hypothesis on record: cross-session .377 → ≥ .45)

Session scatter was estimated on the training speakers only, in the LDA space, from same-word cross-session differences (174–256 pairs per fold); a second estimate used the offsets between whole-session means. WCCN whitens it (λ = fraction of identity mixed in); NAP removes its top-k eigenvectors.

| condition (layer 9) | A top-1 / 3 / 5 | A severe | B cross-session top-1 / 3 / 5 |
|---|---|---|---|
| raw cosine | .339 / .451 / .486 | .284 | .169 / .294 / .362 |
| raw + session norm | .345 / .474 / .516 | .295 | .245 / .381 / .438 |
| LDA | .443 / .632 / .688 | .336 | .361 / .554 / .638 |
| **LDA + session norm (August baseline)** | .452 / .651 / .711 | .348 | .384 / .593 / .675 |
| LDA + WCCN λ=0.1 + sn | .460 / .664 / .724 | .355 | .385 / .598 / .663 |
| **LDA + WCCN λ=0.5 + sn** | **.469 / .675 / .739** | **.364** | **.412 / .605 / .675** |
| LDA + NAP k=1…16 (+sn) | .440–.456 | .334–.350 | .341–.382 |
| LDA + NAP on session offsets k=1…4 (+sn) | .444–.456 | .332–.350 | .356–.392 |

WCCN at λ = 0.5 is worth +1.7 / +2.8 points; NAP is worth nothing at any rank, on either scatter estimate. Per speaker the WCCN gain is small and mixed (M04 .20 → .23, M02 .35 → .36, F04 .59 → .62).

**Why the gain is small, the "which session?" probe.** A 5-fold logistic regression asked to identify the session from the projected features (six multi-session speakers, primary microphone, majority-class chance .613):

| features | session identifiable |
|---|---|
| LDA | .810 |
| LDA + WCCN λ=0.1 / 0.5 | .789 / .807 |
| LDA + NAP k=4 / 8 | .803 / .806 |
| LDA + session norm (his own sessions) | **.514** |

A transform learned from other speakers' sessions leaves his sessions almost as identifiable as before; only statistics from *his own* session remove the information, because session drift is speaker-specific. **Hypothesis not met by WCCN (.412 < .45); met by the estimator fix in §6 (.454 for layer 9 alone, .497 with statistics pooling).** Once the estimator is fixed, WCCN's apparent gain reverses (§6.2: −2.9 cross-session, −1.2 one-shot). Design consequence: `T` in `projection.npz` stays identity; the session vector stays the main mechanism, with the caveats from §6.5 that most of its value is the *enrolment-side* centring and that the live mean must be shrunk, never estimated from a handful of probe words.

### 4.1 Reproduced with the shipped code path

The same comparison run with the repository's own tools (`scripts/matcher/embed_manifest.py --torgo`, then `scripts/matcher/train_projection.py --eval`, 2 September 2026, mean+std features, α = 0.1). This protocol folds the microphone into the session id, so "cross-session" here includes same-take/other-microphone pairs and the absolute numbers are higher; the direction is identical:

| variant | cross-session top-1, macro [n-weighted] | session probe (chance .17–.50) |
|---|---|---|
| raw | .566 [.547] | .75 |
| **LDA** | **.689 [.669]** | .72 |
| LDA + WCCN | .674 [.655] | .71 |
| LDA + NAP k = 1 / 2 / 4 / 8 / 16 | .685 / .684 / .680 / .670 / .658 | .70–.72 |
| LDA + per-session mean subtraction (the runtime's session vector) | .733 [.717] | .23 |

Per speaker (LDA → +WCCN): F01 .814 → .799, F03 .594 → .579, F04 .790 → .777, M01 .721 → .701, M02 .664 → .647, M03 .908 → .913, M04 .545 → .530, M05 .476 → .445. This is the table the T-2 run reproduces on project hardware (`docs/measurements.md`); the "winner" written into `projection.npz` is *none*.

## 5. R2: open-set rejection with cohort normalisation (hypothesis on record: AUROC .74 → ≥ .85)

*(Design later removed from the runtime; kept as the record of why.)*

The cohort was the other seven speakers' word recordings in the same LOSO-projected space (2,000–3,300 items). AS-norm: `z = ½[(s − μ_q)/σ_q + (s − μ_e)/σ_e]` over each side's top-k cohort scores; Z-norm and T-norm are the one-sided halves.

| features | normalisation | AUROC (top-1) | AUROC severe | TPR @ 10% FAR | closed-set top-1 kept |
|---|---|---|---|---|---|
| LDA + sn | none | .692 | .624 | .335 | .543 |
| LDA + sn | AS-norm, top-50 / top-200 / all | .659 / .654 / .683 | .605 / .599 / .622 | .236 / .241 / .321 | .52 |
| LDA + sn | **AS-norm top-200, cohort excluding his vocabulary words** | **.745** | **.678** | **.434** | .532 |
| LDA + WCCN + sn | same | .750 | .684 | .419 | .546 |
| LDA + sn | Z-norm only (query side), other-word cohort, R2b | .758 | .687 | .420 | -|
| LDA + sn | logistic rejector on score features, trained on his own out-of-vocabulary words (R2b) | .711 | .638 | .360 | -|

Three lessons. (1) The cohort must consist of *other words*: when it contains other people saying his vocabulary, every correct match looks like a typical cohort score and normalisation *hurts* (.692 → .654). (2) Query-side z-normalisation alone is as good as the two-sided formula; the exemplar-side statistics add nothing with one exemplar per word. (3) A trained rejector on the speaker's own out-of-vocabulary speech did not beat plain z-norm with the data TORGO offers (a few dozen positives per fold); Lea et al.'s 98% false-positive reduction from explicit negative classes came with thousands of examples.

**Hypothesis not met (.76 vs .85).** The severe speakers sit at .68. With one exemplar per word, score-based rejection stays weak, exactly as Apple found (false-accept .34). §7.3 measures how much multi-session enrolment moves it.

## 6. R6: original ideas: what the matcher should compute (TORGO)

This section keeps the August pipeline (frozen WavLM → linear projection fitted on other speakers → session norm → cosine) and changes one thing at a time.

### 6.1 The estimator was the bottleneck

August's projection was scikit-learn's `LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')`, which estimates a Ledoit–Wolf-shrunk covariance *per word* and averages them. With 2–10 takes per word each per-class estimate collapses toward identity, so the pooled within-class covariance is badly over-regularised. Estimating **one pooled within-class covariance from all residuals** and shrinking it a fixed fraction α toward `tr(S)/d · I` (then the generalized eigenproblem `S_b v = λ S_w v`, top 128 eigenvectors) is textbook LDA and is what ships:

| layer-9 mean, LDA + session norm | A top-1 / 3 / 5 | A severe | B cross-session top-1 / 3 / 5 | B severe |
|---|---|---|---|---|
| scikit-learn per-class shrinkage (August) | .452 / .651 / .711 | .348 | .384 / .593 / .675 | .338 |
| pooled, Ledoit–Wolf α (it picks 0.002) | .473 / .690 / .741 | .333 | .415 / .624 / .701 | -|
| pooled, α = 0.02 | .497 / .714 / .776 | .364 | .440 / .669 / .743 | .335 |
| pooled, α = 0.05 | .495 / .722 / .779 | .355 | .460 / .678 / .751 | .365 |
| **pooled, α = 0.1** | **.502 / .714 / .782** | .359 | **.454 / .676 / .755** | .347 |
| pooled, α = 0.2 | .492 / .722 / .777 | .350 | .435 / .673 / .748 | .335 |
| pooled, α = 0.4 | .478 / .697 / .749 | .354 | .420 / .639 / .720 | -|

Too little shrinkage (Ledoit–Wolf's automatic 0.002) and too much (scikit-learn's per-class) both lose; α ≈ 0.05–0.2 is a broad optimum. `train_projection.py` gained a ten-line NumPy LDA and an `--alpha` flag, default 0.1.

### 6.2 What to pool, and how many layers

All rows: pooled-covariance LDA, α as stated, + session norm.

| features (dim) | α | A top-1 / 3 / 5 | A severe | B top-1 / 3 / 5 | B severe |
|---|---|---|---|---|---|
| layer 9 mean (768) | 0.1 | .502 / .714 / .782 | .359 | .454 / .676 / .755 | .347 |
| **layer 9 mean + std (1,536)** | 0.05 | .519 / .740 / .802 | .367 | .495 / .714 / .774 | .377 |
| **layer 9 mean + std (1,536)** | **0.1** | **.517 / .746 / .810** | .362 | **.497 / .721 / .796** | **.389** |
| layer 9 mean + std (1,536) | 0.2 | .522 / .745 / .814 | .375 | .508 / .723 / .793 | .395 |
| layers 8+9+10 means (2,304) | 0.1 | .507 / .744 / .804 | .372 | .483 / .721 / .791 | .386 |
| layers 8+9+10 means + std (3,072) | 0.1 | .535 / .755 / .816 | .419 | .523 / .740 / .797 | .445 |
| layers 8+9+10 + std, + WCCN λ=0.5 | 0.1 | .523 / .740 / .806 | .391 | .494 / .717 / .786 | .412 |
| layers 8+9+10 + std, + WCCN λ=0.2 | 0.1 | .503 / .714 / .782 | .371 | .472 / .680 / .752 | .389 |
| layers 8+9+10 + std, **no session norm at all** | 0.1 | .525 / .737 / .802 | .391 | .491 / .724 / .802 | .415 |

Layer sweep (August estimator, for the record): word identity peaks at layer 9 in both protocols (L8 .442 / .399, L9 .452 / .384, L10 .445 / .333, L7 .417 / .327, L12 .429 / .350); layer 10 is best for the *severe* group one-shot (.365) but worst cross-session. A score-level ensemble of separate per-layer matchers adds ≤ 1 point.

### 6.3 Unseen-word controls (is any of this memorised vocabulary?)

Each speaker's vocabulary is split in half by a hash of the word; the projection is fitted on the *other* speakers with that half's words removed; evaluation uses only that half (cross-session protocol, α = 0.1):

| features | fitted **without** the test words, B top-1 / 3 / 5 (severe) | fitted with them |
|---|---|---|
| layer 9 mean | .495 / .734 / .808 (.433) | .522 / .765 / .831 |
| **layer 9 mean + std** | **.531 / .737 / .820 (.478)** | .571 / .803 / .851 |
| layers 8+9+10 means + std | .522 / .743 / .807 (.472) | -|

Statistics pooling is a real +3.6 top-1 (+4.5 severe) on words the projection never saw. The extra layers are worth **nothing** on unseen words (.522 vs .531), their +2 on the full vocabulary is memorised TORGO word list. Vocabulary memorisation itself is worth ~3–4 points, which argues for fitting the shipped projection on recordings of *his* vocabulary (consented family voices, in the client phase) plus the dysarthric corpora.

**The recipe that ships: WavLM-base-plus layer 9, mean ⊕ standard deviation over frames (1,536-d) → pooled-covariance LDA, α = 0.1, 128-d → enrolment bank centred per session, live running mean shrunk by n/(n+20) → cosine.** No `T`, no WCCN.

### 6.4 Things that did not work (all with the August estimator unless stated)

| idea | A top-1 / 3 / 5 | B top-1 / 3 / 5 | verdict |
|---|---|---|---|
| layer-9 mean, baseline | .452 / .651 / .711 | .384 / .593 / .675 | -|
| segmental pooling, 3 equal segments, shared projection per segment | .326 / .466 / .518 | .224 / .362 / .427 | order information hurts, as DTW did |
| segmental pooling, 5 segments | .283 / .413 / .462 | .175 / .283 / .327 | worse still |
| mean-pool ⊕ 3-segment fusion | .388 / .578 / .630 | .331 / .508 / .584 | the segmental half drags the mean down |
| unsupervised CORAL (recolour test session to enrolment covariance, λ = 0.5) | -| .384 / .588 / .650 | nothing |
| NAP on session scatter, k = 1…16 | .440–.456 | .341–.382 | nothing |
| NAP on session offsets, k = 1…4 | .444–.456 | .356–.392 | nothing |
| probe-anchored session shift, 5 / 10 / 20 anchor words (test words exclude anchors) | -| .300 / .263 / .207 vs none .369 / .303 / .286 | hurts at every count |
| anchor affine map (ridge, λ = 10) | -| .220 / .162 / .093 | much worse |
| oracle session mean on the same subsets | -| .391 / .303 / .307 | the ceiling itself is only +0–2 |

In the projected space the whole-session shift is *small* (oracle centring gains 0–2 points on these subsets) while the per-take noise of a severe speaker is *large*, so any shift estimate from a few takes has more noise than signal. Centring only pays when the mean is over many utterances, hence the shrinkage in §6.5.

### 6.5 How many live utterances before the session vector helps?

Cross-session protocol, session-B utterances arriving in random order, session vector = running mean of what has been heard so far, shrunk by n/(n+n₀); enrolment side centred on its own full session mean (always known offline). Layers 8–10, Ledoit–Wolf α:

| test-side session vector | B top-1 / 3 / 5 | severe | top-1 in utterances 1–10 / 11–30 / 31+ |
|---|---|---|---|
| none (enrolment side centred only) | .418 / .596 / .673 | .344 | .412 / .407 / .421 |
| running mean, n₀ = 0 | .414 / .603 / .673 | .344 | **.344** / .387 / .427 |
| running mean, n₀ = 5 | .422 / .611 / .686 | .348 | .444 / .387 / .426 |
| **running mean, n₀ = 20** | **.428 / .610 / .685** | .349 | .425 / .407 / .432 |
| running mean, n₀ = 80 | .424 / .605 / .682 | .345 | .425 / .404 / .427 |
| oracle full-session mean | .421 / .615 / .684 | .350 | -|

Two design facts: the unshrunk running mean costs 7 points in the first ten utterances of a session (it subtracts the utterance from itself); and the live vector is worth ≈ +1 overall, the larger share of "session norm" (+3 in §6.2) is the enrolment side, which is free. This became `WILLSPEAK_MATCH_SESSION_WARMUP = 20`.

## 7. R3 / R4: EasyCall: many sessions, another language (Italian)

31 dysarthric speakers, ~67 commands each, one take per command per session, 2–8 sessions; phone microphone at 8 kHz (upsampled to 16 kHz). Severe = TOM 4–5 (F10, M05, M06, M07, M11, M14). Chance ≈ 1.5%. *(EasyCall is not used by the project this semester; these numbers are context only.)*

### 7.1 Cross-lingual transfer (enrol session 1, test every other session; n-weighted; recipe of §6.3)

| projection | all: top-1 / 3 / 5 | severe (TOM 4–5) | severe per speaker (top-1) |
|---|---|---|---|
| raw cosine | .423 / .546 / .595 | .202 / .322 / .385 | f10 .26 m05 .24 m06 .30 m07 .10 m11 .09 m14 .41 |
| raw + session norm | .487 / .634 / .683 | .278 / .416 / .487 | .29 .37 .38 .16 .12 .55 |
| **English (TORGO) projection, unchanged, + sn** | **.536 / .708 / .769** | **.329 / .507 / .586** | .29 .46 .46 .20 .12 .59 |
| English projection + TORGO WCCN + sn | .485 / .652 / .721 | .288 / .438 / .536 | -|
| Italian projection from 4 *healthy* Italian speakers (same commands) + sn | .585 / .734 / .787 | .325 / .503 / .577 | -|
| Italian projection, other dysarthric + healthy speakers (LOSO) + sn | .712 / .848 / .884 | .477 / .641 / .707 | .26 .73 .75 .28 .17 .61 |

The English-fitted projection helps Italian dysarthric speakers by +5 points (+5 severe) with no change of any kind, and WCCN, fitted on English sessions, hurts (−5). An in-language, in-vocabulary projection is worth another +18: the cheapest lever for the client phase is to fit the projection on consented recordings of his own vocabulary by other voices plus the dysarthric corpora.

### 7.2 Enrolment scaling: sessions, prototypes, recency (test = session 6; speakers with ≥ 6 sessions; n = 1,623)

English projection + sn, top-1:

| enrolment | max over takes | **mean prototype** | top-2 mean | mean ⊕ max | severe: max / mean |
|---|---|---|---|---|---|
| session 1 only | .547 | .547 | .547 | .547 | .333 / .333 |
| session 5 only (most recent) | .625 | .625 | .625 | .625 | .333 / .333 |
| sessions 1–2 | .654 | **.669** | .649 | .667 | .419 / .438 |
| sessions 1–3 | .709 | **.712** | .707 | .723 | .461 / .476 |
| sessions 1–4 | .742 | **.759** | .744 | .759 | .506 / .547 |
| sessions 1–5 | .760 | **.786** | .773 | .787 | .517 / .554 |

Three conclusions. (1) **Each additional day of enrolment is worth ~5 points, with no plateau at five.** (2) **The mean prototype beats max-over-takes** once the projection is in place, by 2–4 points, and by 4 on severe speakers, this is why the label score is the cosine to the mean of all takes. (3) **Recency matters**: the most recent session alone beats the oldest alone by 8 points. *(A nearest-session weighting was sketched from (3) and carried as a zero-weight option through 0.8.0; it was removed unmeasured on 10 Sep 2026.)*

### 7.3 Open-set rejection vs enrolment size (half the vocabulary enrolled, test = session 6, z-norm against an other-word cohort)

| projection | sessions enrolled | AUROC top-1 | **AUROC z-norm** | TPR @ 10% FAR | severe: AUROC z / TPR |
|---|---|---|---|---|---|
| English (+WCCN, conservative) | 1 | .715 | .808 | .566 | .703 / .395 |
| | 3 | .777 | .874 | .678 | .752 / .435 |
| | 5 | .811 | .898 | .737 | .784 / .503 |
| Italian LOSO | 1 | .853 | .943 | .846 | .854 / .641 |
| | 3 | .891 | .957 | .890 | .873 / .692 |
| | 5 | .903 | .962 | .892 | .890 / .717 |

The R2 hypothesis (≥ .85) is met at three enrolment sessions with the English projection and at one with an in-language projection. What moves rejection is the number of enrolled days, not the threshold.

## 8. R7: zero-shot Parakeet-TDT vs Whisper-small on severe sentences

TORGO sentence prompts of the four severe speakers, primary microphone, 287 clips (F01 20, M01 89, M02 92, M04 86), 40 minutes of audio; Parakeet-TDT-0.6B-v2 int8 via sherpa-onnx greedy, Whisper-small int8 via faster-whisper greedy (no context carry-over); WER after lower-casing and punctuation removal. Zero-shot, no fine-tuning, no leakage possible.

| speaker | Parakeet WER | Whisper-small WER |
|---|---|---|
| F01 | .560 | .674 |
| M01 | .588 | .749 |
| M02 | .462 | .625 |
| M04 | .818 | .905 |
| **all** | **.618** | **.753** |

Parakeet wins on every speaker by 9–16 points. Speed while sharing the CPU with another job: Parakeet 3.3 s per clip (RTF 0.38), Whisper-small 10.4 s (RTF 1.22). Failure modes: Parakeet produced 8 empty transcripts and no runaways. (Whisper was removed from the code in 0.7.0; the recorded T-1 run is the project's zero-shot baseline, and the fine-tune's baseline.)

## 9. What changed in the code, and what was later undone

*Status, 10 September 2026.* Rows marked **shipped** are in `src/willspeak/matcher/`; rows marked **removed** existed in 0.8.0 and were taken out because they were never built or measured for any real speaker; rows marked **client phase** wait on IRB approval.

| # | Change | Where | Evidence | Status |
|---|---|---|---|---|
| 1 | Projection = pooled-covariance LDA, α = 0.1; input = layer-9 mean ⊕ std (1,536-d); `T` = identity | `train_projection.py`, `embed_manifest.py`, `projection.npz` | §6.1–6.3 | **shipped** |
| 2 | Fit the shipped projection on consented recordings of **his vocabulary** by family voices + the dysarthric corpora; refit when the vocabulary changes | `train_projection.py` | §7.1 | client phase |
| 3 | Label score = cosine to the **mean prototype** of all takes | `scoring.py` | §7.2 | **shipped** |
| 4 | Enrolment bank centred per session (offline); live session vector shrunk n/(n+20) | `scoring.py`, `engine.py` | §6.5 | **shipped** |
| 5 | Cohort = other-word recordings only; query-side z-norm | -| §5, §7.3 | **removed** (never built; `vocab_override` off is the safeguard) |
| 6 | `WILLSPEAK_MATCH_REJECT_Z` set from his own ROC after ≥ 3 sessions | -| §5, §7.3 | **removed** with row 5 |
| 7 | "How is he today" policy from the session vector | `handsfree.py` | §2.2, §7.2, §6.5 | client phase; the nearest-session bonus that would have fed it was **removed** |
| 8 | Parakeet fine-tune recipe of §2.4 in `scripts/asr/train.py`: LR 2e-5 cosine, tokenizer unchanged, SpecAugment + speed {0.85…1.15}, dev-WER early stopping, best-4 checkpoint average, encoder frozen from the bottom only as far as the card forces | `scripts/asr/train.py` | §2.4, §8 | **shipped 10 Sep 2026**; first run = the TORGO experiment ([finetune-torgo.md](finetune-torgo.md)) |
| 9 | Contextual biasing via sherpa-onnx hotwords | -| §2.4 | **removed** (the stock int8 release has no BPE vocabulary; it could never activate) |
| 10 | Recording protocol: one pass over the vocabulary **every** session, across days; the probe list is measurement only | `docs/plan.md` | §7.2, §5, §2.4 | client phase |

Explicitly rejected this round, with numbers: NAP, CORAL, WCCN (after the estimator fix), multi-layer concatenation, segmental pooling, probe-anchor calibration, a trained score-feature rejector on TORGO-scale data, audio-visual input, cloud/LLM fallbacks.

## 10. Honest limitations

- TORGO is lab-read speech with a known word list and 8 speakers; the severe group is four people, and F01 has 16 test items. Gains are consistent across speakers, but the absolute numbers for the client depend on his severity and microphone far more than on anything here.
- EasyCall's "sessions" are repetition blocks whose spacing in time the paper does not state; the recency effect (§7.2) says they drift, but "one session = one day" is an assumption. The Italian corpus is 8 kHz phone audio.
- All session-norm numbers in §4–§7 are the *oracle* version (mean over the whole session); §6.5 is the only live simulation, and it says the live vector is worth ≈ +1 with proper warm-up.
- The open-set experiments use recordings of *other vocabulary words* as impostors; real out-of-vocabulary speech (conversation, laughter, coughs) is a different distribution.
- The Parakeet/Whisper timing in §8 was measured under CPU contention; the recorded Pi 5 bench (RTF 0.13) is the latency to quote.
- No fine-tuning was run this round (no GPU in the CPU pilot).
- Two literature items could not be read at source and are flagged, not relied on: Hojo et al. ASSETS 2025 (paywalled) and the medRxiv 2026 training-free severity estimator.

## 11. Sources (this round; Round 2's list still applies)

Asia-Pacific: takahashi25 Interspeech 2025 (ISCA) · arXiv 2509.16718 · 2608.01865 · 2606.13253 · 2605.22120 · 2604.21628 · 2504.12279 · CSSP 2026 10.1007/s00034-026-03515-4 · SIVP 2025 10.1007/s11760-025-05023-5 · ASSETS 2025 10.1145/3663547.3759757 · MDPI Appl. Sci. 15(3):1416 · Strathclyde preprint (strathprints 95189) · 2512.17474 · 2601.21347 · CDSD licence page · NKU-HLT/PB-DSR · 01Zhangbw/Awesome-Disordered-Speech.
Severe end: ReCANVo (Sci. Data 2023) · voc2vec 2502.16298 · Talkitt (Front. Psychol. 2023) · Lea et al. nonverbal sound detection (CMU) · MMDSR ICASSP 2024 · TORGO page (video not released) · NP-TORGO 2411.00980 · SALR (PLOS Digit. Health 2024) · DSSCNet 2509.13442 · npj Digit. Med. 2025 listener effort · MDPI Computers 13(1):10 · 2509.19946 · 2402.01796 · 2412.12111 · 2412.16874 · musaelab C36 · SAP challenge 2507.22047 · Universal Personalizer 2509.15516 · Apple LPM 2306.05446 · Euphonia Frontiers 2025.
Toolkit: Hatch 2006 (WCCN) · Campbell 2010 (WNAP) · CORAL 1612.01939 · CORAL+ 1812.10260 · Kaldi PLDA · 1803.09153 · Dehak 2011 · 1910.11643 · 2408.13355 · Rajan et al. 2014 (multi-enrolment PLDA) · 2104.01541 · Matejka 2017 (AS-norm) · Cumani 2023 · LIE-TAS-norm 2504.04512 · Zigel & Cohen 2003 · 2606.11279 · Wang et al. 2409.00099 · MMFA (Electronics 2026) · 2401.17632 · Zhang & Wu 2020 · 2410.04091 · Settle & Livescu 1611.02550 · GeMCL 2605.13075 · CLAP-IPA (NAACL 2024) · 2605.02804 · 2104.01454 · 2506.11169.
Recipes: NeMo fine-tuning docs · NeMo word-boosting docs · NeMo #14140, #14772 · TurboBias 2508.07014 / 2608.21343 · sherpa-onnx hotwords docs · 2406.07096 · DrivenData Parakeet tutorial · 2508.06391 (EUSIPCO 2025) · 2606.31722 · ValSub 2503.09906 · heyneo Parakeet CPU case study · 2606.19823 · Bhat et al. Interspeech 2022 · CUHK augmentation study · Adapt4Me 2603.20112 · DSUTA (EMNLP 2024) · SGEM 2306.01981 · DARS 2603.01369 · sanchez25 Interspeech 2025 · 2508.18732 · 2505.12991.
Products: Voiceitt site/FAQ/CP page/Intel brief · MassAITC and Pittsburgh pilots · Project Relate help centre and Play Store listing · Shor et al. Interspeech 2019 · Apple Personal Voice / accessibility pages · bioengineer.org (Azure + SAP) · SAP site · MDPI Electronics 12(20):4278 survey · Whispp, Repeech, SpeechAgent 2510.20113.
Data: TORGO (U. Toronto) · EasyCall (Turrisi et al. 2021, arXiv 2104.02542; HF mirror luigimontaleone/easycall2) · WavLM-base-plus (MIT) · Parakeet-TDT-0.6B-v2 int8 (sherpa-onnx release) · faster-whisper small.
