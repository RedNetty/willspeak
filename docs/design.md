# Design

Paths and commands below are relative to `app/` unless they refer to documentation.

- [Architecture](#architecture)
- [Design and safety](#design-and-safety)
- [Other voices in the room](#other-voices-in-the-room)
- [Corpus and recording](#corpus-and-recording)
- [Word error rate](#word-error-rate)
- [Repository organization](#repository-organization)


---

## Architecture

WillSpeak is a local FastAPI service with a static browser UI. Runtime data is
stored in SQLite and WAV files under `data/`.

### Components

```text
browser
  -> FastAPI routes (server.py)
     -> Store (db.py) -> data/corpus.db
     -> audio ingest/QC (audio.py) -> data/clips/, data/raw/
     -> prompt scheduler (scheduler.py, g2p.py)
     -> ParakeetASR (asr/parakeet.py)
     -> Matcher (matcher/engine.py, matcher/scoring.py)
     -> hands-free policy (handsfree.py)
```

Parakeet is the only open-vocabulary recognizer. The matcher is a separate,
closed-vocabulary path for enrolled words and control phrases.

For design rationale, evidence boundaries, and current research status, see
the [system design paper](#design).

### Startup

`run.py` adds `src/` to the import path and starts Uvicorn. Importing
`willspeak.server` opens the data store and seeds prompts. The FastAPI lifespan
hook optionally builds the matcher bank in a background thread.

The server binds to `127.0.0.1` unless explicitly configured otherwise.

### Recording flow

1. The browser records audio with gain control, echo cancellation, and noise
   suppression disabled.
2. `POST /api/clip` decodes and resamples to 16 kHz mono.
3. Capture QC rejects unusable duration or level and warns about clipping,
   noise, or edge silence. The [recording quality gate](#recording-quality-gate)
   then checks background noise, echo and a second voice.
4. Canonical audio is written before metadata is committed.
5. Review records the actual transcript and status without deleting audio.

### Live flow

1. The browser's adaptive energy gate creates an utterance (its limits in a
   room with other voices: [Other voices in the room](#other-voices-in-the-room)).
2. The server saves the clip as uploaded (edges included, like a recorder
   take) and decodes a copy with the edge silence trimmed.
3. Parakeet produces the sentence transcript.
4. The matcher ranks enrolled phrases from the same audio.
5. `handsfree.decide` applies sleep, wake, stop, and optional phrase-override
   rules.
6. The browser displays and speaks the decision text.
7. Confirmation or correction returns the utterance to the corpus.

### Export and training

`export.py` assigns normalized prompt text to sticky train, development, and
test splits. All takes of the same prompt remain in the same split. Critical
control phrases remain in training.

`scripts/asr/train.py` converts train/dev exports to NeMo manifests,
fine-tunes the model, and saves a `.nemo` checkpoint or an `adapter.pt`.
`scripts/asr/export.py` creates sherpa-onnx files for deployment, with an
adapter folded into the encoder graph when one is given. The recorded runs
and their scores are mapped in the [report map](../app/reports/Report%20Context.md);
the recipe and its traps are [research/finetune-torgo.md](research/finetune-torgo.md).
`scripts/asr/evaluate.py` reads only the selected manifest and writes
generated reports under `reports/`.

The full research and release protocol is documented in
[measurements.md](measurements.md).

### Runtime model contract

The Parakeet model directory contains:

```text
encoder.int8.onnx
decoder.int8.onnx
joiner.int8.onnx
tokens.txt
```

`ParakeetASR` loads the recognizer lazily and reuses it across requests. Audio
is always passed to sherpa-onnx as 16 kHz mono float samples.

### Important environment variables

The [device runbook](device.md#runtime-configuration) owns environment variables
and defaults; [matcher configuration](device.md#configuration) covers bank settings.

---

## Design and safety

WillSpeak speaks recognized text in another person's name. The design therefore
prioritizes recoverability, traceability, and local operation.

### Two recognition jobs

Parakeet handles open-vocabulary sentences. The exemplar matcher handles
enrolled words, short phrases, and control phrases. Both inspect the same live
utterance, but neither silently expands the other's vocabulary.

The matcher may replace a sentence transcript only when a caregiver enables
the override and the measured score and margin thresholds are met.

### Hands-free behavior

The live page opens an utterance with an adaptive energy gate and closes it
after a configurable pause, or at a caregiver-set cap (default 10 s). The
speaker does not need to press a control.

The gate measures level, not identity: another person talking near the
microphone opens it, and a conversation rarely contains the pause that
closes it. Measured on 11 Sep 2026 and designed around in
[Other voices in the room](#other-voices-in-the-room), microphone distance, the cap, a
threshold that follows the speaker's own level, and a proposed speaker gate.
Until the speaker gate exists, someone talking within arm's reach of the
microphone can have their sentence spoken in his voice. That is a known,
listed failure.

Three recorded control phrases are available:

- stop cancels a pending utterance or retracts a recent one;
- sleep pauses spoken output;
- wake resumes spoken output.

Each phrase is detected by transcript similarity and independently by the
speaker's enrolled exemplars. Control phrases must be distinctive, uncommon in
ordinary conversation, and recorded across multiple sessions.
Text detection compares whole-word windows, so a control such as "stop" does
not trigger just because those letters occur inside "unstoppable". Garbled
whole words can still match at the caregiver's chosen similarity threshold,
but every word of the phrase has to resemble the word it lines up with: "to go
to sleep" shares three words with "voice go to sleep" and would otherwise pass
on those alone, muting the device when he says he is tired. A window one word
narrower or wider than the phrase, compared without spaces, catches a
transcript that merged or split the words ("purpleelephant", "purple ele
phant"); those windows must be near-exact. A transcript with fewer words than
the phrase must also be near-exact, so "go to sleep" on its own is not the
sleep phrase.
With only one enrolled label there is no runner-up, so its margin is zero and
the control must pass the absolute matcher score threshold.

### Safety properties

- Every live utterance is saved before recognition.
- Empty recognition output is never spoken.
- The displayed text is the text sent to speech synthesis.
- A configurable preview delay gives time to cancel before speech.
- Recent output can be confirmed, corrected, or rejected in the caregiver
  panel; until then it is unverified and excluded from training and evaluation.
- The same waveform arriving twice within seconds is stored once.
- The microphone is ignored while the device is speaking to reduce feedback.
- Critical phrases can be spoken directly from a caregiver board without ASR.
- The application works without network access after model setup.

Recognition runs in a worker thread; status and caregiver requests remain
available during a decode. The live page serializes recognition requests and
queues replies in capture order. Pending speech waits for recording and
recognition to finish. Stop-phrase tests use a preview request that leaves the
sleep state, prior output and live session mean unchanged. Losing microphone
packets for five seconds, a track ending, or a processor failure stops the
listening status and discards stale replies. Browser voices are restricted to
local services ([voice API](https://developer.mozilla.org/en-US/docs/Web/API/SpeechSynthesisVoice/localService)).

### Known limits

- sherpa-onnx exposes the greedy path's per-token log-probabilities, which the
  server logs as `logp=` on each live clip line; they are not calibrated, so
  the UI does not display a confidence or underline uncertain words.
- With zero preview delay, an incorrect transcript may be spoken before a stop
  phrase can retract it. The retraction window counts from the end of
  playback (the page reports both the start and the end), because the
  microphone is off while the device speaks.
- The stop phrase cannot interrupt playback while the microphone is suppressed.
- Phrase-matcher thresholds do not transfer safely between speakers or
  microphones; they must be measured.
- Published dysarthric-speech results do not predict an individual's result.

### Expo comparison demo (built 22 to 27 October 2026, running on the Pi since 4 November)

For the expo I show several speakers' models side by side with stock, running the real models on the Pi. The interface is a new page in `app/static/`; the models and the stock numbers already exist.

Flow for one demonstration:

1. Choose a speaker (M01, M02 or M04) and one of that speaker's held-out test prompts.
2. The Pi loads that speaker's model (the pretrained adapter exported with `--int8-ops matmul`), decodes the clip and
   scores the transcript against the reference text.
3. It decodes the same clip with the stock model and scores that.
4. The page shows both transcripts, each clip's word errors and the difference between them.
5. Everything plays through the speakers so the audience hears it: the original recording, then each transcript in the
   synthetic voice. Speakers are compared one after the other: the page holds one speaker model at a time and keeps a
   running tally of every clip shown, across speakers, under How this works.

Rules that keep the demo fair:

- Only held-out test clips (their prompts are disjoint from the speaker's training data), from a list fixed before the
  expo that includes clips the model gets wrong. Single-word clips are the weak spot (about 75% WER) and are not hidden.
- Raw model output, no phrase list. The model for each speaker was trained on the other severe speakers and then on that
  speaker's own training recordings, so I say "per-speaker model" and make no claim about an unseen speaker or the intended user.
- Aggregate numbers sit beside the live ones, from recorded runs: stock 76.0% pooled and 62.0% on sentences against 41.9% and
  28.3% for the pretrained adapter (NeMo, three seeds; `docs/measurements.md`). They are replaced by the Pi's int8 figures once F-4 is done.

Model handling (decided 21 October): stock stays loaded for the whole demo. Choosing a speaker in the UI unloads the
previous speaker's model and loads the new one; after that, picking prompts just decodes. Two models are in memory at a
time (stock and one speaker), about 2 to 2.6 GB at the 0.9 to 1.3 GB measured for one, on an 8 GB Pi. The swap happens on
speaker selection, not when a prompt is chosen, and the page shows a loading state until it finishes.

Sound (22 October): the page plays a clip's original recording (`GET /api/compare/audio/<clip id>`, served from `data/demo/` for ids on the
fixed list only) and reads each transcript with the browser's own local voice. Chromium on the Pi finds voices only when `speech-dispatcher`
and `espeak-ng` are installed (`setup.sh` installs them; a Pi set up by unzipping the app needs `sudo apt install speech-dispatcher espeak-ng`
once, then Chromium restarted). The page fits one screen without scrolling from 1024 × 600 up.

What a visitor sees (27 October), kept deliberately plain: three numbered steps (pick a speaker, pick a recording, listen
and compare); three lines of ordinary text, the prompt, the standard model's words and the speaker model's words, each
wrapping on its own, with wrong words in red, missed words struck through and words a model added with a dotted
underline; each model's "N of M words wrong" and a one-sentence verdict; and two bars for the recorded all-words WER.
The words were lined up in shared columns until 31 October; I found the uneven gaps that made in the prompt line hard
to read, so each line now keeps normal word spacing. Nothing plays by itself (changed 27 October at my request): each row
has its own play button (the recording, or that model's answer read aloud with the spoken word highlighted where the
voice reports word boundaries), and Play all runs the three in order. Memory and swap times,
voice settings, presenter keys (1, 2, 3, N, P, R, T, Esc), the sentence and int8 figures and the demo's running totals are in
the How this works window.

Testing other models (4 November, at my request): a Model list beside the speaker buttons loads any complete model folder
under `models/` for the trained side, still decoding the chosen speaker's recordings (`POST /api/compare/speaker` takes an
optional `model`, a path inside `models/`; anything outside is refused). "Speaker's own" stays the default. Another
model's answers are labelled with its folder name and kept out of the tally so the expo totals only count the speaker
models. Choosing a speaker with the same model folder already loaded skips the reload.

Added 30 October, for showing it off at the table:

- **Try your voice** (key R): a visitor records up to 8 seconds from the Pi's microphone and both models (stock and the
  loaded speaker's) write down what they hear. Nothing is saved and nothing is scored, because there is no reference text;
  the page says the speaker's model was trained on one person and may do worse on a typical voice. It is a talking point,
  not the typical-speech check in `docs/TODO.md`: no number from it goes on the poster.
- **Tour** (key T): for an unattended table. It works through every recording of the loaded speaker (decode, play the
  recording and both answers, pause), then moves to the next speaker. Any key or click stops it.
- **The tally on screen**: under the recorded bars, the running word error rate of every test recording shown at the
  table, standard against trained. It is kept in the browser across reloads until Reset (How this works); try-your-voice
  recordings are not counted.
- **Decode time**: each answer shows how long it took on this device, so the Pi's speed is visible next to its accuracy.

Built: `src/willspeak/compare.py` holds stock (shared with the Speak page when it is the same model) and one swappable
speaker recogniser, with its own endpoints (`/api/compare/state`, `/speaker`, `/decode`, `/try`, `/audio/<clip id>`,
`/release`); `scripts/corpus/demo_clips.py` draws the fixed list. Setup steps: [pi5.md](pi5.md#comparison-demo).
Still open: the swap time on the Pi (one model took 3.25 s to load; the adapter model is untested) and whether memory
really falls after an unload (the page reports both); whether TORGO's licence allows playing its recordings in public.

### Release checks

Before deployment, verify the real model with `scripts/asr/check.py`, run
the full test suite, score the sticky test split, measure target-device latency,
and test stop/sleep/wake phrases in the actual room and microphone position.


---

## Other voices in the room

*From my live test on 11 September 2026. Section 5's self-tracking gate was
implemented on 27 October 2026 in `static/live.html`; it has not yet been
measured (P-G).*

### 1. The finding (11 Sep 2026, living room, Pi 5)

Talking with my mother in the room, with the Speak page listening, I said
"Hello, how are you". The page's *Hearing you…* status appeared 5–10 s after
I started speaking, and the reply reported 3741 ms for the recogniser -
against the 0.40 s the same box decodes a 3 s utterance in when it is idle
([pi5.md](pi5.md#running-willspeak-on-the-raspberry-pi-5-and-measuring-it), my run `raspberrypi-20260908-165319`).

Turning the sensitivity slider down (a bigger margin over the room) made the
device fast again and it transcribed only me. That worked because I was
closer to the microphone than my mother was, so it is not a fix. The
client's voice strength varies from day to day, and any fixed margin that
separates him from the room on a good day silences him on a quiet one. For
someone who cannot press anything, "the box stopped hearing me" is the
worst failure the device has. It also has to work in church, at a family
dinner or in a waiting room without the caregiver or the client
understanding any of this.

### 2. What the page actually does, and why a talking room breaks it

From `static/live.html` (`vadTick`), the gate is an energy gate:

- The level meter arrives ~23 times a second. The page keeps a **noise
  floor** as a slow average of the level, but it only updates the floor
  from moments that are *not* already above the gate and not while the
  device is talking.
- A block counts as speech when it is more than `margin` dB above that floor;
  the caregiver's sensitivity slider *is* that margin (6–20 dB, default 10).
  Three speech blocks in a row (~130 ms) open the gate; the configured pause
  (default 1.5 s) with nothing above the gate closes it.
- A cap ends the utterance regardless. It was a constant, 15 s; it is now
  a caregiver slider (§4).

Two consequences follow. First, the floor cannot rise to meet a
talking room: another person's speech is above the gate, so it is excluded
from the floor estimate, so the floor stays at the last quiet moment and
everything they say opens the gate. Second, a conversation rarely contains a
1.5 s silence, so once the gate opens on the other person it stays open until
the cap. The clip that went to the server was therefore not "Hello, how are
you", it was up to 15 s of both people. At the Pi's measured 0.13–0.25 ×
real time (idle to loaded) that is 2–4 s of decoding.

**Correction, same day.** I then noticed that the server did not load the
model until the first utterance, and until 11 Sep `latency_ms` *included*
that load. My measured load on the Pi is 3.25 s and a short decode
~0.45 s, together 3.7 s, which matches 3741 ms at least as well as the
long-clip explanation does. If "Hello, how are you" was the first thing said
after the server started, the load is the likelier cause of the 3741 ms and
the gate is the cause of the late *Hearing you…*. The two effects are
independent, and both are now handled: the server loads and warms the model
in a background thread at start-up (`WILLSPEAK_ASR_EAGER`, default on; the
page says *Warming up* until it is ready), `latency_ms` is the decode alone,
and a load that does happen inside a request is reported as `load_ms`. The
new `live clip` log line tells the two cases apart: `dur` long with
few `words` is the gate; `load=` present is the cold start.

The late *Hearing you…* has the same cause. That status is only set when the
gate *opens*. If the gate was already open on my mother, my words were
appended to a recording already in progress, and the next chance for the
status to fire was after that clip had been sent, decoded, and spoken back -
and the page ignores the microphone while the device is speaking, for as long
as the spoken text is estimated to take. Ten seconds of waiting is about what
that loop costs when the clip is long.

Two smaller facts. `latency_ms` on the page is Parakeet
alone (`asr/parakeet.py`); the matcher, if installed, runs after it and is
not in that number. Uvicorn's access log only records that a POST
happened, so the server now prints one line per utterance with the
clip's trimmed duration, decode time, SNR, speech level and the decision, a
long `dur` with few `words` is the signature of this failure.

### 3. Why no amount of tuning fixes it

An energy gate answers "is there sound above the floor?". In a room where
the noise *is* speech at roughly his level, the answer is yes,
continuously. Raising the margin makes it miss him on quiet days; lowering it
records the room. Noise suppression (RNNoise, DeepFilterNet, the browser's
own) is aimed at hum, fans and traffic and does nothing against another
person talking, and it is off in the capture chain on purpose, because it
gates breathy onsets and weak fricatives ([Design and safety](#design-and-safety)). In my
living room, volume stood in for identity only because the geometry favoured
me. The design needs layers that do not depend on that.

### 4. The four layers

**Geometry, the microphone.** Sound pressure falls 6 dB per doubling of
distance, so where the capsule sits decides how much room there is for any
software. The deployment mic is the **Movo USB lavalier** (bought; the mic
budget is spent), clipped at his collar, or mounted on the chair; **never at
his face**. The arithmetic for a person talking 1 m away:

| where the capsule is | his distance | his advantage over the other voice |
|---|---:|---:|
| lavalier at the collar | ~20 cm | ~14 dB |
| chair mount near the shoulder | ~35 cm | ~9 dB |
| either, with someone leaning in to 40 cm | -| 0–6 dB |

His day-to-day loudness swing is probably 6–10 dB (the corpus will give the
real figure: QC stores `speech_dbfs` for every take and the protocol tags
the day's condition). At the collar the advantage is bigger than the swing,
so a threshold that follows him (§5) has room to work. On the chair mount the
two are about equal, and on a quiet day no threshold exists that separates him
from the room, then §6 is not optional. Those numbers alone make the collar
the position to insist on. Lavaliers are usually omnidirectional. That is
fine, since proximity does the work, but the mic must stay clipped at the
*same* spot: the matcher compares embeddings recorded
weeks apart, and a capsule that migrates from collar to chest is a channel
change.

**A cap, bound the damage.** Added after the test: the 15 s constant is the
caregiver slider *Longest single utterance*, 5–15 s, default 10 s, saved
locally like the other capture knobs. At 10 s the worst case is ~1.5–2.5 s of
decoding whatever the room does, and the reply line says *cut at 10 s, a
long sentence, or other voices in the room* so a caregiver sees the cause in
words. A cap alone still leaves the worse failure, where the clip contains
someone else's sentence and the device speaks it in his voice. The next two
layers are for that.

**A threshold that follows him -§5.** Removes the daily-variance problem
without anyone setting a number.

**An identity gate -§6.** Removes the need for the energy gate to
discriminate at all.

### 5. The self-tracking gate (implemented 27 Oct, not yet measured)

The device already measures how loud he is: every confirmed utterance goes
through QC, and the transcribe response now carries `qc.speech_dbfs` and
`qc.noise_dbfs` for the clip. The page's meter and those fields are on the
same scale (dB relative to full scale), so the page can compare them.

Replace the fixed margin with a threshold that tracks his own recent speech.
State: the noise floor the page already keeps; a running record of the
`speech_dbfs` of the last N = 20 utterances the device actually spoke
(the ones `/api/live/confirm` was posted for, not ignored ones, not other
voices a caregiver marked *Not him*); and the time the gate last opened.

Three rules.

1. **Open threshold** = `max(floor + margin_room, his_level − margin_him)`,
   where `his_level` is the median of the record, `margin_room` is the
   existing slider's margin (keep it, default 10 dB; it protects against a
   quiet room's hum), and `margin_him` is about 12 dB, enough that a
   quiet day still opens the gate, small enough that a voice a metre away at
   the collar does not. With an empty record the second term is absent and
   the gate behaves exactly as today.
2. **Learn from what was spoken.** After `/api/live/confirm`, push that
   clip's `speech_dbfs` and drop the oldest beyond N. A quiet week lowers
   `his_level` by the second or third sentence; a loud day raises it.
3. **The leak.** The tracker can wedge: on a bad day the threshold is too
   high, nothing is confirmed, nothing is learned, it never comes down. So if
   the gate has not opened for M = 3 minutes while the meter has shown
   bursts within 6 dB *below* the threshold, lower `his_level` by 1 dB per
   such burst until it does open (never below `floor + margin_room`). A
   caregiver's *New session* resets the record.

What the caregiver sees: no new slider. The status line gains one plain
sentence when the leak has been running, *Having trouble hearing him today*
-and the existing *Not him* button in Recent is the way to tell it that a
clip was another voice (which must not enter the record).

How to know it works (P-G in [measurements.md](measurements.md)):
I will record myself at three deliberate loudness levels on two days with a
second person talking at a metre, log the threshold the tracker settles on
and whether each utterance opened the gate and whether the other voice did,
and put that table next to the manual-slider one from 11 Sep.

Where it lives: `vadTick`, `beginUtterance` and the confirm path in
`static/live.html`, with the constants `GATE_RECORD` (20), `HIM_MARGIN_DB`
(12), `LEAK_AFTER_MS` (3 min) and `LEAK_NEAR_DB` (6). Choices made while
building it: the tracked threshold decides both opening and staying open, so a
quieter voice cannot hold a clip open either (the noise floor still learns only
from sound below the room gate). A burst for the leak is three consecutive
meter blocks within 6 dB under the threshold, the same length that opens the
gate, and a leak ends when a spoken sentence is learned. The record is kept in
the browser's local storage, so a reload keeps it and New session clears it.
The page sends the threshold each clip opened at, which the server prints as
`gate=` in the `live clip` line for P-G. Tests: `tests/js/live.test.cjs` (the
three rules, Not him, New session, the 20-sentence median, the room-gate floor)
and `tests/integration/test_session_pipeline.py` (the log field). It has not
run against a real microphone or room.

### 6. The identity gate (H5, deferred to spring)

Once "is this him?" is answered by *his voice* rather than his volume, the
energy gate no longer has to discriminate: it can sit low and permissive -
"is the room quiet or not", and a quiet day costs nothing, because the gate
that rejects the other person is the identity gate.

Design: a caregiver enrols him once from the panel ("record 30 seconds of
him talking"; his accepted corpus takes work too). Every clip gets one
question before it is spoken: cosine similarity between its speaker embedding
and the enrolment centroid, against a threshold with hysteresis. Below it,
the clip is not spoken, lands in Recent as *ignored, didn't sound like him*
(recoverable, never silently lost), and the status line says *hearing other
voices*. Stage two, once that works on whole clips, runs the same check on the
first second after the gate opens, so the device stops listening to someone
else instead of recording 10 s of them.

Which model: a dedicated speaker-verification embedding rather than the
matcher's WavLM. WavLM layer 9 is tuned to *what* was said and costs seconds
per clip on the Pi in PyTorch; sherpa-onnx ships speaker-embedding
extractors (3D-Speaker ERes2Net, WeSpeaker ECAPA/CAM++ -~25 MB ONNX) that
answer "same person?" in tens of milliseconds on a CPU, with no new runtime.
Their speed on the Pi is a claim to verify with `bench.py`, not a number.

The caveat: those models are trained on typical speech. His voice is
still his voice, the embeddings mostly encode vocal tract and pitch, so it
should transfer, but it is unmeasured, and it is measurable this semester
without the IRB (T-4): enrol each TORGO speaker on one session, test that
speaker's other sessions and the other seven speakers, report false-accept
and false-reject with the threshold that trades them. The trade-off must be
stated on the poster: for someone who cannot press anything a false reject
(he speaks, the box stays silent) costs more than a false accept, so the
threshold sits on the permissive side.

Cost and the cut: the plumbing (model fetch, the extractor behind
`/api/live/transcribe`, the enrolment endpoint) is about a day, followed by
the decision rule, enrolment UI copy and T-4. The semester plan is
full through the Nov 3 prototype, so this displaces something, by the proposal's cut
order, F-3's middle points first. Decided 27 October: spring, because it is too much for
this semester (recorded in `docs/TODO.md`). Until then the limitation in §7 goes on the
poster.

### 7. The limit worth being honest about

Someone leaning in and talking 40 cm from his head is within a few dB of him
on any microphone that is not at his mouth. Geometry, the cap and the tracker
do not catch that; only identity does, and until §6 exists the device will
sometimes speak a caregiver's sentence in his voice. That goes on the failure
list and on the poster's limitation panel.


---

## Corpus and recording

### Storage

The source corpus consists of:

- `data/corpus.db`: sessions, prompts, clips, reviews, and corrections;
- `data/clips/`: canonical 16 kHz mono PCM16 WAV files;
- `data/raw/`: original uploads when they differ from canonical audio;
- `data/split_assignments.json`: stable prompt-level split assignments.

Audio is written as soon as an upload is decoded. Rejecting or retaking a clip
changes status in SQLite but does not delete the audio.

### Prompt and transcript

`prompt_text` is what the speaker was asked to say. `transcript` is what the
reviewer heard in the recording. They are separate because substitutions,
partial readings, and self-corrections are training-relevant.

Accept only after reviewing the actual recording. Empty or unusable takes
should be rejected, not given the prompt as a transcript.

Live utterances from the Speak page have no prompt; their `prompt_text` is the
placeholder `(live)`. The recogniser's output is stored on them only as a
suggestion, and the clip stays **pending** until a person accepts or fixes it
(review page, or Save in the caregiver panel). The server refuses to accept the
placeholder as a transcript, and any clip that was accepted that way by an
earlier build is returned to the review queue when the database opens. A
model trained or scored on its own guesses confirms its mistakes, and for a speaker it misrecognises most of the time those guesses are
wrong exactly where correction is needed.

### Capture format

The browser requests mono audio with automatic gain control, echo
cancellation, and noise suppression disabled. The server downmixes and
resamples with a polyphase filter.

Capture QC checks:

- duration between 0.35 and 25 seconds;
- usable RMS level and dynamic range;
- clipping ratio;
- estimated SNR;
- leading and trailing silence;
- source sample rate and channel count.

Warnings keep the take and ask the operator to correct the setup. Hard
errors reject the take from normal training use.

### Recording quality gate

Capture QC measures what can be read straight from the samples: length, level,
clipping. Three problems that ruin a training take cannot be measured that way,
so a small network estimates them for every take on the Record page:

- **Background noise** as SNR, in dB: how far the speech sits above the noise.
  40 dB is a quiet room; 0 dB means the noise is as loud as the voice.
- **Echo** as C50, in dB: sound arriving in the first 50 ms (mostly the direct
  path) against everything later (reflections). Above 25 dB is a close
  microphone; near 0 dB is a far microphone in a hard room.
- **A second voice**: the probability that someone else is talking.

It judges the recording, not the speech, and never decodes words.

**Decision.** A take passes when estimated SNR ≥ 11.2 dB, C50 ≥ 8 dB and
second-voice probability ≤ 0.64. The thresholds come from the validation
split: 97.5% of validation takes that were truly fine (SNR ≥ 15 dB, C50 ≥ 8 dB,
no second voice, fixed before training) pass each check. The first failing take
on a prompt is stored with status `retake`, the page shows what to fix ("Too
much background noise. Turn off the TV or fan…") and the prompt comes back. If
the next take on that prompt fails again, it is kept with a warning and marked
**QUALITY FLAG** in Review, because repeating a prompt costs the speaker real
effort. Every take stays on disk with its scores in `qc.gate`, so thresholds can
change later and old takes can be re-filtered. `WILLSPEAK_QUALITY_GATE=0`
switches the gate off.

**Network.** Log-mel features (64 bands, 25 ms windows every 10 ms), the clip's
mean log value subtracted so input gain cannot affect the result, four 1-D
convolution layers (48 channels, dilations 1, 2, 4, 8), mean and standard
deviation over time, and a two-layer head with three outputs: 53,699 weights.
The server runs it in NumPy (`src/willspeak/quality.py`) from
`src/willspeak/models/quality_gate.npz` (205 KB), about 0.7 ms per second of
audio on an x86 core; not timed on the Pi.

**Training data.** Nobody labels clips. `scripts/quality/mix.py` starts from a
clean recording and damages it by known amounts, and the amount applied is the
label: a room impulse response (label: that room's C50), another speaker at
5 dB louder to 15 dB quieter than the main voice (label: second voice), noise at
an SNR drawn from −5 to 35 dB (label: that SNR), then random gain, occasional
muffling and occasional clipping with no label, so loudness and microphone tone
carry no information. Half the base recordings are dysarthric (TORGO head
microphone) and half typical (LibriSpeech), under every label, so recognizing
the voice cannot help predict a label. Base recordings already noisier than
25 dB are skipped.

Sources and splits (nothing in test is seen in training):

- Dysarthric speech, TORGO: train F01, F03, M02, M03, M05; validation M01;
  test M04 and F04.
- Typical speech, LibriSpeech dev-clean (split by speaker) and test-clean (test only).
- Real recorded noise, DEMAND (17 places). Test places: living room, café, car,
  office; validation: kitchen, train station.
- Household sounds, ESC-50 by its own folds (fold 5 test). Breathing, coughing,
  sneezing, snoring and laughing are left out: those are sounds the speaker makes.
- OpenSLR 28: point-source noises (hashed split) and recorded diffuse room noise
  (test only); 60,000 simulated rooms for training and validation, 325 real
  recorded rooms for test only.

**Test results (synthetic tests, design context only, 5 November).** These are
not project results; the measurement that counts is on real recordings (TODO).

- Held-out mixtures, 3,000 clips in real rooms: SNR error 5.2 dB on noisy clips;
  C50 error 5.9 dB on echoed clips; second-voice AUROC 0.90 (the chance a clip
  with a second voice scores higher than one without; 0.5 is chance). It kept
  90.5% of truly fine clips and rejected 78.5% of bad ones. Errors cluster at
  the threshold: clips between 15 and 25 dB SNR were kept 85% of the time.
  Steady hum (car, office ventilation) is the weakest noise type, about 10 dB
  of SNR error against about 4 dB for the living room and café.
- Untouched held-out recordings: kept 93.9% of M04 (severe; 95% interval
  90.9 to 96.3%), 99.6% of F04 (mild) and 98.3% of LibriSpeech test-clean
  (97.8 to 98.8%). M01, the severe validation speaker, is kept 98.2%. M04's
  rejected takes have noisier recordings of their own (median 22.8 dB against
  31.6 dB for kept takes), so most of the gap tracks the recordings; a small
  effect of the voice cannot be ruled out with one test speaker.
- TORGO's array microphone, about 60 cm away, against the head microphone on
  the same utterances (no mixing, a condition never trained on): the far
  microphone scored noisier on at least 89% of pairs for every speaker. Most
  far takes still pass; for M04 about half and for M05 85% are rejected.

TORGO's own head-microphone files for F04, M04 and some of M01 are clipped
(one M01 file has 8,590 samples at full scale), which Capture QC reports.

**Retraining.** `./willspeak.sh quality-gate fetch | prepare | train | evaluate |
export` in the training environment (`scripts/quality/gate.py`; TORGO from
`./willspeak.sh torgo`). `export` rewrites the weights file and refuses to finish
if NumPy and PyTorch disagree. Training runs on a CPU in about an hour.

**Limits.** The thresholds are set on mixtures. Whether a flagged take actually
hurts fine-tuning is unmeasured; the direct test is fine-tuning on gated against
ungated data. Phone and laptop microphones were not in training.

### Sessions and variability

Record multiple sessions across days. Keep the deployment microphone and its
position consistent. Record difficult days and label them; do not filter them
out because recognition quality is lower.

The session record supports condition, self-rated speech, fatigue, a fixed
rater's severity label, microphone, room, and notes. Start/end ratings allow
within-session drift to be measured.

### Export

Export copies accepted clips into train, development, and test layouts. Split
assignment is based on normalized prompt text, so repeated takes of one prompt
cannot leak across splits. Existing assignments remain sticky across exports.
Live utterances have no prompt, so they are keyed on their verified transcript
instead; keying them on the `(live)` placeholder would put every live utterance
into one split for good. Clips whose only text is the placeholder are skipped
and listed under `skipped` in the export summary.

Critical control phrases are forced into training. Probe clips are used for
longitudinal measurement and are not enrolled in the phrase matcher.

### Backup

Use the Coverage page or `POST /api/backup` after every session. A backup
contains a consistent SQLite snapshot and every audio file, including rejected
takes, plus split assignments and caregiver settings. Each completed archive
has a unique name; failed backups do not replace an earlier one. Store at least
one copy outside the working machine.

Exports are built in a temporary directory before replacing the previous export
and split assignments. Invalid split maps are refused. All takes keep their
sticky prompt split; critical and probe sets stay in train. If normalized prompts
or corrected transcripts overlap across splits, export stops for review. It
does not silently move previously assigned prompts. Export destinations cannot
replace the corpus, recordings or backups.


---

## Word error rate

Word error rate (WER) measures substitutions, deletions, and insertions after
aligning a reference transcript with a hypothesis:

```text
WER = (substitutions + deletions + insertions) / reference words
```

WER can exceed 100% because insertions add errors without adding reference
words.

For a silence-only reference, empty output scores zero and hallucinated output
is reported as an insertion count (the [jiwer convention](https://jitsi.github.io/jiwer/)).
Report silence behavior separately from ordinary WER. The confidence-interval
script refuses references with zero scorable words rather than inventing a
denominator.

### Example

```text
reference:  i need cold water
hypothesis: i need water please
```

One deletion (`cold`) and one insertion (`please`) over four reference words
gives 50% WER.

### Alignment

`willspeak.evaluate.align` uses dynamic programming to find a minimum-edit token
alignment. `evaluate` aggregates operation counts across the complete corpus
before dividing. It does not average per-utterance percentages, which would
give short clips disproportionate weight.

### Normalization

`willspeak.normalize` lowercases, standardizes punctuation and apostrophes,
expands supported contractions, and normalizes numbers. The same normalizer
must be used for every comparison. Changing it requires re-scoring history.

The current frozen normalizer also expands apostrophe-free `ill` and `id` as
`I will` and `I would`. Those forms are ambiguous: `I feel ill` and `ID card`
can be altered even when transcribed correctly. This can change word counts
and hide or create errors. Keep the current form for the planned reproduction
checks; correcting it needs a named scoring revision and re-scoring both sides
of every comparison. Split assignments must remain frozen during that revision.

### Related metrics

- CER applies the same calculation to characters.
- PER applies it to phonemes generated by the project's G2P layer.
- Per-phoneme difficulty maps repeated phoneme errors into scheduler weights.

### Honest comparisons

A reported change is meaningful only when all of these remain fixed:

- the test manifest and audio;
- prompt-level split assignments;
- transcript normalization;
- model decoding settings;
- grouping and aggregation rules.

Use development data for training choices. Use the sticky test split for named
comparisons, and keep the generated `results.json` and hypotheses together.
The interval script checks recomputed scores against those saved results before
writing an interval. Its paired bootstrap resamples clips, so the interval
is conditional on the fitted models and observed speakers/sessions; it does
not include training-seed variation or establish performance on new people or
days. Repeated takes of a prompt are also not independent prompt observations.
Use `--cluster-by text` to resample their complete groups, or `--cluster-by
speaker session` for recording visits. Combining speaker and session avoids
merging different speakers' `Session1` recordings. A single held-out session
cannot estimate between-session spread, and intervals from a few groups are
unstable. The method remains a paired percentile bootstrap; both models use
the same sampled units and WER is total errors divided by total reference words,
including when groups have different sizes. See the [SciPy bootstrap
documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
for the distinction between paired sampling and interval methods.

New intervals identify the manifest, reference and candidate reports, scoring
files, resampling unit and seed. Where report provenance exists, the script
refuses changed manifests, changed outputs and different recorded audio bytes.
Legacy reports without provenance are labeled unverifiable on that point.
`ci.json` holds the latest comparison; `ci/<comparison_id>.json` keeps each
distinct baseline/comparison so a later invocation cannot erase the earlier one.

The current TORGO test sets have informed repeated recipe choices. The recorded
scores remain useful descriptions, but later comparisons on those sets are
exploratory. Do not select a seed, recipe or prototype from the best test WER and
then present its interval as confirmation. Specify the decision metric and
recipe on development data, report every planned seed and comparison, and use
new untouched evaluation data for confirmation. A nominal 95% interval is not
a simultaneous guarantee across all the comparisons tried. These limits follow
from [research on adaptive holdout reuse](https://arxiv.org/abs/1506.02629).

`tests/unit/test_wer.py` cross-checks the implementation against `jiwer` and covers
alignment invariants, empty input, insertions, and aggregate arithmetic.


---

## Repository organization

The code has two distinct recognition paths: Parakeet transcribes sentences;
the WavLM exemplar matcher recognizes enrolled phrases. Their commands and
implementation modules are grouped separately so a matcher experiment cannot
be mistaken for another sentence-ASR backend.

### Directory ownership

```text
README.md                       entry point
docs/                           task list, eight guides, research/ (four references)
  wordlists.txt                 candidate vocabulary, not loaded by the app
app/                            copy/package this folder to run elsewhere
  README.md                     destination-machine quickstart
  run.py                        cross-platform server entry point
  willspeak.sh, willspeak.bat        command menu and dispatch
  lab.py                        the lab tool: stages, queues, settings, space, wandb
  package.py                    clean portable ZIP, optional models
  pyproject.toml                package metadata and dependency definitions
  requirements*.txt             install from pyproject.toml
  src/willspeak/
    asr/                        sentence-recognition runtime
    matcher/                    exemplar engine, scoring, projection training
    models/                     shipped projection and quality-gate weights
    server.py                   HTTP routes and application wiring
    audio.py, db.py, export.py   audio quality, corpus and sticky splits
    quality.py                  recording-quality gate for the Record page (NumPy)
    artifacts.py, bundles.py     provenance and portable snapshots/releases
    backup.py                   recovery backups and validated restore
    evaluate.py, report.py       metrics and HTML reports
    g2p.py, normalize.py         phonetic/text processing
    scheduler.py, handsfree.py   prompt selection and live control policy
    paths.py                    application storage roots
  scripts/
    fetch_models.py             shared model setup
    _cli.py                     shared logging
    asr/                        check, bench, train, export, evaluate, ci, nemo_hyp;
                                lab helpers (labkit, lab_space, lab_doctor,
                                long_runs and its three .sh entry points) and
                                the lab-test scripts
    corpus/                     acquisition, snapshots, recovery
    matcher/                    embeddings, projections, evaluation
    quality/                    recording-quality gate: data, training, tests, export
  static/                       browser pages and shared assets
  prompts/                      active recording prompts
  deploy/pi5/                   installer, audio checks, boot service, kiosk
  tests/unit/, tests/integration/, tests/js/
  reports/                      notes, report map, lab result files
  logs/                         per-command and lab logs (lab results tracked)
  models/, data/, datasets/, transfers/   local runtime files
```

The application resolves paths from its own `pyproject.toml`; it does not need
Git, the outer repository directory, or personal docs. Launchers and scripts
also work when invoked from another working directory. Create a fresh Python
virtual environment on each destination and install from `app/requirements.txt`.

`python app/package.py --out willspeak.zip` (from the repository root) packages
only application inputs. `--with-models` adds downloaded model files. The
shipped projection and quality-gate weights are always included. Data, generated reports, virtual
environments and caches are never included. Keep private corpus transfers
separate using `scripts/corpus/transfer.py`. `.gitignore` covers all runtime
paths above except `models/.gitkeep`, public experiment notes,
the [report map](../app/reports/Report%20Context.md), and selected T-1 baseline evidence.
Lab result files (decodes, score tables, `logs/`, `models/*/run.json`) reach Git
only when the lab computer force-adds them (`python lab.py space git-results`
puts them on a side branch); once tracked they are kept as evidence.
Weights & Biases run folders (`logs/wandb/`) and audio stay out of Git.

### Prompt sets

**Prompts.** At start-up the server inserts new lines from `prompts/*.txt` into
the prompt table (existing rows are never overwritten; `#` lines are comments):
`critical.txt` (critical and control phrases; forced into the train split,
eligible for the matcher; target 15 takes), `functional.txt` (starter vocabulary
to extend with the speaker and family; 4), `harvard_seed.txt` and
`phonetic_arctic.txt` (phonetically balanced sentences; ARCTIC fetched by
`scripts/corpus/fetch_prompts.py`; 1), `probe.txt` (the fixed longitudinal list
served at session start and end, excluded from the matcher and from coverage
scheduling, changing an item starts a new measurement series, so document it
rather than edit silently). [candidate vocabulary](wordlists.txt) holds the editable reference sections
(`functional-starter`, `critical-phrases`, `control-phrase-candidates`) that are
not loaded directly; promote entries into `prompts/`. Choose control phrases
from measured cross-session performance, not pronunciation preference.

### Where changes belong

- Put reusable recognition behavior under `src/willspeak/asr/` or
  `src/willspeak/matcher/`, not in shell wrappers. The matcher package's
  `engine.py` owns the live bank; `scoring.py` and `training.py` stay NumPy-only.
- Keep cross-component storage and policy modules at the package root. Avoid
  introducing a one-file package for every helper or a generic `utils/` drawer.
- Put an executable research/operations workflow in the relevant `scripts/`
  group. Use full grouped paths in documentation. ASR evaluation and matcher
  evaluation are different commands even though both files are named `evaluate.py`.
- `willspeak.sh` / `willspeak.bat` only choose the interpreter and dispatch to a
  script; no pipeline logic lives in them. Add a new command there and to the
  launcher test in `tests/integration/test_entrypoints.py`.
- Put reusable test data generation in `tests/helpers.py`; keep imports from
  other test modules out of the suite. Use integration tests when filesystem,
  subprocess, HTTP, or multiple subsystem boundaries are the subject of a test.
- Keep research detail. Operational instructions can link to derivations
  rather than duplicate them, but moving a paper is no reason to shorten it.

The old flat script/module locations are not kept as compatibility shims.
All in-repository imports and commands use the canonical grouped paths. External
personal scripts should use the [application quickstart](../app/README.md).

### Dependency and environment setup

`pyproject.toml` is the single dependency source. From the application directory (`app/`),
`python -m pip install -r requirements.txt` delegates to `-e .`, and
`python -m pip install -r requirements-dev.txt` delegates to `-e .[dev]`.
Training remains the `train` extra. Install the machine-appropriate PyTorch
build separately, following the [training runbook](training.md#parakeet-training-and-evaluation).

The browser UI and prompts travel inside `app/`. Install and run from that
folder; a wheel alone does not include the complete application. Runtime data
belongs in `app/data/` or the explicitly configured corpus directory.

### Checks after moving or renaming files

```bash
python -m pytest tests/unit
python -m pytest tests/integration
python scripts/asr/check.py
```

The first two commands exercise automatic regression coverage. The third loads
the installed model and decodes silence; add `--audio <approved.wav>` for speech.
It does not replace an evaluation on real speech or a real training/export run.

Repository regression tests check documented local links and command targets,
safe `--help` behavior, and launchers from a different working directory. Native
Windows execution still needs a Windows check; parsing its batch wrappers on
macOS is not equivalent. When adding a new group, update the root README,
command inventory, test guide, and relevant runbook together.
