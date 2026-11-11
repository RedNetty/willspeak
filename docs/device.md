# The device: Speak page, matcher, recogniser

Paths and commands below are relative to `app/` unless they refer to documentation.

- [Runtime configuration](#runtime-configuration)
- [Hands-free operation](#hands-free-operation)
- [Exemplar matcher operation](#exemplar-matcher-operation)
- [Parakeet: install, validate, run, troubleshoot](#parakeet-install-validate-run-troubleshoot)


---

## Runtime configuration

| Variable | Purpose | Default |
|---|---|---|
| `WILLSPEAK_DATA` | Corpus directory | `data/` |
| `WILLSPEAK_PARAKEET_DIR` | Parakeet ONNX model directory | `models/parakeet-tdt-0.6b-v2-int8/` |
| `WILLSPEAK_ASR_THREADS` | sherpa-onnx CPU threads | up to 4 |
| `WILLSPEAK_ASR_EAGER` | Load and warm Parakeet at startup | `1`; `0` loads on first utterance |
| `WILLSPEAK_MATCH` | Enable the exemplar matcher | `1` |
| `WILLSPEAK_MATCH_DEVICE` | Matcher device | `auto` |
| `WILLSPEAK_MATCH_EAGER` | Build the phrase bank at startup | `1` |
| `WILLSPEAK_QUALITY_GATE` | Check Record-page takes for noise, echo and a second voice ([design](design.md#recording-quality-gate)) | `1`; `0` keeps only the level checks |
| `PORT` | HTTP port | `8000` |

See [matcher configuration](#configuration) for additional bank settings.

## Hands-free operation

The Speak page takes speech in and gives speech out without the speaker
pressing a button.

### Start

1. Verify the model: `python scripts/asr/check.py`.
2. Start WillSpeak: `python run.py`.
3. Open `/live` and confirm the status shows the expected Parakeet directory.
   The server loads the recogniser in the background as soon as it starts
   (about 3 s on the Pi 5, plus one warm-up decode); the page says *Warming
   up, a few seconds* until then and *Listening for speech* after. Before
   11 Sep 2026 the load happened inside the first utterance, which is why a
   first reply used to take ~4 s. `latency_ms` is the decode alone; when a
   load did happen inside a request the response carries `load_ms` too.
4. Open the Caregiver panel and review stop, sleep, wake, voice, delay, endpoint
   pause, and sensitivity settings.
   The voice list contains local English voices only. Install one if it says
   **Voice needs setup**; an online voice would break offline operation.
5. Press **New session** when the day, room, or microphone position changes.

### Live loop

The browser keeps a one-second pre-roll, starts recording after sustained
energy above the local noise estimate, and ends after the configured pause
or at the *Longest single utterance* cap (default 10 s). The server saves the
upload as recorded, trims a copy, transcribes, matches, and returns one
decision, and prints one line
per utterance (`[willspeak] live clip N: dur=… decode=… snr=… speech=…
gate=… action=… words=…`), the first place to look when a reply was slow or wrong.

The gate that starts a recording follows his voice. Until something has been
spoken it opens at the room level (the noise floor plus the sensitivity
margin). After that it opens at the higher of the room level and his own
recent speaking level less 12 dB, where his level is the median of the last 20
sentences the device spoke. A voice in the room that is quieter than that at
the microphone no longer starts or holds a recording. On a quiet or loud day
the gate moves with him without anyone changing a setting. If the
gate has not opened for three minutes while sound keeps arriving just under
it, it lowers itself 1 dB at a time and the status line adds *Having trouble
hearing him today*; the next sentence spoken resets that. Device details shows
his learned level and where the gate opens; `gate=` in the log line is that
level for each clip. It still measures level, not who is speaking, so someone
leaning in close can open it. Why, and what would fix that:
[design.md](design.md#other-voices-in-the-room).

Parakeet supplies sentence text. The matcher supplies enrolled-phrase
candidates. The screen displays the chosen text and browser speech synthesis
speaks it.

Utterances are decoded in capture order. Spoken replies wait while another
utterance is being recorded or decoded, so a stop phrase that follows can
cancel the pending reply. A reply is marked as spoken when playback starts
and again when it ends; the retraction window counts from the end.
Microphone reconnection discards replies from the previous connection. A
recogniser failure comes back as "Not spoken" with the reason, and the clip is
kept; it is not a server error.

What was spoken is recorded on the clip as an unverified suggestion; the clip
stays pending. It joins the corpus (and, if short, the phrase bank on the next
rebuild) only after a person confirms or corrects it in the caregiver panel's
Recent list or on the review page. Control-phrase detections are handled the
same way, so a false stop/sleep/wake never becomes a labelled take on its own.
The Recent list adds eligible corrected takes immediately. The review page
removes stale bank entries; rebuild after reviewing to enroll the corrected takes.

An utterance that arrives twice within a few seconds with the same waveform
(a second Speak tab on the same microphone, or a re-sent upload) is answered
with `ok: false` and not stored a second time. Run one Speak page per
microphone: a second tab also records the device's own voice, because it cannot
see that the first tab is speaking.

### Stop, sleep, and wake

Use phrases that do not occur in normal conversation. Each saved phrase
becomes a critical recording prompt.

- Stop cancels an item during the preview delay.
- Stop shortly after speech announces a retraction.
- Sleep pauses translation but keeps listening for wake.
- Wake resumes translation.

Test all three after microphone, room, model, or threshold changes. A control
phrase with zero accepted takes cannot trigger through the matcher.
**Test the stop phrase** previews recognition without speaking, changing the
sleep state, retracting previous output, or updating the live session mean.
The default text similarity is still 0.85. A matcher margin of 0 disables that
alternative detector; the score threshold still applies. Similarity scores are
not probabilities of being correct.

### Caregiver tasks

- Confirm or correct recent transcripts (Save) and reject non-speaker audio
  (Not him). Until saved, a live clip is unverified and is not exported.
  Not him also removes the clip from the gate's record of his level.
- After moving the microphone or on a new day, choose **New session**: it
  resets the matcher's session and the gate relearns his level from scratch.
- Rebuild the phrase bank after reviewing new recordings.
- Add vocabulary as prompts, then record and accept it before expecting a match.
- Set a preview delay when immediate errors are too costly.
- Use the critical phrase board when recognition is unavailable.

### Recovery

- **Microphone needs attention**: the track ended, the audio processor failed,
  or no audio packets arrived for five seconds. Check the connection and choose
  **Reconnect microphone** in Device details. A silent room still produces
  packets and does not trigger this warning.
- **Voice needs attention**: check the local voice and speaker. Failed playback
  is not marked as spoken.
- `no sentence recognizer`: run `scripts/fetch_models.py --parakeet-only` and
  `scripts/asr/check.py` in the same Python environment as the server.
- matcher unavailable: install the correct PyTorch build, fetch WavLM, and
  rebuild the bank.
- frequent early cuts: increase the endpoint pause.
- frequent false starts: lower sensitivity or move away from steady noise.
- *Having trouble hearing him today*: the gate has not opened for three
  minutes and is lowering itself. Check the microphone is still clipped at the
  collar; it recovers on the next sentence the device speaks.
- device hears itself: reduce speaker volume, increase microphone separation,
  and confirm capture is suppressed during playback.


---

## Exemplar matcher operation

Model setup saves WavLM in `models/wavlm-base-plus/`, so `package.py
--with-models` can carry it to another machine. Older user-cache models are
still readable; rerun `scripts/fetch_models.py` to save a portable copy.


The matcher recognizes short labels backed by the speaker's accepted clips.
It runs alongside Parakeet, mainly for control phrases and known functional
vocabulary.

### Requirements

- PyTorch and Transformers
- `models/wavlm-base-plus/` (or an existing local Hugging Face cache)
- `src/willspeak/models/projection.npz`
- accepted clips with short transcripts

Fetch and verify assets with:

```bash
python scripts/fetch_models.py --check
```

### Bank lifecycle

The bank is built from accepted clips whose normalized transcript has at most
`WILLSPEAK_MATCH_MAXWORDS` words. Probe clips are excluded.

`data/match_cache.npz` stores rebuildable embeddings. Rebuild after a recording
session or review changes:

```bash
curl -X POST http://127.0.0.1:8000/api/live/match/rebuild
```

Reset the live session mean after a new day or microphone move:

```bash
curl -X POST http://127.0.0.1:8000/api/live/match/new_session
```

### Scoring

Each clip's WavLM layer-9 frames are pooled to a mean and a standard deviation
(1,536-d). The matcher session-centers the bank, applies the packaged
projection (shrinkage LDA to 128-d, no session transform), length-normalizes,
and builds one prototype per label -- the mean of that label's takes. A live
utterance is centered on the shrunk live session vector and scored by cosine
to every prototype. The top-two margin distinguishes a clear win from a
crowded score. These choices came out of the Round 3 CPU pilot runs on
TORGO and EasyCall ([research/round-3.md](research/round-3.md)); the recorded
TORGO measurement of the same recipe is the number the project reports.
The per-take view stays available in each candidate (`best_take`, `session`).

An exact Parakeet transcript match may add a small vote to the corresponding
label. Fuzzy transcript fusion is not used.

The matcher has no open-set rejection: it always returns the nearest enrolled
label, and whether that label is *spoken* is decided in `handsfree.py`, only
when the caregiver setting `vocab_override` is on and both its thresholds are
cleared.

### Configuration

| Variable | Default | Purpose |
|---|---:|---|
| `WILLSPEAK_MATCH` | `1` | Enable matcher |
| `WILLSPEAK_MATCH_DEVICE` | `auto` | `cpu`, `cuda`, or automatic |
| `WILLSPEAK_MATCH_PROJECTION` | packaged file | Projection path |
| `WILLSPEAK_MATCH_MAXWORDS` | `6` | Longest enrolled label |
| `WILLSPEAK_MATCH_SESSION_WARMUP` | `20` | Utterances before the live session vector and the bank-mean prior weigh the same (0 = plain running mean; measured to hurt early in a session) |
| `WILLSPEAK_MATCH_EAGER` | `1` | Build the bank in a background thread at server start |

### Evaluation

Run the pure scoring tests and corpus evaluation:

```bash
python -m pytest tests/unit/test_scoring.py tests/unit/test_scatter.py
python scripts/matcher/evaluate.py --data data
```

Choose thresholds from cross-session results on the speaker who will use the
device. Record control phrases across several days before relying on them
(in the EasyCall CPU pilot runs every additional enrolment day was worth
about five points of closed-set accuracy, context, not a project result).

### Troubleshooting

#### No usable exemplars

Confirm clips are accepted, have actual transcripts, belong to an included
prompt set, are within the word limit, and exist on disk. Rebuild after review.

#### WavLM download fails

Run `python scripts/fetch_models.py` on the setup machine, then package with
`python package.py --out /path/to/willspeak.zip --with-models`. This includes the
local WavLM directory as well as Parakeet. Verify with `--check` on the destination
before offline use.

#### Same-day accuracy is high, cross-day accuracy is low

This is the failure to expect. Check microphone consistency,
session metadata, takes per label, session centering, and projection. Record
more sessions before lowering thresholds.

#### Every unknown phrase becomes a known phrase

That is normal closed-set nearest-neighbor behavior; the matcher has no
rejection. Leave override disabled; if a rejection capability is ever built,
it needs disjoint negative data and a measured threshold, not a raw cosine
cut-off picked from a few examples.

#### Bank rebuild is slow

The first run embeds uncached clips; later runs reuse raw vectors. Confirm the
cache path is writable and IDs are stable. Measure rebuild separately from
single-query latency.

#### Projection appears to hurt

Check that input vectors are session-centered exactly as in fitting, the file
dimension matches WavLM, and the evaluation speaker was not in the projection
training data. Keep the raw baseline visible; a packaged transform still has
to beat it on current evidence.

### Deployment gate

- Multiple sessions exist for control phrases.
- Cross-session top-1/top-3 are reported.
- Unknown speech, silence, and noise were tested.
- If override is enabled, its score and margin thresholds are frozen from
  development evidence and the false-override count is measured.
- Cold-start and steady-state session behavior are measured.
- Target-device query and rebuild latency are recorded.
- Sentence recognition remains available for content outside the bank.

For the representation choice, projection equations, historical results and
negative findings, see
[research/matcher.md](research/matcher.md).


---

## Parakeet: install, validate, run, troubleshoot

This tutorial takes a clean checkout from Python setup through a real
sherpa-onnx decode and explains what each check proves. The supported
sentence model is NVIDIA Parakeet-TDT-0.6B v2; there is no runtime selector.

### 1. Choose the machine role

The deployment/runtime machine needs:

- Python 3.11 or newer;
- enough memory for the int8 graph;
- `sherpa-onnx`, NumPy, SoundFile, FastAPI, and the willspeak package;
- local model files after the first download.

The training/export machine additionally needs Linux or WSL2, a CUDA-enabled
PyTorch build, NeMo, ONNX, and ONNX Runtime. Do not install the training stack
on a low-power deployment machine unless it also serves as the trainer.

Run all commands from the application directory (`app/`).

### 2. Create the runtime environment

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

`requirements.txt` intentionally does not choose a PyTorch build because the
right wheel depends on CPU, CUDA, operating system, and architecture. Install
PyTorch separately if the exemplar matcher will be used. Parakeet inference
itself runs through sherpa-onnx.

Confirm the active interpreter and core runtime:

```bash
python -c "import sys, sherpa_onnx; print(sys.executable); print(sherpa_onnx.__version__)"
```

If that command points outside `.venv`, fix environment activation before
continuing. Installing packages into one interpreter and running another
commonly causes false missing-dependency errors.

### 3. Inspect model readiness without downloading

```bash
python scripts/fetch_models.py --check --parakeet-only
```

This checks two things:

1. `sherpa_onnx` imports in the active interpreter.
2. The selected model directory has an encoder, decoder, joiner, and token
   file.

A fresh checkout should report missing model files. That is expected; the
repository is not broken.

The default location is:

```text
models/parakeet-tdt-0.6b-v2-int8/
```

Override it without moving files:

```bash
WILLSPEAK_PARAKEET_DIR=/absolute/path/to/model \
  python scripts/fetch_models.py --check --parakeet-only
```

On Windows PowerShell:

```powershell
$env:WILLSPEAK_PARAKEET_DIR = "D:\models\parakeet-tdt-0.6b-v2-int8"
python scripts/fetch_models.py --check --parakeet-only
```

### 4. Download the supported graph

```bash
python scripts/fetch_models.py --parakeet-only
```

The downloader streams the archive through a temporary file, extracts the
expected model directory, and verifies the final layout. Rerun the check after
download:

```bash
python scripts/fetch_models.py --check --parakeet-only
```

A complete runtime directory normally contains:

```text
encoder.int8.onnx
decoder.int8.onnx
joiner.int8.onnx
tokens.txt
```

Do not combine an encoder from one download with a decoder or token file
from another.

### 5. Load the real graph

File presence is not enough. Run:

```bash
python scripts/asr/check.py
```

The command creates one second of silence in a temporary WAV, constructs the
same `ParakeetASR` adapter used by the server, loads the actual ONNX graph, and
performs one decode. It prints a JSON response and removes the temporary file.

Expected properties:

- process exit code is zero;
- `model` names the selected model directory;
- `device` is `cpu` after the graph loads;
- `latency_ms` is a non-negative integer;
- silence normally produces empty text.

Silence should produce empty text. If it does not, investigate silence and
noise behavior by hand before deployment.

### 6. Decode representative speech

Use reviewed 16 kHz or higher-quality source audio:

```bash
python scripts/asr/check.py --audio data/clips/example.wav
```

The adapter accepts other sample rates and channel counts because the audio
module downmixes and resamples, but deployment measurements should use the
actual microphone path.

Repeat with:

- a short functional phrase;
- a longer sentence;
- a difficult-day recording;
- room silence;
- ordinary background noise;
- the intended stop phrase.

This is a functional check, not a WER evaluation. One correct transcript does
not establish model quality.

### 7. Run the application

```bash
python run.py
```

Open `http://127.0.0.1:8000`. The server binds to loopback by default. The
model loads and warms up in the background at start-up (`WILLSPEAK_ASR_EAGER=0`
makes it load on the first utterance instead).

Useful runtime settings:

```bash
WILLSPEAK_ASR_THREADS=2 python run.py
python run.py --model-dir models/parakeet-ft/onnx-int8
```

Benchmark thread counts `1`, `2`, and `4` on the target CPU. More threads can
add contention and are not always faster for short utterances.

### 8. Use a fine-tuned export

Fine-tuned exports come from `scripts/asr/export.py` ([training.md](training.md#5-export-for-deployment));
the recorded ones and their scores are in the [report map](../app/reports/Report%20Context.md).
After training and export:

```bash
python scripts/asr/check.py \
  --model-dir models/parakeet-ft/onnx-int8
```

Then evaluate it:

```bash
python scripts/asr/evaluate.py \
  --model-dir models/parakeet-ft/onnx-int8 \
  --out reports/parakeet-ft \
  --label parakeet-ft
```

Select it at runtime only after the graph check and evaluation pass:

```bash
python run.py --model-dir models/parakeet-ft/onnx-int8
```

Keep the previous model directory until the new deployment has passed the
target-device checks. Switch back by restarting with the previous `--model-dir`;
no environment rebuild or replacement of model files is needed. On the Pi the
boot service takes the folder from `setup.sh --service --model-dir <folder>`,
which writes it into the unit as `WILLSPEAK_PARAKEET_DIR`.

### 9. Verify the adapter with automated tests

```bash
python -m pytest tests/unit/test_parakeet.py
```

These tests verify:

- int8 files are preferred over fp32 files;
- incomplete model directories are rejected;
- a fake sherpa-onnx recognizer exercises the real adapter contract with a
  temporary 16 kHz WAV.

They do not load the 0.6B graph. `scripts/asr/check.py` is the real-graph test.

### 10. Troubleshooting

#### `sherpa-onnx is not installed`

Print the interpreter path, activate the intended environment, and reinstall
`requirements.txt` with that interpreter.

#### `model files are missing`

Run the downloader without `--check`, or set `WILLSPEAK_PARAKEET_DIR` to the
complete directory. Check spelling and nesting; the model files must be
directly inside the selected directory.

#### ONNX load failure

Likely causes are an interrupted archive, mixed checkpoint files, an
incompatible sherpa-onnx version, or an export missing expected metadata.
Redownload the stock graph as one unit. For a personalized graph, rerun export
and inspect the complete file list.

#### Decode succeeds but text is always empty

Confirm the input has audible speech, is not an empty container, and can be
read by SoundFile. Inspect sample rate, channels, duration, and RMS. Then test
the same file with the stock graph to separate the fine-tune's effect from audio
plumbing.

#### Decode is slow

Separate cold model load from steady-state decode. Test thread counts, close
competing CPU workloads, record processor and utterance duration, and compare
real-time factor. Do not infer deployment latency from a training GPU.

### 11. Completion checklist

- Active interpreter and sherpa-onnx version recorded.
- Complete model directory present.
- Readiness check passes.
- Real silence graph decode passes.
- Representative speech graph decode passes.
- Full automated test suite passes.
- Sticky test split is evaluated.
- Critical and difficult-condition errors are reviewed.
- Cold/warm latency and memory are measured on the target machine (the Pi 5, `scripts/asr/bench.py`).
- Previous model remains available for rollback.

For the evidence behind the choice and the migration gate, see
[Parakeet model decision and technical review](research/parakeet.md).
