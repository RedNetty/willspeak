# Raspberry Pi 5

Paths and commands below are relative to `app/` unless they refer to documentation.

- [Running WillSpeak on the Raspberry Pi 5, and measuring it](#running-willspeak-on-the-raspberry-pi-5-and-measuring-it)
- [The deploy folder](#the-deploy-folder)


---

## Running WillSpeak on the Raspberry Pi 5, and measuring it


This section explains each piece of the system, what each command does, and
how to read every number that comes out, including the ones that mean "this
box is not good enough."

---

### 1. What I am actually testing

WillSpeak turns speech that is hard for strangers to understand into clear speech
out of a speaker. The chain, for one utterance, is:

```
microphone  ->  browser records until you stop speaking
            ->  the recording is uploaded to a local server on the same box
            ->  the server resamples it to 16 kHz mono and runs quality checks
            ->  Parakeet (the recogniser) turns audio into text
            ->  the browser speaks that text out loud
```

Nothing starts until you stop speaking, so the whole middle of that chain is
silence in a conversation while the other person waits. On a small computer,
the recogniser is the biggest piece of that silence.

**Parakeet** is NVIDIA's Parakeet-TDT-0.6B: a neural network with about 600
million numbers in it that maps audio to text. It is not run through PyTorch
here. It has been *exported* to ONNX (a portable format for neural networks)
and *quantised to int8* (its numbers squeezed from 32-bit floats down to 8-bit
integers, which makes it about four times smaller and much faster, at a small
cost in accuracy). It is executed by **sherpa-onnx**, a small C++ runtime that
does nothing but run models like this one on a CPU. That is why a Pi can run it
at all: there is no GPU, CUDA or Python maths in the hot loop.

The benchmark answers one question:

> **How long does the Pi 5 take to turn one spoken sentence into text?**

The rest of this section makes that number trustworthy, or explains it when it
is bad.

#### What this does *not* measure

Someone will ask about these, so state them before presenting anything:

- **Accuracy.** Nothing here checks whether the text is *right*. That is word
  error rate, it needs real recordings, and it comes from
  `scripts/asr/evaluate.py`. A box can be fast and useless.
- **Microphone and speaker latency.** Real milliseconds, measured separately
  (measurement P-B).
- **The matcher and the text-to-speech**, both of which add time in the live
  loop.
- **A long thermal soak.** The repeats hint at throttling; they are not twenty
  minutes of sustained load.

So the number here is the floor of the total delay.

---

### 2. What is in this zip

Build the portable application archive from the repository root:

```bash
python app/package.py --out willspeak.zip --with-models
```

The ZIP contains `app/`: the application, active prompts, browser assets,
requirements, tests, model tools and `deploy/pi5/`. Personal notes and research
stay outside the ZIP. Without `--with-models`, setup downloads the model on the
Pi. `scripts/fetch_models.py` also unpacks the model release's test recordings
into `models/<model>/test_wavs/` for the benchmark.

**About the tests.** `tests/unit/test_repository_layout.py` checks that every
link and script path in the documentation exists; a failure there is a
documentation problem, not a recogniser problem. `setup.sh` says so and keeps
going; pass `--strict-tests` if you would rather it stop.

---

### 3. Bringing the Pi up

#### 3.1 The operating system

Flash **Raspberry Pi OS 64-bit *with desktop***. The desktop matters: the Speak
page runs in Chromium, and Chromium needs a graphical session. In the Raspberry
Pi Imager's settings, set the username and password, enable SSH, and leave
desktop auto-login on.

Two OS versions both work. Trixie ships Python 3.13, Bookworm ships 3.11.
WillSpeak needs 3.11 or newer, so either is fine.

64-bit is not optional. The Python wheels for sherpa-onnx are built for
`aarch64`; a 32-bit image cannot install them.

#### 3.2 Power supply: read this one

The Pi 5 expects a **5 V / 5 A USB-C supply**. On a weaker one (a laptop
charger, a phone brick, a 3 A Pi 4 supply) it boots and works fine, and then
quietly caps its own clock speed under load. In a benchmark that looks exactly
like a slow CPU, and it is the most common way to get a bad number on a Pi and
blame the wrong thing.

`bench.py` reads the firmware's throttle flags before and after every run and
prints them, so throttling shows up by name.

#### 3.3 Storage

Models plus the virtual environment are about 4 GB. A 32 GB card is enough. A
USB SSD is noticeably faster for everything, including the roughly 630 MB the
recogniser reads at every start. The benchmark reports which one you booted
from and how fast it is.

#### 3.4 Audio

The Pi 5 has **no analogue audio jack**. Output is HDMI, a USB speaker or DAC,
or a HAT. Use USB: a caregiver can replace it by unplugging it.

Plug in the USB microphone (the Movo USB lavalier, clipped at the collar -
see `docs/design.md` for why the distance matters) and the
speaker before the first boot.

#### 3.5 Install

Copy `willspeak.zip` to `~/willspeak/` on the Pi, unpack it there, then:

```bash
cd ~/willspeak/app
bash deploy/pi5/setup.sh
```

That does, in order:

1. **OS packages.** `libsndfile1` (audio file reading), `alsa-utils`
   (`arecord`/`aplay` for the audio check), `ffmpeg`, `chromium`, and
   `speech-dispatcher` + `espeak-ng`. Those last two are what Chromium's
   speech synthesis talks to on Linux, without them the caregiver panel's
   **Voice** list says *no voices* and the device is mute.
2. **A virtual environment** in `.venv`. This is a private folder of
   Python packages, so the Pi's system Python is left alone.
3. **WillSpeak and its dependencies**, from `pyproject.toml`. This includes
   sherpa-onnx, which publishes prebuilt `aarch64` wheels, no compiler needed.
4. **The Parakeet model**, about 480 MB compressed and 630 MB unpacked, from
   the sherpa-onnx release page.
5. **The test suite**, then **one real decode** to prove the whole chain works.

Expect 15–30 minutes, mostly downloading.

PyTorch is *not* installed by default. It is only needed by the exemplar
matcher (the part that recognises the recorded phrases, including the stop
phrase), it is about 200 MB, and the sentence recogniser does not touch it.
Add it when you want the matcher:

```bash
bash deploy/pi5/setup.sh --with-matcher --skip-apt
```

Without it the Speak page still runs; the matcher just reports itself
unavailable in the caregiver panel.

#### 3.6 Check the audio before anything else

```bash
bash deploy/pi5/check-audio.sh
```

It lists the capture and playback devices, shows the PipeWire defaults (which
are what Chromium uses, set them from the volume icon on the task bar, or with
`wpctl set-default <id>`), records five seconds, and prints:

- **peak**, the loudest sample, in dBFS. 0 dBFS is the maximum a digital
  recording can hold. If it reaches 0, the recording is *clipped*: the tops of
  the waveform are cut flat, and no recogniser can recover them. Turn the gain
  down in `alsamixer`.
- **rms**, the average loudness. Around −20 dBFS is comfortable.
- **noise**, the quietest tenth of the recording, standing in for the room.
- **snr**, the difference between the two: how far the speech stands above the
  room. Under about 15 dB and the recogniser is working against the room as
  much as the speaker.

Then it plays the recording back, which is how you find out that the speaker is
plugged into the wrong socket.

Fix any problem here, before demo day.

#### 3.7 Run it by hand once

```bash
.venv/bin/python run.py
```

Open `http://127.0.0.1:8000/live` in Chromium, allow the microphone, say
something. This is the Speak page, the device. If it transcribes and speaks,
the system works on this box.

---

### 4. The measurement

```bash
bash deploy/pi5/bench.sh
```

About five minutes. `bench.sh` is a thin wrapper: it uses the checkout's
virtual environment, warns you if Chromium is running (a benchmark competing
with a browser is measuring the browser), warns you if the CPU governor is not
`performance`, and then runs `scripts/asr/bench.py`.

Useful variations:

```bash
bash deploy/pi5/bench.sh --quick                    # ~2 minutes, first look
bash deploy/pi5/bench.sh --threads 1 2 4 --durations 1 3 10
bash deploy/pi5/bench.sh --with-app                 # again, with the webapp running
bash deploy/pi5/bench.sh --with-app --app-hold-seconds 60
bash deploy/pi5/bench.sh --label pi5-performance-governor
bash deploy/pi5/bench.sh --skip-asr                 # hardware baseline only
```

Every run writes a readable `.txt` and a complete `.json` into
`reports/bench-pi5/`. The `.json` has every individual timing, not just the
medians, so a run can be re-analysed later without re-running it.

#### How it is built, and why

Each timed configuration runs in **its own subprocess**. Loading the model is
expensive and leaves roughly a gigabyte of arrays behind; measuring a second
configuration in the same process would report a load time that never happened
and a memory figure that belongs to both. One process per configuration means
each load time and each peak-memory figure is real.

The audio is **real speech**, taken from the recordings shipped with the model
release. This matters because Parakeet is a *transducer*, which
emits text token by token, so its cost tracks how many words come out. Silence
or noise decodes almost instantly and would flatter the box by a factor of
several. For lengths longer than the reference recording, the recording is
repeated with a short gap, a 30-second clip is several utterances, not one
impossibly long sentence.

The benchmark prefers to load the recogniser through
`src/willspeak/asr/parakeet.py`, the app's own code, so the number describes
the code path the Speak page really runs. If that import fails it falls back to
driving sherpa-onnx directly with identical settings, and the report says which
path it used. That way a broken checkout cannot be mistaken for slow hardware.

---

### 5. Reading the report, line by line

#### HARDWARE

Every field here is a plausible cause of a slow decode, or changes what to do
about one.

- **board / cpu**, the Pi's own name for itself, and the core. A Pi 5 should
  report `Cortex-A76`, four cores, 2400 MHz.
- **simd**, the instruction-set extensions this CPU has. This is the field
  that explained Sky-2. Modern CPUs have special instructions that do arithmetic
  on many numbers at once; the int8 kernels inside the ONNX runtime are written
  against them. On x86 the relevant one is **AVX2**, and Sky-2's Celeron does
  not have it, which is most of why it takes 1.535 s and why no setting fixes
  that. On ARM the equivalents are **asimd** (NEON, always present),
  **asimddp** (dot product) and **i8mm** (int8 matrix multiply). A Cortex-A76
  has asimd and dotprod but *not* i8mm, that arrived in later cores. The
  report lists what is present and what it looked for and did not find.
- **governor**, how Linux chooses clock speed. `powersave` leaves a free 20–30%
  on the table:
  ```bash
  sudo apt install linux-cpupower
  sudo cpupower frequency-set, g performance
  ```
  Run the benchmark before and after with different `--label` values to get a
  measured result.
- **memory**, total and what was free when the run started. Parakeet int8
  peaks around 1.0–1.35 GB. On 8 GB there is room for the model, the server,
  and Chromium.
- **storage**, whether you booted from an SD card or something faster, and its
  write and read speed. This shows up as model load time and nothing else. The
  read figure is partly served from cache; treat it as an upper bound.
- **compute**, two crude scores that bracket the workload. `matmul` is the
  vectorised, multi-threaded, cache-friendly best case, roughly what the
  encoder does. `scalar` is single-core interpreted work, roughly what
  everything around it does. They are for comparing boxes, not for absolute
  claims.
- **edge tpu**, is there a Coral accelerator on this machine. Sky-2 was bought
  believing it had one and 0.7.0 retired that path. The check costs nothing and
  settles the question.
- **temp before / after**, every thermal zone plus the Pi firmware's own
  reading. A Pi 5 idles around 45–50 °C and starts reducing its clock in the
  low 80s.
- **THROTTLE before / after**, the firmware's flags, in plain words. If you see
  `under-voltage` here, stop and change the power supply; every timing in the
  report is describing your charger, not the Pi.

#### MODEL

Which model directory, which code path, which audio file, and the transcript it
produced. **Read the transcript.** It is a cheap sanity check: if it is empty or
gibberish, the timings are timing a failure.

#### THREAD SWEEP

The same 3 seconds of speech decoded at 1, 2 and 4 threads, with the model load
time and the peak memory for each.

More threads is *not* reliably faster on a small box. Past the point where the
memory system can feed them, threads fight over cache and bandwidth instead of
sharing work. On Sky-2: 1 thread 2.838 s, 2 threads 1.535 s, **4 threads
2.329 s**, the "obvious" setting of one thread per core was 1.5× slower than
the right one. WillSpeak uses a single value for `WILLSPEAK_ASR_THREADS`, and this
sweep picks it.

The report names the winner. Put it in `deploy/pi5/willspeak.service`:

```
Environment=WILLSPEAK_ASR_THREADS=<winner>
```

The Pi has four full cores with no hyperthreading, so 4 has a better chance of
winning here than it did on Sky-2, but measure it.

#### LENGTH SWEEP

The same decode at 1, 3, 5, 10 and 30 seconds of audio, at the winning thread
count, reported as **real-time factor**: decode time divided by audio duration.

- **Under 1.0** means the box transcribes faster than a person speaks. That is
  the condition for running live at all.
- It should **improve** with length, because the fixed per-utterance cost -
  building the feature buffer, setting up the decode, gets spread over more
  audio. Sky-2 was flat at 0.48–0.51 from 1 s to 30 s.
- If it gets *worse* on long clips, the box is running out of memory or heat
  rather than compute. The report says so when it sees that.

**drift** is the last repeat measured against the first. Negative is normal -
caches warming up. A large positive number is the box slowing down under
sustained work, which on a small board means heat or power. A device that is
fast for the first minute of a conversation and half-speed by the fifth is
worse than one that is steadily slower throughout.

#### IDLE vs LOADED

`--with-app` measures what the idle run cannot. The server runs whenever the
device is in use, and on a four-core board it takes a real share of the
machine.

The benchmark starts `run.py` from this checkout, waits for it to answer, opens
`/` and `/live` so it loads whatever it loads lazily, then posts one utterance
to `/api/live/transcribe` so the server loads its own copy of the model;
without that step the run would measure an idle web process. It points the server at a throwaway data directory, so a
benchmark run never writes clips into your corpus.

Then it reports three things:

- **loaded**, the same bare decode as before, with the server sitting next to
  it, and the **penalty** as a percentage. This is contention for CPU and
  memory bandwidth.
- **server memory**, the server's resident memory. Both processes hold their
  own copy of the model, so this is a worst case for memory while the CPU
  contention is realistic.
- **end to end**, the same audio sent through the server's real HTTP path:
  upload, ingest, resample, quality checks, silence trim, decode, database
  write. **This is the number the device delivers.** The bare decode is its
  floor, and the difference is everything the application does around the
  model.

`--app-hold-seconds 60` pauses after the server is up so you can open the Speak
page in Chromium during the measurement. Do this at least once: the browser is
still not measured directly, and on hardware this size it is often the bigger
cost.

#### VERDICT

- **headline**, the median decode of one 3-second sentence, from the loaded
  run if there was one, otherwise idle.
- **band**, under 1 s GOOD, 1–2 s USABLE, 2–4 s SLUGGISH, over 4 s TOO SLOW.
  The thresholds come from conversation: a person waiting
  for a reply starts filling the silence after about a second.
- **demo-box rule**, the benchmark acceptance rule: **idle real-time
  factor under 0.7 and the Pi is the demo box.** Otherwise Sky-2 is, and the
  Pi becomes the "next box" slide.
- **note** lines appear only when the run found something worth saying:
  throttling, a real-time factor that got worse with length, or a thread count
  that beat the core count.

---

### 6. Comparing boxes

```bash
bash deploy/pi5/bench.sh --compare \
  reports/bench-pi5/<your-pi-run>.json \
  deploy/pi5/reference/sky2-2026-09-01.json
```

The Sky-2 file holds **my** 1 September measurement, Celeron J6412, no AVX2,
governor `powersave`, 2 threads, 1.535 s for a 3-second sentence, verdict
USABLE, transcribed into this tool's report shape so the comparison works. It
was not produced by `bench.py`, so only the fields the table reads are filled
in, and the file says so at the top.

Use the same command to compare my own runs: before and after the governor
change, SD card against USB SSD, idle against `--with-app`.

---

### 7. If the number comes back bad

In order, cheapest first:

1. **Check the throttle flags in the report.** Under-voltage means the power
   supply, not the Pi. A 5 V / 5 A supply is the fix, and the cheapest one.
2. **Set the governor to `performance`** (`sudo cpupower frequency-set, g
   performance`). Typically 20–30%.
3. **Use the thread count the sweep picked**, which may not be 4.
4. **Close Chromium before measuring**, and check the idle temperature and
   drift. If it is throttling, fix airflow, an active cooler on a Pi 5 is
   about ten dollars.
5. **Boot from a USB SSD instead of an SD card** if the storage numbers are
   poor. This mostly buys back model load time.

Only after all of those:

6. **A smaller ASR model.** That trade costs accuracy, and accuracy is the
   point of the project.
7. **Different hardware**, which, given the current prices, means the Sky-2
   I already own.

---

### 8. The rest of the bring-up

Once the numbers look right:

```bash
bash deploy/pi5/setup.sh --service
sudo reboot
```

That installs a systemd service (the server starts at boot, before anyone logs
in) and an autostart entry that opens the Speak page full-screen in Chromium
with the microphone pre-granted, once the server answers. For the expo, add
`--kiosk-page /compare` so it boots into the comparison page instead; the
choice is the argument on the `Exec=` line of
`~/.config/autostart/willspeak-kiosk.desktop`, so it can also be edited there.

The acceptance test is the roadmap's:

> **power-cycle → the Pi comes up speaking. No keyboard, no login.**

Useful afterwards:

```bash
systemctl status willspeak          # is the server up
journalctl -u willspeak -f          # its log
sudo systemctl restart willspeak    # after a git pull
```

`Alt+F4` gets you out of the kiosk with a keyboard. Without one, SSH in.

#### Using the box from a laptop

VNC works but lags and looks poor. The service listens on the network instead
(`WILLSPEAK_HOST=0.0.0.0` in `deploy/pi5/willspeak.service`), so a laptop on the
same network (the phone hotspot; campus Wi-Fi may block device-to-device
traffic) opens the pages directly:

```
http://raspberrypi.local:8000/live       Speak
http://raspberrypi.local:8000/compare    Compare
```

If the name does not resolve, use the Pi's address from the hotspot's device
list. Click **Click to start** once. The laptop drives the page and speaks out
of its own speakers; the **box's** lavalier does the listening and the box's
model does the recognising. A browser on another computer has no microphone
over plain http, so `static/pimic.js` takes the box's microphone from
`/api/pimic/stream` instead (raw 16-bit PCM at 48 kHz from `arecord`, through
the desktop's PipeWire so the kiosk can share it). The badge in the bottom
left corner names the microphone or says what is wrong.
`/api/pimic/status` says the same.

- `?mic=pi` forces the box's microphone, for example through an SSH tunnel to
  `127.0.0.1`. `?mic=local` uses the browser's own microphone.
- `WILLSPEAK_PI_MIC=<alsa device>` picks the microphone.
  `WILLSPEAK_PI_MIC_CMD="<command>"` replaces the capture with any command that
  writes raw s16le mono PCM at 48 kHz. For a rehearsal without a microphone:
  `ffmpeg -re -stream_loop -1 -i clip.wav -f s16le -ar 48000 -ac 1 -`.
- Anyone on the same network can open these pages and hear the room through
  them. Use the hotspot, not an open network.
- Keep the kiosk on `/compare`. A kiosk on `/live` at the same time would
  transcribe the same speech twice.

#### Comparison demo

The expo page (`/compare`, tab 05 in the kiosk's navigation) needs two things
that are not in the zip: the speaker models and the fixed clip list. Both are
made on the lab computer, in `app/`:

```bash
python lab.py --set trim_models=off run 28 --rerun     # models/lab-28-export-{M01,M02,M04}-matmul
python scripts/corpus/demo_clips.py --dry-run          # check the draw, then without --dry-run
```

`demo_clips.py` writes `data/demo/clips.json` and copies each chosen
recording; it refuses to replace an existing list, so the list stays the one
fixed before the expo. Copy both to the Pi over the local network (never cloud
sync: `data/` stays off it):

```bash
rsync -a models/lab-28-export-M01-matmul models/lab-28-export-M02-matmul models/lab-28-export-M04-matmul \
    jdev@<pi-address>:Desktop/willspeak/app/models/
rsync -a data/demo jdev@<pi-address>:Desktop/willspeak/app/data/
```

On the Pi, check each folder before the expo
(`.venv/bin/python scripts/asr/check.py --model-dir models/lab-28-export-M01-matmul`),
then restart the service. A speaker whose model is missing shows "no model" and
cannot be chosen; a folder elsewhere is set with
`WILLSPEAK_DEMO_MODEL_M01=<folder>` in the service's environment. The page's
How this works window shows the load time and memory for each speaker swap:
those are the numbers the expo plan still needs from the Pi.

The **Model** list beside the speaker buttons tests any other complete model
folder in `models/` (up to three levels down, so `lab-28-export-M02-default/model`
counts) on the chosen speaker's recordings, in place of that speaker's own
model. "Speaker's own" is the expo setting. Results from another model are
labelled with its folder name and are not added to the tally.

Before visitors arrive, check the two features that use the room: press **R**
(Try your voice) and say a sentence to confirm Chromium can use the lavalier
from the Compare page, and press **T** (Tour) to see one speaker cycle with
sound. Clear the table's tally with Reset in How this works so the day starts
at zero.

---

### 9. The other measurements

This benchmark is P-A, and it needs no microphone, room or person.
The other measurements (P-B end-to-end latency, P-D cold start, P-E stop
phrase, P-F battery, P-G other voices in the room) are mine and need the box
on the table. Their commands and tables are in [measurements.md](measurements.md);
the list is [TODO.md](TODO.md). Each is a `reports/<name>/notes.md` and counts
only if it comes from that box.


### 10. Language-model rescoring on the Pi

Only after [lab test 36](../app/reports/lab-36-lm-rescoring/notes.md) shows a gain worth its cost. The question
here is narrower: does the language model fit next to Parakeet, and how long does it add to one utterance?
Nothing is decoded on the Pi for this; it scores the candidate answers lab test 36 already made.

1. Copy two things to the Pi: the pools (`reports/lab-36-lm-rescoring/pools/stock-severe.json` is enough) and the
   one or two model folders worth testing (`models/lm/<tag>/`; `python scripts/asr/lm_score.py fetch --models <tag>`
   brings back a file the test deleted), to the same paths under `app/`.
2. On the Pi, in the app's environment:

   ```bash
   .venv/bin/python, m pip install "llama-cpp-python>=0.3.32" \
     --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
   ```

   If no ready-made build is offered for the Pi, pip compiles it (about ten minutes; it needs `cmake` and a
   compiler: `sudo apt install cmake build-essential`).
3. With the WillSpeak service running (Parakeet loaded) and Chromium open, as in use:

   ```bash
   .venv/bin/python scripts/asr/lm_score.py score --model qwen25-05b-q8 \
     --pool reports/lab-36-lm-rescoring/pools/stock-severe.json \
     --threads 4 --out reports/p-lm-rescoring/qwen25-05b-q8-pi.json
   free, m
   ```

   The script prints tokens a second and its own peak memory; `free -m` while it runs shows what is left with
   everything loaded. Seconds per clip is the sum over every answer in one clip's pool, so it is
   the time rescoring adds to one utterance before any sharing of work between answers.

What it cannot tell you: the cost of the extra decodes. Each view is another Parakeet decode (0.4 to 0.8 s on
the Pi), beam search is not available through sherpa-onnx, and the audio score needs the model run through
onnxruntime. Those are the build, not this measurement.


---

## The deploy folder

Everything needed to put WillSpeak on a Raspberry Pi 5 and measure how it copes.
The full walkthrough, with every number explained, is
[Running WillSpeak on the Raspberry Pi 5, and measuring it](#running-willspeak-on-the-raspberry-pi-5-and-measuring-it).

| file | what it is |
|---|---|
| `setup.sh` | one-shot installer: OS packages, venv, dependencies, the Parakeet model, the test suite, one real decode. `--service` also installs the boot service and the kiosk autostart. |
| `bench.sh` | runs `scripts/asr/bench.py` with this checkout's venv, after warning about a running browser or a `powersave` governor |
| `check-audio.sh` | lists devices, records, reports level and signal-to-noise, plays it back |
| `willspeak.service` | systemd unit template (`@ROOT@`, `@USER@` filled in by `setup.sh`) |
| `kiosk.sh` | waits for the server, opens `/live` (or the page given as its argument, such as `/compare`) in Chromium kiosk mode with the microphone pre-granted |
| `willspeak-kiosk.desktop` | XDG autostart entry that runs `kiosk.sh` when the desktop session starts |
| `reference/sky2-2026-09-01.json` | the Sky-2 result from 1 September 2026, in this tool's report shape, so `bench.sh --compare` can put the two boxes side by side |
