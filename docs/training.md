# TORGO, training, evaluation, transport

Paths and commands below are relative to `app/` unless they refer to documentation.

- [Public corpora: TORGO](#public-corpora-torgo)
- [Parakeet training and evaluation](#parakeet-training-and-evaluation)
- [Tutorial: evaluate Parakeet reproducibly](#tutorial-evaluate-parakeet-reproducibly)
- [Laptop collection → desktop training → laptop testing](#laptop-collection--desktop-training--laptop-testing)


---

## Public corpora: TORGO

How to get the public dysarthric speakers onto a machine, score the two
recognisers on them, and build the fine-tune exports. This is the data path
for the Fall 2026 semester ([plan.md](plan.md#fall-2026-plan-the-public-corpus-semester));
nothing here touches the private corpus in `data/`, and nothing here needs
consent, see [plan.md](plan.md#ethics-review-and-the-data-this-semester-uses) for why.

### What TORGO is

Eight speakers with dysarthria (cerebral palsy or ALS: F01, F03, F04, M01–M05)
and seven controls, recorded at the University of Toronto reading single words
and sentences, one to three sessions each (sessions are separate visits), on
a head-mounted and an array microphone. Free for academic use; cite Rudzicz,
Namasivayam & Wolff (2012). The paper's intelligibility groups are in
`willspeak.torgo.SEVERITY`: severe = F01, M01, M02, M04; moderate–severe = M05;
moderate = F03; mild = F04, M03; controls = FC01–03, MC01–04.

Prompts are not all speech: some are bracketed instructions ("[say Ah-P-Eee
repeatedly]"), some are picture descriptions given as image paths, a few are
lost ("xxx"), and some words carry a reader's hint in brackets ("tear [as in
tear up that paper]") that is stripped. Only words and sentences carry a
transcript and are scored. Audio is 16 kHz mono already.

### Fetch

```bash
python scripts/corpus/fetch_torgo.py             # F + M archives, 3.6 GB down
python scripts/corpus/fetch_torgo.py --check     # which speakers are present
python scripts/corpus/fetch_torgo.py --controls  # add FC + MC (typical speech) for the forgetting check
```

Lands in `data/public/torgo/`: `archives/` (the downloads; resume if
interrupted), `TORGO/<speaker>/Session*/{wav_headMic,wav_arrayMic,prompts}/`
and `SOURCE.md` (terms and citation). `data/` is git-ignored, so the corpus
never enters the repository.

### Manifests and exports

```bash
python scripts/corpus/torgo_manifest.py --export
python scripts/corpus/torgo_manifest.py --export --curve M01    # + 1 / 5 / 15-minute training subsets (F-3)
```

Writes `data/public/torgo/manifests/`:

| file | contents |
|---|---|
| `all.jsonl` | every usable utterance, one file each (head mic preferred), with its split |
| `severe-sentences.jsonl` | F01, M01, M02, M04 sentences (T-1) |
| `all-sentences.jsonl` | every dysarthric speaker's sentences |
| `words.csv` | single words for `scripts/matcher/embed_manifest.py --manifest` |
| `summary.md` | counts per speaker / kind / split, read this first |

and `data/public/torgo/export/`:

| folder | contents |
|---|---|
| `loso-<SPEAKER>/` | train/dev = the other dysarthric speakers, test = this speaker's test prompts (F-1, H2) |
| `<SPEAKER>/` | one speaker: train/dev from the earlier sessions, test from the last session, prompt-disjoint as well (F-2, H3) |
| `<SPEAKER>-min1\|5\|15/` | the same with the training set cut to the first N minutes of audio (F-3) |
| `controls/` | the control speakers (typical-speech check) |

Each row has `id`, `audio`, `text`, `duration_s`, `speaker`, `session`,
`mic`, `severity`, `kind`, `prompt_set`, the same shape as the recorder's
own export, so `train.py` and `evaluate.py` read it unchanged, and
`train.py --check` validates it the same way (split isolation, duplicate
audio, 16 kHz mono).

**One microphone per utterance.** The head microphone when it exists, the
array otherwise: they capture the same take, and scoring both would double
count, or, worse, put the same speech on both sides of a split.

**Splits.** Every prompt is train, dev or test by a stable hash of its
normalised text, and the assignment is the same for every speaker
(`willspeak.torgo.assign_split`), so no test prompt is in anyone's training
data. Standard TORGO splits share almost all prompts between train and test,
which inflates fine-tuned WER (Hui, Zhang & Mohan 2024), the reason this
project reports against prompt-disjoint splits. The fine-tune design is in
[research/finetune-torgo.md](research/finetune-torgo.md).

### Score the sentence recogniser

```bash
python scripts/asr/evaluate.py \
  --manifest data/public/torgo/manifests/severe-sentences.jsonl \
  --audio-root data/public/torgo/TORGO --group-by speaker \
  --out reports/torgo-severe-zero-shot
```

`--group-by` takes any manifest field (`speaker`, `severity`, `session`,
`kind`, `mic`). `--limit 20` is for checking the path on a slow box; the
report is stamped as a plumbing check. Every report records the machine it
ran on (`provenance.json` → `machine`), because int8 output differs slightly
between CPU families.

### Score the matcher

```bash
python scripts/matcher/embed_manifest.py --manifest data/public/torgo/manifests/words.csv --out data/emb/torgo-words.npz
python scripts/matcher/train_projection.py --emb data/emb/torgo-words.npz --eval      # T-2: the projection's gain
python scripts/matcher/evaluate.py --emb data/emb/torgo-words.npz --k 1               # T-3: per-speaker protocols
```

Embedding needs PyTorch and WavLM (`scripts/fetch_models.py`); ~4,200 clips
at a few per second on a laptop CPU. Both evaluations hold one speaker out
at a time and fit the projection on the rest, so no speaker is ever scored
by a projection that saw them. The vocabulary is shared across TORGO, so
these numbers measure transfer across *speakers*, not across words, say so
wherever they are quoted. `embed_manifest.py --torgo <root>` is the older
route that keeps both microphones as separate sessions.

To refit the shipped projection (only after comparing candidates, and with
the numbers recorded):

```bash
python scripts/matcher/train_projection.py --emb data/emb/torgo-words.npz --out src/willspeak/models/projection.npz
```

### Provenance and citation

`data/public/torgo/SOURCE.md` carries the terms and the citation; the poster
and any write-up cite: Rudzicz, F., Namasivayam, A.K., Wolff, T. (2012). The
TORGO database of acoustic and articulatory speech from speakers with
dysarthria. *Language Resources and Evaluation*, 46(4), 523–541.


---

## Parakeet training and evaluation

**Status (1 Oct 2026).** TORGO fine-tuning, export comparisons and adapter
experiments have run on the 4060 under WSL2. The prototype model is still
undecided; [Report Context](../app/reports/Report%20Context.md) maps the recorded
results. Whole-block runs currently train blocks 0–7. Adapter runs train only
the added layer and currently have no Pi export. Recording and training on
the intended user's speech remains a later, IRB-gated phase.

Training runs in a Linux CUDA environment with NeMo (the 4060
desktop under WSL2). Deployment uses a sherpa-onnx int8 export on CPU (the
Raspberry Pi 5).

### 1. Prepare data

Review accepted clips, then export from the Coverage page. Confirm these files
exist and are non-empty:

```text
data/export/train.jsonl
data/export/dev.jsonl
data/export/test.jsonl
```

The training script reads train and development only. The evaluator reads the
selected manifest, which defaults to test.

For laptop/desktop work, use the [snapshot handoff](#laptop-collection--desktop-training--laptop-testing), then:

```bash
python scripts/asr/train.py --check --export datasets/session-01/export
```

This validates train/dev paths, duration, split isolation, and duplicate audio
without loading NeMo or reading test. Substitute that snapshot path for
`data/export` in the training/evaluation commands below.

### 2. Install training dependencies

NeMo does not run on native Windows. On the 4060 desktop the training
environment is Ubuntu under WSL2. The Windows NVIDIA driver provides the GPU;
the CUDA pieces NeMo compiles against come from pip packages inside the training
environment, so no system CUDA toolkit is installed in Ubuntu.

Training uses its own environment, `~/venvs/willspeak-train`, on Python 3.12,
separate from `app/.venv`. Current Ubuntu ships Python 3.14, which is too new for
the training stack (`onnx` has no 3.14 package and fails to build; NeMo is tested
on older Python). The app, `ci.py` and `evaluate.py` stay on `app/.venv`.

1. In an administrator PowerShell: `wsl --install -d Ubuntu-24.04`, reboot,
   create the Linux user. Update the Windows NVIDIA driver from nvidia.com if
   it is older than 2024.
2. In Ubuntu: `nvidia-smi` must list the 4060. If it does not, the Windows
   driver is the problem, not Ubuntu.
3. Clone the repository **inside the Linux filesystem** (`~/willspeak`, not
   `/mnt/c/...`, training reads thousands of small files and the Windows
   mount is ten times slower), then:

One command does all of it, from `app/` inside WSL. It is safe to re-run to
repair a broken environment:

```bash
./willspeak.sh train-setup
```

For a separate Linux lab computer running `lab.py`, use
`./willspeak.sh lab-setup` instead. It also prepares public TORGO files/exports,
WavLM, the stock NeMo checkpoint, CPU matching dependencies and the separate
candidate-model environments. See [the lab setup instructions](../app/README.md)
for the earlier fine-tunes and decode reports that must be copied from Yack.
Neither setup command trains those earlier models or reproduces their results.
Alternatively, `python lab.py run 1`
rebuilds the missing baselines for stage 1 in separate folders before the
stage starts; the app README describes reuse, logs and remaining stage dependencies.

It ends by compiling one tiny GPU kernel through numba (the same path the
training loss takes) and prints `training environment ready` or the exact
problem and fix. `train.py` runs the same check before loading the model, so a
broken environment fails in seconds, not after the 2.4 GB download.

What it runs, for doing it by hand (**one line at a time**: pasting the
`curl ... | sh` line together with the next lets the installer swallow it):

```bash
sudo apt install -y libsndfile1 ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
uv venv --python 3.12 --seed ~/venvs/willspeak-train
echo 'export CUDA_HOME=$HOME/venvs/willspeak-train/lib/python3.12/site-packages/nvidia/cuda_nvcc' >> ~/venvs/willspeak-train/bin/activate
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/nvvm/lib64:$LD_LIBRARY_PATH' >> ~/venvs/willspeak-train/bin/activate
mkdir -p ~/.cache/willspeak-tmp
echo 'export TMPDIR=$HOME/.cache/willspeak-tmp' >> ~/venvs/willspeak-train/bin/activate
source ~/venvs/willspeak-train/bin/activate
cd ~/willspeak/app
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e ".[train]"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`uv` downloads its own Python 3.12, so it works whatever version Ubuntu ships.

**Never write training output to `/tmp`.** Current Ubuntu mounts `/tmp` as a
RAM-backed tmpfs capped at half the RAM: every checkpoint written there (about
2.5 GB each) is taken out of memory, and the disk "fills" at about 6 GB
(`No space left on device`, 23 Sep 2026). Outputs go under `models/` (ignored by
git), and the `TMPDIR` line above moves NeMo's own staging folder to disk too.
`train.py` checks free space in both places before loading the model.

**Checkpoint size.** Training checkpoints keep only what training can change
(the trainable layers plus batch-norm statistics); frozen layers are the base
model's and are not saved again. With `--freeze-encoder-layers 16` a checkpoint
is about 0.8 GB instead of 2.5 GB, and with 24 frozen about 40 MB, so a run with
four kept checkpoints needs roughly 6 GB, not 12.5. The final `parakeet-ft.nemo`
is always the complete model (2.5 GB), since export needs all of it.

**Memory under WSL2.** WSL2 gets half of Windows' RAM and only 2 GB of swap by
default. Saving and averaging checkpoints after a validation pass is a short
memory peak; with the defaults Linux ran out and killed the run with a bare
`Killed` (23 Sep 2026). The fix is swap, not more RAM: the peak is brief, so
spilling to disk costs seconds, and Windows keeps the RAM it needs. Create
`C:\Users\<you>\.wslconfig` in Windows. Give WSL about three quarters of the
installed RAM (12GB on a 16 GB PC, 24GB on a 32 GB PC) and leave the rest to Windows:

```ini
[wsl2]
memory=12GB
swap=16GB
```

Then run `wsl --shutdown` in PowerShell and reopen Ubuntu; `free -h` shows the
new `Mem:` and `Swap:` totals. With this the 4060 desktop got past the checkpoint
save that was killed before. Its RAM was upgraded from 16 GB on 24 Sep 2026; WSL
now gets 24GB, and `free -h` shows 23 Gi memory and 18 Gi swap.
`train.py` warns when RAM + swap together are under 14 GB.
The two `echo` lines make every activation point numba at the pip-installed
libNVVM; without them the first training step fails with *libNVVM cannot be
found*. Every new terminal: `source ~/venvs/willspeak-train/bin/activate` before
`train.py` or `export.py`.

The last line must print `True NVIDIA GeForce RTX 4060`. The TORGO data
comes over as a copy of `data/public/torgo/` (or run `fetch_torgo.py` again
here; it is public). The `.nemo` checkpoint downloads on the first `train.py`
run (2.4 GB, cached under `~/.cache`).

Memory: the card has 8 GB. `train.py --freeze-encoder-layers 16` with
`--batch 2 --grad-accum 8` fits; 12 is the next thing to try; 5 and below
need a 16 GB card ([research/finetune-torgo.md](research/finetune-torgo.md)
§5 has the accounting). The export step needs 16 GB of *system* RAM; WSL on the desktop has had that since the 24 Sep upgrade.

Smoke test before any real run, with a throwaway output directory:

```bash
rm -rf models/smoke-test
python scripts/asr/train.py --export data/public/torgo/export/F01 --out models/smoke-test --max-steps 2 --batch 1 --freeze-encoder-layers 24
```

`train.py` refuses to write into an existing output folder, hence the `rm`.
Always pass `--export`: the default is `data/export`, the recorder's own corpus.

That proves the manifests, the model download, the GPU and the loss all
work. It takes a few minutes and writes nothing worth keeping.

Fallback if WSL2 refuses to cooperate for more than an evening: a Kaggle
notebook (30 GPU hours a week, 16 GB T4/P100) running the same commands on
an uploaded copy of `data/public/torgo/export/`. Public data only, never
the client's recordings.

### 3. Establish a stock baseline

Follow [the evaluation baseline procedure](#3-establish-the-stock-baseline),
saving the exact manifest and report before fine-tuning.

### 4. Fine-tune

```bash
python scripts/asr/train.py \
  --export data/export \
  --out models/parakeet-ft
```

Useful controls include `--freeze-encoder-layers` or `--train-parts` (or `--adapter-after-blocks`), `--epochs`, `--batch`,
`--grad-accum`, `--lr`, `--weight-decay`, `--patience`, `--average-top`, `--select`, `--dev-scoring`, `--keep-stock`,
`--speed-perturb` / `--no-speed-perturb`, `--time-stretch`, `--white-noise`, `--extra-aug-prob`,
`--precision`, `--no-freeze-batchnorm`, `--target-form`, and `--curriculum`.
How the released model is chosen: by default (`--select dev`), dev WER, scored the way
`evaluate.py` scores (`--dev-scoring evaluate`), picks and averages the best `--average-top`
checkpoints, and the run stops after `--patience` epochs without improvement. Stock is scored on
dev first and released unchanged if no checkpoint beats it (`--keep-stock`). When a speaker's dev
set is too small to choose on, `--select last --no-keep-stock` trains exactly `--epochs` epochs and
averages the last `--average-top` of them, and dev decides nothing (the F-2 speaker runs).
Batch-norm statistics stay at the base model's values by default. With
`--no-freeze-batchnorm`, batches of two clips rewrite them in every layer, frozen
or not: on F01 that alone cost 19.6 WER points and made the model answer silence
(the smoke test against the batch-norm diagnosis, 24 Sep 2026). Curriculum changes
order; it does not remove hard-day data, and it has no effect on the TORGO exports, whose
every row carries the condition `unknown`. The TORGO experiments use the
exports under `data/public/torgo/export/` in place of `data/export`.

**Target form.** `--target-form` chooses how the transcripts are written for training; dev WER
normalises both sides whatever the form, so the form changes no score.

- `scoring` (the default, and the form of every recorded run): lowercase, no punctuation,
  contractions expanded, exactly as `evaluate.py` compares them. The fine-tuned model learns
  to write that way. The scoring normaliser also expands the apostrophe-less forms
  `ill`, `id`, `wont`, `cant` and `lets` (to `i will`, `i would`, `will not`, `cannot`, `let us`),
  because recognisers drop apostrophes; a prompt that is literally one of those words is
  therefore trained as the expanded phrase.
- `stock`: the model's own output form, a capital first letter and one terminal mark
  (`tomato` becomes `Tomato.`; a sentence keeps its own punctuation). The stock tokenizer
  splits `Tomato.` and `tomato` into different pieces, so this form keeps the targets on the
  token paths the base model already uses and leaves nothing about formatting for a frozen
  decoder and joint to re-learn through an adapter. The two characters the tokenizer has no
  piece for, `;` and `"`, are dropped. Not yet run on any speaker.
- `raw`: the manifest text as written (the two 24 Sep runs before `--normalize-targets` existed).

The run also checks every train and dev target against the tokenizer and warns when a target
contains a character it cannot spell (it would be trained as the unknown token).
`--weight-decay` (default 1e-3, as in every recorded run) applies to every trainable parameter,
including an adapter, which it pulls back toward identity at each step.

Every run changes the training clips in two ways unless told otherwise: speed perturbation (each
clip is resampled by a factor drawn from `--speed-perturb`, default 0.85-1.15 in five steps, which
moves tempo and pitch together) and the base model's own SpecAugment (`--no-spec-augment` turns it
off). Two more waveform changes are off by default and added with flags, so an old command line
trains exactly as it did before. `--time-stretch MIN MAX` changes tempo without moving pitch (1.25
is 25 percent faster, 0.75 is 25 percent slower; five steps across the range), and
`--white-noise MIN_DB MAX_DB` adds white noise at a whole-number level in dB below full scale,
drawn from MIN_DB up to but not including MAX_DB (for example `-70 -40`; MIN_DB must be lower
than MAX_DB, both between -120 and 0). Each runs after speed perturbation on a fraction of clips,
`--extra-aug-prob` (default 0.5; it is an error without one of the two flags). Both are NeMo's own
perturbation classes. Before the model loads the run tries each one on synthetic audio, so a NeMo or
librosa version that cannot run them stops the run at once instead of inside a data-loader worker. The
run header and `run.json` record the chain in use (`augmentation`). The first comparison is
`reports/f2-augmentation-variants/notes.md`.

`--adapter-after-blocks N` trains something different from the options above: one small layer,
`h + W h + b` (1,049,600 numbers, all zero at the start so the model begins exactly as the
base), inserted after encoder block N counted from the bottom, with every other weight frozen,
the decoder and joint included. It needs a much larger `--lr` than whole-block training (about
3e-4; the run warns at 5e-5 or below) and cannot be combined with `--train-parts` or
`--freeze-encoder-layers`. The run saves `adapter.pt` and no `.nemo`;
`nemo_hyp.py --adapter <file>` attaches it to the same base model and refuses a different
one, and `export.py --adapter <file>` (with the same `--stock` or `--nemo` base) exports the
model with the adapter inside its encoder graph. The first test is
`reports/f2-m02-adapter8/notes.md`.

The run writes the released model, `parakeet-ft.nemo`, and a
`run.json` with options, input/script/checkpoint hashes and runtime versions.
Output directories must be new. A tiny corpus validates plumbing, not model quality.

#### Reading the training log

Each line starts with a time and `[train]`. The opening summary reports unique
audio minutes, clip uses per epoch (including repeats), batch size, gradient
accumulation, estimated updates, learning rate, selection rule, seed and output.
The environment summary identifies the GPU, precision and optimizer placement.
The opening update estimate is approximate. After NeMo builds each phase's
loader, `TRAINING PLAN` reports its actual clip uses, batches and updates,
including whether a final partial batch is dropped. A restored model may
exclude very short clips through its dataset settings. The test split is never
opened by training.

Every 30 seconds a progress line shows the batch, optimizer step, current loss,
learning rate, batches per second and remaining training time for that epoch.
The estimate excludes validation and saving. Long validation passes also show
dev-batch progress. Set `--log-every-seconds 10` for
more frequent updates. `--verbose` turns on NeMo setup messages and development
predictions; by default they and the animated progress bars are hidden.
The console also hides expected restore-time configuration notices, repeated
small-grid utilization warnings and startup chatter, though all of it still goes
to the saved log. Other warnings and errors stay visible. `nemo_hyp.py` uses
the same console view and accepts `--verbose` for troubleshooting.

An epoch summary shows dev WER as a percentage, change from the base in
percentage points, selection status, mean batch loss, cumulative optimizer
steps, ending learning rate, separate training and validation times, and peak
GPU reserved memory in decimal GB. Positive
base gain means fewer errors. Loss is restored to its value before Lightning's
gradient-accumulation division; it is not an accuracy percentage.

`curve.csv` and `run.json` keep the same epoch records, including the curriculum
phase. Records include updates within the epoch, cumulative updates, batch
count, learning rate, training/validation/epoch seconds, and GPU allocated and
reserved memory. Console epochs and `display_epoch` start at 1; saved `epoch`
values stay zero-based. CSV fields are blank when a value was not measured.
Failed or interrupted runs keep their error and completed epoch records.
JSON records are replaced atomically, so an interrupted write leaves the
previous complete copy in place.

The final summary separates the best individual checkpoint's dev WER from the
released model's dev WER. A mean of checkpoints can score differently from its
best member, so averaged weights are validated again. With the base guard on,
a dev-selected average that fails to beat the base falls back to the best
individual checkpoint. A fixed schedule still uses dev only for reporting.
The completion summary gives the stop reason, completed epochs and actual
optimizer updates. `--max-steps` plumbing checks validate the final weights
even if the limit falls inside a later epoch; missing validation cannot reuse
an earlier dev score. A fixed epoch schedule rejects `--max-steps`, since that
would shorten the specified schedule. A release score that is unavailable is shown as `not measured`. Test WER comes
from the separate evaluation step below. Existing run records are not rewritten.
Completion elapsed time includes loading the model, baseline scoring, training,
release validation and saving after the run record is initialized. It is not
the sum of training-batch times alone.

`nemo_hyp.py` writes a sibling `<name>.provenance.json` recording the manifest,
audio, checkpoint/adapter hashes and decoding environment. `evaluate.py --hyp`
verifies those records before scoring and includes them in the report. Keep the
sidecar with the hypotheses when moving them between machines. Older hypotheses
without a sidecar still score with an explicit warning; their original decoder
inputs cannot be verified retrospectively. `--stock` uses the standard base;
`--pretrained NAME` supports another model name supplied to `train.py --base`.

### 5. Export for deployment

```bash
python scripts/asr/export.py \
  --nemo models/parakeet-ft/parakeet-ft.nemo \
  --out models/parakeet-ft/onnx-int8
python scripts/asr/export.py --stock --adapter models/parakeet-ft-f2-M02-adapter8/adapter.pt
```

An adapter is exported on the base it was trained on (`--stock`, or `--nemo` with that run's
base file); the script refuses any other base. The output folder defaults to `onnx-int8` next
to the `.nemo` or the adapter, and `export.json` records the source, the adapter and the
quantisation settings.

**Quantisation.** The default, `--int8-ops all` with one scale per tensor and uint8 encoder
weights, is sherpa-onnx's own recipe and the one every recorded export used. It quantises the
depthwise convolutions with their BatchNorm folded in, where one outlying channel sets the
scale for all 1,024 channels and can round the others' weights to zero. After quantising, the
script loads the int8 encoder and prints the **int8 weight check**: how many quantised weight
tensors have more than half of their values at the zero point (those values dequantise to
exactly 0 and do nothing on the device), with the worst nodes named. `--int8-ops matmul`
leaves every convolution in fp32; `--per-channel` gives each output channel its own scale.
Either produces a different model for the Pi, so it has to be scored again on the F-2 clips
before it replaces an export whose number is on record (`export.json` says which settings an
export used).

**Validation.** The script decodes one second of silence (the stock export says nothing; a
model that answers it is flagged) and, when a recording is at hand (`--check-audio`, or the
stock model's `test_wavs/`), one real clip, and keeps both transcripts in `export.json`. Neither
is an accuracy measurement; that is `evaluate.py` on the test manifest.

Verify the exported graph before evaluation:

```bash
python scripts/asr/check.py \
  --model-dir models/parakeet-ft/onnx-int8
```

### 6. Evaluate

Follow [the candidate evaluation procedure](#5-evaluate-a-fine-tuned-export)
on the same manifest and audio root as the stock baseline. It covers the
comparison settings, result files, and critical-phrase regressions to inspect.

### 7. Deploy

```bash
python run.py --model-dir models/parakeet-ft/onnx-int8
bash deploy/pi5/setup.sh --service --model-dir models/parakeet-ft/onnx-int8   # the boot service on the Pi
```

The service unit records the model folder (`WILLSPEAK_PARAKEET_DIR`) and sets the CPU governor
to `performance` at every boot, so the measured decode times do not depend on who last ran
`cpupower`. The server names the model by its folder and parent (`parakeet:parakeet-ft-M02/onnx-int8`
on the status page and in reports), since every export folder is called `onnx-int8`.

Repeat the model check and latency measurement on the deployment box (the Pi 5: `scripts/asr/bench.py`;
`--with-app --model-dir <folder>` starts the server under test on that same folder).

### 8. A personal model for the device (no scoring)

This builds a model for one real person so the device works for them. It is not
a research run: it uses the best recorded recipe (stage P) unchanged and never
decodes, scores or reports anything about their speech. The IRB confirmed on
5 November that this needs no review as long as it stays that way; keep their
written answer with your project files, and apply any conditions in it (a
consent form, deleting the recordings after the expo) to every copy of the
person's folder.

What `scripts/asr/personal.py` does, and why each part is there:

| Step | What happens | Privacy |
|---|---|---|
| `base` | F-1's recipe on every severe TORGO speaker (F01, M01, M02, M04): stage P's phase 1 with nobody left out. Once, reused for anyone. | TORGO only |
| `export` | Every accepted, transcribed clip of theirs goes to `train`. There is no `test`, and dev is 24 public TORGO clips, because `train.py` needs a dev set; with `--select last` it chooses nothing. Their `split_assignments.json` is read, never written. | stays in `data/people/<name>/` |
| `train` | Stage P's adapter on the base: block 8, `--lr 6e-4`, 15 epochs, last epoch kept. The dev WER it prints is about the TORGO clips. | logs in `data/people/<name>/logs/`, wandb off |
| `deploy` | `export.py` with `--int8-ops matmul` (lab tests 19 and 28: no measurable cost) into `models/personal/<name>/`. Its smoke check decodes the stock sample recording. | `models/personal/` is skipped by the lab bundler |

1. **Record** on whichever machine you take to them, in their own corpus:

   ```bash
   WILLSPEAK_DATA=data/people/<name> ./willspeak.sh run
   ```

   Record on the Record page and accept clips on the Review page; only accepted
   clips with a transcript train. Stage P's speakers had 72 to 156 training clips
   (F-2 sizes: M01 72, M04 82, M02 156), so aim for at least 150. `<name>` is a
   short lowercase folder name; it never reaches Git (`data/` and `models/` are
   ignored), but use initials if you prefer.

2. **Move the folder to the training machine** with `scp` or a copy, never Git
   or cloud sync. From WSL on Yack, with the Pi on the Ethernet cable:

   ```bash
   scp, r jdev@192.168.137.170:~/Desktop/willspeak/app/data/people/<name> data/people/
   ```

3. **Build the base once** (TORGO must already be fetched):

   ```bash
   ./willspeak.sh torgo manifest --export --severe-base --only severe-base
   ./willspeak.sh personal base
   ```

4. **Train and export** their model (`--low-memory` if batch 2 runs out of GPU
   memory; `--dry-run` prints the commands first):

   ```bash
   ./willspeak.sh personal all --name <name>
   ```

   The steps also run one at a time: `personal export`, `personal train`,
   `personal deploy`, each with `--name <name>`. Retraining after more
   recordings: `personal export`, then `personal train --retrain`, then
   `personal deploy`; the previous version is kept as `<folder>.previous`.

5. **Put it on the Pi** and use it:

   ```bash
   scp, r models/personal/<name> jdev@192.168.137.170:~/Desktop/willspeak/app/models/personal/
   ```

   On the Compare page, load any speaker, choose `personal/<name>` in the Model
   list, and press **Try your voice** (R): stock and their model answer side by
   side, labelled `personal/<name>`, and nothing is stored or scored. Leave the
   recorded-clip rows and the tour alone while it is loaded: they play TORGO
   clips through a model that was not trained for that speaker.
   For the Speak page, make it the boot model:
   `deploy/pi5/setup.sh --service --skip-apt --skip-tests --model-dir models/personal/<name>`.

Do not run `evaluate.py`, `nemo_hyp.py` or `battery.py` on their recordings,
and do not put anything about their speech on the poster or in `reports/`.
Those would make it a measurement, which is the research phase below.

### Troubleshooting

- empty train/dev: review more clips and export again;
- NeMo import failure: verify the active interpreter and Linux environment
  (`python --version` must say 3.12 inside `~/venvs/willspeak-train`);
- `Failed to build onnx` during `pip install -e ".[train]"`: the environment is
  Python 3.14; make the 3.12 one with `uv` as above;
- `libNVVM cannot be found`: the two `echo` lines were not added to the
  activate script, or the environment was not re-activated after adding them;
- `No supported GPU compute capabilities found`: `numba-cuda` is missing
  (`python -m pip install "numba-cuda[cu12]"`); numba's built-in CUDA support
  does not know the pip CUDA version;
- `No module named 'resampy'` warnings: speed perturbation is being skipped;
  `python -m pip install resampy`;
- `No space left on device` while saving the `.nemo`: the output or the temp
  folder is on the RAM-backed `/tmp`; use `--out models/<name>` and re-run
  `./willspeak.sh train-setup` (adds `TMPDIR`);
- a bare `Killed` right after a validation pass: Linux ran out of memory
  while saving checkpoints; add swap (`.wslconfig` above).
  `sudo dmesg | grep -i "killed process"` confirms it;
- any of the errors below: `./willspeak.sh train-setup` repairs the environment;
- `module 'numpy' has no attribute 'row_stack'`: numpy is too new for
  `numba-cuda`; `python -m pip install "numpy<2.3"` (the `train` extra pins it);
- no CUDA: stop rather than attempting a practical full fine-tune on CPU;
- training suddenly slows to a fraction of its usual speed (below 1 it/s) with
  `nvidia-smi` showing the 8 GB almost full and the GPU cool: Windows is
  spilling GPU memory into system RAM. Close other Windows apps and set
  NVIDIA Control Panel > Manage 3D settings > "CUDA, Sysmem Fallback Policy"
  to "Prefer No Sysmem Fallback", so a full GPU stops with an error instead of
  crawling (30 Sep 2026). To make room on the card without changing the method:
  `--offload-optimizer` keeps AdamW's running averages in system RAM (about
  1.7 GB less on the GPU with `--train-parts all`), and `--batch 1 --grad-accum 16`
  holds one recording at a time instead of two while each update still uses 16.
  The epoch summary reports peak GPU reserved memory;
- out of memory: lower batch size and increase gradient accumulation;
- export incomplete: keep encoder, decoder, joiner, and tokens from the same
  checkpoint and rerun the graph check;
- worse test WER: inspect overfitting, split leakage, transcript errors, session
  coverage, and decoding settings before adding epochs.

The model-family evidence, training hypotheses, and migration gate are in
[research/parakeet.md](research/parakeet.md). The complete experiment and
release protocol is in [measurements.md](measurements.md).


---

## Tutorial: evaluate Parakeet reproducibly

This tutorial produces a stock or personalized Parakeet report without
changing the corpus. It assumes the model passes the
[Parakeet setup](device.md#parakeet-install-validate-run-troubleshoot).

### 1. Review before export

Evaluation references are only as good as the review process. In the Review
page, listen to each accepted clip and check that the transcript matches the
audio, not just the prompt.

Before exporting, check:

- accepted clips have non-empty actual transcripts;
- incomplete or unusable clips are rejected or explicitly transcribed;
- session condition and microphone metadata are filled when known;
- critical, functional, phonetic, probe, and live clips have the intended
  prompt set;
- a backup exists.

### 2. Export stable splits

Use the Coverage page's export action. The expected layout is:

```text
data/export/
  train.jsonl
  dev.jsonl
  test.jsonl
  train/*.wav
  dev/*.wav
  test/*.wav
data/split_assignments.json
```

All recordings with the same normalized prompt text remain in the same split.
The assignment file is sticky: once a prompt belongs to development or test,
future exports keep it there.

Inspect counts without modifying anything:

```bash
wc -l data/export/train.jsonl \
      data/export/dev.jsonl \
      data/export/test.jsonl
```

Do not use `train.jsonl` for a claimed model result. Training WER is useful for
debugging but says little about generalization.

### 3. Establish the stock baseline

```bash
python scripts/asr/evaluate.py \
  --manifest data/export/test.jsonl \
  --audio-root data/export \
  --out reports/parakeet-stock \
  --label parakeet-stock
```

The evaluator transcribes every manifest row through the supported adapter and
writes:

```text
reports/parakeet-stock/
  results.json
  hypotheses.jsonl
  difficulty.json
  report.html
  provenance.json
```

`results.json` contains aggregate WER, CER, PER, error operations, utterance
details, and the selected group breakdown. `hypotheses.jsonl` keeps model
output by manifest ID. `difficulty.json` records phoneme weights together with
their source split and manifest hash. `report.html` is a readable summary.

`provenance.json` binds those outputs to manifest/audio/model hashes, source
hashes, package versions, and decoder settings. Report directories must be new;
use a run-specific `--out`. Imported hypotheses require every manifest ID once,
with no extras. Use `"hyp": ""` for a genuine recognition deletion. Even imported
evaluation keeps the original audio inventory for comparison.

For reproducible desktop runs, evaluate the unpacked `datasets/<snapshot>/export`
(created by `transfer.py unpack`) rather than an export directory that may change. The full transport, release,
and rollback procedure is in [Laptop collection → desktop training → laptop testing](#laptop-collection--desktop-training--laptop-testing).

Reports are generated artifacts and are ignored by Git. Copy the model
configuration, exact hypotheses, and summarized result into an experiment or
release record if the comparison matters long term.

### 4. Group by recording condition

The default groups by prompt set. Run a second report for session condition:

```bash
python scripts/asr/evaluate.py \
  --group-by condition \
  --out reports/parakeet-stock-condition \
  --label parakeet-stock-condition
```

Look for an aggregate improvement hiding a difficult-day regression. Small
groups should be interpreted with their utterance and reference-word counts.

### 5. Evaluate a fine-tuned export

The recorded fine-tuned exports and their scores are in `app/reports/` (the
[report map](../app/reports/Report%20Context.md)). For a new export, first prove
that the graph itself loads:

```bash
python scripts/asr/check.py \
  --model-dir models/parakeet-ft/onnx-int8
```

Then use the identical manifest and audio root:

```bash
python scripts/asr/evaluate.py \
  --manifest data/export/test.jsonl \
  --audio-root data/export \
  --model-dir models/parakeet-ft/onnx-int8 \
  --out reports/parakeet-ft \
  --label parakeet-ft
```

Compare only when these are fixed:

- manifest contents and split assignments;
- reference transcripts;
- normalization code;
- grouping;
- decoder settings;
- deployment graph type.

Review aggregate WER, insertions, deletions, critical/functional groups,
condition groups, empty hypotheses, worst utterances, and latency. A lower
total WER is not enough when a critical phrase regresses.

### 6. Evaluate imported hypotheses

Use this mode when recognition ran on another machine or through a source NeMo
checkpoint:

```bash
python scripts/asr/evaluate.py \
  --hyp path/to/hypotheses.jsonl \
  --out reports/imported \
  --label imported-system
```

Each non-empty line must be JSON:

```json
{"id": "<manifest id>", "hyp": "recognized text"}
```

IDs must match the manifest. Keep the external model and decoder configuration
beside the hypotheses; an unlabeled text file is not a reproducible result.

This mode is useful for quantization parity: create hypotheses from the source
checkpoint and the int8 deployment graph, then score both with the same
normalizer.

### 7. Read a result correctly

The WER definition, the worked example and the sum-then-divide rule live in
[WER](design.md#word-error-rate), one definition for every report. Use CER for
spelling-level near misses and PER for phoneme-error scheduling. Neither
replaces listening to safety-relevant errors.

### 8. Feed difficulty back into recording

Use **development-set** errors to choose new recording prompts. Choosing what
to record from test errors lets the test set influence the next training set,
even when none of its exact prompts is copied. New difficulty records include
their source split; the server accepts feedback records from `dev.jsonl` and
refuses records from test or training splits. Test-derived weights remain in
the report for analysis. Manual phone-to-weight objects remain supported;
their values must be finite, greater than zero and at most 5, and their source
cannot be checked automatically.

```bash
python scripts/asr/evaluate.py --manifest data/export/dev.jsonl \
  --audio-root data/export --model-dir models/parakeet-ft/onnx-int8 \
  --out reports/parakeet-ft-dev
curl -X POST localhost:8000/api/difficulty \
  -H 'content-type: application/json' -d @reports/parakeet-ft-dev/difficulty.json
```

After collecting and training, evaluate the frozen candidate on the unchanged
test split. If test results have already guided recipes, call subsequent
comparisons exploratory; a new untouched evaluation is needed for confirmation.

### 9. Preserve a named experiment

For a decision-changing run, record:

```text
question and prediction
repository commit
corpus/export identity
split-assignment identity
model directory and graph hashes
training run.json
decoder settings
hardware and thread count
results and hypotheses
decision and accepted regressions
```

A generated folder name such as `run` is not enough. Prefer names that encode
the comparison, such as `parakeet-v2-stock-2026-09-01`.

### 10. Common mistakes

- Comparing reports after editing reference transcripts without rescoring both
  models.
- Evaluating a `.nemo` checkpoint for one side and int8 ONNX for the other
  without labeling the conversion difference.
- Selecting epochs or thresholds on the test split.
- Averaging utterance WER values.
- Treating punctuation-only differences as acoustic evidence.
- Reporting a small condition group without counts.
- Assuming one correct live phrase validates the corpus.
- Committing large generated HTML and JSON report directories.

### 11. Completion checklist

- Test manifest is prompt-disjoint from train.
- References are reviewed.
- Exact deployment graph produced the hypotheses.
- Plain and experimental configurations use the same inputs.
- Aggregate and condition/prompt groups are inspected.
- Critical errors and insertions are reviewed manually.
- Latency is measured on the target device.
- Hypotheses and experiment identity are kept for named comparisons.

For the full release protocol and statistical cautions, see
[Evaluation protocol and release evidence](measurements.md).


---

## Laptop collection → desktop training → laptop testing

> **Where this fits (Fall 2026).** This is the transport loop for a
> fine-tune: the corpus goes to the 4060 desktop as a fixed snapshot, the
> trained int8 model comes back as a release bundle, and the release is then
> copied to the deployment box (the Raspberry Pi 5, `deploy/pi5/`). The
> semester's TORGO experiment uses the same loop with a TORGO export in place
> of the personal corpus ([research/finetune-torgo.md](research/finetune-torgo.md)).
> "Laptop" below means whichever machine holds the editable corpus.

The laptop owns the editable corpus. The desktop trains on a fixed, verified
export; it does not edit the laptop's SQLite database. An evaluated model comes
back as a separate release directory. Keep the previous working model for rollback.

Git moves source, tests, prompt seeds, and documentation. `scripts/corpus/transfer.py`
moves private data and model artifacts. It does not upload or sync anything;
the resulting ZIP moves by hand on an approved encrypted drive or through an
authenticated file-transfer service. A private repository is not the default corpus store.

### 1. Machine roles and directories

| Location | Role | Mutable? | Move to the other machine? |
|---|---|---|---|
| laptop `data/corpus.db`, `data/clips/`, `data/raw/` | authoritative corpus | through the app | recovery backup only |
| laptop `data/export/` | accepted, trimmed, split audio | replaced by Export corpus | package a snapshot |
| laptop `data/split_assignments.json` | persistent prompt assignments | new prompts added | in every corpus bundle |
| desktop `datasets/session-01/export/` | training/evaluation snapshot | treat as immutable | no need to return |
| desktop `models/session-01/` | run, `.nemo`, ONNX export | one run only | deployable release files only |
| desktop `reports/session-01-test/` | evaluation and provenance | one evaluation only | included with model |
| laptop `models/candidate-01/model/` | received candidate ONNX files | treat as immutable | keep alongside prior model |
| either `transfers/` | transport ZIPs | new filename per bundle | approved transport |

Generated/private directories are ignored by Git, and nothing under `data/` or
`models/` (other than `models/.gitkeep`) is tracked. `data/split_assignments.json`
was untracked on 2 Sep 2026: its keys are the speaker's own verified live
transcripts, so it is private data and travels inside every corpus bundle
instead. Before pulling that commit on the desktop, copy its
`data/split_assignments.json` aside -- Git deletes a tracked file that the
incoming commit removes -- then put it back. No Git LFS attributes remain and
git-lfs is not required on any machine. Do not force-add private artifacts
during a normal source commit.

Keep the checkout outside OneDrive/iCloud/Dropbox-style live sync folders.
Copy completed ZIP files, not an actively written SQLite database. A corpus
bundle is a training snapshot, not a disaster-recovery backup: it omits rejected
takes, raw recordings, corrections history, and the database.

### 2. Set up each machine

Use the same source commit and record it with the experiment. Create a separate
virtual environment on each machine; do not move one between operating systems.
Follow the [runtime setup](device.md#2-create-the-runtime-environment) on the
collection machine and [training dependencies](#2-install-training-dependencies)
on the Linux/WSL2 desktop. Verify each with `python scripts/asr/check.py` and
`python -m pytest`.

The one-line commands below also work in PowerShell. Run from the application directory (`app/`);
replace the example session/run names with meaningful new names.

### 3. Collect, review, export, and package on the laptop

1. Follow the [Recording protocol](plan.md#recording-protocol). Record condition
   and fatigue; vary days, not only repetitions within one sitting.
2. Review transcripts and accept usable clips. Fix a wrong transcript rather
   than treating the ASR guess as ground truth.
3. Create a full recovery backup in the app and store it separately.
4. Use **Export corpus** on Coverage and wait for completion. Do not export again
   while packaging; export replaces `data/export/`.
5. Package the snapshot:

```bash
python scripts/corpus/transfer.py pack-corpus --out transfers/session-01.zip
python scripts/corpus/transfer.py inspect transfers/session-01.zip
```

The packer requires nonempty train/dev/test splits and checks:

- unique clip IDs and normalized transcript/prompt isolation across splits;
- identical audio bytes accidentally placed in different splits;
- existing, non-symlink audio inside the declared split directory;
- mono 16 kHz audio, finite positive durations, and agreement with WAV duration;
- matching sticky prompt assignments (critical/probe are training-only exceptions);
- a SHA-256 inventory of each manifest, referenced WAV, and assignment file.

Only referenced exported audio is included. Unrelated files in `data/export/`
are not swept into the bundle. `bundle_id` identifies inventory and metadata;
`dataset_id` identifies split manifests/audio. Hashes detect changed bytes, not
whether a transcript is correct.

If a split is empty, collect distinct eligible prompts and re-export. Do not
copy training clips into test. If sticky assignments disagree, find the matching
assignment file; do not delete it to obtain a convenient split.

### 4. Verify and train on the desktop

Transfer `session-01.zip` through the approved channel, then:

```bash
python scripts/corpus/transfer.py unpack transfers/session-01.zip --into datasets/session-01
python scripts/asr/train.py --check --export datasets/session-01/export
```

Unpack checks archive paths, limits, and every checksum in a temporary directory
before publishing the destination. Existing destinations are refused, even if
empty. Symlinks, absolute/traversal paths, Windows reserved names, duplicate or
case-colliding paths, and unlisted entries are rejected. Limits are 100,000
entries and 20 GiB uncompressed per bundle. Budget disk space for ZIP, extraction,
training checkpoints, and ONNX export simultaneously.

Preflight uses only train/dev: it never reads the test manifest or test audio.
It needs neither CUDA nor NeMo, reports clips excluded by `--max-duration`, and
rejects empty remaining training data. It creates a diagnostic log, not a run.

```bash
python scripts/asr/train.py --export datasets/session-01/export --out models/session-01
python scripts/asr/export.py --nemo models/session-01/parakeet-ft.nemo --out models/session-01/onnx-int8
```

Training records CLI options, input hashes, package/CUDA versions, device,
script hash, and the completed checkpoint hash. A `running` status means
completion was never recorded, and the run is not releasable. Training picks checkpoints on development WER
by default, or by a fixed schedule with `--select last`. Export requires a real sherpa-onnx smoke decode; an export/decode
exception publishes no model directory. Neither overwrites an earlier run.

### 5. Evaluate a frozen candidate

Tune on development data. Once model/decoder settings are fixed, evaluate stock
and candidate on the same held-out snapshot:

```bash
python scripts/asr/evaluate.py --manifest datasets/session-01/export/test.jsonl --audio-root datasets/session-01/export --out reports/session-01-stock
python scripts/asr/evaluate.py --manifest datasets/session-01/export/test.jsonl --audio-root datasets/session-01/export --model-dir models/session-01/onnx-int8 --out reports/session-01-test
```

Unset `WILLSPEAK_PARAKEET_DIR` for the stock command, or specify the stock directory.

`provenance.json` records input/model/output hashes, packages, source hashes,
decoder settings, and recognition time including cold load. Report directories
must be new. Imported hypotheses require exact ID correspondence and are marked
as imported. An explicit empty hypothesis means deletion; a missing row is an error.

Compare WER/CER, insertions/deletions, conditions, and worst utterances. Critical
and probe phrases forced into training require a **separate future-session
closed-set evaluation**. The automatic report does not calculate every release
statistic in the [evaluation protocol](measurements.md), including
cluster-bootstrap intervals, warm p95 latency, and live false activations.
Measure and record those separately.

### 6. Package and return the evaluated model

```bash
python scripts/corpus/transfer.py pack-model --model-dir models/session-01/onnx-int8 --evaluation reports/session-01-test --out transfers/candidate-01.zip
```

This includes the self-contained int8 trio, tokens, recognized
attribution/export metadata files, and the matching evaluation report.
It excludes `.nemo`, optimizer checkpoints, and recordings. Reports still contain
transcripts, IDs, and error examples: model releases are private artifacts too.

Packaging fails for changed model bytes or report outputs, non-test splits,
and imported predictions. The gate shows that the artifacts belong together. It does not
show that the splits were labeled correctly, that training never leaked, or that the model
suits client use.
Human review remains required. Bundles are neither encrypted nor digitally signed.

### 7. Install, test, and roll back on the laptop

After copying the release ZIP back:

```bash
python scripts/corpus/transfer.py unpack transfers/candidate-01.zip --into models/candidate-01
python scripts/asr/check.py --model-dir models/candidate-01/model
python scripts/asr/check.py --model-dir models/candidate-01/model --audio /path/to/approved-local-sample.wav
python run.py --model-dir models/candidate-01/model
```

`--model-dir` selects this server process's model without changing a saved default
or replacing stock files. Stop the existing server first, then check the selected
model on Speak. Test microphone capture, normal/fatigued speech, silence and
background speech, phrase confirmation, speech output, and stop-phrase behavior
with whoever will use it (the client only after IRB approval). Measure warm
latency on the deployment box (the Pi 5).

To roll back, stop the server and launch with the previous model directory.
Use stock explicitly if needed; `python run.py` alone still honors
`WILLSPEAK_PARAKEET_DIR` when set. Keep the prior release until acceptance criteria
are met. Installing a candidate does not approve automatic speech.

### Recovery and privacy checklist

- Corrupt bundle: recopy and compare IDs through a trusted channel; do not bypass
  checks or unpack over a partial prior installation.
- Recovery backup: stop the server, then run
  `python scripts/corpus/restore_backup.py transfers/backup.zip --into data-recovered`.
  The destination must not exist. `--dry-run` checks paths/ZIP integrity without
  writing; actual restore also checks SQLite integrity. There is no `--force`
  mode. Select the recovered corpus explicitly with `WILLSPEAK_DATA` and keep the
  original until the counts and sample recordings check out.
- Two recording laptops: designate one authoritative corpus. This tool does
  not merge SQLite databases or provide globally unique cross-corpus clip IDs.
- Changed source: keep the prior revision as well as the weights. Decoder/audio
  preprocessing changes can alter results with identical model files.
- Access/deletion changes: account for source history, snapshots, reports,
  derived models, ZIPs, external storage, and recovery backups.
- No cloud inference is required. Stock-model/dependency downloads are separate
  from transferring private data.
