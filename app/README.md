# Run WillSpeak

This folder is the complete application. Copy it to a Windows, macOS, or Linux
machine with Python 3.11 or newer. Create a new virtual environment there;
virtual environments cannot be moved between machines.

The short way, from this folder: `./willspeak.sh setup` (macOS, Linux, WSL, the
Pi) or `.\willspeak.bat setup` (Windows) creates `.venv`, installs the app and
test requirements, and downloads Parakeet. Then `test` runs the tests and `run`
starts the app. Training setup is in the repository's top-level README.

For the lab tests on a Linux NVIDIA computer, run from this folder:

```bash
./willspeak.sh lab-setup
```

After setup, `./willspeak.sh lab-remaining` pulls the code, repairs what it can,
checks the machine and the plan, and queues every lab experiment that is not
finished yet in the background, pushing results to GitHub as it goes
(`docs/measurements.md`, **Lab tests**, in the repository).
`./willspeak.sh lab-lm` does the same for lab test 36 alone (language-model
rescoring), including its language-model environment and downloads.

This installs the app, CPU PyTorch for word matching, the GPU training
environment and separate Hugging Face/Qwen environments. It downloads public
TORGO audio, creates the M01/M02/M04 and leave-one-out exports plus `words.csv`,
and downloads WavLM, Parakeet int8 and the stock NeMo checkpoint. It uses several
GB of downloads and disk space. Candidate models download on first use.
Word embeddings are generated when a lab stage first needs them.
Setup runs no training or research decodes.

To rebuild the earlier models locally instead of transferring them, turn on
the runner's `rebuild_missing` setting. For example, stage 1:

```bash
python lab.py --set rebuild_missing=1 run 1
```

Lab stages 16 and later, the letter stages and the long studies turn it on by
themselves; stages 1 to 15 need the setting. It prepares only the baseline
types needed by the selected stages and reuses
completed files on later invocations. New checkpoints go under
`models/lab-baselines/`; their decodes and scores go under
`reports/finetune-torgo/lab-baselines/`. These are new runs with current code,
not the original Yack results, and GPU training can vary. Stage 1 rebuilds
the three whole-block models and stock/whole-block decodes. Stages needing
F-1 models or adapters rebuild those too and can take much longer. Preparation
logs end in `-prepare.txt`. A failed training folder is reported for inspection
and preserved. Stage 8b still needs stage 8's output; rebuilding prepares
baselines, not the results of other lab experiments. Write predictions for
the lab stages before running them, as described in the research guides.

Earlier fine-tuned checkpoints and decode reports are **not downloadable**.
Copy the required `models/parakeet-ft-*/` folders and
`reports/finetune-torgo/` from Yack into the same paths on the lab computer,
including decode provenance files. Copy the matching
`data/public/torgo/export/` too when using those earlier results. Git carries
the decodes and score tables the lab computer committed, but never model
weights or audio, and the portable ZIP includes none of them. Linux paths are case
sensitive: speaker folders are `M01`, `M02`, `M04`.

Lab setup preserves complete existing exports and refuses to replace partial
ones, to protect splits used by recorded results. If it reports incomplete
exports, finish copying the matching exports from Yack and rerun setup.
`setup` alone prepares the app; `train-setup` prepares training software alone.

The same steps by hand, from this folder:

```bash
python -m venv .venv
```

Activate with `source .venv/bin/activate` on macOS/Linux, or
`.venv\Scripts\Activate.ps1` in Windows PowerShell. Then:

```bash
python -m pip install -r requirements.txt
python scripts/fetch_models.py --parakeet-only
python run.py
```

Open http://127.0.0.1:8000/live in Chrome or Edge and allow microphone access.
Model setup needs internet once; if `models/` was included, existing Parakeet
files are reused. The application then runs locally. On Linux, browser speech
output also needs Speech Dispatcher and a voice such as espeak-ng; the Pi
installer below installs these system packages.

For enrolled-phrase matching, install a PyTorch build for the destination and
run `python scripts/fetch_models.py` without `--parakeet-only`. Sentence
recognition does not need PyTorch. Fine-tuning additionally needs the `train`
extra and a Linux/WSL2 CUDA training environment.

On a Raspberry Pi or Debian machine, `bash deploy/pi5/setup.sh --service`
installs dependencies, downloads the model, and configures boot and kiosk.
Add `--with-matcher` for phrase matching.

`willspeak.sh` (macOS/Linux) and `willspeak.bat` (Windows) provide a command menu
(they use an activated environment, else `.venv` here, else `.venv` one level
up at the repository root);
`python run.py` works on all three. Run `python package.py --out /path/to/willspeak.zip`
to build a clean portable ZIP. Add `--with-models` to include downloaded models.
Extract it, enter the included `app/` folder, and follow the commands above.

The archive excludes recordings, reports, virtual environments, caches and
personal documentation. Its tests are included; install `requirements-dev.txt`
and run `python -m pytest` to check a destination machine.
With Node.js installed, `node --test tests/js/live.test.cjs` checks the live
page's speech queue, cancellation and recovery logic without a microphone.

Runtime files live in `data/`, `models/`, `reports/`, and `logs/` inside this
folder. `WILLSPEAK_DATA` selects a different corpus; `WILLSPEAK_PARAKEET_DIR` selects
a different Parakeet graph. Copy a corpus through the backup/transfer commands,
not by copying a database while the server is running.
