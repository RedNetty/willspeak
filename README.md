# WillSpeak

The repository has two parts:

- [app/](app/) is the complete runnable application: source, launcher,
  requirements, browser assets, active prompts, model tooling, tests, and Pi setup.
- [docs/](docs/) holds personal notes, plans, measurements, and research.

## Quick start after cloning

Every command runs from `app/`. `setup` makes `app/.venv` (Python 3.11 or
newer), installs the app and test requirements, and downloads Parakeet. It is
safe to run again.

Windows (PowerShell or cmd):

```bat
cd app
.\willspeak.bat setup
.\willspeak.bat test
.\willspeak.bat run
```

macOS, Linux, WSL or the Pi:

```bash
cd app
./willspeak.sh setup
./willspeak.sh test
./willspeak.sh run
```

Then open http://127.0.0.1:8000/live in Chrome or Edge.

### Training (Ubuntu under WSL2, NVIDIA GPU)

NeMo does not run on native Windows, so fine-tuning happens in WSL2. Clone the
repository inside the Linux filesystem (`~/willspeak`, not `/mnt/c/...`, which is
about ten times slower for training reads), then:

```bash
git clone https://github.com/RedNetty/WillSpeak.git ~/willspeak
cd ~/willspeak/app
./willspeak.sh setup                      # app environment: tests, evaluate.py, ci.py
./willspeak.sh train-setup                # training environment ~/venvs/willspeak-train (Python 3.12, NeMo, CUDA)
./willspeak.sh torgo --controls           # download TORGO, dysarthric and control speakers (several GB, once)
./willspeak.sh torgo manifest --export    # build every train/dev/test folder under data/public/torgo/export/
./willspeak.sh train --export data/public/torgo/export/F01 --out models/smoke-test --max-steps 2 --batch 1 --freeze-encoder-layers 24
rm -rf models/smoke-test
```

The last training line is a two-step smoke test: it proves the data, the model
download (2.4 GB, once), the GPU and the loss all work. `./willspeak.sh train` and
`./willspeak.sh export` use `~/venvs/willspeak-train` on their own; nothing needs
activating first. Before a long run, set WSL's memory and swap
(`.wslconfig`, [docs/training.md](docs/training.md) §2).

Each experiment's exact commands are in its notes file,
`app/reports/<experiment>/notes.md`, under `### Commands`. To run them from
`app/`:

```bash
bash <(sed -n '/^### Commands/,$p' reports/<experiment>/notes.md | awk '/^```bash/{f=1;next} /^```/{f=0} f')
```

## Moving the app to another machine

To run on another machine, package `app/`, extract it there, install its
requirements, and start `run.py`. Follow [the application quickstart](app/README.md)
for the exact commands and one-time model setup. Python environments are
machine-specific and are recreated on the destination.

From the repository root, make a clean archive with:

```bash
python app/package.py --out willspeak.zip
```

Add `--with-models` to carry downloaded models too. Recordings, virtual
environments, generated reports, and personal documentation are excluded.
The output ZIP is ignored by Git. The package needs no files from this
repository's outer directory.

All operational paths and commands in the notes are relative to `app/`.
The [repository map](docs/design.md#repository-organization) explains where
application changes belong. Candidate vocabulary lives in
[docs/wordlists.txt](docs/wordlists.txt); only `app/prompts/` is loaded at runtime.

## Notes and research

Start with [Report Context](app/reports/Report%20Context.md) for the proposal's
hypotheses, experiment goals, results map, and note naming scheme. Each public
experiment note has a short overview and an expandable original run record.
This repository carries the notes only. The raw result files they cite
(decodes, score tables, `logs/` and `models/*/run.json`) are kept out of it;
rerunning a note's commands regenerates them at the same paths.

`docs/` is one task list, six guides and four research references, plus the
candidate vocabulary in `wordlists.txt` mentioned above:

| file | what it is |
|---|---|
| [docs/TODO.md](docs/TODO.md) | the only task list; every item carries its instructions |
| [docs/plan.md](docs/plan.md) | the Fall 2026 semester: dates, project work, the storyboard, ethics review, the client-phase recording protocol |
| [docs/measurements.md](docs/measurements.md) | every number for the poster, the command that produces it and the table it fills (T-n, F-n, P-n) |
| [docs/design.md](docs/design.md) | architecture, safety, other voices in the room, corpus and splits, WER, repository organisation |
| [docs/device.md](docs/device.md) | the Speak page and caregiver panel, the matcher, Parakeet install and troubleshooting |
| [docs/pi5.md](docs/pi5.md) | the Raspberry Pi 5: bring-up, the benchmark and every number in it, the deploy folder |
| [docs/training.md](docs/training.md) | TORGO, the fine-tune runbook (WSL2), evaluation, moving corpus and models between machines |
| [docs/research/](docs/research/) | reference, not tasks: `finetune-torgo.md` (read §2 once), `matcher.md`, `parakeet.md`, `round-3.md` (CPU pilot work, context, not project measurements) |
