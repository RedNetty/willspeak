#!/usr/bin/env python3
"""willspeak lab."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections import deque
from pathlib import Path

CODE = Path(__file__).resolve().parent
for _sub in ("src", "scripts", "scripts/asr"):
    sys.path.insert(0, str(CODE / _sub))

import lab_doctor  # noqa: E402

EX = "data/public/torgo/export"
FT = "reports/finetune-torgo"
HOME = Path.home()
TRAIN_PY = HOME / "venvs/willspeak-train/bin/python"
SETTINGS_FILE = ".lab/settings.json"
EXPERIMENTS_FILE = "lab_experiments.json"
SEVERE = ("M01", "M02", "M04")
BAR_WORDS = 148


def now() -> str:
    return time.strftime("%H:%M:%S")


VARIABLES: dict[str, tuple] = {
    "speakers": ("M01 M02 M04", "SPK", "speakers every test runs on; put the hardest first, e.g. 'M04 M01'"),
    "seeds": ("", None, "seeds for every sweep, e.g. '17 18 19'; empty = each sweep's own seeds"),
    "xflags": ("", "XFLAGS", "extra train.py flags added to every training run, e.g. --offload-optimizer"),
    "long_flags": ("", None, "resource flags for the three long experiments, e.g. --offload-optimizer or --batch 4 --grad-accum 4; xflags does not change their fixed recipes"),
    "trim_models": ("auto", "TRIM_MODELS", "delete a fine-tuned model's weights after its decodes: on | off | auto (auto = under 100 GB free)"),
    "rebuild_missing": (False, "REBUILD_MISSING", "train missing baselines into models/lab-baselines/ instead of needing Yack's"),
    "rerun": (False, "RERUN", "rename a half-finished earlier attempt to NAME-first-run instead of refusing"),
    "tune_tags": ("v2 v3 1p1b rnnt06", "TUNE_TAGS", "lab test 7: base models to fine-tune (candidates.py lists the tags)"),
    "arms9": ("b2:0:2 b4:0:4 b12:0:12 b16:0:16 mid8:8:8", "ARMS9", "lab test 9: arms as label:first block:blocks"),
    "rounds": (3, "ROUNDS", "lab test 12: pseudo-label rounds"),
    "margin": (5.0, "MARGIN", "lab test 10: anti-silence margin in nats"),
    "kper": (8, "KPER", "lab test 10: non-speech clips per kind per epoch"),
    "layers": ("all", "LAYERS", "lab test 5: encoder layers to screen"),
    "wb_m02_nemo": ("", "WB_M02_NEMO", "M02's whole-block model (default: the M02e-A3 run)"),
    "wb_m02_hyp": ("", "WB_M02_HYP", "M02's whole-block decode (default: the M02e-A3 decode)"),
    "min_free_gb": (0, "MIN_FREE_GB", "queue: free disk to keep between experiments (0 = 14 with trimming, 40 without)"),
    "wandb": ("auto", None, "Weights & Biases: auto | online | offline | off"),
    "wandb_project": ("willspeak-torgo", None, "Weights & Biases project"),
    "wandb_entity": ("", None, "Weights & Biases team or user name (empty = your default)"),
    "hfpy": (str(HOME / "venvs/willspeak-hf/bin/python"), "HFPY", "Python of the Hugging Face environment (lab test 4)"),
    "qwenpy": (str(HOME / "venvs/willspeak-qwen/bin/python"), "QWENPY", "Python of the Qwen environment (lab test 4)"),
    "cpupy": ("", "SC", "Python for the CPU scripts (empty = .venv/bin/python, else this one)"),
    "export_to": ("", "EXPORT_TO", "where finished models go to free disk: rclone:gdrive:willspeak-models (Google Drive) or a folder; used by the queue when disk runs low and by `export-models`"),
    "auto_fix": (True, "AUTO_FIX", "queue: let the doctor repair what it safely can before each experiment (never deletes)"),
    "allow_download": (False, "ALLOW_DOWNLOAD", "queue: let the doctor download missing public data it needs (the TORGO control speakers, about 2.5 GB)"),
    "push_results": (False, "PUSH_RESULTS", "queue: push each saved lab-results-* branch to origin as soon as it is built, so results reach the other computer without a manual step"),
    "wait_busy": (False, "WAIT_BUSY", "queue: if another lab run is going, wait for it to finish instead of stopping"),
    "stage_retries": (1, "STAGE_RETRIES", "queue: how many times to retry an experiment that failed for a fixable reason (disk, memory, network)"),
    "synth_severity": (0.0, "SYNTH_SEVERITY", "lab tests 17-18: how severe the synthetic speech is, 0.2-1.2 (0 = the value lab test 16 calibrated, else 0.7)"),
    "synth_scale": (2.0, "SYNTH_SCALE", "lab tests 17-18: synthetic clips mixed in per clip of his own training data"),
    "synth_pre_scale": (6.0, "SYNTH_PRE_SCALE", "lab test 17: synthetic clips per own clip for the synthetic-only pre-training arm"),
    "synth_severities": ("0.3 0.5 0.7 0.9 1.1", "SYNTH_SEVERITIES", "lab test 16: severities to try on healthy recordings"),
    "cv_folds": (4, "CV_FOLDS", "lab tests 20 and 23: folds that hold out whole prompts"),
    "cv_arms": ("adf1:8:15:f1 ad:8:15:stock adf1e30:8:30:f1 adf1b4:4:15:f1", "CV_ARMS", "lab test 20: recipes as label:block:epochs:f1|stock; the first is the control. Lab test 23 needs adf1 and ad"),
    "cv_min_gain": (1.0, "CV_MIN_GAIN", "lab test 20: points of WER a recipe must beat the control by, on top of the paired interval"),
    "selector_folds": (5, "SELECTOR_FOLDS", "lab test 23: folds for the selector's own leave-prompt-out score"),
    "tts_backend": ("auto", "TTS_BACKEND", "lab test 21: speech engine, auto | espeak | piper (python scripts/asr/tts_synth.py backends lists what is installed)"),
    "tts_voices": ("", "TTS_VOICES", "lab test 21: folder of piper .onnx voices (empty = the default places)"),
    "tts_variants": (4, "TTS_VARIANTS", "lab test 21: voice and speed variants per sentence"),
    "tts_max_clips": (2000, "TTS_MAX_CLIPS", "lab test 21: most synthesised clips per speaker"),
    "tts_extra_text": ("", "TTS_EXTRA_TEXT", "lab test 21: a text file of extra phrases to speak, one per line (e.g. what a caregiver typed in)"),
    "tta_chunks": (4, "TTA_CHUNKS", "lab test 22: chunks the test clips arrive in (chunk 0 is decoded before anything is learned)"),
    "tta_order": ("hash", "TTA_ORDER", "lab test 22: hash = chunks mix the prompts, id = chunks are runs of sorted ids"),
    "tta_agree_min": (0.5, "TTA_AGREE_MIN", "lab test 22: how much the other systems must agree with an answer before it is trained on (0-1)"),
    "tta_max_pseudo": (1.0, "TTA_MAX_PSEUDO", "lab test 22: most pseudo-labelled clips, as a share of his labelled clips"),
    "acoustic_severe_epochs": (30, "ACOUSTIC_SEVERE_EPOCHS", "lab test 32: epochs of the acoustic recipe on the other severe speakers (about 38 updates an epoch)"),
    "acoustic_pooled_epochs": (30, "ACOUSTIC_POOLED_EPOCHS", "lab test 33: epochs of the acoustic recipe on every dysarthric speaker at once (about 120 updates an epoch)"),
    "pooled_positions": ("4 8 12 16 24", "POOLED_POSITIONS", "lab test 33: adapter positions on the pooled model (after encoder block N, 1-24)"),
    "pretrained_positions": ("8 12 16 24", "PRETRAINED_POSITIONS", "lab test 34: adapter positions on stage P's pretrained model (after encoder block N, 1-24)"),
    "lm_models": ("", "LM_MODELS", "lab test 36: language models that re-rank the candidates: tags or groups (pi, large, ceiling; python scripts/asr/lm_score.py list); empty = pi"),
    "lm_systems": ("stock prototype folds", "LM_SYSTEMS", "lab test 36: which candidate pools to build: stock (stock Parakeet on every severe sentence), prototype (stage P's adapter and stock on the test clips), folds (stage P's recipe cross-validated on the train and dev clips, and stock on the same clips)"),
    "lm_views": ("greedy beam4 beam8 tempo1.1 tempo1.2 tempo1.3 pitch0.9 pitch0.95 pitch1.05", "LM_VIEWS", "lab test 36: the decodes that propose answers (python scripts/asr/lm_pool.py --help)"),
    "lm_threads": (8, "LM_THREADS", "lab test 36: CPU threads the language model may use here (speed on the Pi is measured on the Pi)"),
    "lmpy": (str(HOME / "venvs/willspeak-lm/bin/python"), "LMPY", "Python of the language-model environment (lab test 36; ./willspeak.sh lm-setup builds it)"),
    "routing_arms": ("word sentence", "ROUTING_ARMS", "lab test 37: adapters trained besides the control (every clip): word (single-word clips only; required) and sentence (sentence clips only)"),
    "routing_kind_epochs": (15, "ROUTING_KIND_EPOCHS", "lab test 37: epochs of the single-word and sentence adapters (15 is stage P's recipe: roughly 75 updates on a speaker's word clips and 30 on his sentence clips, about 105 for the control, which always trains 15)"),
    "lm_label_model": ("qwen25-05b-q4", "LM_LABEL_MODEL", "lab test 38: the one language model that chooses pseudo-labels, a tag from python scripts/asr/lm_score.py list (its weights are not tuned: lm_weight and lm_bonus are fixed)"),
    "lm_label_arms": ("all top margin", "LM_LABEL_ARMS", "lab test 38: which label rules to run, one or more of: all (every sentence the language model chose), top (only the lm_top_share of sentences with the widest margin over the runner-up), margin (only sentences at least lm_min_margin nats ahead); single words keep their plain answer in every arm"),
    "lm_weight": (0.7, "LM_WEIGHT", "lab test 38: weight of the language-model score against the audio score (0.7 is lm_report.py's preset, not chosen on any speaker)"),
    "lm_bonus": (3.0, "LM_BONUS", "lab test 38: bonus in nats per word of an answer (3 is lm_report.py's preset)"),
    "lm_top_share": (0.5, "LM_TOP_SHARE", "lab test 38: arm top keeps this share of the sentence clips seen so far, those with the widest margin (0-1; a margin needs no transcript)"),
    "lm_min_margin": (3.0, "LM_MIN_MARGIN", "lab test 38: arm margin keeps sentences whose chosen answer beats the runner-up by at least this many nats of total score (a value fixed before any run, not tuned)"),
    "wide_extras": ("seeds positions both", "WIDE_EXTRAS", "lab test 39: which wider pools to build besides the baseline (lab 36's pools): seeds (stage P's seeds 18 and 19), positions (the block-12 and block-16 adapters), both (all four extra models in one pool); a pool needs only the extra models its arms use"),
    "wide_systems": ("prototype folds", "WIDE_SYSTEMS", "lab test 39: prototype (stage P's adapter on the test clips) and/or folds (stage P's recipe cross-validated on every labelled clip; includes the test clips)"),
    "wide_extra_views": ("greedy", "WIDE_EXTRA_VIEWS", "lab test 39: what each extra model proposes: greedy, optionally with beamN (for example greedy beam4)"),
    "wide_lm_models": ("qwen25-05b-q4", "WIDE_LM_MODELS", "lab test 39: language models that choose among the wider pools: tags or groups (python scripts/asr/lm_score.py list)"),
}


def _coerce(name: str, value):
    default = VARIABLES[name][0]
    if isinstance(default, bool):
        return str(value).strip().lower() in ("1", "true", "on", "yes", "y")
    if isinstance(default, int):
        return int(float(value))
    if isinstance(default, float):
        return float(value)
    text = str(value).strip()
    if name == "trim_models":
        text = {"1": "on", "true": "on", "yes": "on", "0": "off", "false": "off", "no": "off", "": "auto"}.get(text.lower(), text.lower())
        if text not in ("on", "off", "auto"):
            raise ValueError("trim_models is on, off or auto")
    if name == "wandb" and text not in ("auto", "online", "offline", "off"):
        raise ValueError("wandb is auto, online, offline or off")
    return text


def smi(query, extra=""):
    try:
        done = subprocess.run(["nvidia-smi", "--query-gpu=" + query, "--format=csv,noheader"],
                              capture_output=True, text=True, check=False)
    except OSError:
        return subprocess.CompletedProcess([], 1, "", "")
    return done


class Settings:
    def __init__(self, root: Path, overrides: dict | None = None, env: dict | None = None):
        self.path = root / SETTINGS_FILE
        self.file: dict = {}
        if self.path.is_file():
            try:
                self.file = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self.file = {}
        self.overrides = dict(overrides or {})
        self.env = os.environ if env is None else env

    def source(self, name: str) -> tuple[object, str]:
        if name in self.overrides:
            return _coerce(name, self.overrides[name]), "--set"
        variable = VARIABLES[name][1]
        if variable and variable in self.env and self.env[variable] != "":
            return _coerce(name, self.env[variable]), f"env {variable}"
        if name in self.file:
            return _coerce(name, self.file[name]), "saved"
        return VARIABLES[name][0], "default"

    def __getitem__(self, name: str):
        return self.source(name)[0]

    def save(self, name: str, value) -> None:
        if name not in VARIABLES:
            raise KeyError(f"No variable {name!r}; `lab.py vars` lists them")
        self.file[name] = _coerce(name, value)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.file, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def unset(self, name: str) -> None:
        self.file.pop(name, None)
        if self.path.parent.is_dir():
            self.path.write_text(json.dumps(self.file, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_sets(items: list[str] | None) -> dict:
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--set wants NAME=VALUE, not {item!r}")
        key, value = item.split("=", 1)
        key = key.strip().replace("-", "_")
        if key not in VARIABLES:
            raise SystemExit(f"No variable {key!r}; `python lab.py vars` lists them")
        out[key] = value
    return out


def split_flags(tokens: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    current = None
    for token in tokens:
        if token.startswith("--"):
            current = token
            out[current] = []
        elif current is not None:
            out[current].append(token)
    return out


def merge_flags(base: list[str], extra: list[str]) -> list[str]:
    merged = split_flags(base)
    for flag, values in split_flags(extra).items():
        for alt in (flag, "--no-" + flag[2:], "--" + flag[5:] if flag.startswith("--no-") else None):
            if alt and alt in merged and alt != flag:
                del merged[alt]
        merged[flag] = values
    return [token for flag, values in merged.items() for token in (flag, *values)]


def safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9.+_-]+", "-", text).strip("-")


class Lab:
    def __init__(self, root: Path | str = CODE, settings: Settings | None = None, dry: bool = False,
                 stamp: str | None = None, quiet: bool = False, strict: bool = False):
        self.root = Path(root)
        self.cfg = settings or Settings(self.root)
        self.dry = dry
        self.out_of_disk = False
        self.strict = strict
        self.keep_weights = False
        self.last_output: deque = deque(maxlen=300)
        self.quiet = quiet
        self.stamp = stamp or time.strftime("%m%d-%H%M")
        self.commands: list[list[str]] = []
        self.lines: list[str] = []
        self.log = None
        self.stage = "-"
        self.failed_stages: list[str] = []
        self.logdir = self.root / "logs"
        self.catalog = build_catalog(self)

    def say(self, *parts) -> None:
        text = " ".join(str(p) for p in parts)
        self.emit(text)

    def emit(self, text: str) -> None:
        self.lines.append(text)
        if not self.quiet:
            print(text, flush=True)
        if self.log is not None:
            self.log.write(text + "\n")
            self.log.flush()

    def banner(self, text: str) -> None:
        self.say("")
        self.say(f"######## {text}, {now()}")

    @property
    def sc(self) -> str:
        chosen = self.cfg["cpupy"]
        if chosen:
            return chosen
        venv = self.root / ".venv/bin/python"
        return str(venv) if venv.exists() else sys.executable

    @property
    def gpu_py(self) -> str:
        if self.dry:
            return "python"
        return str(TRAIN_PY)

    def act(self) -> bool:
        if self.dry or TRAIN_PY.is_file():
            return True
        self.say(f"MISSING training environment {TRAIN_PY}: run ./willspeak.sh lab-setup")
        return False

    def child_env(self, extra: dict | None = None) -> dict:
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["WILLSPEAK_WANDB"] = str(self.cfg["wandb"])
        env["WANDB_PROJECT"] = str(self.cfg["wandb_project"])
        if self.cfg["wandb_entity"]:
            env["WANDB_ENTITY"] = str(self.cfg["wandb_entity"])
        env["WILLSPEAK_WANDB_GROUP"] = f"{self.stamp}-{self.stage}"
        env["WILLSPEAK_WANDB_TAGS"] = f"lab,stage-{self.stage}"
        env.update(extra or {})
        return env

    def run(self, cmd: list, env: dict | None = None) -> bool:
        cmd = [str(c) for c in cmd]
        self.commands.append(cmd)
        self.last_output.clear()
        self.say("+ " + " ".join(shlex.quote(c) for c in cmd))
        if self.dry:
            return True
        try:
            proc = subprocess.Popen(cmd, cwd=self.root, env=self.child_env(env), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1)
        except OSError as exc:
            self.say(f"cannot start {cmd[0]}: {exc}")
            return False
        for line in proc.stdout:
            self.emit(line.rstrip("\n"))
            self.last_output.append(line.rstrip("\n"))
        return proc.wait() == 0

    def why_failed(self):
        return lab_doctor.classify_failure("\n".join(self.last_output))

    def remedy(self, why) -> bool:
        if self.dry or not why.retryable:
            return False
        if why.category == "disk":
            self.say("RETRY: freeing the weights of finished runs first")
            self.run([sys.executable, "scripts/asr/lab_space.py", "clean", "--apply", "--lab-models", "--long-models"])
            return True
        if why.category == "network":
            self.say("RETRY: waiting a minute for the network")
            time.sleep(60)
            return True
        return why.category in ("oom", "killed")

    def capture(self, cmd: list) -> str:
        done = subprocess.run([str(c) for c in cmd], cwd=self.root, capture_output=True, text=True, check=False,
                              env=self.child_env())
        return done.stdout

    def exists(self, path: str) -> bool:
        return (self.root / path).exists()

    def aside(self, path: str) -> None:
        n, i = f"{path}-first-run", 2
        while self.exists(n):
            n, i = f"{path}-first-run-{i}", i + 1
        self.say(f"RERUN: moving {path} to {n}")
        if not self.dry:
            (self.root / path).rename(self.root / n)

    def need_free(self, *paths: str) -> bool:
        for p in paths:
            if self.exists(p) and self.cfg["rerun"]:
                self.aside(p)
                if p.endswith(".jsonl"):
                    for q in (p[:-6] + ".provenance.json", p[:-6]):
                        if self.exists(q):
                            self.aside(q)
            elif self.exists(p):
                self.say(f"ALREADY EXISTS {p}, stopping this stage")
                return False
        return True

    def have(self, path: str) -> bool:
        return (self.dry and not self.strict) or self.exists(path)

    def buildable(self, path: str) -> bool:
        if not (self.strict and self.rebuild):
            return False
        if "lab-baselines" in path:
            return True
        if path.startswith(EX) and "/controls" not in path:
            return self.exists("data/public/torgo/TORGO")
        return False

    def need_have(self, *paths: str) -> bool:
        for p in paths:
            if self.buildable(p):
                continue
            if p.endswith("/hypotheses.jsonl"):
                q = p[: -len("/hypotheses.jsonl")]
                if not ((self.dry and not self.strict) or (self.root / p).is_file() or (self.root / q).is_file()):
                    self.say(f"MISSING hypotheses at {q}, stopping this stage")
                    self.say("Copy the matching decode and provenance from reports/finetune-torgo/ on Yack.")
                    return False
            elif not self.have(p):
                self.say(f"MISSING {p}, stopping this stage")
                if p.startswith(("models/parakeet-ft-", "reports/")):
                    self.say("Copy the earlier run's checkpoint/report from Yack with the same path; setup cannot download a fine-tune.")
                elif p.startswith("data/"):
                    self.say("Run ./willspeak.sh lab-setup for public corpus/model assets; preserve exports used by earlier results.")
                return False
        return True

    def has_hyp(self, folder: str) -> bool:
        if self.dry and not self.strict:
            return True
        return (self.root / folder / "hypotheses.jsonl").is_file() or (self.root / folder).is_file()

    def need_hyp(self, *folders: str) -> bool:
        return all(self.need_have(f + "/hypotheses.jsonl") for f in folders)

    @property
    def spk(self) -> list[str]:
        return str(self.cfg["speakers"]).split()

    @property
    def rebuild(self) -> bool:
        return bool(self.cfg["rebuild_missing"])

    def stock_h(self, s: str) -> str:
        return f"{FT}/lab-baselines/stock-{s}" if self.rebuild else f"{FT}/{s}-test-stock-nemo"

    def wb_h(self, s: str) -> str:
        if self.rebuild:
            return f"{FT}/lab-baselines/whole-block-{s}"
        if s == "M02":
            return self.cfg["wb_m02_hyp"] or f"{FT}/M02e-A3-test-nemo"
        return f"{FT}/f2-{s}-bottom8-5epochs-test-nemo"

    def wb_m(self, s: str) -> str:
        if self.rebuild:
            return f"models/lab-baselines/whole-block-{s}/parakeet-ft.nemo"
        if s == "M02":
            return self.cfg["wb_m02_nemo"] or "models/parakeet-ft-M02e-A3/parakeet-ft.nemo"
        return f"models/parakeet-ft-f2-{s}-bottom8-5epochs/parakeet-ft.nemo"

    def f1_h(self, s: str) -> str:
        return f"{FT}/lab-baselines/f1-{s}" if self.rebuild else f"{FT}/f1-loso-{s}-bottom8-on-F2-test-nemo"

    def f1_m(self, s: str) -> str:
        return f"models/lab-baselines/f1-{s}/parakeet-ft.nemo" if self.rebuild else f"models/parakeet-ft-f1-loso-{s}-bottom8/parakeet-ft.nemo"

    def ad_h(self, s: str) -> str:
        if self.rebuild:
            return f"{FT}/lab-baselines/adapter-{s}"
        return f"{FT}/f2-M02-adapter8-pos-test-nemo" if s == "M02" else f"{FT}/f2-{s}-adapter8-test-nemo"

    def adf1_h(self, s: str) -> str:
        if self.rebuild:
            return f"{FT}/lab-baselines/adapter-f1-{s}"
        return f"{FT}/f2-M02-adapter8-on-f1-seed17-test-nemo" if s == "M02" else f"{FT}/f2-{s}-adapter8-on-f1-test-nemo"

    def base8(self, s: str, seed: int) -> str:
        if self.rebuild:
            return f"{FT}/lab-baselines/adapter-{s}" if seed == 17 else f"{FT}/lab-baselines/adapter-{s}-seed{seed}"
        if seed == 17:
            return f"{FT}/f2-M02-adapter8-pos-test-nemo" if s == "M02" else f"{FT}/f2-{s}-adapter8-test-nemo"
        return f"{FT}/f2-M02-adapter8-fixed-seed{seed}-test-nemo" if s == "M02" else f"{FT}/f2-{s}-adapter8-pos-seed{seed}-test-nemo"

    def sysfn(self, label: str, func, spk: list[str] | None = None) -> str:
        return label + "=" + ",".join(f"{s}:{func(s)}" for s in (spk or self.spk))

    def sysfmt(self, label: str, pattern: str, spk: list[str] | None = None) -> str:
        return label + "=" + ",".join(f"{s}:{pattern.replace('@', s)}" for s in (spk or self.spk))

    def trim_on(self) -> bool:
        mode = self.cfg["trim_models"]
        if mode in ("on", "off"):
            return mode == "on"
        return shutil.disk_usage(self.root).free < 100 * 2**30

    def trim_model(self, name: str) -> bool:
        if not self.trim_on():
            return True
        d = f"models/parakeet-ft-{name}"
        if not self.exists(d):
            return True
        self.say(f"TRIM_MODELS: removing the weights of {d} (decodes kept)")
        if not self.dry:
            for part in ("ckpt", "parakeet-ft.nemo", "adapter.pt"):
                target = self.root / d / part
                if target.is_dir():
                    shutil.rmtree(target, ignore_errors=True)
                elif target.exists():
                    target.unlink()
        return True

    def score(self, speaker: str, name: str, split: str = "test") -> bool:
        ok = self.run([self.sc, "scripts/asr/evaluate.py", "--manifest", f"{EX}/{speaker}/{split}.jsonl",
                       "--audio-root", f"{EX}/{speaker}", "--hyp", f"{FT}/{name}-{split}-nemo.jsonl",
                       "--group-by", "kind", "--out", f"{FT}/{name}-{split}-nemo"])
        if not ok:
            self.say(f"scoring failed for {name}")
        else:
            self.track_eval(f"{FT}/{name}-{split}-nemo/results.json", name)
        return ok

    def decode(self, speaker: str, name: str, model_flags: list[str], split: str = "test") -> bool:
        cmd = [self.gpu_py, "scripts/asr/nemo_hyp.py", *model_flags, "--manifest", f"{EX}/{speaker}/{split}.jsonl",
               "--audio-root", f"{EX}/{speaker}", "--out", f"{FT}/{name}-{split}-nemo.jsonl"]
        ok = self.run(cmd)
        if not ok and not self.dry:
            why = self.why_failed()
            if why.category in ("oom", "killed"):
                self.say(f"decode of {name} ran out of memory; trying again with batch 1")
                ok = self.run(cmd + ["--batch", "1"])
            elif self.remedy(why):
                ok = self.run(cmd)
        return ok and self.score(speaker, name, split)

    def track_eval(self, results: str, name: str) -> None:
        if self.dry or self.cfg["wandb"] == "off" or not self.exists(results):
            return
        try:
            import wandb_log
            wandb_log.log_eval(name, self.root / results, group=f"{self.stamp}-{self.stage}", tags=["lab", f"stage-{self.stage}"],
                               run_id=hashlib.sha1(f"eval:{results}".encode()).hexdigest()[:32])
        except Exception as exc:  # noqa: BLE001
            self.say(f"[wandb] skipped: {exc}")

    def track_table(self, table: str) -> None:
        if self.dry or self.cfg["wandb"] == "off" or not self.exists(table):
            return
        try:
            import wandb_backfill
            import wandb_log
            data = json.loads((self.root / table).read_text(encoding="utf-8"))
            handle = wandb_log.start(Path(table).parent.name + "-" + Path(table).stem, {"source": table}, job_type="lab-table",
                                     group=f"{self.stamp}-{self.stage}", tags=["lab", f"stage-{self.stage}"],
                                     run_id=hashlib.sha1(f"table:{table}:{self.stamp}".encode()).hexdigest()[:32])
            wandb_log.summary(handle, wandb_backfill.flat_numbers(data))
            wandb_log.log_file(handle, self.root / table)
            wandb_log.finish(handle)
        except Exception as exc:  # noqa: BLE001
            self.say(f"[wandb] table skipped: {exc}")

    def pooled(self, *args: str) -> bool:
        ok = self.run([self.sc, "scripts/asr/pool_table.py", *args])
        if not ok:
            self.say("pool_table failed (a decode above failed, or a path is missing)")
        else:
            out = args[args.index("--out") + 1] if "--out" in args else None
            if out:
                self.track_table(out)
        return ok

    def arm_table(self, folder: str, arms: list[str], name: str = "table.json") -> bool:
        args = ["--system", self.sysfn("stock", self.stock_h), "--system", self.sysfn("whole-block", self.wb_h)]
        for a in arms:
            label, pattern = a.split("=", 1)
            args += ["--system", self.sysfmt(label, pattern)]
        return self.pooled(*args, "--baseline", "stock", "--out", f"{folder}/{name}")

    def candidates(self, family: str | None = None) -> list[tuple[str, str, str]]:
        cmd = [self.sc, "scripts/asr/candidates.py", "--lines"] + (["--family", family] if family else [])
        rows = []
        for line in self.capture(cmd).splitlines():
            parts = line.split("|")
            if len(parts) >= 3:
                rows.append((parts[0], parts[1], parts[2]))
        return rows

    def ensure_words(self) -> bool:
        if self.have("data/emb/torgo-words.npz"):
            return True
        self.say("the WavLM word embeddings are missing; making them (CPU, a few minutes)")
        return self.run([self.sc, "scripts/matcher/embed_manifest.py", "--manifest", "data/public/torgo/manifests/words.csv",
                         "--out", "data/emb/torgo-words.npz"])

    BAT = f"{EX}/silence-battery"

    def ensure_battery(self) -> bool:
        if not self.dry and self.exists(f"{self.BAT}/manifest.jsonl"):
            return True
        return self.run([self.sc, "scripts/asr/battery.py", "build", "--out", self.BAT, "--seed", "2000",
                         "--edge-from", f"{EX}/M02/test.jsonl", "--edge-audio-root", f"{EX}/M02"])

    def battery_decode(self, label: str, model_flags: list[str]) -> bool:
        out = f"{FT}/lab-13-battery-{label}.jsonl"
        if not self.dry and self.exists(out):
            self.say(f"reusing {out}")
            return True
        return self.run([self.gpu_py, "scripts/asr/nemo_hyp.py", *model_flags, "--manifest", f"{self.BAT}/manifest.jsonl",
                         "--audio-root", self.BAT, "--out", out])

    def stock_screen(self, speaker: str, tag: str, family: str, kind: str, name: str) -> bool:
        out = f"{FT}/{speaker}-test-stock-{tag}-{family}"
        if not self.dry and self.exists(f"{out}/hypotheses.jsonl"):
            self.say(f"reusing {out}")
            return True
        if family == "nemo":
            ok = self.run([self.gpu_py, "scripts/asr/nemo_hyp.py", "--pretrained", name, "--manifest", f"{EX}/{speaker}/test.jsonl",
                           "--audio-root", f"{EX}/{speaker}", "--out", out + ".jsonl"])
        else:
            hp = self.cfg["qwenpy"] if kind == "qwen3-asr" else self.cfg["hfpy"]
            if not self.dry and not Path(str(hp)).exists():
                self.say(f"MISSING {hp}: run ./willspeak.sh lab-setup (or set hfpy / qwenpy)")
                return False
            ok = self.run([hp, "scripts/asr/hf_hyp.py", "--kind", kind, "--model", name, "--manifest", f"{EX}/{speaker}/test.jsonl",
                           "--audio-root", f"{EX}/{speaker}", "--out", out + ".jsonl"])
        if not ok:
            return False
        return self.run([self.sc, "scripts/asr/evaluate.py", "--manifest", f"{EX}/{speaker}/test.jsonl", "--audio-root", f"{EX}/{speaker}",
                         "--hyp", out + ".jsonl", "--group-by", "kind", "--out", out])

    def newest_train_log(self) -> Path | None:
        logs = sorted(self.logdir.glob("train_parakeet-*.log"), key=lambda p: p.stat().st_mtime)
        return logs[-1] if logs else None

    KEY_LINES = re.compile(r"Training audio|parameters exact|Updates +:|kept epochs|averaged|Released|Optimizer updates|Elapsed|silence-aware|saved .*adapter")

    def show_train_log(self) -> None:
        if self.dry:
            return
        newest = self.newest_train_log()
        if newest:
            self.say(f"-- log lines from {newest}")
            for line in newest.read_text(encoding="utf-8", errors="replace").splitlines():
                if self.KEY_LINES.search(line):
                    self.say(line)

    RECIPES = {
        "f2": ["--epochs", "5", "--select", "last", "--no-keep-stock", "--warmup-steps", "10"],
        "f1": ["--keep-stock"],
        "wb": ["--train-parts", "all", "--epochs", "5", "--select", "last", "--no-keep-stock", "--warmup-steps", "10"],
        "ad8": ["--adapter-after-blocks", "8", "--lr", "3e-4", "--epochs", "15", "--select", "last", "--no-keep-stock", "--warmup-steps", "10"],
    }

    def train(self, export: str, name: str, recipe: list[str], extra: list[str] | None = None, batch: int = 2, accum: int = 8,
              seed: int | None = None, parts: str | None = None) -> bool:
        flags = ["--export", export, "--out", f"models/parakeet-ft-{name}"]
        if parts is not None:
            flags += ["--train-parts", parts]
        common = ["--batch", str(batch), "--grad-accum", str(accum), "--freeze-batchnorm", "--normalize-targets", "--dev-scoring", "evaluate"]
        if seed is not None:
            common += ["--seed", str(seed)]
        flags = merge_flags(flags + common + recipe, shlex.split(str(self.cfg["xflags"])))
        flags = merge_flags(flags, extra or [])
        ok = self.run([self.gpu_py, "scripts/asr/train.py", *flags])
        if ok:
            self.show_train_log()
        return ok

    def train_retrying(self, export: str, name: str, recipe: list[str], extra: list[str] | None = None, seed: int | None = None) -> bool:
        self.out_of_disk = False
        if self.train(export, name, recipe, extra, seed=seed):
            return True
        if self.dry:
            return False
        why = self.why_failed()
        self.out_of_disk = why.category == "disk"
        if why.category in ("disk", "network") and self.remedy(why) and self.need_free(f"models/parakeet-ft-{name}"):
            if self.train(export, name, recipe, extra, seed=seed):
                self.out_of_disk = False
                return True
            self.out_of_disk = self.why_failed().category == "disk"
        return False

    def fit(self, export: str, name: str, recipe: str, extra: list[str] | None = None, parts: str = "all", seed: int | None = None) -> str | None:
        flags = self.RECIPES[recipe]
        if self.train(export, name, flags, extra, parts=parts, seed=seed):
            return name
        if not self.dry:
            why = self.why_failed()
            if why.category in ("exists", "changed", "interrupted"):
                self.say(f"FAILED {name}: {why.category}. {why.remedy}. Not retrying with a smaller batch.")
                return None
            if why.category in ("disk", "network") and self.remedy(why):
                if self.need_free(f"models/parakeet-ft-{name}") and self.train(export, name, flags, extra, parts=parts, seed=seed):
                    return name
        self.say(f"FAILED {name} with batch 2; retrying with batch 1 x 16 accumulation as {name}-b1")
        if not self.need_free(f"models/parakeet-ft-{name}-b1"):
            return None
        if self.train(export, f"{name}-b1", flags, extra, batch=1, accum=16, parts=parts, seed=seed):
            return f"{name}-b1"
        return None

    def adapter_flags(self, name: str, base: str | None) -> list[str]:
        path = f"models/parakeet-ft-{name}/adapter.pt"
        if base is None:
            return ["--stock", "--adapter", path]
        if base.endswith(".nemo") or base.startswith("models/"):
            return ["--nemo", base, "--adapter", path]
        return ["--pretrained", base, "--adapter", path]

    BASELINE_NEEDS = {
        "1": ("stock", "whole-block"), "2": ("stock", "whole-block"),
        "3": ("stock", "whole-block", "f1", "adapter", "adapter-f1"),
        "4": ("stock",), "5": ("whole-block", "f1"), "6": ("stock", "whole-block"), "7": ("stock",),
        "8": ("stock", "whole-block"), "8b": ("stock", "whole-block"), "9": ("stock", "whole-block"),
        "10": ("stock", "whole-block"), "11": ("stock", "whole-block", "f1"), "12": ("stock", "whole-block"),
        "13": ("stock", "whole-block", "f1"), "14": ("stock", "whole-block", "adapter"), "15": (),
        "16": ("stock",), "17": ("stock", "whole-block", "adapter"), "18": ("stock", "f1", "adapter-f1"),
        "19": ("stock", "whole-block", "f1", "adapter-f1"),
        "20": ("stock", "f1", "adapter-f1"), "21": ("stock", "f1", "adapter-f1"), "22": ("stock", "f1", "adapter-f1"),
        "23": ("stock", "f1", "adapter", "adapter-f1"),
        "24": ("stock", "f1", "adapter-f1"), "25": ("stock", "f1"), "26": ("stock", "f1", "adapter-f1"), "27": ("stock", "f1", "adapter-f1"),
        "28": ("stock",), "29": ("stock", "whole-block", "f1", "adapter", "adapter-f1"), "30": ("stock",),
        "31": ("stock", "adapter"), "32": ("stock", "adapter"), "33": ("stock", "adapter"), "34": ("stock", "adapter"),
        "35": ("stock", "whole-block"),
        "36": (),
        "37": ("stock",),
        "38": ("stock", "f1"),
        "39": (),
        "L": ("stock",), "T": ("stock",), "E": ("stock",), "S": ("stock", "f1"),
        "C": ("stock",), "A": ("stock", "adapter"), "N": ("stock", "adapter"),
        "F": ("stock", "adapter"), "B": ("stock", "adapter"), "U": ("stock",), "P": ("stock",), "G": ("stock",), "G2": ("stock",), "R": ("stock",), "K": ("stock",), "W": ("stock", "whole-block", "f1", "adapter", "adapter-f1"),
    }

    def prepare_baseline(self, kind: str, speaker: str, seed: int = 17) -> bool:
        export = f"{EX}/{speaker}"
        suffix = "" if seed == 17 else f"-seed{seed}"
        report = f"{FT}/lab-baselines/{kind}-{speaker}{suffix}"
        weights = f"models/lab-baselines/{kind}-{speaker}{suffix}/" + ("adapter.pt" if kind.startswith("adapter") else "parakeet-ft.nemo")
        if self.exists(report + "/hypotheses.jsonl") and self.exists(report + "/results.json"):
            if kind == "stock" or not (kind == "f1" or self.keep_weights) or self.exists(weights):
                return True
        flags = ["--stock"]
        if kind != "stock":
            file = "adapter.pt" if kind.startswith("adapter") else "parakeet-ft.nemo"
            model = f"models/lab-baselines/{kind}-{speaker}{suffix}/{file}"
            flags = ["--nemo", model]
            train_export = export
            common = ["--batch", "2", "--grad-accum", "8", "--freeze-batchnorm", "--normalize-targets", "--dev-scoring", "evaluate", "--seed", str(seed)]
            if kind == "f1":
                train_export = f"{EX}/loso-{speaker}"
                recipe = ["--train-parts", "all", "--keep-stock"]
            elif kind == "whole-block":
                recipe = ["--train-parts", "all", "--warmup-steps", "10"]
                recipe += ["--keep-stock"] if speaker == "M02" else ["--epochs", "5", "--select", "last", "--no-keep-stock"]
            else:
                flags = ["--stock", "--adapter", model]
                recipe = ["--adapter-after-blocks", "8", "--lr", "3e-4", "--warmup-steps", "10"]
                if kind == "adapter-f1":
                    if not self.prepare_baseline("f1", speaker):
                        return False
                    base = f"models/lab-baselines/f1-{speaker}/parakeet-ft.nemo"
                    recipe += ["--base", base]
                    flags = ["--nemo", base, "--adapter", model]
                recipe += (["--keep-stock"] if kind == "adapter-f1" and speaker == "M02"
                           else ["--epochs", "15", "--select", "last", "--no-keep-stock"])
            folder = str(Path(model).parent)
            if not self.exists(model):
                if self.exists(folder):
                    self.aside(folder)
                if not self.run([self.gpu_py, "scripts/asr/train.py", *merge_flags(
                        ["--export", train_export, "--out", folder, *common, *recipe],
                        shlex.split(str(self.cfg["xflags"])))]):
                    return False
        hypotheses = report + ".jsonl"
        if not (self.exists(f"{report}/hypotheses.jsonl") and self.exists(f"{report}/results.json")):
            if not self.exists(hypotheses) and not self.run([self.gpu_py, "scripts/asr/nemo_hyp.py", *flags, "--manifest", export + "/test.jsonl",
                                                             "--audio-root", export, "--out", hypotheses]):
                return False
            if not self.run([self.sc, "scripts/asr/evaluate.py", "--manifest", export + "/test.jsonl", "--audio-root", export,
                             "--hyp", hypotheses, "--group-by", "kind", "--out", report]):
                return False
        if not self.dry and not self.cfg["trim_models"] == "off" and kind != "stock":
            shutil.rmtree(self.root / folder / "ckpt", ignore_errors=True)
            if self.stage in AUTO_REBUILD and kind != "f1" and not self.keep_weights:
                (self.root / model).unlink(missing_ok=True)
        return True

    def clean_baseline_weights(self):
        if self.dry or self.cfg["trim_models"] == "off":
            return
        for folder in (self.root / "models/lab-baselines").glob("*"):
            if folder.is_dir():
                shutil.rmtree(folder / "ckpt", ignore_errors=True)
                for name in ("parakeet-ft.nemo", "adapter.pt"):
                    (folder / name).unlink(missing_ok=True)

    def prepare_baselines(self, stage: str) -> bool:
        self.say("Rebuilding missing baselines with current code, seed 17. These are new runs, not original Yack results.")
        speakers = ("M02",) if stage in ("5", "F") else SEVERE if stage in ("A", "N", "W", "B") else self.spk
        required = [f"{EX}/{s}/{split}.jsonl" for s in speakers for split in ("train", "dev", "test")]
        if not all(self.exists(path) for path in required):
            if not self.need_have("data/public/torgo/TORGO"):
                return False
            if not self.run([self.sc, "scripts/corpus/torgo_manifest.py", "--export", "--missing-only", "--speakers", *speakers]):
                return False
        if any(k in ("f1", "adapter-f1") for k in self.BASELINE_NEEDS.get(stage, ())):
            if not all(self.exists(f"{EX}/loso-{s}/train.jsonl") and self.exists(f"{EX}/loso-{s}/dev.jsonl") for s in speakers):
                if not self.run([self.sc, "scripts/corpus/torgo_manifest.py", "--export", "--missing-only", "--speakers", *speakers]):
                    return False
        for kind in self.BASELINE_NEEDS.get(stage, ()):
            for speaker in speakers:
                seeds = (17, 18, 19) if kind == "adapter" and stage in ("A", "N", "F", *ACOUSTIC_QUEUE) else (17,)
                for seed in seeds:
                    if not self.prepare_baseline(kind, speaker, seed):
                        return False
        return True


def summary_lines(lab: Lab, failed: list[str]) -> None:
    lab.say("")
    lab.say("######## summary")
    lab.say("failed:" + (" " + " ".join(failed) if failed else " none"))


def stage_1(lab: Lab) -> bool:
    n = "reports/lab-01-weight-interpolation"
    lab.say(f"######## 1: weight interpolation, stock to whole-block, started {now()}")
    for s in lab.spk:
        if not lab.need_have(lab.wb_m(s), lab.wb_h(s) + "/hypotheses.jsonl", lab.stock_h(s) + "/hypotheses.jsonl"):
            return False
        for w in (25, 50, 75):
            if not lab.need_free(f"{FT}/lab-01-w{w}-{s}-test-nemo.jsonl"):
                return False
    if not lab.need_free(f"{n}/table.json") or not lab.act():
        return False
    failed = []
    for s in lab.spk:
        for w in (25, 50, 75):
            lab.banner(f"{s}, {w}% of the way from stock to his whole-block model")
            if not lab.decode(s, f"lab-01-w{w}-{s}", ["--stock", "--mix", f"{lab.wb_m(s)}=0.{w}"]):
                lab.say(f"FAILED {s} {w}")
                failed.append(f"{s}:{w}")
    summary_lines(lab, failed)
    lab.pooled("--system", lab.sysfn("stock", lab.stock_h), "--system", lab.sysfmt("w25", f"{FT}/lab-01-w25-@-test-nemo.jsonl"),
               "--system", lab.sysfmt("w50", f"{FT}/lab-01-w50-@-test-nemo.jsonl"),
               "--system", lab.sysfmt("w75", f"{FT}/lab-01-w75-@-test-nemo.jsonl"),
               "--system", lab.sysfn("whole-block", lab.wb_h), "--baseline", "stock", "--out", f"{n}/table.json")
    lab.say(f"1 finished {now()}")
    return True


def stage_2(lab: Lab) -> bool:
    n = "reports/lab-02-weight-merging"
    lab.say(f"######## 2: merging weight changes across speakers, started {now()}")
    for s in lab.spk:
        if not lab.need_have(lab.wb_m(s), lab.wb_h(s) + "/hypotheses.jsonl", lab.stock_h(s) + "/hypotheses.jsonl"):
            return False
        if not lab.need_free(f"{FT}/lab-02-others-{s}-test-nemo.jsonl", f"{FT}/lab-02-all-{s}-test-nemo.jsonl"):
            return False
    if not lab.need_free(f"{n}/geometry.json", f"{n}/table.json") or not lab.act():
        return False
    failed = []
    lab.say("")
    lab.say("######## how far each model moved, and in what direction relative to the others")
    if not lab.run([lab.gpu_py, "scripts/asr/weights.py", "--base", "auto", "--tuned", *[f"{s}={lab.wb_m(s)}" for s in SEVERE],
                    "--out", f"{n}/geometry.json"]):
        lab.say("FAILED geometry")
        failed.append("geometry")
    for s in lab.spk:
        others = [o for o in lab.spk if o != s]
        mix = [x for o in others for x in ("--mix", f"{lab.wb_m(o)}=0.5")]
        lab.banner(f"{s}: the average of {' '.join(others)}'s changes, nothing from {s}")
        if not lab.decode(s, f"lab-02-others-{s}", ["--stock", *mix]):
            lab.say(f"FAILED others {s}")
            failed.append(f"others:{s}")
        mix = [x for o in lab.spk for x in ("--mix", f"{lab.wb_m(o)}=0.3333333")]
        lab.banner(f"{s}: the average of all three speakers' changes")
        if not lab.decode(s, f"lab-02-all-{s}", ["--stock", *mix]):
            lab.say(f"FAILED all {s}")
            failed.append(f"all:{s}")
    summary_lines(lab, failed)
    lab.pooled("--system", lab.sysfn("stock", lab.stock_h), "--system", lab.sysfmt("others", f"{FT}/lab-02-others-@-test-nemo.jsonl"),
               "--system", lab.sysfmt("all-three", f"{FT}/lab-02-all-@-test-nemo.jsonl"), "--system", lab.sysfn("whole-block", lab.wb_h),
               "--baseline", "stock", "--out", f"{n}/table.json")
    lab.say(f"2 finished {now()}")
    return True


def stage_3(lab: Lab) -> bool:
    n = "reports/lab-03-hypothesis-bounds"
    lab.say(f"######## 3: bounds on combining the systems that exist, started {now()}")
    if not lab.need_free(f"{n}/bounds"):
        return False
    for s in lab.spk:
        if not lab.need_hyp(lab.stock_h(s), lab.wb_h(s), lab.f1_h(s), lab.ad_h(s), lab.adf1_h(s)):
            return False
    systems = ["--system", lab.sysfn("stock", lab.stock_h), "--system", lab.sysfn("F-1", lab.f1_h),
               "--system", lab.sysfn("adapter-on-stock", lab.ad_h), "--system", lab.sysfn("adapter-on-F-1", lab.adf1_h),
               "--system", lab.sysfn("whole-block", lab.wb_h)]
    for v in ("stock", "whole-block", "adapter-on-stock"):
        p = f"reports/lab-14-route-scoring/route-{v}/routed"
        if all(lab.exists(f"{p}/{s}/hypotheses.jsonl") for s in lab.spk):
            systems += ["--system", lab.sysfmt(f"routed-{v}", p + "/@")]
        else:
            lab.say(f"no routed hypotheses for {v} (stage 14 has not run); leaving that system out")
    if not lab.run([lab.sc, "scripts/asr/hyp_bounds.py", *systems, "--out", f"{n}/bounds"]):
        lab.say("FAILED hyp_bounds")
    lab.say("")
    lab.say("######## summary")
    lab.say(f"3 finished {now()}")
    return True


def stage_4(lab: Lab) -> bool:
    n = "reports/lab-04-stock-model-screen"
    lab.say(f"######## 4: stock severe-speech screen, started {now()}")
    if not lab.need_free(f"{n}/table.json"):
        return False
    for s in lab.spk:
        if not lab.need_have(f"{EX}/{s}/test.jsonl", lab.stock_h(s) + "/hypotheses.jsonl"):
            return False
    failed: list[str] = []
    systems = ["--system", lab.sysfn("v2", lab.stock_h)]
    if not lab.act():
        return False
    for fam in ("nemo", "hf"):
        if fam == "hf" and not lab.dry and not Path(str(lab.cfg["hfpy"])).exists():
            lab.say(f"MISSING Hugging Face environment at {lab.cfg['hfpy']}; run ./willspeak.sh lab-setup; skipping the non-NeMo models")
            continue
        for tag, kind, name in lab.candidates(fam):
            if tag == "v2":
                continue
            lab.banner(f"{tag} ({name}): stock decode on the three test sets")
            ok = True
            for s in lab.spk:
                if not lab.stock_screen(s, tag, fam, kind, name):
                    ok = False
                    lab.say(f"FAILED {tag} {s}")
                    failed.append(f"{tag}:{s}")
            if ok:
                systems += ["--system", lab.sysfmt(tag, f"{FT}/@-test-stock-{tag}-{fam}")]
    summary_lines(lab, failed)
    lab.pooled(*systems, "--baseline", "v2", "--out", f"{n}/table.json")
    lab.say(f"4 finished {now()}")
    return True


def stage_5(lab: Lab) -> bool:
    n = "reports/lab-05-layer-screen"
    e = "data/emb"
    words_csv = "data/public/torgo/manifests/words.csv"
    wavlm = "data/emb/torgo-words.npz"
    lab.say(f"######## 5: encoder layer screen, started {now()}")
    if not lab.need_have(words_csv, lab.wb_m("M02"), lab.f1_m("M02")):
        return False
    if not lab.need_free(f"{n}/results.json", f"{n}/results-last-session.json"):
        return False
    if not lab.cfg["rerun"] and not lab.need_free(f"{e}/lab-05-v2-stock-layers.npz", f"{e}/lab-05-v2-whole-block-m02-layers.npz",
                                                  f"{e}/lab-05-v2-f1-loso-m02-layers.npz"):
        return False
    if not lab.ensure_words() or not lab.act():
        return False
    failed: list[str] = []
    for label, flags in (("v2-stock", ["--stock"]), ("v2-whole-block-m02", ["--nemo", lab.wb_m("M02")]), ("v2-f1-loso-m02", ["--nemo", lab.f1_m("M02")])):
        out = f"{e}/lab-05-{label}-layers.npz"
        if lab.cfg["rerun"] and not lab.dry and lab.exists(out):
            probe = "import numpy as np, sys; z=np.load(sys.argv[1]); sys.exit(not np.isfinite(z['raw']).all())"
            if subprocess.run([lab.sc, "-c", probe, str(lab.root / out)], check=False).returncode == 0:
                lab.say(f"keeping {out} from the first run")
                continue
            lab.say(f"non-finite or unreadable embeddings: rebuilding {label}")
            lab.aside(out)
        lab.banner(f"embedding every word clip with every layer of {label}")
        if not lab.run([lab.gpu_py, "scripts/matcher/embed_layers.py", *flags, "--manifest", words_csv, "--out", out]):
            lab.say(f"FAILED {label}")
            failed.append(label)
    emb = ["--emb", f"wavlm={wavlm}"]
    for label in ("v2-stock", "v2-whole-block-m02", "v2-f1-loso-m02"):
        if lab.have(f"{e}/lab-05-{label}-layers.npz"):
            emb += ["--emb", f"{label}={e}/lab-05-{label}-layers.npz"]
    summary_lines(lab, failed)
    lab.say("-- every layer, enrolling on one session and testing on the others (the session mean is the whole test session's)")
    if not lab.run([lab.sc, "scripts/matcher/layer_screen.py", *emb, "--baseline", "wavlm", "--layers", str(lab.cfg["layers"]), "--causal",
                    "--out", f"{n}/results.json"]):
        lab.say("FAILED layer_screen")
        failed.append("screen")
    lab.say("-- last session only (the F-2 split), the fair setting for the fine-tuned models")
    if not lab.run([lab.sc, "scripts/matcher/layer_screen.py", *emb, "--baseline", "wavlm", "--layers", str(lab.cfg["layers"]), "--last-session",
                    "--out", f"{n}/results-last-session.json"]):
        lab.say("FAILED layer_screen last session")
        failed.append("last-session")
    lab.say(f"5 finished {now()}")
    return not failed


def stage_6(lab: Lab) -> bool:
    n = "reports/lab-06-reference-score"
    lab.say(f"######## 6: reference score and silence margin, started {now()}")
    for s in lab.spk:
        if not lab.need_have(lab.wb_m(s), lab.wb_h(s) + "/hypotheses.jsonl", lab.stock_h(s) + "/hypotheses.jsonl"):
            return False
        if not lab.need_free(f"{n}/{s}-stock.json", f"{n}/{s}-whole-block.json"):
            return False
    if not lab.act():
        return False
    failed = []
    for s in lab.spk:
        common = ["--manifest", f"{EX}/{s}/test.jsonl", "--audio-root", f"{EX}/{s}"]
        lab.banner(f"{s}, stock")
        if not lab.run([lab.gpu_py, "scripts/asr/reference_score.py", "--stock", *common, "--hyps", lab.stock_h(s), "--out", f"{n}/{s}-stock.json"]):
            lab.say(f"FAILED {s} stock")
            failed.append(f"{s}:stock")
        lab.banner(f"{s}, whole-block")
        if not lab.run([lab.gpu_py, "scripts/asr/reference_score.py", "--nemo", lab.wb_m(s), *common, "--hyps", lab.wb_h(s),
                        "--out", f"{n}/{s}-whole-block.json"]):
            lab.say(f"FAILED {s} whole-block")
            failed.append(f"{s}:whole-block")
    summary_lines(lab, failed)
    lab.say(f"6 finished {now()}")
    return True


def stage_7(lab: Lab) -> bool:
    n = "reports/lab-07-stock-vs-fine-tuned"
    tags = str(lab.cfg["tune_tags"]).split()
    spk = ["M02", "M04"]
    lab.say(f"######## 7: stock order against fine-tuned order, models: {' '.join(tags)}, speakers: {' '.join(spk)}, started {now()}")
    if not lab.need_free(f"{n}/rank.json"):
        return False
    for s in spk:
        if not lab.need_have(f"{EX}/{s}/train.jsonl", f"{EX}/{s}/dev.jsonl", f"{EX}/{s}/test.jsonl"):
            return False
        for t in tags:
            if not lab.need_free(f"models/parakeet-ft-lab-07-{t}-{s}", f"{FT}/lab-07-{t}-{s}-test-nemo.jsonl"):
                return False
    if not lab.act():
        return False
    known = {tag: (kind, name) for tag, kind, name in lab.candidates()}
    failed: list[str] = []
    good: list[str] = []
    for t in tags:
        if t not in known:
            lab.say(f"unknown candidate tag {t}")
            failed.append(t)
            continue
        kind, name = known[t]
        lab.banner(f"{t} ({name})")
        ok = True
        if t != "v2":
            for s in spk:
                if not lab.stock_screen(s, t, "nemo", kind, name):
                    ok = False
                    lab.say(f"FAILED stock {t} {s}")
                    failed.append(f"stock:{t}:{s}")
        if not ok:
            lab.say(f"skipping the training runs for {t}")
            continue
        base = [] if t == "v2" else ["--base", name]
        extra = base + (["--offload-optimizer"] if t == "1p1b" else [])
        for s in spk:
            lab.banner(f"{s}, {t}, F-2 whole-block recipe")
            fitted = lab.fit(f"{EX}/{s}", f"lab-07-{t}-{s}", "f2", extra)
            if fitted and lab.decode(s, f"lab-07-{t}-{s}", ["--nemo", f"models/parakeet-ft-{fitted}/parakeet-ft.nemo"]):
                lab.trim_model(fitted)
            else:
                ok = False
                lab.say(f"FAILED {t} {s}")
                failed.append(f"{t}:{s}")
        if ok:
            good.append(t)
    summary_lines(lab, failed)
    args: list[str] = []
    for t in good:
        args += ["--stock", lab.sysfn(t, lab.stock_h, spk) if t == "v2" else lab.sysfmt(t, f"{FT}/@-test-stock-{t}-nemo", spk)]
        args += ["--tuned", lab.sysfmt(t, f"{FT}/lab-07-{t}-@-test-nemo.jsonl", spk)]
    if not lab.run([lab.sc, "scripts/asr/rank_check.py", "--speakers", "M02,M04", *args, "--out", f"{n}/rank.json"]):
        lab.say("FAILED rank_check (it needs at least two models that finished)")
    lab.say(f"7 finished {now()}")
    return True


def ff_arm(lab: Lab, label: str, parts: str, failed: list[str], arms: list[str]) -> None:
    ok = True
    for s in lab.spk:
        lab.banner(f"{s}, only {parts} in blocks 0-7, F-2 recipe")
        fitted = lab.fit(f"{EX}/{s}", f"lab-08-{label}-{s}", "f2", parts=parts)
        if fitted and lab.decode(s, f"lab-08-{label}-{s}", ["--nemo", f"models/parakeet-ft-{fitted}/parakeet-ft.nemo"]):
            lab.trim_model(fitted)
        else:
            ok = False
            lab.say(f"FAILED {label} {s}")
            failed.append(f"{label}:{s}")
    if ok:
        arms.append(f"{label}={FT}/lab-08-{label}-@-test-nemo.jsonl")


def stage_8(lab: Lab) -> bool:
    n = "reports/lab-08-feed-forward-only"
    lab.say(f"######## 8: feed-forward modules only, blocks 0-7, started {now()}")
    for s in lab.spk:
        if not lab.need_have(f"{EX}/{s}/train.jsonl", lab.wb_h(s) + "/hypotheses.jsonl", lab.stock_h(s) + "/hypotheses.jsonl"):
            return False
        if not lab.need_free(f"models/parakeet-ft-lab-08-ff-{s}", f"{FT}/lab-08-ff-{s}-test-nemo.jsonl"):
            return False
    if not lab.need_free(f"{n}/table.json") or not lab.act():
        return False
    failed: list[str] = []
    arms: list[str] = []
    ff_arm(lab, "ff", "feed_forward1,feed_forward2", failed, arms)
    summary_lines(lab, failed)
    if not lab.arm_table(n, arms):
        failed.append("table")
    lab.say(f"8 finished {now()}")
    return not failed


def stage_8b(lab: Lab) -> bool:
    n = "reports/lab-08-feed-forward-only"
    lab.say(f"######## 8b: bisect, feed_forward1 only and feed_forward2 only, started {now()}")
    for s in lab.spk:
        if not lab.need_have(f"{EX}/{s}/train.jsonl", lab.wb_h(s) + "/hypotheses.jsonl", lab.stock_h(s) + "/hypotheses.jsonl",
                             f"{FT}/lab-08-ff-{s}-test-nemo.jsonl"):
            return False
        if not lab.need_free(f"models/parakeet-ft-lab-08-ff1-{s}", f"models/parakeet-ft-lab-08-ff2-{s}",
                             f"{FT}/lab-08-ff1-{s}-test-nemo.jsonl", f"{FT}/lab-08-ff2-{s}-test-nemo.jsonl"):
            return False
    if not lab.need_free(f"{n}/table-bisect.json") or not lab.act():
        return False
    failed: list[str] = []
    arms = [f"ff={FT}/lab-08-ff-@-test-nemo.jsonl"]
    ff_arm(lab, "ff1", "feed_forward1", failed, arms)
    ff_arm(lab, "ff2", "feed_forward2", failed, arms)
    summary_lines(lab, failed)
    if not lab.arm_table(n, arms, "table-bisect.json"):
        failed.append("table")
    lab.say(f"8b finished {now()}")
    return not failed


def stage_9(lab: Lab) -> bool:
    n = "reports/lab-09-depth-sweep"
    arms9 = str(lab.cfg["arms9"]).split()
    lab.say(f"######## 9: depth sweep, arms: {' '.join(arms9)}, started {now()}")
    for s in lab.spk:
        if not lab.need_have(f"{EX}/{s}/train.jsonl", lab.wb_h(s) + "/hypotheses.jsonl", lab.stock_h(s) + "/hypotheses.jsonl"):
            return False
    for a in arms9:
        for s in lab.spk:
            if not lab.need_free(f"models/parakeet-ft-lab-09-{a.split(':')[0]}-{s}", f"{FT}/lab-09-{a.split(':')[0]}-{s}-test-nemo.jsonl"):
                return False
    if not lab.need_free(f"{n}/table.json") or not lab.act():
        return False
    failed: list[str] = []
    arms: list[str] = []
    for a in arms9:
        label, first, blocks = a.split(":")
        extra = ["--train-blocks", blocks, "--train-first-block", first] + (["--offload-optimizer"] if int(blocks) >= 12 else [])
        ok = True
        for s in lab.spk:
            lab.banner(f"{s}, blocks {first} to {int(first) + int(blocks) - 1} ({label}), F-2 recipe")
            fitted = lab.fit(f"{EX}/{s}", f"lab-09-{label}-{s}", "f2", extra)
            if fitted and lab.decode(s, f"lab-09-{label}-{s}", ["--nemo", f"models/parakeet-ft-{fitted}/parakeet-ft.nemo"]):
                lab.trim_model(fitted)
            else:
                ok = False
                lab.say(f"FAILED {label} {s}")
                failed.append(f"{label}:{s}")
        if ok:
            arms.append(f"{label}={FT}/lab-09-{label}-@-test-nemo.jsonl")
    summary_lines(lab, failed)
    lab.arm_table(n, arms)
    lab.say(f"9 finished {now()}")
    return True


def stage_10(lab: Lab) -> bool:
    n = "reports/lab-10-silence-margin"
    margin, kper = lab.cfg["margin"], lab.cfg["kper"]
    lab.say(f"######## 10: margin {margin} nats and {kper} non-speech clips per kind, started {now()}")
    for s in lab.spk:
        if not lab.need_have(f"{EX}/{s}/train.jsonl", lab.wb_h(s) + "/hypotheses.jsonl", lab.stock_h(s) + "/hypotheses.jsonl"):
            return False
        for arm in ("margin", "noise", "both"):
            if not lab.need_free(f"models/parakeet-ft-lab-10-{arm}-{s}", f"{FT}/lab-10-{arm}-{s}-test-nemo.jsonl"):
                return False
    if not lab.need_free(f"{n}/table.json", f"{n}/battery.json") or not lab.ensure_battery() or not lab.act():
        return False
    failed: list[str] = []
    if not lab.battery_decode("stock", ["--stock"]):
        failed.append("battery:stock")
    bsys = ["--system", f"stock={FT}/lab-13-battery-stock.jsonl"]
    for s in lab.spk:
        if lab.battery_decode(f"whole-block-{s}", ["--nemo", lab.wb_m(s)]):
            bsys += ["--system", f"whole-block-{s}={FT}/lab-13-battery-whole-block-{s}.jsonl"]
        else:
            failed.append(f"battery:whole-block-{s}")
    flags = {"margin": ["--empty-margin", str(margin)], "noise": ["--empty-clips-per-kind", str(kper)],
             "both": ["--empty-margin", str(margin), "--empty-clips-per-kind", str(kper)]}
    for s in lab.spk:
        for arm in ("margin", "noise", "both"):
            lab.banner(f"{s}, {arm}, F-2 recipe")
            fitted = lab.fit(f"{EX}/{s}", f"lab-10-{arm}-{s}", "f2", flags[arm])
            if not fitted:
                lab.say(f"FAILED {arm} {s}")
                failed.append(f"{arm}:{s}")
                continue
            good = lab.decode(s, f"lab-10-{arm}-{s}", ["--nemo", f"models/parakeet-ft-{fitted}/parakeet-ft.nemo"])
            if not good:
                lab.say(f"FAILED decode {arm} {s}")
                failed.append(f"decode:{arm}:{s}")
            if lab.battery_decode(f"{arm}-{s}", ["--nemo", f"models/parakeet-ft-{fitted}/parakeet-ft.nemo"]):
                bsys += ["--system", f"{arm}-{s}={FT}/lab-13-battery-{arm}-{s}.jsonl"]
                if good:
                    lab.trim_model(fitted)
            else:
                failed.append(f"battery:{arm}:{s}")
    summary_lines(lab, failed)
    arms = [f"{arm}={FT}/lab-10-{arm}-@-test-nemo.jsonl" for arm in ("margin", "noise", "both")
            if all(lab.have(f"{FT}/lab-10-{arm}-{s}-test-nemo.jsonl") for s in lab.spk)]
    if not lab.arm_table(n, arms):
        failed.append("table")
    lab.say("-- false inserts on clips with no speech (lab test 13's battery)")
    if not lab.run([lab.sc, "scripts/asr/battery.py", "score", "--manifest", f"{lab.BAT}/manifest.jsonl", *bsys, "--out", f"{n}/battery.json"]):
        lab.say("FAILED battery score")
        failed.append("battery-score")
    lab.say(f"10 finished {now()}")
    return not failed


def manifests_hash(lab: Lab) -> str:
    digest = hashlib.sha256()
    for s in lab.spk:
        for split in ("train", "dev", "test"):
            path = lab.root / EX / s / f"{split}.jsonl"
            if path.is_file():
                digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def stage_11(lab: Lab) -> bool:
    n = "reports/lab-11-severe-pretraining"
    lab.say(f"######## 11: the other severe speakers, then his own sessions, started {now()}")
    for s in lab.spk:
        if not lab.need_have(f"{EX}/{s}/train.jsonl", lab.f1_h(s) + "/hypotheses.jsonl", lab.wb_h(s) + "/hypotheses.jsonl",
                             lab.stock_h(s) + "/hypotheses.jsonl"):
            return False
        if not lab.need_free(f"models/parakeet-ft-lab-11-pre-{s}", f"models/parakeet-ft-lab-11-then-{s}",
                             f"{FT}/lab-11-pre-{s}-test-nemo.jsonl", f"{FT}/lab-11-then-{s}-test-nemo.jsonl"):
            return False
    if not lab.need_free(f"{n}/table.json"):
        return False
    if not lab.ensure_severe_exports():
        return False
    if not lab.act():
        return False
    failed: list[str] = []
    for s in lab.spk:
        lab.banner(f"{s}, phase 1: the other severe speakers, F-1 recipe")
        pre = lab.fit(f"{EX}/loso-severe-{s}", f"lab-11-pre-{s}", "f1")
        if not pre:
            lab.say(f"FAILED phase 1 {s}")
            failed.append(f"pre:{s}")
            continue
        pre_model = f"models/parakeet-ft-{pre}/parakeet-ft.nemo"
        if not lab.decode(s, f"lab-11-pre-{s}", ["--nemo", pre_model]):
            lab.say(f"FAILED decode pre {s}")
            failed.append(f"decode:pre:{s}")
        lab.banner(f"{s}, phase 2: his own earlier sessions, F-2 recipe, from the phase 1 model")
        then = lab.fit(f"{EX}/{s}", f"lab-11-then-{s}", "f2", ["--base", pre_model])
        if then and lab.decode(s, f"lab-11-then-{s}", ["--nemo", f"models/parakeet-ft-{then}/parakeet-ft.nemo"]):
            lab.trim_model(then)
            lab.trim_model(pre)
        else:
            lab.say(f"FAILED phase 2 {s}")
            failed.append(f"then:{s}")
    summary_lines(lab, failed)
    lab.pooled("--system", lab.sysfn("stock", lab.stock_h), "--system", lab.sysfn("F-1", lab.f1_h),
               "--system", lab.sysfmt("severe-others-only", f"{FT}/lab-11-pre-@-test-nemo.jsonl"),
               "--system", lab.sysfn("whole-block", lab.wb_h),
               "--system", lab.sysfmt("severe-others-then-his-own", f"{FT}/lab-11-then-@-test-nemo.jsonl"),
               "--baseline", "stock", "--out", f"{n}/table.json")
    lab.say(f"11 finished {now()}")
    return True


def stage_12(lab: Lab) -> bool:
    n = "reports/lab-12-pseudo-label-loop"
    rounds = int(lab.cfg["rounds"])
    lab.say(f"######## 12: pseudo-label loop, {rounds} rounds, started {now()}")
    for s in lab.spk:
        if not lab.need_have(f"{EX}/{s}/train.jsonl", f"{EX}/{s}/test.jsonl", f"{EX}/loso-{s}/train.jsonl", f"{EX}/loso-{s}/dev.jsonl",
                             lab.wb_h(s) + "/hypotheses.jsonl", lab.stock_h(s) + "/hypotheses.jsonl"):
            return False
        for r in range(1, rounds + 1):
            if not lab.need_free(f"{EX}/lab-12-{s}-r{r}", f"models/parakeet-ft-lab-12-{s}-r{r}", f"{FT}/lab-12-{s}-r{r}-test-nemo.jsonl",
                                 f"{FT}/lab-12-{s}-r{r}-teacher.jsonl"):
                return False
    if not lab.need_free(f"{n}/table.json") or not lab.act():
        return False
    failed: list[str] = []
    for s in lab.spk:
        teacher, who, prev = ["--stock"], "the stock model's", None
        for r in range(1, rounds + 1):
            lab.banner(f"{s}, round {r}: {who} answers on his earlier recordings")
            if not lab.run([lab.gpu_py, "scripts/asr/nemo_hyp.py", *teacher, "--manifest", f"{EX}/{s}/train.jsonl", "--audio-root", f"{EX}/{s}",
                            "--out", f"{FT}/lab-12-{s}-r{r}-teacher.jsonl"]):
                lab.say(f"FAILED teacher {s} {r}")
                failed.append(f"teacher:{s}:{r}")
                break
            if not lab.run([lab.sc, "scripts/asr/pseudo_label.py", "--manifest", f"{EX}/{s}/train.jsonl", "--audio-root", f"{EX}/{s}",
                            "--hyps", f"{FT}/lab-12-{s}-r{r}-teacher.jsonl", "--phrases-from", f"{EX}/loso-{s}/train.jsonl",
                            f"{EX}/loso-{s}/dev.jsonl", "--test-manifest", f"{EX}/{s}/test.jsonl", "--audit", "--out", f"{EX}/lab-12-{s}-r{r}"]):
                lab.say(f"FAILED labels {s} {r}")
                failed.append(f"labels:{s}:{r}")
                break
            lab.banner(f"{s}, round {r}: fine-tune the stock model on those labels, F-2 recipe")
            fitted = lab.fit(f"{EX}/lab-12-{s}-r{r}", f"lab-12-{s}-r{r}", "f2")
            if not fitted:
                lab.say(f"FAILED {s} {r}")
                failed.append(f"{s}:{r}")
                break
            teacher, who = ["--nemo", f"models/parakeet-ft-{fitted}/parakeet-ft.nemo"], f"the round {r} model's"
            if not lab.decode(s, f"lab-12-{s}-r{r}", teacher):
                lab.say(f"FAILED decode {s} {r}")
                failed.append(f"decode:{s}:{r}")
                break
            if prev:
                lab.trim_model(prev)
            prev = fitted
        if prev:
            lab.trim_model(prev)
    summary_lines(lab, failed)
    lab.arm_table(n, [f"round{r}={FT}/lab-12-@-r{r}-test-nemo.jsonl" for r in range(1, rounds + 1)])
    lab.say(f"12 finished {now()}")
    return True


def stage_13(lab: Lab) -> bool:
    n = "reports/lab-13-silence-false-inserts"
    lab.say(f"######## 13: the silence battery on the models that exist, started {now()}")
    if not lab.need_free(f"{n}/score.json"):
        return False
    for s in lab.spk:
        if not lab.need_have(lab.wb_m(s), lab.f1_m(s)):
            return False
    if not lab.ensure_battery() or not lab.act():
        return False
    failed: list[str] = []
    bsys: list[str] = []
    if lab.battery_decode("stock", ["--stock"]):
        bsys += ["--system", f"stock={FT}/lab-13-battery-stock.jsonl"]
    else:
        failed.append("stock")
    for s in lab.spk:
        for label, model in ((f"whole-block-{s}", lab.wb_m(s)), (f"f1-{s}", lab.f1_m(s))):
            if lab.battery_decode(label, ["--nemo", model]):
                bsys += ["--system", f"{label}={FT}/lab-13-battery-{label}.jsonl"]
            else:
                failed.append(label)
    summary_lines(lab, failed)
    if not lab.run([lab.sc, "scripts/asr/battery.py", "score", "--manifest", f"{lab.BAT}/manifest.jsonl", *bsys, "--out", f"{n}/score.json"]):
        lab.say("FAILED battery score")
    lab.say(f"13 finished {now()}")
    return True


def stage_14(lab: Lab) -> bool:
    n = "reports/lab-14-route-scoring"
    lab.say(f"######## 14: route-aware scoring, started {now()}")
    for s in lab.spk:
        if not lab.need_hyp(lab.stock_h(s), lab.wb_h(s), lab.ad_h(s)):
            return False
    if not lab.need_free(f"{n}/route-stock", f"{n}/route-whole-block", f"{n}/route-adapter-on-stock", f"{n}/table.json"):
        return False
    if not lab.ensure_words():
        return False
    failed: list[str] = []
    systems = ["--system", lab.sysfn("stock", lab.stock_h)]
    for label, func in (("stock", lab.stock_h), ("whole-block", lab.wb_h), ("adapter-on-stock", lab.ad_h)):
        speakers = [x for s in lab.spk for x in ("--speaker", f"{s}={func(s)}")]
        lab.banner(f"routing with {label} as the sentence recognizer")
        if lab.run([lab.sc, "scripts/matcher/route_score.py", "--emb", "data/emb/torgo-words.npz", *speakers, "--min-scores", "0.3", "0.5",
                    "--out", f"{n}/route-{label}"]):
            if label != "stock":
                systems += ["--system", lab.sysfn(label, func)]
            systems += ["--system", lab.sysfmt(f"routed-{label}", f"{n}/route-{label}/routed/@"),
                        "--system", lab.sysfmt(f"routed-{label}-min0.3", f"{n}/route-{label}/routed-min0.3/@"),
                        "--system", lab.sysfmt(f"routed-{label}-min0.5", f"{n}/route-{label}/routed-min0.5/@")]
        else:
            lab.say(f"FAILED route {label}")
            failed.append(label)
    summary_lines(lab, failed)
    lab.pooled(*systems, "--baseline", "stock", "--out", f"{n}/table.json")
    lab.say(f"14 finished {now()}")
    return True


def stage_15(lab: Lab) -> bool:
    n = "reports/lab-15-int8-screen"
    lab.say(f"######## 15: int8 screen, started {now()}")
    if not lab.need_free(f"{n}/screen.json") or not lab.act():
        return False
    args = ["--stock"]
    for tag, kind, name in lab.candidates("nemo"):
        if tag == "v2":
            continue
        probe = "import sys; sys.path.insert(0, 'scripts/asr'); import weights; weights.stock_nemo(sys.argv[1])"
        if lab.dry or subprocess.run([lab.gpu_py, "-c", probe, name], cwd=lab.root, capture_output=True, check=False).returncode == 0:
            args += ["--cached", f"{tag}={name}"]
        else:
            lab.say(f"{tag} ({name}) is not in the Hugging Face cache (decode it once, stage 4); leaving it out")
    for s in lab.spk:
        if lab.have(lab.wb_m(s)):
            args += ["--nemo", f"whole-block-{s}={lab.wb_m(s)}"]
        if lab.have(lab.f1_m(s)):
            args += ["--nemo", f"f1-loso-{s}={lab.f1_m(s)}"]
    for d in sorted((lab.root / "models").glob("parakeet-ft-lab-*/parakeet-ft.nemo")):
        args += ["--nemo", f"{d.parent.name.removeprefix('parakeet-ft-')}={d.relative_to(lab.root)}"]
    for f in sorted((lab.root / "models").glob("*/onnx-int8/encoder.int8.onnx")) + sorted((lab.root / "models").glob("*/encoder.int8.onnx")):
        folder = f.parent.parent if f.parent.name == "onnx-int8" else f.parent
        args += ["--onnx", f"int8-{folder.name.removeprefix('parakeet-')}={f.relative_to(lab.root)}"]
    ok = lab.run([lab.gpu_py, "scripts/asr/quant_screen.py", *args, "--out", f"{n}/screen.json"])
    if not ok:
        lab.say("FAILED quant_screen")
    lab.say("")
    lab.say("######## summary")
    lab.say(f"15 finished {now()}")
    return ok


def ensure_severe_exports(self: Lab) -> bool:
    if all(self.exists(f"{EX}/loso-severe-{s}") for s in self.spk):
        return True
    before = manifests_hash(self) if not self.dry else ""
    self.say("making the leave-one-out exports that train on the other severe speakers only")
    if not self.run([self.sc, "scripts/corpus/torgo_manifest.py", "--export", "--speakers", *self.spk, "--severe-loso"]):
        self.say("export step failed, stopping this stage")
        return False
    if not self.dry:
        after = manifests_hash(self)
        self.say(f"the three speakers' manifests before: {before}, after: {after}")
        if before != after:
            self.say("their own manifests changed, stopping this stage (every earlier result used the old ones)")
            return False
    return True


Lab.ensure_severe_exports = ensure_severe_exports


def wrong_words(lab: Lab, folder: str) -> int | None:
    path = lab.root / folder / "results.json"
    if not path.is_file():
        return None
    word = json.loads(path.read_text(encoding="utf-8"))["word"]
    return word["sub"] + word["dele"] + word["ins"]


def adapter_run(lab: Lab, speaker: str, name: str, seed: int, blocks: int = 8, base: str | None = None, extra: list[str] | None = None,
                epochs: int = 15, batch: int = 2, accum: int = 8, export: str | None = None, battery: str | None = None) -> bool:
    flags = merge_flags(Lab.RECIPES["ad8"], ["--adapter-after-blocks", str(blocks), "--epochs", str(epochs)] + (["--base", base] if base else []) + (extra or []))
    if not lab.train(export or f"{EX}/{speaker}", name, flags, None, batch=batch, accum=accum, seed=seed):
        return False
    model_flags = lab.adapter_flags(name, base)
    if not lab.decode(speaker, name, model_flags):
        return False
    if battery and not lab.battery_decode(battery, model_flags):
        lab.say(f"-- silence battery failed for {name} (test result unaffected)")
    if lab.have(f"{EX}/{speaker}/dev.jsonl") and (lab.dry or (lab.root / EX / speaker / "dev.jsonl").stat().st_size > 0):
        if not lab.decode(speaker, name, model_flags, split="dev"):
            lab.say(f"-- dev scoring failed for {name} (test result unaffected)")
    lab.trim_model(name)
    return True


def wrong_table(lab: Lab, rows: list[tuple[str, list[str]]]) -> None:
    lab.say(f"wrong words on each test set and pooled over the 425 test words (M01 169, M02 95, M04 161); {BAR_WORDS} or fewer is 35% or lower")
    lab.say(f"{'model':<46} {'M01':>5} {'M02':>5} {'M04':>5} {'pooled':>7} {'WER':>7}")
    for label, folders in rows:
        per = [wrong_words(lab, f) for f in folders]
        total = None if any(v is None for v in per) else sum(per)
        cells = " ".join(f"{('-' if v is None else v):>5}" for v in per)
        lab.say(f"{label:<46} {cells} {('-' if total is None else total):>7} {('-' if total is None else '%.1f%%' % (100.0 * total / 425)):>7}")


def stage_W(lab: Lab) -> bool:
    pb = "reports/phrasebook-matching"
    thresholds = "0.20,0.30,0.40,0.50,0.55,0.60,0.65,0.70,0.80"
    lab.say(f"######## W: matching single words, and the whole test set, started {now()}")
    if not lab.need_free(f"{pb}/words", f"{pb}/device-both"):
        return False
    original = {
        "stock": lambda s: f"{FT}/{s}-test-stock-nemo",
        "F-1": lambda s: f"{FT}/f1-loso-{s}-bottom8-on-F2-test-nemo",
        "adapter-on-stock": lambda s: f"{FT}/f2-M02-adapter8-pos-test-nemo" if s == "M02" else f"{FT}/f2-{s}-adapter8-test-nemo",
        "adapter-on-F-1": lambda s: f"{FT}/f2-M02-adapter8-on-f1-seed17-test-nemo" if s == "M02" else f"{FT}/f2-{s}-adapter8-on-f1-test-nemo",
        "whole-block": lambda s: (lab.cfg["wb_m02_hyp"] or f"{FT}/M02e-A3-test-nemo") if s == "M02" else f"{FT}/f2-{s}-bottom8-5epochs-test-nemo",
    }
    found: dict[str, dict[str, str]] = {}
    for label, fn, kind in (("stock", lab.stock_h, "stock"), ("F-1", lab.f1_h, "f1"), ("adapter-on-stock", lab.ad_h, "adapter"),
                            ("adapter-on-F-1", lab.adf1_h, "adapter-f1"), ("whole-block", lab.wb_h, "whole-block")):
        hits = {}
        for s in SEVERE:
            hit = next((p for p in (fn(s), original[label](s), f"{FT}/lab-baselines/{kind}-{s}") if lab.has_hyp(p)), None)
            if hit:
                hits[s] = hit
        if len(hits) == len(SEVERE):
            found[label] = hits
        elif label == "stock":
            lab.say("MISSING the stock model's answers for " + ", ".join(s for s in SEVERE if s not in hits) + "; stopping this stage")
            lab.say("Run `python lab.py run 1` or copy the stock decodes from reports/finetune-torgo/ on Yack.")
            return False
        else:
            lab.say(f"-- leaving out {label}: no answers for " + ", ".join(s for s in SEVERE if s not in hits))
    lab.say("models matched: " + ", ".join(found))

    models = [item for label, hits in found.items() for item in ("--model", label + "=" + ",".join(f"{s}:{hits[s]}" for s in SEVERE))]
    marks: dict[str, int] = {}
    for kinds, out in (("word", f"{pb}/words"), ("sentence,word", f"{pb}/device-both")):
        lab.say("")
        lab.say(f"==== --kinds {kinds} -> {out}")
        marks[kinds] = len(lab.lines)
        lab.run([lab.sc, "scripts/asr/phrasebook.py", "--kinds", kinds, "--pool-glob", f"{EX}/*/train.jsonl", f"{EX}/*/dev.jsonl", f"{EX}/*/test.jsonl",
                 *[x for s in SEVERE for x in ("--speaker", f"{s}={EX}/{s}/test.jsonl")], *models, "--sizes", "10,25,50,100", "--repeats", "20",
                 "--thresholds", thresholds, "--out", out])
    lab.say("")
    lab.say("######## summary")
    for kinds, start in marks.items():
        lab.say("")
        lab.say(f"== --kinds {kinds}, whole pool, pooled over M01, M02 and M04")
        block = lab.lines[start:]
        for i, line in enumerate(block):
            if line.startswith("-- ALL, phrasebook of the whole pool"):
                for x in block[i:]:
                    lab.say(x)
                break
    lab.say(f"W finished {now()}")
    return True


B_DONE = "reports/f2-base-model-ceiling/finished.json"


def stage_B(lab: Lab) -> bool:
    lab.say(f"######## B: larger and other base models, started {now()}")
    bases = [("1p1b", "nvidia/parakeet-tdt-1.1b"), ("v3", "nvidia/parakeet-tdt-0.6b-v3"), ("110m", "nvidia/parakeet-tdt_ctc-110m")]
    for s in SEVERE:
        if not lab.need_have(lab.base8(s, 17) + "/results.json"):
            return False
    if not lab.act():
        return False
    failed: list[str] = []
    use_b1: set[str] = set()
    finished: set[str] = set()
    names: list[str] = []
    rows: list[tuple[str, list[str]]] = [("0.6B v2 stock", [f"{FT}/{s}-test-stock-nemo" for s in SEVERE]),
                                         ("0.6B v2 + adapter after block 8 (seed 17)", [lab.base8(s, 17) for s in SEVERE])]
    for tag, model in bases:
        lab.banner(f"{tag} ({model}): stock decode on the three test sets")
        ok = True
        for s in SEVERE:
            if not lab.stock_screen(s, tag, "nemo", "", model):
                ok = False
                lab.say(f"FAILED stock decode {s} {tag}")
                failed.append(f"stock:{s}:{tag}")
        if not ok:
            lab.say(f"skipping the training runs for {tag}")
            continue
        blocks = {"1p1b": (8, 14), "110m": (6, 8)}.get(tag, (8,))
        rows.append((f"{tag} stock", [f"{FT}/{s}-test-stock-{tag}-nemo" for s in SEVERE]))
        stop = False
        for n in blocks:
            folders = []
            for s in SEVERE:
                name = f"f2-{s}-adapter{n}-{tag}-seed17"
                if not lab.need_free(f"models/parakeet-ft-{name}", f"{FT}/{name}-test-nemo"):
                    failed.append(f"{s}:{n}:{tag}")
                    continue
                lab.banner(f"{s}, {tag}, adapter after {n} blocks, seed 17")
                done = False
                if tag not in use_b1:
                    done = adapter_run(lab, s, name, 17, n, model)
                if not done:
                    if tag not in use_b1:
                        lab.say(f"FAILED {s} {n} {tag} with batch 2; retrying with batch 1 x 16 accumulation as {name}-b1, and using that for the rest of {tag}")
                    name += "-b1"
                    done = adapter_run(lab, s, name, 17, n, model, batch=1, accum=16)
                    if done:
                        use_b1.add(tag)
                if done:
                    finished.add(tag)
                    names.append(name)
                    folders.append(f"{FT}/{name}-test-nemo")
                else:
                    lab.say(f"FAILED {s} {n} {tag} again, continuing")
                    failed.append(f"{s}:{n}:{tag}")
                    if tag not in finished:
                        lab.say(f"no {tag} run has finished, so the rest of the {tag} runs are skipped (look at the first error above)")
                        stop = True
                        break
            if len(folders) == len(SEVERE):
                rows.append((f"{tag} + adapter after block {n} (seed 17)", folders))
            if stop:
                break
    summary_lines(lab, failed)
    wrong_table(lab, rows)
    lab.say("-- ci.py, each new adapter model against the 0.6B v2 adapter after block 8 (positive gain = the new model is better)")
    for name in names:
        for s in SEVERE:
            if f"-{s}-" not in name:
                continue
            lab.say(f"== {name} against {lab.base8(s, 17)}")
            if lab.dry:
                continue
            out = lab.capture([lab.sc, "scripts/asr/ci.py", "--manifest", f"{EX}/{s}/test.jsonl", "--stock", lab.base8(s, 17), "--finetuned", f"{FT}/{name}-test-nemo"])
            for line in out.splitlines():
                if re.search(r"Gain|Baseline|Candidate", line):
                    lab.say(line)
    if not lab.dry:
        (lab.root / B_DONE).parent.mkdir(parents=True, exist_ok=True)
        (lab.root / B_DONE).write_text(json.dumps({"finished": now(), "runs": names, "failed": failed}, indent=1) + "\n")
    lab.say(f"B finished {now()}")
    return True


AUG = {"speedwide": ["--speed-perturb", "0.7", "1.3"], "nospeed": ["--no-speed-perturb"], "nospec": ["--no-spec-augment"],
       "tempo": ["--time-stretch", "0.75", "1.25"], "noise": ["--white-noise", "-70", "-40"],
       "tempo-noise": ["--time-stretch", "0.75", "1.25", "--white-noise", "-70", "-40"]}


def aug_stage(lab: Lab, st: str, variants: list[str]) -> bool:
    lab.say(f"######## {st}: augmentation variants ({' '.join(variants)}), adapter after block 8, fixed schedule, started {now()}")
    for v in variants:
        lab.say(f"variant {v}: {' '.join(AUG[v])}")
    seeds = (17, 18, 19)
    for s in SEVERE:
        for seed in seeds:
            if not lab.need_have(lab.base8(s, seed) + "/results.json"):
                return False
    for v in variants:
        for s in SEVERE:
            for seed in seeds:
                if not lab.need_free(f"models/parakeet-ft-f2-{s}-adapter8-aug-{v}-seed{seed}", f"{FT}/f2-{s}-adapter8-aug-{v}-seed{seed}-test-nemo"):
                    return False
    if st == "N":
        lab.say("-- how loud the training clips are, against the white-noise levels used here (whole-clip RMS in dBFS)")
        if not lab.run([lab.sc, "scripts/asr/clip_levels.py", *[f"{EX}/{s}" for s in SEVERE], "--noise-db", "-70", "-40"]):
            lab.say("clip levels failed (not needed for the runs)")
    if not lab.act():
        return False
    failed: list[str] = []
    for seed in seeds:
        for v in variants:
            for s in SEVERE:
                name = f"f2-{s}-adapter8-aug-{v}-seed{seed}"
                lab.banner(f"{s}, {v} ({' '.join(AUG[v])}), seed {seed}")
                if not adapter_run(lab, s, name, seed, 8, None, AUG[v]):
                    lab.say(f"FAILED {s} {v} {seed}, continuing")
                    failed.append(f"{s}:{v}:{seed}")
    summary_lines(lab, failed)
    stock = [wrong_words(lab, lab.stock_h(s)) for s in SEVERE]
    lab.say(f"wrong words, adapter after block 8, fixed 15-epoch schedule; stock is {sum(v or 0 for v in stock)} of 425 pooled")
    lab.say(f"{'variant':<12} {'seed 17 / 18 / 19 (pooled)':<30} {'mean':>7} {'minus default':>14}")
    means: dict[str, float] = {}
    for v in ("default", *variants):
        totals = []
        for seed in seeds:
            per = [wrong_words(lab, lab.base8(s, seed) if v == "default" else f"{FT}/f2-{s}-adapter8-aug-{v}-seed{seed}-test-nemo") for s in SEVERE]
            totals.append(None if any(x is None for x in per) else sum(per))
        if None in totals:
            lab.say(f"{v:<12} {'missing seeds':<30}")
            continue
        means[v] = sum(totals) / 3.0
        diff = "" if v == "default" or "default" not in means else f"{means[v] - means['default']:+.1f}"
        lab.say(f"{v:<12} {' / '.join(str(t) for t in totals):<30} {means[v]:>7.1f} {diff:>14}")
    lab.say(f"{st} finished {now()}")
    return not failed


def stage_A(lab: Lab) -> bool:
    return aug_stage(lab, "A", ["speedwide", "nospeed", "nospec"])


def stage_N(lab: Lab) -> bool:
    return aug_stage(lab, "N", ["tempo", "noise", "tempo-noise"])


def stage_F(lab: Lab) -> bool:
    lab.say(f"######## F: data-quantity curve, M02, adapter after block 8, started {now()}")
    minutes = [1, 3, 5, 7]
    seeds = (17, 18, 19)
    if not lab.need_have("data/public/torgo/TORGO"):
        return False
    for seed in seeds:
        if not lab.need_have(lab.base8("M02", seed) + "/results.json"):
            return False
    for m in minutes:
        if not lab.need_free(f"{EX}/M02-min{m}"):
            return False
        for arm in ("epochs", "updates"):
            for seed in seeds:
                if not lab.need_free(f"models/parakeet-ft-f3-M02-min{m}-{arm}-seed{seed}", f"{FT}/f3-M02-min{m}-{arm}-seed{seed}-test-nemo"):
                    return False
    own = [f"{EX}/M02/{x}.jsonl" for x in ("train", "dev", "test")]

    def digest() -> str:
        h = hashlib.sha256()
        for p in own + sorted(str(x.relative_to(lab.root)) for x in (lab.root / EX / "loso-M02").glob("*.jsonl")):
            if (lab.root / p).is_file():
                h.update((lab.root / p).read_bytes())
        return h.hexdigest()[:12]

    before = "" if lab.dry else digest()
    lab.say(f"M02 manifests before: {before}")
    if not lab.run([lab.sc, "scripts/corpus/torgo_manifest.py", "--export", "--speakers", "M02", "--curve", "M02", "--curve-minutes", *map(str, minutes)]):
        lab.say("manifest step failed, stopping this stage")
        return False
    if not lab.dry and digest() != before:
        lab.say("M02's own or loso-M02's manifests changed, stopping this stage (every earlier M02 result used the old ones)")
        return False

    def updates_per_epoch(clips: int) -> int:
        return (((clips + 1) // 2) + 7) // 8

    def count(path: str) -> int:
        file = lab.root / path
        return sum(1 for _ in file.open(encoding="utf-8")) if file.is_file() else 0

    n_all = count(f"{EX}/M02/train.jsonl") if not lab.dry else 72
    target = 15 * updates_per_epoch(n_all)
    lab.say(f"M02 whole set: {n_all} clips, {target} updates in 15 epochs (the target for the equal-updates arm)")
    if not lab.act():
        return False
    failed: list[str] = []
    for seed in seeds:
        for m in minutes:
            n = count(f"{EX}/M02-min{m}/train.jsonl") if not lab.dry else 30
            upe = max(1, updates_per_epoch(n))
            equal = min(150, (target + upe - 1) // upe)
            for arm in ("epochs", "updates"):
                epochs = 15 if arm == "epochs" else equal
                name = f"f3-M02-min{m}-{arm}-seed{seed}"
                lab.banner(f"M02, {m} minutes ({n} clips), arm {arm}, {epochs} epochs, seed {seed}")
                ok = lab.train(f"{EX}/M02-min{m}", name, merge_flags(Lab.RECIPES["ad8"], ["--epochs", str(epochs)]), None, seed=seed) \
                    and lab.decode("M02", name, lab.adapter_flags(name, None))
                if not ok:
                    lab.say(f"FAILED {name}, continuing")
                    failed.append(name)
                lab.trim_model(name)
    summary_lines(lab, failed)
    stock = wrong_words(lab, lab.stock_h("M02"))
    lab.say(f"wrong words on M02's 95 test words; stock {stock}")
    lab.say(f"{'minutes':<8} {'arm':<8} {'seeds 17 18 19':<18} {'mean':>6}")
    for m in minutes:
        for arm in ("epochs", "updates"):
            vals = [wrong_words(lab, f"{FT}/f3-M02-min{m}-{arm}-seed{seed}-test-nemo") for seed in seeds]
            done = [v for v in vals if v is not None]
            mean = f"{sum(done) / len(done):6.1f}" if done else "     -"
            lab.say(f"{m:<8} {arm:<8} {' '.join(f'{v!s:>4}' for v in vals):<18} {mean}")
    lab.say(f"F finished {now()}")
    return not failed


def stage_U(lab: Lab) -> bool:
    seeds = [17, 18, 19, 20, 21]
    lab.say(f"######## U: adapter soup, {len(seeds)} seeds averaged per speaker, started {now()}")
    for sp in lab.spk:
        if not lab.need_have(f"{EX}/{sp}/train.jsonl", f"{EX}/{sp}/test.jsonl", lab.stock_h(sp) + "/hypotheses.jsonl"):
            return False
    if not lab.act():
        return False
    failed: list[str] = []
    table: dict[str, dict] = {}
    for sp in lab.spk:
        good: list[str] = []
        for seed in seeds:
            name = f"big-u-ad8-{sp}-s{seed}"
            path = f"models/parakeet-ft-{name}/adapter.pt"
            ok = True
            if lab.dry or not lab.exists(path):
                lab.banner(f"{sp}, adapter, seed {seed}")
                ok = lab.train_retrying(f"{EX}/{sp}", name, Lab.RECIPES["ad8"], None, seed=seed)
            if ok and (lab.dry or not lab.exists(f"{FT}/{name}-test-nemo/results.json")):
                ok = lab.decode(sp, name, lab.adapter_flags(name, None))
            if ok:
                good.append(path)
            else:
                lab.say(f"FAILED {name}, continuing")
                failed.append(name)
        soup = f"big-u-soup-{sp}"
        soup_path = f"models/parakeet-ft-{soup}/adapter.pt"
        if len(good) >= 2:
            lab.banner(f"{sp}: average of {len(good)} adapters")
            ok = lab.run([lab.gpu_py, "scripts/asr/adapter_soup.py", "--inputs", *good, "--out", soup_path]) \
                and lab.decode(sp, soup, lab.adapter_flags(soup, None))
            if not ok:
                lab.say(f"FAILED {soup}")
                failed.append(soup)
        else:
            lab.say(f"{sp}: fewer than two adapters finished; no average")
        for seed in seeds:
            lab.trim_model(f"big-u-ad8-{sp}-s{seed}")
        lab.trim_model(soup)
        if not lab.dry:
            table[sp] = {"seeds": [wrong_words(lab, f"{FT}/big-u-ad8-{sp}-s{seed}-test-nemo") for seed in seeds],
                         "soup": wrong_words(lab, f"{FT}/{soup}-test-nemo"), "stock": wrong_words(lab, lab.stock_h(sp))}
    summary_lines(lab, failed)
    if not lab.dry:
        lab.say(f"{'speaker':<8} {'stock':>6} {'seeds ' + ' '.join(map(str, seeds)):<30} {'mean':>6} {'soup':>6}")
        pooled_mean, pooled_soup = 0.0, 0
        for sp, row in table.items():
            done = [v for v in row["seeds"] if v is not None]
            mean = sum(done) / len(done) if done else float("nan")
            lab.say(f"{sp:<8} {row['stock']!s:>6} {' '.join(f'{v!s:>4}' for v in row['seeds']):<30} {mean:>6.1f} {row['soup']!s:>6}")
            if row["soup"] is None or not done:
                pooled_soup = None
            elif pooled_soup is not None:
                pooled_mean += mean
                pooled_soup += row["soup"]
        if pooled_soup is not None:
            lab.say(f"pooled: individual seeds average {pooled_mean:.1f} wrong, the averaged adapter {pooled_soup}")
        out = lab.root / "reports/lab-big-u"
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps({"seeds": seeds, "rows": table}, indent=2) + "\n", encoding="utf-8")
        lab.track_table("reports/lab-big-u/summary.json")
    lab.say(f"U finished {now()}")
    return not failed


class Arm:
    def __init__(self, label: str, recipe: str, flags: list[str] | None = None, init: str = "stock"):
        self.label, self.recipe, self.flags, self.init = label, recipe, flags or [], init


def axis_flags(key: str, value: str) -> list[str]:
    if key == "flags":
        return shlex.split(value)
    flag = "--" + key.replace("_", "-")
    low = value.lower()
    if low in ("on", "true", "yes"):
        return [flag]
    if low in ("off", "false", "no"):
        return ["--no-" + key.replace("_", "-")]
    if low in ("-", "none", "default"):
        return []
    return [flag, *shlex.split(value)]


def lr_arms() -> list[Arm]:
    arms = [Arm("wb-default", "wb")] + [Arm(f"wb-lr{v}", "wb", ["--lr", v]) for v in ("5e-6", "1e-5", "4e-5", "8e-5")]
    return arms + [Arm("ad8-default", "ad8")] + [Arm(f"ad8-lr{v}", "ad8", ["--lr", v]) for v in ("1e-4", "6e-4", "1e-3")]


def target_arms() -> list[Arm]:
    return [Arm("wb-default", "wb"), Arm("wb-stockform", "wb", ["--target-form", "stock"]),
            Arm("ad8-default", "ad8"), Arm("ad8-stockform", "ad8", ["--target-form", "stock"])]


def epoch_arms() -> list[Arm]:
    return ([Arm("wb-default", "wb")] + [Arm(f"wb-ep{v}", "wb", ["--epochs", v]) for v in ("8", "12", "20")]
            + [Arm("ad8-default", "ad8")] + [Arm(f"ad8-ep{v}", "ad8", ["--epochs", v]) for v in ("8", "30")])


def seed_arms() -> list[Arm]:
    return [Arm("wb-default", "wb"), Arm("ad8-default", "ad8"), Arm("ad8-on-f1", "ad8", init="f1")]


def pre_arms() -> list[Arm]:
    return [Arm("wb-default", "wb"), Arm("pre-wb", "wb", init="pre"), Arm("pre-ad8", "ad8", init="pre"), Arm("ad8-default", "ad8")]


def stack_arms() -> list[Arm]:
    longer = ["--epochs", "30"]
    faster = ["--lr", "6e-4"]
    return [Arm("ad8-ep30-lr6e-4", "ad8", longer + faster),
            Arm("pre-ad8-default", "ad8", init="pre"),
            Arm("pre-ad8-ep30", "ad8", longer, init="pre"),
            Arm("pre-ad8-lr6e-4", "ad8", faster, init="pre"),
            Arm("pre-ad8-ep30-lr6e-4", "ad8", longer + faster, init="pre"),
            Arm("ad8-default", "ad8"),
            Arm("pre-wb-lr8e-5-ep20", "wb", ["--lr", "8e-5", "--epochs", "20"], init="pre"),
            Arm("pre-wb-lr1.6e-4-ep20", "wb", ["--lr", "1.6e-4", "--epochs", "20"], init="pre")]


def whole_block_arms() -> list[Arm]:
    return [Arm("wb-default", "wb"),
            Arm("wb-lr8e-5-ep20", "wb", ["--lr", "8e-5", "--epochs", "20"]),
            Arm("wb-lr1.6e-4-ep20", "wb", ["--lr", "1.6e-4", "--epochs", "20"]),
            Arm("wb-lr8e-5-ep40", "wb", ["--lr", "8e-5", "--epochs", "40"])]


def whole_block_far_arms() -> list[Arm]:
    return whole_block_arms() + [Arm("wb-lr1.6e-4-ep40", "wb", ["--lr", "1.6e-4", "--epochs", "40"]),
                                 Arm("wb-lr8e-5-ep60", "wb", ["--lr", "8e-5", "--epochs", "60"])]


def regularize_arms() -> list[Arm]:
    return [Arm("ad8-default", "ad8"),
            Arm("ad8-wd0", "ad8", ["--weight-decay", "0"]),
            Arm("ad8-wd1e-2", "ad8", ["--weight-decay", "1e-2"]),
            Arm("ad8-wd1e-1", "ad8", ["--weight-decay", "1e-1"]),
            Arm("ad8-warmup30", "ad8", ["--warmup-steps", "30"]),
            Arm("ad8-avg1", "ad8", ["--average-top", "1"]),
            Arm("ad8-avg8", "ad8", ["--average-top", "8"]),
            Arm("ad8-bn-trainable", "ad8", ["--no-freeze-batchnorm"])]


def kind_arms() -> list[Arm]:
    return [Arm("ad8-default", "ad8"),
            Arm("ad8-words-only", "ad8", ["--kinds", "word"]),
            Arm("ad8-sentences-only", "ad8", ["--kinds", "sentence"])]


GRIDS: dict[str, tuple] = {
    "L": ("learning-rate sweep, whole-block and adapter recipes", lr_arms, [17, 18, 19],
          "The recipes use 2e-5 (whole-block) and 3e-4 (adapter) without a sweep; docs/TODO.md lists the learning rate as an untried lever."),
    "T": ("targets in the model's own form (--target-form stock) against scoring form", target_arms, [17, 18, 19],
          "The other untried lever in docs/TODO.md."),
    "E": ("schedule length: more or fewer epochs", epoch_arms, [17, 18, 19],
          "Both recipes use a fixed schedule averaged over the last four epochs; this asks whether it is too short or too long."),
    "S": ("five more seeds of the three recipes that matter, to size the run-to-run noise", seed_arms, [20, 21, 22, 23, 24],
          "Single runs differ by 4 to 5 points per speaker; the F-1 adapter's 11-error lead over the stock adapter is inside that."),
    "C": ("severe-speaker pretraining, then the whole-block or adapter recipe, several seeds", pre_arms, [17, 18, 19],
          "Lab test 11 ran this once; here with seeds, and with the adapter as the second phase."),
    "P": ("stack the winners: severe pretraining with a longer or higher-rate adapter, and whole-block from pretrained", stack_arms, [17, 18, 19],
          "Pretraining took the adapter from 213 to 182 wrong; 30 epochs and 6e-4 each gained about 13 alone. Do they add? Two whole-block arms start from the same pretrained models."),
    "G": ("whole-block training with a higher rate and more epochs, to find where it stops improving", whole_block_arms, [17, 18],
          "Whole-block improved at every step of the L and E sweeps (305 to 257 and 267 wrong) and had not levelled off."),
    "G2": ("whole-block training further out, and a third seed of G's arms", whole_block_far_arms, [17, 18, 19],
           "G's trend had not levelled off at 8e-5 and 40 epochs; this reuses G's runs, adds seed 19, and goes to 1.6e-4 at 40 epochs and 8e-5 at 60."),
    "R": ("adapter regularisation: weight decay, warmup, checkpoint averaging, batch-norm", regularize_arms, [17, 18, 19],
          "None of these has been varied; the defaults were set once and kept."),
    "K": ("adapter trained on single words only, sentences only, or both", kind_arms, [17, 18, 19],
          "Words stay near 70% WER while sentences reach 36%; this asks whether the two kinds compete."),
}


def custom_experiments(root: Path) -> list[dict]:
    path = root / EXPERIMENTS_FILE
    if not path.is_file():
        return []
    try:
        return list(json.loads(path.read_text(encoding="utf-8")).get("experiments", []))
    except (OSError, ValueError):
        return []


def save_custom(root: Path, experiments: list[dict]) -> None:
    (root / EXPERIMENTS_FILE).write_text(json.dumps({"experiments": experiments}, indent=2) + "\n", encoding="utf-8")


def custom_arms(spec: dict) -> list[Arm]:
    axes = spec.get("axes") or {}
    keys = list(axes)
    base_flags = shlex.split(spec.get("flags", ""))
    arms: list[Arm] = []
    if spec.get("default", True):
        arms.append(Arm(f"{spec['recipe']}-default", spec["recipe"], list(base_flags), spec.get("init", "stock")))
    for combo in itertools.product(*[axes[k] for k in keys]) if keys else []:
        label = "-".join(f"{k}{v}" for k, v in zip(keys, combo))
        flags = list(base_flags)
        for k, v in zip(keys, combo):
            flags = merge_flags(flags, axis_flags(k, str(v)))
        arms.append(Arm(safe(f"{spec['recipe']}-{label}"), spec["recipe"], flags, spec.get("init", "stock")))
    return arms


def base_for(lab: Lab, init: str, speaker: str) -> str | None:
    if init == "f1":
        return lab.f1_m(speaker)
    if init == "pre":
        return pretrained_model(speaker)
    return None


RUN_TAG = {"G2": "g"}


def run_tag(gid: str) -> str:
    return RUN_TAG.get(gid, gid).lower()


def run_grid(lab: Lab, gid: str, title: str, arms: list[Arm], seeds: list[int], speakers: list[str] | None = None,
             prepare=None, cleanup=None) -> bool:
    speakers = speakers or lab.spk
    seeds = [int(x) for x in str(lab.cfg["seeds"]).split()] or seeds
    total = len(arms) * len(seeds) * len(speakers)
    lab.say(f"######## {gid}: {title}")
    lab.say(f"{len(arms)} arms x {len(seeds)} seeds x {len(speakers)} speakers = {total} runs, started {now()}")
    for arm in arms:
        lab.say(f"  arm {arm.label}: recipe {arm.recipe}, init {arm.init}, flags {' '.join(arm.flags) or '(none)'}")
    for s in speakers:
        if not lab.need_have(f"{EX}/{s}/train.jsonl", f"{EX}/{s}/test.jsonl", lab.stock_h(s) + "/hypotheses.jsonl"):
            return False
        if any(a.init == "f1" for a in arms) and not lab.need_have(lab.f1_m(s)):
            return False
    if not lab.act():
        return False
    todo = [name for name in grid_run_names(lab, gid, arms, seeds, speakers) if not lab.exists(f"{FT}/{name}-test-nemo.jsonl")]
    if prepare and (todo or lab.dry) and not prepare():
        return False
    if not todo and not lab.dry:
        lab.say(f"every {gid} run is already decoded; only the summary is rebuilt")
    failed: list[str] = []
    skipped = 0
    done = 0
    out_of_disk = False
    for seed in seeds:
        for arm in arms:
            if out_of_disk:
                break
            for s in speakers:
                name = f"big-{run_tag(gid)}-{safe(arm.label)}-{s}-s{seed}"
                if lab.exists(f"{FT}/{name}-test-nemo.jsonl") and not lab.dry:
                    skipped += 1
                    continue
                folder = f"models/parakeet-ft-{name}"
                final = "adapter.pt" if arm.recipe == "ad8" else "parakeet-ft.nemo"
                if lab.exists(folder) and not lab.exists(f"{folder}/{final}") and not lab.dry:
                    lab.aside(folder)
                base = base_for(lab, arm.init, s)
                lab.banner(f"{s}, {arm.label}, seed {seed}")
                extra = list(arm.flags) + (["--base", base] if base else [])
                ok = True
                if not lab.exists(f"{folder}/{final}") or lab.dry:
                    ok = lab.train_retrying(f"{EX}/{s}", name, Lab.RECIPES[arm.recipe], extra, seed=seed)
                    if not ok and not lab.dry and lab.out_of_disk:
                        lab.say(f"FAILED {name}: still out of disk after freeing finished weights. Stopping {gid} instead of failing every remaining run.")
                        failed.append(name)
                        lab.trim_model(name)
                        out_of_disk = True
                        break
                if ok:
                    model_flags = lab.adapter_flags(name, base) if arm.recipe == "ad8" else ["--nemo", f"{folder}/parakeet-ft.nemo"]
                    ok = lab.decode(s, name, model_flags)
                    if ok and lab.exists(f"{EX}/{s}/dev.jsonl") and not lab.dry and (lab.root / EX / s / "dev.jsonl").stat().st_size:
                        lab.decode(s, name, model_flags, split="dev")
                if ok:
                    lab.trim_model(name)
                    done += 1
                else:
                    lab.say(f"FAILED {name}, continuing")
                    failed.append(name)
                    lab.trim_model(name)
        if out_of_disk:
            break
    if cleanup:
        cleanup()
    lab.say(f"-- {done} runs finished now, {skipped} were already done, {len(failed)} failed")
    summary_lines(lab, failed)
    grid_summary(lab, gid, title, arms, seeds, speakers)
    lab.say(f"{gid} finished {now()}")
    return not failed


def grid_summary(lab: Lab, gid: str, title: str, arms: list[Arm], seeds: list[int], speakers: list[str]) -> None:
    if lab.dry:
        return
    try:
        import labkit
    except Exception as exc:  # noqa: BLE001
        lab.say(f"summary skipped ({exc})")
        return
    manifests = {s: labkit.read_manifest(labkit.test_manifest(s)) for s in speakers}
    sentences = {s: labkit.only_kind(m, "sentence") for s, m in manifests.items()}

    def wrong_and_words(manifest: dict, path: str) -> tuple[int, int]:
        return labkit.totals(labkit.clip_counts(manifest, labkit.read_hyps(path)))

    words = sentence_words = stock = stock_sentences = 0
    for s in speakers:
        wrong, n = wrong_and_words(manifests[s], lab.stock_h(s))
        sw, sn = wrong_and_words(sentences[s], lab.stock_h(s))
        stock, words, stock_sentences, sentence_words = stock + wrong, words + n, stock_sentences + sw, sentence_words + sn
    rows = []
    per_speaker: dict[str, dict[str, float]] = {}
    for arm in arms:
        totals, sentence_totals, by_speaker = [], [], {s: [] for s in speakers}
        for seed in seeds:
            counts, sentence_counts = [], []
            for s in speakers:
                path = f"{FT}/big-{run_tag(gid)}-{safe(arm.label)}-{s}-s{seed}-test-nemo.jsonl"
                if not lab.exists(path):
                    counts = None
                    break
                counts.append(wrong_and_words(manifests[s], path)[0])
                sentence_counts.append(wrong_and_words(sentences[s], path)[0])
            if counts is not None:
                totals.append(sum(counts))
                sentence_totals.append(sum(sentence_counts))
                for s, c in zip(speakers, counts):
                    by_speaker[s].append(c)
        if totals:
            per_speaker[arm.label] = {s: sum(v) / len(v) for s, v in by_speaker.items()}
            rows.append({"arm": arm.label, "recipe": arm.recipe, "init": arm.init, "seeds_done": len(totals), "pooled_by_seed": totals,
                         "mean": sum(totals) / len(totals), "range": max(totals) - min(totals),
                         "sentence_by_seed": sentence_totals, "sentence_mean": sum(sentence_totals) / len(sentence_totals)})
    by_label = {r["arm"]: r for r in rows}
    lab.say(f"wrong words pooled over {', '.join(speakers)} ({words} words); stock is {stock}; {int(0.35 * words)} or fewer is 35% or lower")
    lab.say(f"sentences only: {sentence_words} words, stock {stock_sentences} wrong, {labkit.bar_words(sentence_words)} or fewer is 35% or lower")
    lab.say(f"{'arm':<26} {'seeds':>5} {'pooled by seed':<24} {'mean':>7} {'WER':>6} {'range':>5} {'vs default':>10} {'speakers agree':>14}"
            f" {'sentence mean':>13} {'sent. WER':>9}")
    for r in rows:
        ref = by_label.get(f"{r['recipe']}-default")
        diff = "" if ref is None or ref is r else f"{r['mean'] - ref['mean']:+.1f}"
        agree = ""
        if ref is not None and ref is not r:
            sign = 1 if r["mean"] < ref["mean"] else -1
            n = sum(1 for s in speakers if (per_speaker[ref["arm"]][s] - per_speaker[r["arm"]][s]) * sign > 0)
            agree = f"{n} of {len(speakers)}"
        lab.say(f"{r['arm']:<26} {r['seeds_done']:>5} {' '.join(map(str, r['pooled_by_seed'])):<24} {r['mean']:>7.1f} "
                f"{100 * r['mean'] / words:>5.1f}% {r['range']:>5} {diff:>10} {agree:>14}"
                f" {r['sentence_mean']:>13.1f} {100 * r['sentence_mean'] / max(sentence_words, 1):>8.1f}%")
    lab.say("A change counts only when at least 2 of 3 speakers move the same way and the pooled WER moves by more than about 4.5 points.")
    out = lab.root / f"reports/lab-big-{gid.lower()}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"title": title, "speakers": speakers, "words": words, "stock_wrong": stock,
                                                   "sentence_words": sentence_words, "stock_sentence_wrong": stock_sentences, "rows": rows}, indent=2) + "\n",
                                      encoding="utf-8")
    lab.track_table(f"reports/lab-big-{gid.lower()}/summary.json")


def stage_grid(gid: str):
    def run(lab: Lab) -> bool:
        title, arms, seeds, _ = GRIDS[gid]
        if any(a.init == "pre" for a in arms()):
            return run_pretrained_grid(lab, gid)
        return run_grid(lab, gid, title, arms(), seeds)
    return run


def stage_C(lab: Lab) -> bool:
    return run_pretrained_grid(lab, "C")


def pretrained_model(s: str) -> str:
    return f"models/parakeet-ft-big-c-pre-{s}/parakeet-ft.nemo"


def ensure_pretrained_models(lab: Lab) -> bool:
    if not lab.ensure_severe_exports():
        return False
    for s in lab.spk:
        folder = f"models/parakeet-ft-big-c-pre-{s}"
        has_weights = lab.exists(pretrained_model(s))
        if has_weights and not lab.dry:
            drop_checkpoints(lab, folder)
            continue
        if lab.exists(folder) and not has_weights:
            lab.say(f"{folder} has no weights (removed after an earlier run); moving it aside so phase 1 can train it again")
            lab.aside(folder)
        if not room_for_full_training(lab, folder):
            lab.say(f"FAILED phase 1 {s}: not enough disk")
            return False
        lab.banner(f"{s}, phase 1: the other severe speakers, F-1 recipe (seed 17)")
        if not lab.train(f"{EX}/loso-severe-{s}", f"big-c-pre-{s}", Lab.RECIPES["f1"], None, seed=17):
            lab.say(f"FAILED phase 1 {s}")
            return False
        drop_checkpoints(lab, folder)
    return True


def run_pretrained_grid(lab: Lab, gid: str) -> bool:
    title, arms, seeds, _ = GRIDS[gid]

    def cleanup() -> None:
        for s in lab.spk:
            lab.trim_model(f"big-c-pre-{s}")

    return run_grid(lab, gid, title, arms(), seeds, prepare=lambda: ensure_pretrained_models(lab), cleanup=cleanup)


def custom_stage(spec: dict):
    def run(lab: Lab) -> bool:
        speakers = spec.get("speakers") or None
        return run_grid(lab, "c-" + safe(spec["name"]), spec.get("title") or spec["name"], custom_arms(spec),
                        [int(x) for x in spec.get("seeds", [17])], speakers)
    return run


SYN16 = "reports/lab-16-synthetic-realism"
SYNTH_PY = "scripts/asr/synth_dysarthria.py"
KEEP_BASELINE_WEIGHTS = ("1", "2", "5", "6", "10", "13", "15", "18", "19", "21", "24", "26", "27", "35")


def ensure_controls(self: Lab) -> bool:
    if self.have(f"{EX}/controls/train.jsonl") and self.have(f"{EX}/controls/test.jsonl"):
        return True
    if not self.have("data/public/torgo/TORGO/FC01"):
        self.say("MISSING the TORGO control speakers (FC, MC), about 2.5 GB: python lab.py doctor --fix --allow-download")
        return False
    self.say("making the controls export")
    return self.run([self.sc, "scripts/corpus/torgo_manifest.py", "--export", "--missing-only", "--only", "controls"])


Lab.ensure_controls = ensure_controls


def adf1_adapter(self: Lab, s: str) -> str:
    if self.rebuild:
        return f"models/lab-baselines/adapter-f1-{s}/adapter.pt"
    return ("models/parakeet-ft-f2-M02-adapter8-on-f1-seed17/adapter.pt" if s == "M02"
            else f"models/parakeet-ft-f2-{s}-adapter8-on-f1/adapter.pt")


Lab.adf1_adapter = adf1_adapter


def synth_severity(lab: Lab, speaker: str) -> tuple[float, str]:
    explicit = float(lab.cfg["synth_severity"])
    if explicit > 0:
        return explicit, "set by the synth_severity variable"
    try:
        calibration = json.loads((lab.root / SYN16 / "summary.json").read_text(encoding="utf-8"))["calibration"]
        entry = calibration.get(speaker) or calibration.get("pooled")
        if entry:
            return float(entry["severity"]), "calibrated by lab test 16"
    except (OSError, ValueError, KeyError):
        pass
    return 0.7, "default: lab test 16 has not run"


def synth_export(lab: Lab, speaker: str, base: str, tag: str, scale: float, only: bool = False) -> str | None:
    severity, why = synth_severity(lab, speaker)
    out = f"{EX}/synth-{tag}-{speaker}-s{severity:.2f}-x{scale:g}"
    lab.say(f"synthetic speech for {speaker}: severity {severity} ({why}), {scale:g} synthetic clips per clip of {Path(base).name}"
            + (", synthetic only" if only else ""))
    cmd = [lab.sc, SYNTH_PY, "build", "--controls", f"{EX}/controls", "--target", base, "--out", out,
           "--severity", str(severity), "--scale", str(scale), "--seed", "17", *(["--only-synthetic"] if only else [])]
    return out if lab.run(cmd) else None


def fit_or_reuse(lab: Lab, export: str, name: str, recipe: str, extra: list[str] | None = None) -> str | None:
    folder = f"models/parakeet-ft-{name}"
    if not lab.dry and lab.exists(f"{folder}/parakeet-ft.nemo"):
        lab.say(f"reusing {folder}")
        return name
    if lab.exists(folder) and not lab.dry:
        lab.aside(folder)
    return lab.fit(export, name, recipe, extra)


def stage_16(lab: Lab) -> bool:
    n = SYN16
    val = f"{EX}/synth-validation"
    severities = [float(x) for x in str(lab.cfg["synth_severities"]).split()]
    lab.say(f"######## 16: healthy recordings made to sound dysarthric, and what the stock model does with them, started {now()}")
    if not lab.need_free(f"{n}/summary.json"):
        return False
    for s in lab.spk:
        if not lab.need_have(lab.stock_h(s) + "/results.json"):
            return False
    if not lab.ensure_controls() or not lab.act():
        return False
    if not lab.run([lab.sc, SYNTH_PY, "validate-set", "--controls", f"{EX}/controls", "--out", val, "--severities",
                    *[f"{x:g}" for x in severities], "--limit", "150", "--seed", "17", "--target", *[f"{EX}/{s}" for s in lab.spk]]):
        lab.say("FAILED building the synthetic validation clips")
        return False
    failed: list[str] = []
    for label in ["clean", *[f"s{x:.2f}" for x in severities]]:
        hyp, folder = f"{n}/{label}.jsonl", f"{val}/{label}"
        if not lab.dry and lab.exists(hyp):
            lab.say(f"reusing {hyp}")
        elif not lab.run([lab.gpu_py, "scripts/asr/nemo_hyp.py", "--stock", "--manifest", f"{folder}/test.jsonl", "--audio-root", folder, "--out", hyp]):
            failed.append(f"decode:{label}")
            continue
        if lab.dry or not lab.exists(f"{n}/{label}/results.json"):
            if not lab.run([lab.sc, "scripts/asr/evaluate.py", "--manifest", f"{folder}/test.jsonl", "--audio-root", folder, "--hyp", hyp,
                            "--group-by", "kind", "--out", f"{n}/{label}"]):
                failed.append(f"score:{label}")
    summary_lines(lab, failed)
    if not lab.run([lab.sc, SYNTH_PY, "summarize", "--results", n, "--real", *[f"{s}={lab.stock_h(s)}" for s in lab.spk], "--out", f"{n}/summary.json"]):
        lab.say("FAILED summarising")
        failed.append("summary")
    else:
        lab.track_table(f"{n}/summary.json")
    lab.say("Reading it: the stock model's WER on healthy recordings made to sound dysarthric, by severity, against its WER on each real speaker.")
    lab.say("`calibration` in summary.json gives the severity at which the two meet; lab tests 17 and 18 use it. Matching WER does not make the")
    lab.say("sound the same: it says the stock model finds the synthetic clips about as hard, nothing more.")
    lab.say(f"16 finished {now()}")
    return not failed


def stage_17(lab: Lab) -> bool:
    n = "reports/lab-17-synthetic-training"
    lab.say(f"######## 17: does training on synthetic dysarthric speech help, started {now()}")
    for s in lab.spk:
        if not lab.need_have(f"{EX}/{s}/train.jsonl", lab.stock_h(s) + "/hypotheses.jsonl", lab.wb_h(s) + "/hypotheses.jsonl",
                             lab.ad_h(s) + "/hypotheses.jsonl"):
            return False
    if not lab.need_free(f"{n}/table.json"):
        return False
    if not lab.ensure_controls() or not lab.ensure_severe_exports() or not lab.act():
        return False
    scale, pre_scale = float(lab.cfg["synth_scale"]), float(lab.cfg["synth_pre_scale"])
    failed: list[str] = []

    def decoded(name: str) -> bool:
        return not lab.dry and lab.exists(f"{FT}/{name}-test-nemo.jsonl")

    for s in lab.spk:
        base = f"{EX}/{s}"
        lab.banner(f"{s}, arm 1: his own sessions plus synthetic speech, adapter after block 8")
        mix = f"lab-17-mix-{s}"
        if decoded(mix):
            lab.say(f"reusing {mix}")
        else:
            export = synth_export(lab, s, base, "mix", scale)
            if not export or not adapter_run(lab, s, mix, 17, 8, None, None, export=export):
                lab.say(f"FAILED {mix}")
                failed.append(f"mix:{s}")
        lab.banner(f"{s}, arm 2: synthetic speech only, then his own sessions")
        pre, then = f"lab-17-pre-{s}", f"lab-17-then-{s}"
        if decoded(then):
            lab.say(f"reusing {then}")
        else:
            export = synth_export(lab, s, base, "pre", pre_scale, only=True)
            fitted = fit_or_reuse(lab, export, pre, "wb") if export else None
            if not fitted:
                failed.append(f"pre:{s}")
            else:
                model = f"models/parakeet-ft-{fitted}/parakeet-ft.nemo"
                if not decoded(pre) and not lab.decode(s, pre, ["--nemo", model]):
                    failed.append(f"decode:pre:{s}")
                done = fit_or_reuse(lab, base, then, "f2", ["--base", model])
                if done and lab.decode(s, then, ["--nemo", f"models/parakeet-ft-{done}/parakeet-ft.nemo"]):
                    lab.trim_model(done)
                    lab.trim_model(fitted)
                else:
                    failed.append(f"then:{s}")
        lab.banner(f"{s}, arm 3: the other severe speakers plus synthetic speech, then his own sessions")
        spre, sthen = f"lab-17-sevpre-{s}", f"lab-17-sevthen-{s}"
        if decoded(sthen):
            lab.say(f"reusing {sthen}")
        else:
            export = synth_export(lab, s, f"{EX}/loso-severe-{s}", "sev", 1.0)
            fitted = fit_or_reuse(lab, export, spre, "f1") if export else None
            if not fitted:
                failed.append(f"sevpre:{s}")
            else:
                model = f"models/parakeet-ft-{fitted}/parakeet-ft.nemo"
                if not decoded(spre) and not lab.decode(s, spre, ["--nemo", model]):
                    failed.append(f"decode:sevpre:{s}")
                done = fit_or_reuse(lab, base, sthen, "f2", ["--base", model])
                if done and lab.decode(s, sthen, ["--nemo", f"models/parakeet-ft-{done}/parakeet-ft.nemo"]):
                    lab.trim_model(done)
                    lab.trim_model(fitted)
                else:
                    failed.append(f"sevthen:{s}")
    summary_lines(lab, failed)
    systems = ["--system", lab.sysfn("stock", lab.stock_h), "--system", lab.sysfn("whole-block", lab.wb_h),
               "--system", lab.sysfn("adapter", lab.ad_h)]
    if all(lab.have(f"{FT}/lab-11-then-{s}-test-nemo.jsonl") for s in lab.spk):
        systems += ["--system", lab.sysfmt("lab11-severe-others-then-own", f"{FT}/lab-11-then-@-test-nemo.jsonl")]
    for label, pattern in (("synthetic-only", "lab-17-pre-@"), ("adapter+synthetic", "lab-17-mix-@"), ("synthetic-then-own", "lab-17-then-@"),
                           ("severe+synthetic-pre", "lab-17-sevpre-@"), ("severe+synthetic-then-own", "lab-17-sevthen-@")):
        if all(lab.have(f"{FT}/{pattern.replace('@', s)}-test-nemo.jsonl") for s in lab.spk):
            systems += ["--system", lab.sysfmt(label, f"{FT}/{pattern}-test-nemo.jsonl")]
    if not lab.pooled(*systems, "--baseline", "stock", "--out", f"{n}/table.json"):
        failed.append("table")
    lab.say("Compare with: adapter 208 (48.9%), whole-block 291 (68.5%), severe-others-then-own 248 (58.4%) of 425 wrong words on the first-round runs.")
    lab.say(f"17 finished {now()}")
    return not failed


def stage_18(lab: Lab) -> bool:
    n = "reports/lab-18-best-system-hardened"
    margin, kper = lab.cfg["margin"], lab.cfg["kper"]
    lab.say(f"######## 18: the adapter on the F-1 model, with silence protection and with synthetic speech, started {now()}")
    for s in lab.spk:
        if not lab.need_have(lab.f1_m(s), f"{EX}/{s}/train.jsonl", lab.stock_h(s) + "/hypotheses.jsonl", lab.f1_h(s) + "/hypotheses.jsonl",
                             lab.adf1_h(s) + "/hypotheses.jsonl"):
            return False
    if not lab.need_free(f"{n}/table.json", f"{n}/battery.json"):
        return False
    if not lab.ensure_controls() or not lab.ensure_battery() or not lab.act():
        return False
    silence = ["--empty-margin", str(margin), "--empty-clips-per-kind", str(kper)]
    failed: list[str] = []
    bsys: list[str] = []
    if lab.battery_decode("stock", ["--stock"]):
        bsys += ["--system", f"stock={FT}/lab-13-battery-stock.jsonl"]
    for s in lab.spk:
        if lab.battery_decode(f"adapter-f1-{s}", ["--nemo", lab.f1_m(s), "--adapter", lab.adf1_adapter(s)]) or not lab.dry and lab.exists(f"{FT}/lab-13-battery-adapter-f1-{s}.jsonl"):
            bsys += ["--system", f"adapter-f1-{s}={FT}/lab-13-battery-adapter-f1-{s}.jsonl"]
        else:
            failed.append(f"battery:plain:{s}")
    for s in lab.spk:
        for arm in ("sil", "synth", "synthsil"):
            name = f"lab-18-{arm}-{s}"
            if not lab.dry and lab.exists(f"{FT}/{name}-test-nemo.jsonl"):
                lab.say(f"reusing {name}")
                if lab.exists(f"{FT}/lab-13-battery-{arm}-{s}.jsonl"):
                    bsys += ["--system", f"{arm}-{s}={FT}/lab-13-battery-{arm}-{s}.jsonl"]
                continue
            lab.banner(f"{s}, {arm}")
            export = None
            if arm in ("synth", "synthsil"):
                export = synth_export(lab, s, f"{EX}/{s}", "mix", float(lab.cfg["synth_scale"]))
                if not export:
                    failed.append(f"{arm}:{s}")
                    continue
            extra = silence if arm in ("sil", "synthsil") else None
            if adapter_run(lab, s, name, 17, 8, lab.f1_m(s), extra, export=export, battery=f"{arm}-{s}"):
                if lab.exists(f"{FT}/lab-13-battery-{arm}-{s}.jsonl") or lab.dry:
                    bsys += ["--system", f"{arm}-{s}={FT}/lab-13-battery-{arm}-{s}.jsonl"]
            else:
                lab.say(f"FAILED {name}")
                failed.append(f"{arm}:{s}")
    summary_lines(lab, failed)
    systems = ["--system", lab.sysfn("stock", lab.stock_h), "--system", lab.sysfn("F-1", lab.f1_h), "--system", lab.sysfn("adapter-on-F-1", lab.adf1_h)]
    for arm in ("sil", "synth", "synthsil"):
        if all(lab.have(f"{FT}/lab-18-{arm}-{s}-test-nemo.jsonl") for s in lab.spk):
            systems += ["--system", lab.sysfmt(f"adapter-on-F-1+{arm}", f"{FT}/lab-18-{arm}-@-test-nemo.jsonl")]
    if not lab.pooled(*systems, "--baseline", "stock", "--out", f"{n}/table.json"):
        failed.append("table")
    lab.say("-- false inserts on clips with no speech (lab test 13's battery)")
    if not lab.run([lab.sc, "scripts/asr/battery.py", "score", "--manifest", f"{lab.BAT}/manifest.jsonl", *bsys, "--out", f"{n}/battery.json"]):
        failed.append("battery-score")
    lab.say(f"18 finished {now()}")
    return not failed


PHRASE24 = "reports/lab-24-phrase-scoring"


def stage_24(lab: Lab) -> bool:
    n = PHRASE24
    lab.say(f"######## 24: acoustic phrase scoring against the phrase list, started {now()}")
    systems = {
        "stock": (lambda s: ["--stock"], lab.stock_h),
        "F-1": (lambda s: ["--nemo", lab.f1_m(s)], lab.f1_h),
        "adapter-on-F-1": (lambda s: ["--nemo", lab.f1_m(s), "--adapter", lab.adf1_adapter(s)], lab.adf1_h),
    }
    for s in lab.spk:
        if not lab.need_have(f"{EX}/{s}/test.jsonl", lab.stock_h(s) + "/hypotheses.jsonl", lab.f1_m(s), lab.f1_h(s) + "/hypotheses.jsonl",
                             lab.adf1_adapter(s), lab.adf1_h(s) + "/hypotheses.jsonl"):
            return False
    if not lab.need_free(f"{n}/report/results.json") or not lab.act():
        return False
    pool = [f"{EX}/*/train.jsonl", f"{EX}/*/dev.jsonl", f"{EX}/*/test.jsonl"]
    failed: list[str] = []
    done: dict[str, dict[str, str]] = {}
    for label, (flags, hyps) in systems.items():
        for s in lab.spk:
            out = f"{n}/{s}-{safe(label)}.json"
            if lab.exists(out) and not lab.dry:
                lab.say(f"reusing {out}")
                done.setdefault(label, {})[s] = out
                continue
            lab.banner(f"{s}, {label}: score the nearest phrases against the audio")
            ok = lab.run([lab.gpu_py, "scripts/asr/phrase_score.py", *flags(s), "--manifest", f"{EX}/{s}/test.jsonl", "--audio-root", f"{EX}/{s}",
                          "--pool-glob", *pool, "--hyps", hyps(s), "--shortlist", "15", "--out", out])
            if ok:
                done.setdefault(label, {})[s] = out
            else:
                lab.say(f"FAILED {s} {label}")
                failed.append(f"{s}:{label}")
    summary_lines(lab, failed)
    args: list[str] = []
    for label, files in done.items():
        if len(files) == len(lab.spk):
            args += ["--system", label + "=" + ",".join(f"{s}:{p}" for s, p in files.items())]
    if not args and not lab.dry:
        lab.say("no system has scores for every speaker; no report")
        return False
    if not lab.run([lab.sc, "scripts/asr/phrase_report.py", *args, "--export-root", EX, "--out", f"{n}/report"]):
        failed.append("report")
    lab.say(f"24 finished {now()}")
    return not failed


INT8_VARIANTS = {"default": [], "matmul": ["--int8-ops", "matmul"], "perchannel": ["--per-channel"]}


def stage_19(lab: Lab) -> bool:
    n = "reports/lab-19-int8-cost"
    lab.say(f"######## 19: what int8 costs, per export recipe, on the whole-block and adapter-on-F-1 models, started {now()}")
    for s in lab.spk:
        if not lab.need_have(lab.wb_m(s), lab.f1_m(s), lab.adf1_adapter(s), lab.wb_h(s) + "/hypotheses.jsonl", lab.adf1_h(s) + "/hypotheses.jsonl"):
            return False
    missing = [f"{m}-{s}-{v}" for m in ("whole-block", "adapter-on-F-1") for s in lab.spk for v in INT8_VARIANTS
               if not lab.exists(f"{n}/{m}-{s}-{v}/results.json")]
    if missing and lab.exists(f"{n}/table.json"):
        lab.say(f"table.json exists but {len(missing)} recipe score(s) are missing ({' '.join(missing)}); rebuilding it")
        lab.aside(f"{n}/table.json")
    elif not lab.need_free(f"{n}/table.json"):
        return False
    if not lab.act():
        return False
    models = {"whole-block": (lambda s: ["--nemo", lab.wb_m(s)], lab.wb_h), "adapter-on-F-1": (lambda s: ["--nemo", lab.f1_m(s), "--adapter", lab.adf1_adapter(s)], lab.adf1_h)}
    failed: list[str] = []
    for s in lab.spk:
        for label, (flags, _ref) in models.items():
            for variant, extra in INT8_VARIANTS.items():
                tag = f"{safe(label)}-{s}-{variant}"
                report, export = f"{n}/{tag}", f"models/lab-19-export-{tag}"
                if not lab.dry and lab.exists(f"{report}/results.json"):
                    lab.say(f"reusing {report}")
                    continue
                lab.banner(f"{s}, {label}, int8 {variant}")
                if lab.exists(export) and not lab.dry:
                    lab.aside(export)
                if not lab.run([lab.gpu_py, "scripts/asr/export.py", *flags(s), "--out", export, *extra]):
                    failed.append(f"export:{tag}")
                    continue
                if not lab.run([lab.sc, "scripts/asr/evaluate.py", "--manifest", f"{EX}/{s}/test.jsonl", "--audio-root", f"{EX}/{s}",
                                "--model-dir", export, "--group-by", "kind", "--out", report]):
                    failed.append(f"score:{tag}")
                    continue
                if lab.trim_on() and not lab.dry:
                    shutil.rmtree(lab.root / export, ignore_errors=True)
    summary_lines(lab, failed)
    systems = ["--system", lab.sysfn("stock", lab.stock_h)]
    for label, (_flags, ref) in models.items():
        systems += ["--system", lab.sysfn(f"{label} NeMo fp32", ref)]
        for variant in INT8_VARIANTS:
            if all(lab.have(f"{n}/{safe(label)}-{s}-{variant}/results.json") for s in lab.spk):
                systems += ["--system", lab.sysfmt(f"{label} int8 {variant}", f"{n}/{safe(label)}-@-{variant}")]
    if not lab.pooled(*systems, "--baseline", "stock", "--out", f"{n}/table.json"):
        failed.append("table")
    lab.say("Read it against the 'NeMo fp32' row of the same model: the difference is what the device's int8 file costs. The sherpa-onnx")
    lab.say("decode is a different program from NeMo, so a small gap can also be the decoder, not the quantisation.")
    lab.say(f"19 finished {now()}")
    return not failed


CV_PY = "scripts/asr/cv_folds.py"
TTS_PY = "scripts/asr/tts_synth.py"
TTA_PY = "scripts/asr/tta.py"
SEL_PY = "scripts/asr/hyp_selector.py"
CV20 = "reports/lab-20-grouped-cv"
TTS21 = "reports/lab-21-tts-synthetic"
TTA22 = "reports/lab-22-test-time-adaptation"
SEL23 = "reports/lab-23-hypothesis-selector"


def cv_arms(lab: Lab) -> list[tuple[str, int, int, str]]:
    arms = []
    for token in str(lab.cfg["cv_arms"]).split():
        parts = token.split(":")
        if len(parts) != 4 or not parts[1].isdigit() or not parts[2].isdigit() or parts[3] not in ("f1", "stock") or not parts[0]:
            raise ValueError(f"cv_arms entry {token!r} should look like adf1:8:15:f1 (label:block:epochs:f1 or stock)")
        arms.append((parts[0], int(parts[1]), int(parts[2]), parts[3]))
    if len(arms) < 2:
        raise ValueError("cv_arms needs a control and at least one other recipe")
    if len({a[0] for a in arms}) != len(arms):
        raise ValueError("cv_arms labels must differ")
    return arms


def hyps_complete(lab: Lab, out: str, manifest: str) -> bool:
    try:
        have = sum(1 for line in (lab.root / out).read_text(encoding="utf-8").splitlines() if line.strip())
        want = sum(1 for line in (lab.root / manifest).read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        return False
    return want > 0 and have == want


def train_adapter_arm(lab: Lab, export: str, name: str, blocks: int, epochs: int, base_model: str | None, seed: int = 17,
                      extra: list[str] | None = None) -> bool:
    folder = f"models/parakeet-ft-{name}"
    if not lab.dry and lab.exists(f"{folder}/adapter.pt"):
        lab.say(f"reusing {folder}")
        return True
    if lab.exists(folder) and not lab.dry:
        lab.aside(folder)
    flags = merge_flags(Lab.RECIPES["ad8"], ["--adapter-after-blocks", str(blocks), "--epochs", str(epochs)] + (["--base", base_model] if base_model else []))
    flags = merge_flags(flags, extra or [])
    if lab.train(export, name, flags, None, seed=seed):
        return True
    if lab.dry or lab.why_failed().category not in ("oom", "killed"):
        return False
    lab.say(f"FAILED {name} with batch 2; retrying with batch 1 x 16 accumulation")
    if lab.exists(folder):
        lab.aside(folder)
    return lab.train(export, name, flags, None, batch=1, accum=16, seed=seed)


def decode_manifest(lab: Lab, model_flags: list[str], manifest: str, audio_root: str, out: str) -> bool:
    if not lab.dry and lab.exists(out):
        if hyps_complete(lab, out, manifest):
            lab.say(f"reusing {out}")
            return True
        lab.aside(out)
    cmd = [lab.gpu_py, "scripts/asr/nemo_hyp.py", *model_flags, "--manifest", manifest, "--audio-root", audio_root, "--out", out]
    if lab.run(cmd):
        return True
    if not lab.dry and lab.why_failed().category in ("oom", "killed"):
        lab.say("decode ran out of memory; trying again with batch 1")
        if lab.exists(out):
            lab.aside(out)
        return lab.run(cmd + ["--batch", "1"])
    return False


def stage_20(lab: Lab) -> bool:
    n = CV20
    folds, gain = int(lab.cfg["cv_folds"]), float(lab.cfg["cv_min_gain"])
    arms = cv_arms(lab)
    control = arms[0][0]
    lab.say(f"######## 20: grouped cross-validation of {len(arms)} adapter recipes, {folds} folds, started {now()}")
    lab.say("Each recipe is trained once per fold on the other folds' prompts and decodes the held-out fold. Choosing uses every labelled clip, not the 112-word dev set.")
    for s in lab.spk:
        paths = [f"{EX}/{s}/train.jsonl", f"{EX}/{s}/dev.jsonl", f"{EX}/{s}/test.jsonl"]
        if any(a[3] == "f1" for a in arms):
            paths.append(lab.f1_m(s))
        if not lab.need_have(*paths):
            return False
    if not lab.need_free(f"{n}/selection.json", f"{n}/table.json") or not lab.act():
        return False
    for s in lab.spk:
        if not (lab.run([lab.sc, CV_PY, "make", "--export", f"{EX}/{s}", "--speaker", s, "--folds", str(folds)])
                and lab.run([lab.sc, CV_PY, "check", "--speaker", s, "--folds", str(folds)])):
            lab.say(f"FAILED building the folds for {s}; python lab.py doctor --fix repairs missing audio")
            return False
    failed: list[str] = []
    for s in lab.spk:
        for label, blocks, epochs, base in arms:
            base_model = lab.f1_m(s) if base == "f1" else None
            for k in range(folds):
                name, export = f"lab-20-{label}-{s}-f{k}", f"{EX}/cv-{s}-f{k}"
                hyp = f"{FT}/{name}-test-nemo.jsonl"
                lab.banner(f"{s}, {label}, fold {k + 1} of {folds}")
                if not lab.dry and hyps_complete(lab, hyp, f"{export}/test.jsonl"):
                    lab.say(f"reusing {hyp}")
                    continue
                if not train_adapter_arm(lab, export, name, blocks, epochs, base_model):
                    lab.say(f"FAILED {name}")
                    failed.append(f"{label}:{s}:f{k}")
                    continue
                if not decode_manifest(lab, lab.adapter_flags(name, base_model), f"{export}/test.jsonl", export, hyp):
                    failed.append(f"decode:{label}:{s}:f{k}")
                    continue
                lab.trim_model(name)
    if failed:
        summary_lines(lab, failed)
        lab.say("Selection needs every fold of every recipe; run this stage again and finished folds are reused.")
        return False
    for s in lab.spk:
        for label, *_ in arms:
            out = f"{n}/oof/{label}-{s}.jsonl"
            if not lab.dry and lab.exists(out):
                continue
            if not lab.run([lab.sc, CV_PY, "merge", "--speaker", s, "--folds", str(folds), "--arm", f"{label}={FT}/lab-20-{label}-{s}-f{{fold}}-test-nemo.jsonl",
                            "--out", f"{n}/oof"]):
                failed.append(f"merge:{label}:{s}")
    pick = [lab.sc, CV_PY, "select", "--speaker", *lab.spk, "--folds", str(folds), "--control", control, "--min-gain-points", f"{gain:g}",
            "--out", f"{n}/selection.json"]
    for label, *_ in arms:
        pick += ["--arm", f"{label}={FT}/lab-20-{label}-{{speaker}}-f{{fold}}-test-nemo.jsonl"]
    if not lab.run(pick):
        failed.append("select")
        summary_lines(lab, failed)
        return False
    selected = arms[1][0]
    if not lab.dry:
        selected = json.loads((lab.root / f"{n}/selection.json").read_text(encoding="utf-8"))["selected"]
    lab.say(f"Chosen on the held-out folds: {selected}" + ("  (the control, so nothing more is trained)" if selected == control else ""))
    systems = ["--system", lab.sysfn("stock", lab.stock_h), "--system", lab.sysfn("F-1", lab.f1_h), "--system", lab.sysfn("adapter-on-F-1 (existing)", lab.adf1_h)]
    if selected != control:
        label, blocks, epochs, base = next(a for a in arms if a[0] == selected)
        for s in lab.spk:
            name = f"lab-20-final-{label}-{s}"
            if not lab.dry and lab.exists(f"{FT}/{name}-test-nemo.jsonl"):
                lab.say(f"reusing {name}")
                continue
            lab.banner(f"{s}, the chosen recipe {label} on all of his labelled clips, test scored once")
            if not adapter_run(lab, s, name, 17, blocks, lab.f1_m(s) if base == "f1" else None, None, epochs=epochs):
                failed.append(f"final:{s}")
        if all(lab.have(f"{FT}/lab-20-final-{label}-{s}-test-nemo.jsonl") for s in lab.spk):
            systems += ["--system", lab.sysfmt(f"chosen-{label}", f"{FT}/lab-20-final-{label}-@-test-nemo")]
    summary_lines(lab, failed)
    if not lab.pooled(*systems, "--baseline", "stock", "--out", f"{n}/table.json"):
        failed.append("table")
    lab.track_table(f"{n}/selection.json")
    lab.say("Read selection.json first: it ranks the recipes by wrong words on held-out prompts and keeps the control unless another beats it clearly.")
    lab.say("The out-of-fold answers in oof/ are what lab test 23's selector learns from.")
    lab.say(f"20 finished {now()}")
    return not failed


def tts_backend_ready(lab: Lab) -> bool:
    if lab.dry:
        return True
    import tts_synth
    status = tts_synth.available(str(lab.cfg["tts_voices"]) or None)
    want = str(lab.cfg["tts_backend"])
    usable = [k for k, (ok, _d) in status.items() if ok and k != "fake"]
    if (want == "auto" and usable) or (want in status and want != "fake" and status[want][0]):
        return True
    for k, (_ok, detail) in status.items():
        if k != "fake":
            lab.say(f"  {k}: {detail}")
    lab.say("No text-to-speech engine is ready. Install one, then run this stage again: `sudo apt install espeak-ng` (small, robotic voices),")
    lab.say("or `pip install piper-tts` and put .onnx voices in a folder named in the tts_voices variable (better voices). `python scripts/asr/tts_synth.py backends` lists what it finds.")
    return False


def stage_21(lab: Lab) -> bool:
    n = TTS21
    scale, backend, voices = float(lab.cfg["synth_scale"]), str(lab.cfg["tts_backend"]), str(lab.cfg["tts_voices"])
    extra = str(lab.cfg["tts_extra_text"])
    lab.say(f"######## 21: training on text-to-speech made to sound dysarthric, started {now()}")
    for s in lab.spk:
        if not lab.need_have(lab.f1_m(s), f"{EX}/{s}/train.jsonl", f"{EX}/{s}/dev.jsonl", f"{EX}/{s}/test.jsonl", f"{EX}/loso-{s}/train.jsonl",
                             f"{EX}/loso-{s}/dev.jsonl", lab.stock_h(s) + "/hypotheses.jsonl", lab.f1_h(s) + "/hypotheses.jsonl",
                             lab.adf1_h(s) + "/hypotheses.jsonl"):
            return False
    if extra and not lab.need_have(extra):
        return False
    if not lab.need_free(f"{n}/table.json", f"{n}/battery.json") or not tts_backend_ready(lab) or not lab.ensure_battery() or not lab.act():
        return False
    failed: list[str] = []
    bsys: list[str] = []
    if lab.battery_decode("stock", ["--stock"]):
        bsys += ["--system", f"stock={FT}/lab-13-battery-stock.jsonl"]
    for s in lab.spk:
        if lab.battery_decode(f"adapter-f1-{s}", ["--nemo", lab.f1_m(s), "--adapter", lab.adf1_adapter(s)]) or not lab.dry and lab.exists(f"{FT}/lab-13-battery-adapter-f1-{s}.jsonl"):
            bsys += ["--system", f"adapter-f1-{s}={FT}/lab-13-battery-adapter-f1-{s}.jsonl"]
    for s in lab.spk:
        tts_dir = f"{EX}/lab-21-tts-{s}"
        lab.banner(f"{s}: speech from the synthesiser, using other speakers' prompts and never his dev or test prompts")
        cmd = [lab.sc, TTS_PY, "make", "--backend", backend, "--out", tts_dir, "--texts-from", f"{EX}/loso-{s}/train.jsonl", f"{EX}/loso-{s}/dev.jsonl",
               "--exclude-from", f"{EX}/{s}/dev.jsonl", f"{EX}/{s}/test.jsonl", "--variants", str(int(lab.cfg["tts_variants"])),
               "--max-clips", str(int(lab.cfg["tts_max_clips"]))]
        if voices:
            cmd[cmd.index("--out"):cmd.index("--out")] = ["--voice-dir", voices]
        if extra:
            cmd += ["--extra-text", extra]
        if not lab.run(cmd):
            failed.append(f"tts:{s}")
            continue
        for arm in ("tts", "ttslow"):
            name = f"lab-21-{arm}-{s}"
            if not lab.dry and lab.exists(f"{FT}/{name}-test-nemo.jsonl"):
                lab.say(f"reusing {name}")
                if lab.exists(f"{FT}/lab-13-battery-{arm}-{s}.jsonl"):
                    bsys += ["--system", f"{arm}-{s}={FT}/lab-13-battery-{arm}-{s}.jsonl"]
                continue
            severity = synth_severity(lab, s)[0] if arm == "tts" else 0.2
            export = f"{EX}/lab-21-{arm}-mix-{s}"
            lab.banner(f"{s}, {arm}: severity {severity:g}, {scale:g} synthetic clips per clip of his own")
            if not lab.run([lab.sc, SYNTH_PY, "build", "--controls", tts_dir, "--target", f"{EX}/{s}", "--out", export, "--severity", f"{severity:g}",
                            "--scale", f"{scale:g}", "--seed", "17"]):
                failed.append(f"mix:{arm}:{s}")
                continue
            if adapter_run(lab, s, name, 17, 8, lab.f1_m(s), None, export=export, battery=f"{arm}-{s}"):
                if lab.dry or lab.exists(f"{FT}/lab-13-battery-{arm}-{s}.jsonl"):
                    bsys += ["--system", f"{arm}-{s}={FT}/lab-13-battery-{arm}-{s}.jsonl"]
            else:
                lab.say(f"FAILED {name}")
                failed.append(f"{arm}:{s}")
    summary_lines(lab, failed)
    systems = ["--system", lab.sysfn("stock", lab.stock_h), "--system", lab.sysfn("F-1", lab.f1_h), "--system", lab.sysfn("adapter-on-F-1", lab.adf1_h)]
    for arm, label in (("tts", "adapter-on-F-1+TTS"), ("ttslow", "adapter-on-F-1+TTS (barely degraded)")):
        if all(lab.have(f"{FT}/lab-21-{arm}-{s}-test-nemo.jsonl") for s in lab.spk):
            systems += ["--system", lab.sysfmt(label, f"{FT}/lab-21-{arm}-@-test-nemo")]
    if all(lab.have(f"{FT}/lab-18-synth-{s}-test-nemo.jsonl") for s in lab.spk):
        systems += ["--system", lab.sysfmt("adapter-on-F-1+recorded controls (lab 18)", f"{FT}/lab-18-synth-@-test-nemo")]
    if not lab.pooled(*systems, "--baseline", "stock", "--out", f"{n}/table.json"):
        failed.append("table")
    if not lab.run([lab.sc, "scripts/asr/battery.py", "score", "--manifest", f"{lab.BAT}/manifest.jsonl", *bsys, "--out", f"{n}/battery.json"]):
        failed.append("battery-score")
    lab.say("Read it against the adapter-on-F-1 row and, when it ran, the lab 18 row (the same recipe with degraded recordings of healthy speakers).")
    lab.say("'barely degraded' separates what the extra words and voices give from what the degrading gives.")
    lab.say(f"21 finished {now()}")
    return not failed


def export_layout_ok(lab: Lab, export: str) -> bool:
    for split in ("train", "dev"):
        path = lab.root / export / f"{split}.jsonl"
        if not path.is_file():
            return False
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and not str(json.loads(line)["audio"]).startswith(f"{split}/"):
                return False
    return True


def tta_loop(lab: Lab, s: str, tag: str, manifests: str, base_model: str, base_h: str, agree: list[str], by_chunk: str,
             chunks: int, order: str, agree_min: float, max_pseudo: float, failed: list[str], seed: int = 17, lr: str | None = None,
             labels: LmLabels | None = None) -> bool:
    test = f"{EX}/{s}/test.jsonl"
    common = ["--test-manifest", test, "--n", str(chunks), "--order", order]
    answers = {0: base_h}
    extra = ["--lr", lr] if lr else None
    for k in range(1, chunks):
        manifest, out = f"{FT}/{manifests}-chunk{k}-manifest.jsonl", f"{FT}/{tag}-chunk{k}-answers.jsonl"
        answers[k] = out
        lab.banner(f"{s}, {tag}, round {k} of {chunks - 1}")
        if not lab.dry and hyps_complete(lab, out, manifest) and (labels is None or labels.ready(k)):
            lab.say(f"reusing {out}")
            continue
        if lab.dry or not lab.exists(manifest):
            if not lab.run([lab.sc, TTA_PY, "manifest", "--manifest", test, "--n", str(chunks), "--order", order, "--index", str(k), "--out", manifest]):
                failed.append(f"manifest:{tag}:{k}")
                return False
        export = f"{EX}/{tag}-r{k}"
        if not lab.dry and lab.exists(export) and not export_layout_ok(lab, export):
            lab.say(f"{export} was built with audio paths train.py refuses; rebuilding it")
            shutil.rmtree(lab.root / export)
        if lab.dry or not lab.exists(export):
            if labels is not None and not labels.round(k):
                failed.append(f"labels:{tag}:{k}")
                return False
            build = [lab.sc, TTA_PY, "build", "--train-export", f"{EX}/{s}", *common, "--round", str(k),
                     *(labels.chunk_hyps(k) if labels is not None else [x for j in range(k) for x in ("--chunk-hyps", f"{j}={answers[j]}")]),
                     "--agree", *agree, "--agree-min", f"{agree_min:g}", "--max-pseudo-share", f"{max_pseudo:g}",
                     *(labels.withhold(k) if labels is not None else []), "--out", export]
            if not lab.run(build):
                failed.append(f"build:{tag}:{k}")
                return False
        name = f"{tag}-r{k}"
        if not train_adapter_arm(lab, export, name, 8, 15, base_model, seed=seed, extra=extra):
            failed.append(f"train:{tag}:{k}")
            return False
        if not decode_manifest(lab, lab.adapter_flags(name, base_model), manifest, f"{EX}/{s}", out):
            failed.append(f"decode:{tag}:{k}")
            return False
        if labels is not None and k < chunks - 1 and not labels.prepare(k, name, manifest, out):
            failed.append(f"pseudo-labels:{tag}:{k}")
            return False
        lab.trim_model(name)
    merged = f"{FT}/{tag}-tta-test-nemo.jsonl"
    if lab.dry or not lab.exists(merged):
        if not lab.run([lab.sc, TTA_PY, "merge", *common, *[x for j in range(chunks) for x in ("--chunk-hyps", f"{j}={answers[j]}")], "--out", merged]):
            failed.append(f"merge:{tag}")
            return False
    if lab.dry or not lab.exists(f"{FT}/{tag}-tta-test-nemo/results.json"):
        if not lab.score(s, f"{tag}-tta"):
            failed.append(f"score:{tag}")
            return False
    if lab.dry or not lab.exists(by_chunk):
        lab.run([lab.sc, TTA_PY, "score", *common, "--system", f"base={base_h}", "--system", f"tta={merged}", "--out", by_chunk])
    return True


def stage_22(lab: Lab) -> bool:
    n = TTA22
    chunks, order = int(lab.cfg["tta_chunks"]), str(lab.cfg["tta_order"])
    lab.say(f"######## 22: test-time adaptation on his own recordings, {chunks} chunks, started {now()}")
    lab.say("No transcripts: each round trains on his labelled clips plus the system's own earlier answers that other systems agree with.")
    for s in lab.spk:
        if not lab.need_have(lab.f1_m(s), f"{EX}/{s}/train.jsonl", f"{EX}/{s}/dev.jsonl", f"{EX}/{s}/test.jsonl", lab.f1_h(s) + "/hypotheses.jsonl",
                             lab.adf1_h(s) + "/hypotheses.jsonl"):
            return False
    if not lab.need_free(f"{n}/table.json") or not lab.act():
        return False
    failed: list[str] = []
    for s in lab.spk:
        tta_loop(lab, s, f"lab-22-{s}", f"lab-22-{s}", lab.f1_m(s), lab.adf1_h(s), [lab.f1_h(s), lab.adf1_h(s)], f"{n}/by-chunk-{s}.json",
                 chunks, order, float(lab.cfg["tta_agree_min"]), float(lab.cfg["tta_max_pseudo"]), failed)
    summary_lines(lab, failed)
    systems = ["--system", lab.sysfn("stock", lab.stock_h), "--system", lab.sysfn("F-1", lab.f1_h), "--system", lab.sysfn("adapter-on-F-1", lab.adf1_h)]
    if all(lab.have(f"{FT}/lab-22-{s}-tta-test-nemo/results.json") for s in lab.spk):
        systems += ["--system", lab.sysfmt("adapter-on-F-1+TTA", f"{FT}/lab-22-@-tta-test-nemo")]
    if not lab.pooled(*systems, "--baseline", "stock", "--out", f"{n}/table.json"):
        failed.append("table")
    lab.say("Chunk 0 is the base system's own answers, so the two rows differ only from chunk 1 on. by-chunk-<speaker>.json shows whether later chunks improve.")
    lab.say("This is pseudo-label self-training; adapting on each clip with an entropy loss is not built.")
    lab.say(f"22 finished {now()}")
    return not failed


TTA25 = "reports/lab-25-tta-on-pretrained"
TTA25_LR = "6e-4"
TTA25_SEEDS = (17, 18, 19)
PRE_AD8 = "big-p-pre-ad8-lr6e-4"


def pretrained_adapter_answers(lab: Lab, s: str, seed: int) -> str | None:
    name = f"{PRE_AD8}-{s}-s{seed}"
    folder = f"{FT}/{name}-test-nemo"
    if not lab.dry and lab.exists(f"{folder}/results.json"):
        return folder
    lab.banner(f"{s}, seed {seed}: the pretrained adapter itself (stage P's arm)")
    ok = (train_adapter_arm(lab, f"{EX}/{s}", name, 8, 15, pretrained_model(s), seed=seed, extra=["--lr", TTA25_LR])
          and lab.decode(s, name, lab.adapter_flags(name, pretrained_model(s))))
    lab.trim_model(name)
    return folder if ok else None


def fresh_pretrained_adapter(lab: Lab, s: str) -> tuple[str, str] | None:
    name = f"lab-pre-ad8-{s}"
    folder, hyps = f"models/parakeet-ft-{name}", f"{FT}/{name}-test-nemo"
    had_weights = lab.exists(f"{folder}/adapter.pt")
    if not train_adapter_arm(lab, f"{EX}/{s}", name, 8, 15, pretrained_model(s), seed=17, extra=["--lr", TTA25_LR]):
        return None
    if not had_weights and not lab.dry:
        for stale in (f"{hyps}.jsonl", f"{hyps}.provenance.json", hyps):
            if lab.exists(stale):
                lab.aside(stale)
    if lab.dry or not lab.exists(f"{hyps}/results.json"):
        if not lab.decode(s, name, lab.adapter_flags(name, pretrained_model(s))):
            return None
    return name, hyps


def stage_25(lab: Lab) -> bool:
    n = TTA25
    chunks, order = int(lab.cfg["tta_chunks"]), str(lab.cfg["tta_order"])
    lab.say(f"######## 25: test-time adaptation on the pretrained adapter, {chunks} chunks, seeds {' '.join(map(str, TTA25_SEEDS))}, started {now()}")
    lab.say("No transcripts: each round trains on his labelled clips plus the system's own earlier answers that F-1 agrees with.")
    for s in lab.spk:
        if not lab.need_have(f"{EX}/{s}/train.jsonl", f"{EX}/{s}/dev.jsonl", f"{EX}/{s}/test.jsonl", lab.stock_h(s) + "/hypotheses.jsonl",
                             lab.f1_h(s) + "/hypotheses.jsonl"):
            return False
    if not lab.need_free(f"{n}/table.json") or not lab.act():
        return False
    if not ensure_pretrained_models(lab):
        return False
    failed: list[str] = []
    for s in lab.spk:
        for seed in TTA25_SEEDS:
            base_h = pretrained_adapter_answers(lab, s, seed)
            if base_h is None:
                failed.append(f"base:{s}-s{seed}")
                continue
            tta_loop(lab, s, f"lab-25-{s}-s{seed}", f"lab-25-{s}", pretrained_model(s), base_h, [lab.f1_h(s), base_h], f"{n}/by-chunk-{s}-s{seed}.json",
                     chunks, order, float(lab.cfg["tta_agree_min"]), float(lab.cfg["tta_max_pseudo"]), failed, seed=seed, lr=TTA25_LR)
        lab.trim_model(f"big-c-pre-{s}")
    summary_lines(lab, failed)
    systems = ["--system", lab.sysfn("stock", lab.stock_h), "--system", lab.sysfn("F-1", lab.f1_h)]
    if all(lab.have(lab.adf1_h(s) + "/hypotheses.jsonl") for s in lab.spk):
        systems += ["--system", lab.sysfn("adapter-on-F-1", lab.adf1_h)]
    if all(lab.have(f"{FT}/lab-22-{s}-tta-test-nemo/results.json") for s in lab.spk):
        systems += ["--system", lab.sysfmt("adapter-on-F-1+TTA (lab 22)", f"{FT}/lab-22-@-tta-test-nemo")]
    for seed in TTA25_SEEDS:
        for label, pattern in ((f"pretrained adapter 6e-4 s{seed}", f"{FT}/{PRE_AD8}-@-s{seed}-test-nemo"),
                               (f"pretrained adapter+TTA s{seed}", f"{FT}/lab-25-@-s{seed}-tta-test-nemo")):
            if lab.dry or all(lab.have(pattern.replace("@", s) + "/results.json") for s in lab.spk):
                systems += ["--system", lab.sysfmt(label, pattern)]
    if not lab.pooled(*systems, "--baseline", "stock", "--out", f"{n}/table.json"):
        failed.append("table")
    seed_means(lab, TTA25_SEEDS, (("pretrained adapter 6e-4", f"{FT}/{PRE_AD8}-@-s%d-test-nemo"), ("pretrained adapter+TTA", f"{FT}/lab-25-@-s%d-tta-test-nemo")))
    lab.say("Chunk 0 is the pretrained adapter's own answers, so each pair of rows differs only from chunk 1 on; by-chunk-<speaker>-s<seed>.json shows the later chunks.")
    lab.say("A change counts only when at least 2 of 3 speakers move the same way and the pooled WER moves by more than about 4.5 points (19 words).")
    lab.say(f"25 finished {now()}")
    return not failed


LM38 = "reports/lab-38-lm-pseudo-labels"
LM38_ARMS = ("all", "top", "margin")


def lm38_settings(lab: Lab) -> dict:
    arms = str(lab.cfg["lm_label_arms"]).split()
    if not arms or len(set(arms)) != len(arms) or any(a not in LM38_ARMS for a in arms):
        raise ValueError(f"lm_label_arms is one or more of {' '.join(LM38_ARMS)}, each once")
    tag = str(lab.cfg["lm_label_model"]).strip()
    if not tag or " " in tag:
        raise ValueError("lm_label_model is one language-model tag (python scripts/asr/lm_score.py list)")
    import lm_score

    lm_score.get(tag)
    cfg = {"arms": arms, "tag": tag, "weight": float(lab.cfg["lm_weight"]), "bonus": float(lab.cfg["lm_bonus"]),
           "share": float(lab.cfg["lm_top_share"]), "margin": float(lab.cfg["lm_min_margin"]), "threads": int(lab.cfg["lm_threads"])}
    if cfg["weight"] < 0 or cfg["bonus"] < 0:
        raise ValueError("lm_weight and lm_bonus cannot be negative")
    if not 0 < cfg["share"] <= 1:
        raise ValueError("lm_top_share is above 0 and at most 1")
    if cfg["margin"] < 0:
        raise ValueError("lm_min_margin cannot be negative")
    return cfg


def lm38_key(arm: str, cfg: dict) -> str:
    return {"all": "lm-all", "top": f"lm-top{100 * cfg['share']:g}", "margin": f"lm-margin{cfg['margin']:g}"}[arm]


def lm38_label(arm: str, cfg: dict) -> str:
    return {"all": "LM labels, every sentence", "top": f"LM labels, widest {100 * cfg['share']:g}% of margins",
            "margin": f"LM labels, margin of {cfg['margin']:g} nats or more"}[arm]


def lm38_chunk_picks(lab: Lab, cfg: dict, flags: list[str], manifest: str, s: str, plain: str, pool: str, picks: str) -> bool:
    if not lab.dry and lab.exists(picks):
        lab.say(f"reusing {picks}")
        return True
    scores, tag = f"{LM38}/scores/{cfg['tag']}.json", cfg["tag"]
    return (lm_pool(lab, flags, manifest, f"{EX}/{s}", pool, ["--no-reference"])
            and lab.run([lm_python(lab, tag), "scripts/asr/lm_score.py", "score", "--model", tag, "--pool", pool, "--threads", str(cfg["threads"]),
                         "--extend", "--out", scores])
            and lab.run([lab.sc, "scripts/asr/lm_pick.py", "pick", "--pool", pool, "--scores", scores, "--plain", plain,
                         "--weight", f"{cfg['weight']:g}", "--bonus", f"{cfg['bonus']:g}", "--out", picks]))


class LmLabels:
    def __init__(self, lab: Lab, s: str, seed: int, arm: str, cfg: dict, chunks: int, base_model: str, picks0: str):
        self.lab, self.s, self.seed, self.arm, self.cfg, self.chunks = lab, s, seed, arm, cfg, chunks
        self.base_model, self.picks0 = base_model, picks0
        self.key = lm38_key(arm, cfg)

    def picks(self, k: int) -> str:
        return self.picks0 if k == 0 else f"{LM38}/picks/{self.key}-{self.s}-s{self.seed}-c{k}.jsonl"

    def round_file(self, k: int) -> str:
        return f"{LM38}/labels/{self.key}-{self.s}-s{self.seed}-r{k}.jsonl"

    def ready(self, k: int) -> bool:
        return k >= self.chunks - 1 or self.lab.exists(self.picks(k))

    def chunk_hyps(self, k: int) -> list[str]:
        return [x for j in range(k) for x in ("--chunk-hyps", f"{j}={self.round_file(k)}")]

    def withhold(self, k: int) -> list[str]:
        return ["--withhold", self.round_file(k)]

    def round(self, k: int) -> bool:
        out = self.round_file(k)
        if not self.lab.dry and self.lab.exists(out):
            return True
        rule = {"all": ["--rule", "all"], "top": ["--rule", "top", "--share", f"{self.cfg['share']:g}"],
                "margin": ["--rule", "margin", "--min-margin", f"{self.cfg['margin']:g}"]}[self.arm]
        return self.lab.run([self.lab.sc, "scripts/asr/lm_pick.py", "filter", "--picks", *[self.picks(j) for j in range(k)], *rule, "--out", out])

    def prepare(self, k: int, name: str, manifest: str, answers: str) -> bool:
        pool = f"{LM38}/pools/{self.key}-{self.s}-s{self.seed}-c{k}.json"
        return lm38_chunk_picks(self.lab, self.cfg, self.lab.adapter_flags(name, self.base_model), manifest, self.s, answers, pool, self.picks(k))


def lm38_static(lab: Lab, s: str, seed: int, cfg: dict, order: str, chunks: int, picks0: str) -> str | None:
    name = f"{PRE_AD8}-{s}-s{seed}"
    folder = f"{FT}/{name}-test-nemo"
    have_answers = not lab.dry and lab.exists(f"{folder}/results.json")
    if have_answers and lab.exists(picks0):
        return folder
    lab.banner(f"{s}, seed {seed}: the pretrained adapter itself (stage P's arm) and chunk 0's pseudo-labels")
    flags = lab.adapter_flags(name, pretrained_model(s))
    if not train_adapter_arm(lab, f"{EX}/{s}", name, 8, 15, pretrained_model(s), seed=seed, extra=["--lr", TTA25_LR]):
        return None
    if not have_answers and not lab.decode(s, name, flags):
        return None
    ok = True
    if lab.dry or not lab.exists(picks0):
        manifest = f"{FT}/lab-38-{s}-chunk0-manifest.jsonl"
        ok = ((not lab.dry and lab.exists(manifest))
              or lab.run([lab.sc, TTA_PY, "manifest", "--manifest", f"{EX}/{s}/test.jsonl", "--n", str(chunks), "--order", order, "--index", "0",
                          "--out", manifest]))
        ok = ok and lm38_chunk_picks(lab, cfg, flags, manifest, s, folder, f"{LM38}/pools/static-{s}-s{seed}-c0.json", picks0)
    lab.trim_model(name)
    return folder if ok else None


def stage_38(lab: Lab) -> bool:
    n = LM38
    cfg = lm38_settings(lab)
    chunks, order = int(lab.cfg["tta_chunks"]), str(lab.cfg["tta_order"])
    agree_min, max_pseudo = float(lab.cfg["tta_agree_min"]), float(lab.cfg["tta_max_pseudo"])
    arms = cfg["arms"]
    lab.say(f"######## 38: self-training on language-model-chosen pseudo-labels, {chunks} chunks, seeds {' '.join(map(str, TTA25_SEEDS))}, "
            f"arms {' '.join(arms)}, started {now()}")
    lab.say(f"No transcripts: lab 25's loop, but a sentence's pseudo-label is the answer that {cfg['tag']} and the recogniser choose together "
            f"(weight {cfg['weight']:g}, bonus {cfg['bonus']:g}); single words keep the plain decode. Control: lab 25's plain labels.")
    for s in lab.spk:
        if not lab.need_have(f"{EX}/{s}/train.jsonl", f"{EX}/{s}/dev.jsonl", f"{EX}/{s}/test.jsonl", lab.stock_h(s) + "/hypotheses.jsonl",
                             lab.f1_h(s) + "/hypotheses.jsonl"):
            return False
    if not lab.need_free(f"{n}/table.json") or not lab.act():
        return False
    import lm_score

    tag, py = cfg["tag"], lm_python(lab, cfg["tag"])
    lmpy = Path(str(lab.cfg["lmpy"]))
    if not lab.dry and py == str(lmpy) and not lmpy.exists():
        lab.say(f"{lmpy} is not here yet; building the language-model environment")
        if not lab.run(["bash", "willspeak.sh", "lm-setup"]) or not lmpy.exists():
            lab.say(f"MISSING {lmpy}: ./willspeak.sh lm-setup builds it and needs the network")
            return False
    if not ensure_pretrained_models(lab):
        return False
    kept = not lab.dry and lm_score.present(lm_score.get(tag))
    if not lab.run([py, "scripts/asr/lm_score.py", "fetch", "--models", tag]):
        lab.say(f"FAILED downloading {tag}")
        return False
    failed: list[str] = []
    for s in lab.spk:
        for seed in TTA25_SEEDS:
            picks0 = f"{n}/picks/static-{s}-s{seed}-c0.jsonl"
            base_h = lm38_static(lab, s, seed, cfg, order, chunks, picks0)
            if base_h is None:
                failed.append(f"base:{s}-s{seed}")
                continue
            agree = [lab.f1_h(s), base_h]
            tta_loop(lab, s, f"lab-25-{s}-s{seed}", f"lab-25-{s}", pretrained_model(s), base_h, agree, f"{n}/control-by-chunk-{s}-s{seed}.json",
                     chunks, order, agree_min, max_pseudo, failed, seed=seed, lr=TTA25_LR)
            for arm in arms:
                key = lm38_key(arm, cfg)
                labels = LmLabels(lab, s, seed, arm, cfg, chunks, pretrained_model(s), picks0)
                tta_loop(lab, s, f"lab-38-{key}-{s}-s{seed}", f"lab-38-{s}", pretrained_model(s), base_h, agree, f"{n}/by-chunk-{key}-{s}-s{seed}.json",
                         chunks, order, agree_min, max_pseudo, failed, seed=seed, lr=TTA25_LR, labels=labels)
        lab.trim_model(f"big-c-pre-{s}")
    if lab.trim_on() and not kept:
        lab.say(f"TRIM_MODELS: removing the files of {tag} (its scores are kept)")
        lab.run([py, "scripts/asr/lm_score.py", "release", "--models", tag])
    summary_lines(lab, failed)
    systems = ["--system", lab.sysfn("stock", lab.stock_h), "--system", lab.sysfn("F-1", lab.f1_h)]
    rows = [("pretrained adapter 6e-4", f"{FT}/{PRE_AD8}-@-s%d-test-nemo"), ("control: plain labels (lab 25)", f"{FT}/lab-25-@-s%d-tta-test-nemo")]
    rows += [(lm38_label(arm, cfg), f"{FT}/lab-38-{lm38_key(arm, cfg)}-@-s%d-tta-test-nemo") for arm in arms]
    for seed in TTA25_SEEDS:
        for label, pattern in rows:
            if lab.dry or all(lab.have((pattern % seed).replace("@", s) + "/results.json") for s in lab.spk):
                systems += ["--system", lab.sysfmt(f"{label} s{seed}", pattern % seed)]
    if not lab.pooled(*systems, "--baseline", "stock", "--out", f"{n}/table.json"):
        failed.append("table")
    seed_means(lab, TTA25_SEEDS, tuple(rows))
    last = chunks - 1
    audit = []
    for seed in TTA25_SEEDS:
        audit.append(f"control (lab 25) s{seed}, trained on=" + ",".join(f"{EX}/lab-25-{s}-s{seed}-r{last}" for s in lab.spk))
        audit.append(f"control (lab 25) s{seed}, before the gate=" + ",".join(
            f"{n}/picks/static-{s}-s{seed}-c0.jsonl@plain" if j == 0 else f"{FT}/lab-25-{s}-s{seed}-chunk{j}-answers.jsonl"
            for s in lab.spk for j in range(last)))
        for arm in arms:
            key, label = lm38_key(arm, cfg), lm38_label(arm, cfg)
            audit.append(f"{label} s{seed}, trained on=" + ",".join(f"{EX}/lab-38-{key}-{s}-s{seed}-r{last}" for s in lab.spk))
            audit.append(f"{label} s{seed}, before the gate=" + ",".join(f"{n}/labels/{key}-{s}-s{seed}-r{last}.jsonl" for s in lab.spk))
    if not lab.run([lab.sc, "scripts/asr/lm_pick.py", "audit", "--truth", *[f"{EX}/{s}/test.jsonl" for s in lab.spk],
                    *[x for item in audit for x in ("--system", item)], "--out", f"{n}/label-quality.json"]):
        failed.append("audit")
    lab.say("Chunk 0 is the pretrained adapter's own answers, so each pair of rows differs only from chunk 1 on; by-chunk-<arm>-<speaker>-s<seed>.json shows the later chunks.")
    lab.say("An arm counts against the control only when at least 2 of 3 speakers move the same way and the pooled WER moves by more than about 4.5 points (19 words), at the three-seed mean.")
    lab.say("The label-quality lines compare each arm's pseudo-labels with the true text (read for that purpose only, after all training); 'plain, same clips' is the plain decode on the clips the arm kept.")
    lab.say(f"38 finished {now()}")
    return not failed


def seed_means(lab: Lab, seeds: tuple, rows: tuple) -> None:
    if lab.dry:
        return
    lab.say(f"{'system':<30} {'seeds ' + ' '.join(map(str, seeds)):<24} {'mean':>6}")
    for label, pattern in rows:
        per_seed = []
        for seed in seeds:
            counts = [wrong_words(lab, (pattern % seed).replace("@", s)) for s in lab.spk]
            per_seed.append(sum(counts) if all(c is not None for c in counts) else None)
        done = [v for v in per_seed if v is not None]
        lab.say(f"{label:<30} {' '.join(f'{v!s:>5}' for v in per_seed):<24} {sum(done) / len(done) if done else float('nan'):>6.1f}")


PRE26 = "reports/lab-26-pretraining-recipe"
TTA27 = "reports/lab-27-tta-settings"
EXP28 = "reports/lab-28-deployable-export"
CORE29 = "reports/lab-29-hard-core"
PHRASE30 = "reports/lab-30-phrase-scoring-pretrained"
PRE26_ARMS = {
    "bottom8": ["--train-parts", "all"],
    "lr4e-5": ["--lr", "4e-5"],
    "longer": ["--epochs", "20", "--patience", "6"],
    "dysarthric-then-severe": None,
}
TTA27_VARIANTS = {
    "c8": (8, "hash", 0.5, 1.0),
    "id": (4, "id", 0.5, 1.0),
    "g0.3": (4, "hash", 0.3, 1.0),
    "g0.7": (4, "hash", 0.7, 1.0),
    "nogate": (4, "hash", 0.0, 1.0),
    "cap0.5": (4, "hash", 0.5, 0.5),
}


def stage_26(lab: Lab) -> bool:
    n = PRE26
    lab.say(f"######## 26: the pretraining recipe under the adapter, {len(PRE26_ARMS)} variants against stage C's phase 1, started {now()}")
    for s in lab.spk:
        if not lab.need_have(f"{EX}/{s}/train.jsonl", f"{EX}/{s}/test.jsonl", lab.stock_h(s) + "/hypotheses.jsonl", lab.f1_m(s)):
            return False
    if not lab.need_free(f"{n}/table.json") or not lab.act():
        return False
    if not ensure_pretrained_models(lab):
        return False
    failed: list[str] = []
    for s in lab.spk:
        for seed in TTA25_SEEDS:
            if pretrained_adapter_answers(lab, s, seed) is None:
                failed.append(f"control:{s}-s{seed}")
        lab.trim_model(f"big-c-pre-{s}")
        for arm, flags in PRE26_ARMS.items():
            pre = f"lab-26-pre-{arm}-{s}"
            base = f"models/parakeet-ft-{pre}/parakeet-ft.nemo"
            if lab.dry or not all(lab.exists(f"{FT}/lab-26-{arm}-{s}-s{seed}-test-nemo/results.json") for seed in TTA25_SEEDS):
                if lab.dry or not lab.exists(base):
                    if lab.exists(f"models/parakeet-ft-{pre}") and not lab.dry:
                        lab.aside(f"models/parakeet-ft-{pre}")
                    lab.banner(f"{s}, phase 1 {arm}: the other severe speakers")
                    extra = ["--base", lab.f1_m(s)] if flags is None else list(flags)
                    if not lab.train(f"{EX}/loso-severe-{s}", pre, Lab.RECIPES["f1"], extra, seed=17):
                        failed.append(f"phase1:{arm}:{s}")
                        continue
                for seed in TTA25_SEEDS:
                    name = f"lab-26-{arm}-{s}-s{seed}"
                    if not lab.dry and lab.exists(f"{FT}/{name}-test-nemo/results.json"):
                        continue
                    lab.banner(f"{s}, {arm}, adapter seed {seed}")
                    if not (train_adapter_arm(lab, f"{EX}/{s}", name, 8, 15, base, seed=seed, extra=["--lr", TTA25_LR])
                            and lab.decode(s, name, lab.adapter_flags(name, base))):
                        failed.append(f"adapter:{arm}:{s}-s{seed}")
                    lab.trim_model(name)
            lab.trim_model(pre)
    summary_lines(lab, failed)
    systems = ["--system", lab.sysfn("stock", lab.stock_h)]
    if all(lab.have(lab.adf1_h(s) + "/hypotheses.jsonl") for s in lab.spk):
        systems += ["--system", lab.sysfn("adapter-on-F-1", lab.adf1_h)]
    rows = [("control: severe, every parameter", f"{FT}/{PRE_AD8}-@-s%d-test-nemo")] + [(arm, f"{FT}/lab-26-{arm}-@-s%d-test-nemo") for arm in PRE26_ARMS]
    for seed in TTA25_SEEDS:
        for label, pattern in rows:
            if lab.dry or all(lab.have((pattern % seed).replace("@", s) + "/results.json") for s in lab.spk):
                systems += ["--system", lab.sysfmt(f"{label} s{seed}", pattern % seed)]
    if not lab.pooled(*systems, "--baseline", "stock", "--out", f"{n}/table.json"):
        failed.append("table")
    seed_means(lab, TTA25_SEEDS, tuple(rows))
    lab.say("Every arm ends with the same adapter (block 8, 15 epochs, 6e-4); only the model it starts from differs. The control is stage P's arm.")
    lab.say(f"26 finished {now()}")
    return not failed


def stage_27(lab: Lab) -> bool:
    n = TTA27
    lab.say(f"######## 27: test-time adaptation settings, {len(TTA27_VARIANTS)} variants against lab test 22, started {now()}")
    for s in lab.spk:
        if not lab.need_have(lab.f1_m(s), f"{EX}/{s}/train.jsonl", f"{EX}/{s}/dev.jsonl", f"{EX}/{s}/test.jsonl", lab.f1_h(s) + "/hypotheses.jsonl",
                             lab.adf1_h(s) + "/hypotheses.jsonl"):
            return False
    if not lab.need_free(f"{n}/table.json") or not lab.act():
        return False
    failed: list[str] = []
    for s in lab.spk:
        for label, (chunks, order, agree_min, max_pseudo) in TTA27_VARIANTS.items():
            tta_loop(lab, s, f"lab-27-{label}-{s}", f"lab-27-n{chunks}-{order}-{s}", lab.f1_m(s), lab.adf1_h(s), [lab.f1_h(s), lab.adf1_h(s)],
                     f"{n}/by-chunk-{label}-{s}.json", chunks, order, agree_min, max_pseudo, failed)
    summary_lines(lab, failed)
    systems = ["--system", lab.sysfn("stock", lab.stock_h), "--system", lab.sysfn("adapter-on-F-1", lab.adf1_h)]
    if all(lab.have(f"{FT}/lab-22-{s}-tta-test-nemo/results.json") for s in lab.spk):
        systems += ["--system", lab.sysfmt("TTA control (lab 22: 4 chunks, hash, gate 0.5)", f"{FT}/lab-22-@-tta-test-nemo")]
    for label in TTA27_VARIANTS:
        if lab.dry or all(lab.have(f"{FT}/lab-27-{label}-{s}-tta-test-nemo/results.json") for s in lab.spk):
            systems += ["--system", lab.sysfmt(f"TTA {label}", f"{FT}/lab-27-{label}-@-tta-test-nemo")]
    if not lab.pooled(*systems, "--baseline", "stock", "--out", f"{n}/table.json"):
        failed.append("table")
    lab.say("Each variant changes one setting of lab test 22. Chunk 0 never changes, so compare the by-chunk tables as well as the pooled rows.")
    lab.say(f"27 finished {now()}")
    return not failed


def stage_28(lab: Lab) -> bool:
    n = EXP28
    lab.say(f"######## 28: the deployable export, pretrained adapter at 6e-4, int8 default and matmul, started {now()}")
    for s in lab.spk:
        if not lab.need_have(f"{EX}/{s}/train.jsonl", f"{EX}/{s}/test.jsonl", lab.stock_h(s) + "/hypotheses.jsonl"):
            return False
    if not lab.need_free(f"{n}/table.json") or not lab.act():
        return False
    if not ensure_pretrained_models(lab):
        return False
    failed: list[str] = []
    variants = {k: v for k, v in INT8_VARIANTS.items() if k != "perchannel"}
    for s in lab.spk:
        fresh = fresh_pretrained_adapter(lab, s)
        if fresh is None:
            failed.append(f"adapter:{s}")
            continue
        name, _hyps = fresh
        for variant, extra in variants.items():
            report, export = f"{n}/{s}-{variant}", f"models/lab-28-export-{s}-{variant}"
            if not lab.dry and lab.exists(f"{report}/results.json"):
                lab.say(f"reusing {report}")
                continue
            lab.banner(f"{s}, int8 {variant}")
            if lab.exists(export) and not lab.dry:
                lab.aside(export)
            if not lab.run([lab.gpu_py, "scripts/asr/export.py", "--nemo", pretrained_model(s), "--adapter", f"models/parakeet-ft-{name}/adapter.pt",
                            "--out", export, *extra]):
                failed.append(f"export:{s}:{variant}")
                continue
            if not lab.run([lab.sc, "scripts/asr/evaluate.py", "--manifest", f"{EX}/{s}/test.jsonl", "--audio-root", f"{EX}/{s}",
                            "--model-dir", export, "--group-by", "kind", "--out", report]):
                failed.append(f"score:{s}:{variant}")
                continue
            if lab.trim_on() and not lab.dry:
                shutil.rmtree(lab.root / export, ignore_errors=True)
        lab.trim_model(name)
        lab.trim_model(f"big-c-pre-{s}")
    summary_lines(lab, failed)
    systems = ["--system", lab.sysfn("stock", lab.stock_h)]
    if lab.dry or all(lab.have(f"{FT}/lab-pre-ad8-{s}-test-nemo/results.json") for s in lab.spk):
        systems += ["--system", lab.sysfmt("pretrained adapter NeMo fp32", f"{FT}/lab-pre-ad8-@-test-nemo")]
    for variant in variants:
        if lab.dry or all(lab.have(f"{n}/{s}-{variant}/results.json") for s in lab.spk):
            systems += ["--system", lab.sysfmt(f"pretrained adapter int8 {variant}", f"{n}/@-{variant}")]
    for variant in variants:
        if all(lab.have(f"reports/lab-19-int8-cost/adapter-on-F-1-{s}-{variant}/results.json") for s in lab.spk):
            systems += ["--system", lab.sysfmt(f"adapter-on-F-1 int8 {variant} (lab 19)", f"reports/lab-19-int8-cost/adapter-on-F-1-@-{variant}")]
    if not lab.pooled(*systems, "--baseline", "stock", "--out", f"{n}/table.json"):
        failed.append("table")
    lab.say("The int8 rows are what the Pi would say (sherpa-onnx); the fp32 row is the same weights in NeMo. The export folder of the")
    lab.say("matmul variant is the file to copy to the Pi for F-4 (regenerate it without trimming, or keep trim_models off).")
    lab.say(f"28 finished {now()}")
    return not failed


def stage_29(lab: Lab) -> bool:
    n = CORE29
    lab.say(f"######## 29: the hard core of errors across the saved answers, started {now()}")
    for s in lab.spk:
        if not lab.need_have(f"{EX}/{s}/test.jsonl", lab.stock_h(s) + "/hypotheses.jsonl"):
            return False
    if not lab.need_free(f"{n}/hard_core.json"):
        return False
    candidates = [("stock", lab.stock_h), ("F-1", lab.f1_h), ("whole-block", lab.wb_h), ("adapter", lab.ad_h), ("adapter-on-F-1", lab.adf1_h),
                  ("pretrained adapter s17", lambda s: f"{FT}/{PRE_AD8}-{s}-s17-test-nemo"),
                  ("adapter-on-F-1+TTA", lambda s: f"{FT}/lab-22-{s}-tta-test-nemo"),
                  ("pretrained adapter+TTA s17", lambda s: f"{FT}/lab-25-{s}-s17-tta-test-nemo")]
    systems = []
    for label, fn in candidates:
        if lab.dry or all(lab.have(fn(s) + "/hypotheses.jsonl") for s in lab.spk):
            systems += ["--system", lab.sysfn(label, fn)]
        else:
            lab.say(f"skipping {label}: no saved answers for every speaker")
    ok = lab.run([lab.sc, "scripts/asr/hard_core.py", *systems, "--export-root", EX, "--out", f"{n}/hard_core.json"])
    lab.say("This reads the test references to describe the errors after the fact; it chooses nothing. Do not tune anything on its lists.")
    lab.say(f"29 finished {now()}")
    return ok


def stage_30(lab: Lab) -> bool:
    n = PHRASE30
    lab.say(f"######## 30: acoustic phrase scoring on the pretrained adapter, started {now()}")
    for s in lab.spk:
        if not lab.need_have(f"{EX}/{s}/train.jsonl", f"{EX}/{s}/test.jsonl", lab.stock_h(s) + "/hypotheses.jsonl"):
            return False
    if not lab.need_free(f"{n}/report/results.json") or not lab.act():
        return False
    if not ensure_pretrained_models(lab):
        return False
    pool = [f"{EX}/*/train.jsonl", f"{EX}/*/dev.jsonl", f"{EX}/*/test.jsonl"]
    failed: list[str] = []
    scores: dict[str, str] = {}
    for s in lab.spk:
        out = f"{n}/{s}-pretrained-adapter.json"
        if lab.exists(out) and not lab.dry:
            lab.say(f"reusing {out}")
            scores[s] = out
            continue
        fresh = fresh_pretrained_adapter(lab, s)
        if fresh is None:
            failed.append(f"adapter:{s}")
            continue
        name, hyps = fresh
        lab.banner(f"{s}, pretrained adapter: score the nearest phrases against the audio")
        if lab.run([lab.gpu_py, "scripts/asr/phrase_score.py", "--nemo", pretrained_model(s), "--adapter", f"models/parakeet-ft-{name}/adapter.pt",
                    "--manifest", f"{EX}/{s}/test.jsonl", "--audio-root", f"{EX}/{s}", "--pool-glob", *pool, "--hyps", hyps, "--shortlist", "15", "--out", out]):
            scores[s] = out
        else:
            failed.append(f"score:{s}")
        lab.trim_model(name)
        lab.trim_model(f"big-c-pre-{s}")
    summary_lines(lab, failed)
    args: list[str] = []
    if len(scores) == len(lab.spk):
        args += ["--system", "pretrained-adapter=" + ",".join(f"{s}:{p}" for s, p in scores.items())]
    earlier = {s: f"{PHRASE24}/{s}-{safe('adapter-on-F-1')}.json" for s in lab.spk}
    if all(lab.have(p) for p in earlier.values()):
        args += ["--system", "adapter-on-F-1=" + ",".join(f"{s}:{p}" for s, p in earlier.items())]
    if not args and not lab.dry:
        lab.say("no system has scores for every speaker; no report")
        return False
    if not lab.run([lab.sc, "scripts/asr/phrase_report.py", *args, "--export-root", EX, "--out", f"{n}/report"]):
        failed.append("report")
    lab.say("Read it beside lab test 24: same shortlist, same cross-speaker choice of thresholds, a better model underneath.")
    lab.say(f"30 finished {now()}")
    return not failed


ACOUSTIC31 = "reports/lab-31-adapter-on-acoustic-only"
ACOUSTIC32 = "reports/lab-32-acoustic-severe-pretraining"
ACOUSTIC33 = "reports/lab-33-pooled-acoustic-adapters"
PRE34 = "reports/lab-34-pretrained-adapter-positions"
ACOUSTIC_SEEDS = (17, 18, 19)
ACOUSTIC_OWN_EPOCHS = 100
ACOUSTIC_RECIPE = ["--train-parts", "all", "--train-blocks", "24", "--freeze-transducer-head", "--lr", "1e-5", "--target-form", "stock",
                   "--warmup-steps", "10", "--select", "last", "--average-top", "4", "--no-keep-stock"]
FULL_ENCODER_GB = 12.5
ACOUSTIC_CODE = ("lab.py", "scripts/asr/train.py", "scripts/asr/block_adapter.py", "scripts/asr/nemo_hyp.py", "scripts/corpus/torgo_manifest.py")


def acoustic_flags(epochs: int) -> list[str]:
    return [*ACOUSTIC_RECIPE, "--epochs", str(epochs)]


def pooled_export(lab: Lab) -> str:
    return f"{EX}/pooled-{'-'.join(lab.spk)}"


def block_positions(lab: Lab, key: str) -> list[int]:
    values = [int(x) for x in str(lab.cfg[key]).replace(",", " ").split()]
    if not values or any(not 1 <= v <= 24 for v in values) or len(set(values)) != len(values):
        raise ValueError(f"{key} lists distinct encoder blocks from 1 to 24, for example '4 8 12 16 24'")
    return values


def pooled_positions(lab: Lab) -> list[int]:
    return block_positions(lab, "pooled_positions")


def update_runs(lab: Lab, folder: str, change) -> None:
    if lab.dry:
        return
    path = lab.root / folder / "runs.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError):
        data = {}
    change(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def note_plan(lab: Lab, folder: str, plan: dict) -> None:
    attempt = {"stamp": lab.stamp, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "code": code_version(lab),
               "sha256": {name: sha12(lab.root / name) for name in ACOUSTIC_CODE}, "xflags": str(lab.cfg["xflags"]),
               "trim_models": str(lab.cfg["trim_models"]), "free_gb": free_gb(lab.root), **plan}
    update_runs(lab, folder, lambda data: data.setdefault("attempts", []).append(attempt))


def note_run(lab: Lab, folder: str, key: str, **fields) -> None:
    fields.update(stamp=lab.stamp, free_gb=free_gb(lab.root), updated=time.strftime("%Y-%m-%d %H:%M:%S"))
    update_runs(lab, folder, lambda data: data.setdefault("runs", {}).setdefault(key, {}).update(fields))


def export_budget(lab: Lab, export: str, epochs: int) -> str:
    path = lab.root / export / "train.jsonl"
    if not path.is_file():
        return f"{export}: not built yet, so no budget to show"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept = [r for r in rows if float(r.get("duration_s", 0)) <= 25]
    minutes = sum(float(r["duration_s"]) for r in kept) / 60
    updates = math.ceil(math.ceil(len(kept) / 2) / 8)
    speakers = len({r.get("speaker") for r in kept})
    return (f"{export}: {len(kept)} clips / {minutes:.1f} min from {speakers} speaker(s); about {updates} updates an epoch, "
            f"{updates * epochs} over {epochs} epochs")


def run_complete(lab: Lab, folder: str) -> bool:
    try:
        return json.loads((lab.root / folder / "run.json").read_text(encoding="utf-8")).get("status") == "complete"
    except (OSError, ValueError, AttributeError):
        return False


def drop_checkpoints(lab: Lab, folder: str) -> None:
    ckpt = lab.root / folder / "ckpt"
    model = lab.root / folder / "parakeet-ft.nemo"
    if lab.dry or not ckpt.is_dir() or not model.is_file() or not run_complete(lab, folder):
        return
    import lab_space

    if not lab_space.nemo_intact(model):
        lab.say(f"kept {folder}/ckpt: the final .nemo does not read back intact")
        return
    size = lab_space.tree_size(ckpt)
    shutil.rmtree(ckpt, ignore_errors=True)
    lab.say(f"freed {size / 2**30:.1f} GB: {folder}/ckpt (the averaged .nemo is the model; the checkpoints are not used again)")


def room_for_full_training(lab: Lab, folder: str) -> bool:
    need = FULL_ENCODER_GB + 2
    if lab.dry or shutil.disk_usage(lab.root).free / 2**30 >= need:
        return True
    lab.say(f"DISK: under {need:.1f} GB free before {folder}; freeing the weights of finished runs first")
    lab.run([sys.executable, "scripts/asr/lab_space.py", "clean", "--apply", "--lab-models", "--long-models"])
    free = shutil.disk_usage(lab.root).free / 2**30
    if free >= need:
        return True
    storage_snapshot(lab, f"too little disk for {folder}")
    lab.say(f"{free:.1f} GB free where {folder} goes; this full-encoder training needs about {need:.0f} GB (four checkpoints and the "
            "final .nemo). Stopping this stage: free space or set export_to, then run it again; finished runs are reused.")
    lab.out_of_disk = True
    return False


def base_id(lab: Lab, model: str) -> str | None:
    try:
        record = json.loads((lab.root / Path(model).parent / "run.json").read_text(encoding="utf-8"))
        return str(record["checkpoint"]["sha256"])[:12]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def acoustic_base(lab: Lab, n: str, export: str, name: str, epochs: int, label: str, dependents: list[str]) -> str | None:
    folder = f"models/parakeet-ft-{name}"
    model = f"{folder}/parakeet-ft.nemo"
    if not lab.dry and lab.exists(model) and run_complete(lab, folder):
        lab.say(f"reusing {folder} (checkpoint {base_id(lab, model)})")
        drop_checkpoints(lab, folder)
        return model
    if not lab.dry and lab.exists(folder):
        lab.say(f"{folder} has no complete model (released after an earlier run, or cut off); moving it aside and training again")
        lab.aside(folder)
    for dependent in dependents:
        if not lab.dry and lab.exists(f"models/parakeet-ft-{dependent}") and not scored(lab, dependent):
            lab.say(f"models/parakeet-ft-{dependent} was trained on an earlier copy of this base; moving it aside")
            lab.aside(f"models/parakeet-ft-{dependent}")
    lab.banner(f"{label}: acoustic recipe (all 24 blocks, text head frozen, 1e-5, {epochs} epochs), seed 17")
    lab.say("  " + export_budget(lab, export, epochs))
    if not room_for_full_training(lab, folder):
        note_run(lab, n, name, kind="base", export=export, epochs=epochs, status="no disk")
        return None
    started = time.monotonic()
    flags = acoustic_flags(epochs)
    ok = lab.train_retrying(export, name, flags, None, seed=17)
    if not ok and not lab.dry and not lab.out_of_disk and lab.why_failed().category in ("oom", "killed"):
        lab.say(f"FAILED {name} with batch 2; retrying with batch 1 x 16 accumulation")
        if lab.exists(folder):
            lab.aside(folder)
        ok = lab.train(export, name, flags, None, batch=1, accum=16, seed=17)
    note_run(lab, n, name, kind="base", export=export, epochs=epochs, status="trained" if ok else "failed",
             minutes=round((time.monotonic() - started) / 60, 1), checkpoint=None if lab.dry or not ok else base_id(lab, model))
    if not ok:
        storage_snapshot(lab, f"after {name} failed")
        return None
    drop_checkpoints(lab, folder)
    storage_snapshot(lab, f"after training {name}")
    return model


def has_dev(lab: Lab, s: str) -> bool:
    path = lab.root / EX / s / "dev.jsonl"
    return lab.dry or (path.is_file() and path.stat().st_size > 0)


def scored(lab: Lab, name: str) -> bool:
    return not lab.dry and lab.exists(f"{FT}/{name}-test-nemo/results.json")


def decode_acoustic(lab: Lab, n: str, s: str, name: str, model_flags: list[str], battery: str | None, failed: list[str], **fields) -> bool:
    started = time.monotonic()
    ok = lab.decode(s, name, model_flags)
    if ok and has_dev(lab, s) and not lab.decode(s, name, model_flags, split="dev"):
        lab.say(f"-- dev scoring failed for {name} (test result unaffected)")
    if ok and battery and not lab.battery_decode(battery, model_flags):
        lab.say(f"-- silence battery failed for {name} (test result unaffected)")
    if not ok:
        failed.append(name)
    note_run(lab, n, name, speaker=s, status="scored" if ok else "failed", wrong=None if lab.dry else wrong_words(lab, f"{FT}/{name}-test-nemo"),
             decode_minutes=round((time.monotonic() - started) / 60, 1), **fields)
    return ok


def acoustic_adapter(lab: Lab, n: str, s: str, name: str, base: str, block: int, seed: int, battery: str | None, failed: list[str]) -> None:
    if scored(lab, name):
        lab.say(f"reusing {FT}/{name}-test-nemo")
        return
    lab.banner(f"{s}, adapter after block {block}, seed {seed}, on {base}")
    started = time.monotonic()
    if not train_adapter_arm(lab, f"{EX}/{s}", name, block, 15, base, seed=seed, extra=["--lr", TTA25_LR]):
        failed.append(name)
        note_run(lab, n, name, kind="adapter", speaker=s, block=block, seed=seed, base=base, status="training failed",
                 minutes=round((time.monotonic() - started) / 60, 1))
        lab.trim_model(name)
        return
    decode_acoustic(lab, n, s, name, lab.adapter_flags(name, base), battery, failed, kind="adapter", block=block, seed=seed, base=base,
                    base_checkpoint=None if lab.dry else base_id(lab, base), minutes=round((time.monotonic() - started) / 60, 1))
    lab.trim_model(name)


def acoustic_reference(lab: Lab) -> tuple[str, str] | None:
    base = lab.root / "reports/f2-acoustic-only"
    for summary in sorted(base.glob("*/summary.json"), reverse=True) if base.is_dir() else []:
        if (summary.parent / "battery.json").is_file():
            try:
                return summary.parent.relative_to(lab.root).as_posix(), str(json.loads(summary.read_text(encoding="utf-8"))["selection"])
            except (OSError, ValueError, KeyError):
                continue
    return None


def word_counts(lab: Lab, folder: str | None) -> tuple[int, int] | None:
    if not folder:
        return None
    try:
        word = json.loads((lab.root / folder / "results.json").read_text(encoding="utf-8"))["word"]
        return int(word["errors"]), int(word["n_ref"])
    except (OSError, ValueError, KeyError):
        return None


def seed_table(lab: Lab, n: str, rows: list[tuple[str, object]], control: str | None) -> None:
    if lab.dry:
        return
    seeds = ACOUSTIC_SEEDS
    table = []
    for label, folder_of in rows:
        by_seed, speakers, words = [], {s: [] for s in lab.spk}, None
        for seed in seeds:
            counts = [word_counts(lab, folder_of(s, seed)) for s in lab.spk]
            if all(c is not None for c in counts):
                by_seed.append(sum(c[0] for c in counts))
                words = sum(c[1] for c in counts)
                for s, c in zip(lab.spk, counts):
                    speakers[s].append(c[0])
            else:
                by_seed.append(None)
        done = [v for v in by_seed if v is not None]
        if done:
            table.append({"system": label, "by_seed": dict(zip(map(str, seeds), by_seed)), "mean": round(sum(done) / len(done), 1),
                          "range": max(done) - min(done), "words": words,
                          "speaker_means": {s: round(sum(v) / len(v), 1) for s, v in speakers.items() if v}})
    ref = next((r for r in table if r["system"] == control), None)
    lab.say(f"wrong words pooled over {', '.join(lab.spk)} at seeds {' '.join(map(str, seeds))}; '-' is a seed that has no result")
    lab.say(f"{'system':<44} {'by seed':<17} {'mean':>6} {'WER':>6} {'range':>5} {'per speaker':<20} {'vs ' + (control or '-'):>12} {'agree':>6}")
    for r in table:
        cells = " ".join(f"{('-' if v is None else v)!s:>5}" for v in r["by_seed"].values())
        per = " ".join(f"{v:g}" for v in r["speaker_means"].values())
        diff = agree = ""
        if ref is not None and r is not ref and set(r["speaker_means"]) == set(ref["speaker_means"]):
            diff = f"{r['mean'] - ref['mean']:+.1f}"
            sign = 1 if r["mean"] < ref["mean"] else -1
            agree = f"{sum(1 for s in lab.spk if (ref['speaker_means'][s] - r['speaker_means'][s]) * sign > 0)} of {len(lab.spk)}"
        lab.say(f"{r['system']:<44} {cells:<17} {r['mean']:>6.1f} {100 * r['mean'] / r['words']:>5.1f}% {r['range']:>5} {per:<20} {diff:>12} {agree:>6}")
    lab.say("A change counts only when at least 2 of 3 speakers move the same way and the pooled WER moves by more than about 4.5 points (19 words).")
    out = lab.root / n / "seeds.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seeds": list(seeds), "speakers": lab.spk, "control": control, "rows": table}, indent=2) + "\n", encoding="utf-8")


def reference_rows(lab: Lab, pretrained: bool, acoustic: bool) -> list[tuple[str, object]]:
    rows: list[tuple[str, object]] = [("adapter on stock (block 8, 3e-4)", lambda s, seed: lab.base8(s, seed))]
    if pretrained:
        rows.append(("stage P: severe pretraining + adapter 6e-4", lambda s, seed: f"{FT}/{PRE_AD8}-{s}-s{seed}-test-nemo"))
    found = acoustic_reference(lab) if acoustic else None
    if found:
        folder, recipe = found
        rows.append((f"acoustic-only study ({recipe})", lambda s, seed: f"{folder}/{recipe}/{s}-seed{seed}-test"))
    return rows


def acoustic_tables(lab: Lab, n: str, rows: list[tuple[str, object]], control: str, battery: list[tuple[str, str]], failed: list[str]) -> None:
    for name in ("table.json", "battery.json"):
        if not lab.dry and lab.exists(f"{n}/{name}"):
            lab.aside(f"{n}/{name}")
    systems = ["--system", lab.sysfn("stock", lab.stock_h)]
    for label, folder_of in rows:
        for seed in ACOUSTIC_SEEDS:
            folders = {s: folder_of(s, seed) for s in lab.spk}
            if all(folders.values()) and (lab.dry or all(lab.has_hyp(f) for f in folders.values())):
                systems += ["--system", f"{label} s{seed}=" + ",".join(f"{s}:{f}" for s, f in folders.items())]
    if not lab.pooled(*systems, "--baseline", "stock", "--out", f"{n}/table.json"):
        failed.append("table")
    seed_table(lab, n, rows, control)
    bsys = [x for label, path in battery if lab.dry or lab.exists(path) for x in ("--system", f"{label}={path}")]
    if bsys:
        lab.say("-- false inserts on clips with no speech (lab test 13's battery), seed 17")
        if not lab.run([lab.sc, "scripts/asr/battery.py", "score", "--manifest", f"{lab.BAT}/manifest.jsonl", *bsys, "--out", f"{n}/battery.json"]):
            failed.append("battery-score")


def acoustic_finish(lab: Lab, n: str, st: str, failed: list[str]) -> bool:
    if failed or lab.dry:
        if failed:
            lab.say(f"not marking {st} finished: {' '.join(failed)}. Run it again; saved results are reused and the tables rebuilt.")
        lab.say(f"{st} stopped {now()}" if failed else f"{st} finished {now()}")
        return not failed
    (lab.root / n / "finished.json").write_text(json.dumps({"stamp": lab.stamp, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                            "code": code_version(lab)}, indent=2) + "\n", encoding="utf-8")
    storage_snapshot(lab, f"after {st}")
    lab.say(f"{st} finished {now()}")
    return True


def release_base(lab: Lab, base_name: str, dependents: list[str]) -> None:
    missing = [name for name in dependents if not scored(lab, name)]
    if missing and not lab.dry:
        lab.say(f"keeping models/parakeet-ft-{base_name} for a rerun: {len(missing)} result(s) still missing ({' '.join(missing[:4])})")
        return
    lab.trim_model(base_name)


def stock_battery(lab: Lab) -> list[tuple[str, str]]:
    return [("stock", f"{FT}/lab-13-battery-stock.jsonl")] if lab.battery_decode("stock", ["--stock"]) else []


def acoustic_start(lab: Lab, n: str, title: str, plan: dict) -> bool:
    lab.say(f"######## {title}, started {now()}")
    for s in lab.spk:
        if not lab.need_have(f"{EX}/{s}/train.jsonl", f"{EX}/{s}/dev.jsonl", f"{EX}/{s}/test.jsonl", lab.stock_h(s) + "/hypotheses.jsonl"):
            return False
    if not lab.need_free(f"{n}/finished.json") or not lab.act() or not lab.ensure_battery():
        return False
    lab.out_of_disk = False
    note_plan(lab, n, plan)
    lab.say("Disk: one full-encoder training at a time (about 12.5 GB at its peak, 2.5 GB kept while its adapters train); its checkpoints")
    lab.say(f"are deleted once the averaged model is saved. Run record: {n}/runs.json; storage snapshots: logs/lab-{lab.stamp}-storage.jsonl")
    return True


def stage_31(lab: Lab) -> bool:
    n = ACOUSTIC31
    plan = {"test": "31", "base_export": f"{EX}/<speaker>", "base_epochs": ACOUSTIC_OWN_EPOCHS, "base_recipe": acoustic_flags(ACOUSTIC_OWN_EPOCHS),
            "adapter": {"block": 8, "epochs": 15, "lr": TTA25_LR}, "seeds": list(ACOUSTIC_SEEDS), "speakers": lab.spk}
    if not acoustic_start(lab, n, "31: an adapter on each speaker's own acoustic-only model", plan):
        return False
    lab.say("The base is the acoustic-only study's chosen recipe on the speaker's own clips (seed 17); the adapter then trains on the same clips.")
    failed: list[str] = []
    battery = stock_battery(lab)
    for s in lab.spk:
        if lab.out_of_disk:
            break
        base_name = f"lab-31-base-{s}"
        adapters = {seed: f"lab-31-ad8-{s}-s{seed}" for seed in ACOUSTIC_SEEDS}
        battery += [(f"acoustic-{s}", f"{FT}/lab-13-battery-lab31-base-{s}.jsonl"), (f"acoustic+adapter-{s}", f"{FT}/lab-13-battery-lab31-ad8-{s}.jsonl")]
        if all(scored(lab, name) for name in (base_name, *adapters.values())):
            lab.say(f"{s}: every result is saved; nothing to train")
            continue
        base = acoustic_base(lab, n, f"{EX}/{s}", base_name, ACOUSTIC_OWN_EPOCHS, f"{s}, own clips", list(adapters.values()))
        if base is None:
            failed.append(f"base:{s}")
            continue
        if not scored(lab, base_name):
            decode_acoustic(lab, n, s, base_name, ["--nemo", base], f"lab31-base-{s}", failed, kind="base alone")
        for seed, name in adapters.items():
            acoustic_adapter(lab, n, s, name, base, 8, seed, f"lab31-ad8-{s}" if seed == 17 else None, failed)
        release_base(lab, base_name, [base_name, *adapters.values()])
    summary_lines(lab, failed)
    rows = reference_rows(lab, pretrained=True, acoustic=True)
    control = next((label for label, _ in rows if label.startswith("acoustic-only study")), rows[0][0])
    rows += [("own acoustic model alone (seed 17)", lambda s, seed: f"{FT}/lab-31-base-{s}-test-nemo" if seed == 17 else None),
             ("own acoustic model + adapter 6e-4", lambda s, seed: f"{FT}/lab-31-ad8-{s}-s{seed}-test-nemo")]
    acoustic_tables(lab, n, rows, control, battery, failed)
    lab.say("The model alone is the study's recipe at seed 17 again: compare it with the study's seed 17 to see how well a rerun repeats.")
    return acoustic_finish(lab, n, "31", failed)


def stage_32(lab: Lab) -> bool:
    n = ACOUSTIC32
    epochs = int(lab.cfg["acoustic_severe_epochs"])
    plan = {"test": "32", "base_export": f"{EX}/loso-severe-<speaker>", "base_epochs": epochs, "base_recipe": acoustic_flags(epochs),
            "adapter": {"block": 8, "epochs": 15, "lr": TTA25_LR}, "seeds": list(ACOUSTIC_SEEDS), "speakers": lab.spk}
    if not acoustic_start(lab, n, f"32: the acoustic recipe on the other severe speakers ({epochs} epochs), then an adapter", plan):
        return False
    lab.say("Only the pretraining changes from stage P: the acoustic recipe instead of every parameter at 2e-5. The adapter is stage P's.")
    if not lab.ensure_severe_exports():
        return False
    failed: list[str] = []
    battery = stock_battery(lab)
    for s in lab.spk:
        if lab.out_of_disk:
            break
        base_name = f"lab-32-pre-{s}"
        adapters = {seed: f"lab-32-ad8-{s}-s{seed}" for seed in ACOUSTIC_SEEDS}
        battery += [(f"severe-acoustic-{s}", f"{FT}/lab-13-battery-lab32-pre-{s}.jsonl"), (f"severe-acoustic+adapter-{s}", f"{FT}/lab-13-battery-lab32-ad8-{s}.jsonl")]
        if all(scored(lab, name) for name in (base_name, *adapters.values())):
            lab.say(f"{s}: every result is saved; nothing to train")
            continue
        base = acoustic_base(lab, n, f"{EX}/loso-severe-{s}", base_name, epochs, f"{s}, the other severe speakers", list(adapters.values()))
        if base is None:
            failed.append(f"base:{s}")
            continue
        if not scored(lab, base_name):
            decode_acoustic(lab, n, s, base_name, ["--nemo", base], f"lab32-pre-{s}", failed, kind="base alone (never heard this speaker)")
        for seed, name in adapters.items():
            acoustic_adapter(lab, n, s, name, base, 8, seed, f"lab32-ad8-{s}" if seed == 17 else None, failed)
        release_base(lab, base_name, [base_name, *adapters.values()])
    summary_lines(lab, failed)
    rows = reference_rows(lab, pretrained=True, acoustic=True)
    rows += [("severe acoustic model alone (seed 17)", lambda s, seed: f"{FT}/lab-32-pre-{s}-test-nemo" if seed == 17 else None),
             ("severe acoustic model + adapter 6e-4", lambda s, seed: f"{FT}/lab-32-ad8-{s}-s{seed}-test-nemo")]
    acoustic_tables(lab, n, rows, rows[1][0], battery, failed)
    lab.say("The control is stage P: the same adapter on stage C's pretrained model. The model alone has never heard the speaker.")
    return acoustic_finish(lab, n, "32", failed)


def ensure_pooled_export(lab: Lab) -> bool:
    export = pooled_export(lab)
    record = lab.root / export / "pooled.json"
    if lab.dry or not (lab.exists(f"{export}/train.jsonl") and record.is_file()):
        if not lab.need_have("data/public/torgo/TORGO"):
            return False
        if not lab.dry and lab.exists(export):
            lab.say(f"{export} is incomplete (no train.jsonl or pooled.json); moving it aside and building it again")
            lab.aside(export)
        before = "" if lab.dry else manifests_hash(lab)
        lab.say("building the pooled export: every dysarthric speaker's training prompts in one set")
        if not lab.run([lab.sc, "scripts/corpus/torgo_manifest.py", "--export", "--missing-only", "--speakers", *lab.spk, "--pooled",
                        "--only", Path(export).name]):
            lab.say("export step failed, stopping this stage")
            return False
        if not lab.dry and manifests_hash(lab) != before:
            lab.say("the speakers' own manifests changed while building the pooled export, stopping this stage (every earlier result used the old ones)")
            return False
    if lab.dry:
        return True
    data = json.loads(record.read_text(encoding="utf-8"))
    if data.get("targets") != lab.spk:
        lab.say(f"MISSING a pooled export for {' '.join(lab.spk)}: {export} was built for {' '.join(data.get('targets') or [])}")
        return False
    if not export_layout_ok(lab, export):
        lab.say(f"{export} does not have train/ and dev/ audio beside its manifests; move it aside and run the stage again")
        return False
    lab.say(f"pooled export {export}: train {data['train_clips']} clips / {data['train_minutes']} min, dev {data['dev_clips']} clips "
            f"(the targets' own dev), {data['bytes'] / 2**20:.0f} MB on disk; {data['test_prompt_overlap']} clips share a prompt with a test clip")
    for spk, entry in data["speakers"].items():
        lab.say(f"  {spk}: train {entry['train_clips']} clips / {entry['train_minutes']} min, dev {entry['dev_clips']} clips")
    return True


def stage_33(lab: Lab) -> bool:
    n = ACOUSTIC33
    epochs = int(lab.cfg["acoustic_pooled_epochs"])
    positions = pooled_positions(lab)
    plan = {"test": "33", "base_export": pooled_export(lab), "base_epochs": epochs, "base_recipe": acoustic_flags(epochs),
            "adapter": {"blocks": positions, "epochs": 15, "lr": TTA25_LR}, "seeds": list(ACOUSTIC_SEEDS), "speakers": lab.spk}
    if not acoustic_start(lab, n, f"33: one acoustic model on every dysarthric speaker ({epochs} epochs), then adapters after blocks "
                                  f"{' '.join(map(str, positions))}", plan):
        return False
    lab.say("One base model serves every speaker; it trains on their own training sessions too, never on a test session or a test prompt.")
    if not ensure_pooled_export(lab):
        return False
    failed: list[str] = []
    battery = stock_battery(lab) + [("pooled-acoustic", f"{FT}/lab-13-battery-lab33-pooled.jsonl")]
    battery += [(f"pooled+adapter{p}-{s}", f"{FT}/lab-13-battery-lab33-ad{p}-{s}.jsonl") for p in positions for s in lab.spk]
    base_name = "lab-33-pooled"
    alone = {s: f"lab-33-pooled-{s}" for s in lab.spk}
    adapters = {(seed, p, s): f"lab-33-ad{p}-{s}-s{seed}" for seed in ACOUSTIC_SEEDS for p in positions for s in lab.spk}
    if all(scored(lab, name) for name in (*alone.values(), *adapters.values())):
        lab.say("every result is saved; nothing to train")
    else:
        base = acoustic_base(lab, n, pooled_export(lab), base_name, epochs, "every dysarthric speaker at once", list(adapters.values()))
        if base is None:
            failed.append("base")
        else:
            for i, s in enumerate(lab.spk):
                if not scored(lab, alone[s]):
                    decode_acoustic(lab, n, s, alone[s], ["--nemo", base], "lab33-pooled" if i == 0 else None, failed, kind="base alone")
            for (seed, p, s), name in adapters.items():
                acoustic_adapter(lab, n, s, name, base, p, seed, f"lab33-ad{p}-{s}" if seed == 17 else None, failed)
            release_base(lab, base_name, [*alone.values(), *adapters.values()])
    summary_lines(lab, failed)
    rows = reference_rows(lab, pretrained=True, acoustic=True)
    rows.append(("pooled acoustic model alone (seed 17)", lambda s, seed: f"{FT}/lab-33-pooled-{s}-test-nemo" if seed == 17 else None))
    rows += [(f"pooled acoustic + adapter block {p}", lambda s, seed, p=p: f"{FT}/lab-33-ad{p}-{s}-s{seed}-test-nemo") for p in positions]
    acoustic_tables(lab, n, rows, rows[1][0], battery, failed)
    lab.say("Seeds run in order (17 for every block, then 18, then 19), so a stopped run still leaves a whole position curve at seed 17.")
    lab.say("Every adapter uses 6e-4 and 15 epochs; the stock position sweep found block 8 best at 3e-4, so other blocks are not tuned here.")
    return acoustic_finish(lab, n, "33", failed)


def stage_34(lab: Lab) -> bool:
    n = PRE34
    positions = block_positions(lab, "pretrained_positions")
    plan = {"test": "34", "base": "stage C phase 1 (models/parakeet-ft-big-c-pre-<speaker>)",
            "adapter": {"blocks": positions, "epochs": 15, "lr": TTA25_LR}, "seeds": list(ACOUSTIC_SEEDS), "speakers": lab.spk}
    if not acoustic_start(lab, n, f"34: stage P's adapter after blocks {' '.join(map(str, positions))} on the severe-pretrained model", plan):
        return False
    lab.say("The base is stage P's: F-1's recipe on the other severe speakers (seed 17). Only the adapter's position changes; block 8 is")
    lab.say("trained again here as the in-test control, because a retrained base does not repeat stage P's scores exactly.")
    if not ensure_pretrained_models(lab):
        return False
    failed: list[str] = []
    battery = stock_battery(lab)
    battery += [(f"pretrained-{s}", f"{FT}/lab-13-battery-lab34-pre-{s}.jsonl") for s in lab.spk]
    battery += [(f"pretrained+adapter{p}-{s}", f"{FT}/lab-13-battery-lab34-ad{p}-{s}.jsonl") for p in positions for s in lab.spk]
    alone = {s: f"lab-34-pre-{s}" for s in lab.spk}
    adapters = {(seed, p, s): f"lab-34-ad{p}-{s}-s{seed}" for seed in ACOUSTIC_SEEDS for p in positions for s in lab.spk}
    for s in lab.spk:
        if not scored(lab, alone[s]):
            decode_acoustic(lab, n, s, alone[s], ["--nemo", pretrained_model(s)], f"lab34-pre-{s}", failed, kind="base alone (never heard this speaker)")
    for (seed, p, s), name in adapters.items():
        acoustic_adapter(lab, n, s, name, pretrained_model(s), p, seed, f"lab34-ad{p}-{s}" if seed == 17 else None, failed)
    for s in lab.spk:
        release_base(lab, f"big-c-pre-{s}", [alone[s], *[name for (_seed, _p, spk), name in adapters.items() if spk == s]])
    summary_lines(lab, failed)
    rows = reference_rows(lab, pretrained=True, acoustic=False)
    rows.append(("pretrained model alone (seed 17)", lambda s, seed: f"{FT}/lab-34-pre-{s}-test-nemo" if seed == 17 else None))
    rows += [(f"pretrained + adapter block {p}", lambda s, seed, p=p: f"{FT}/lab-34-ad{p}-{s}-s{seed}-test-nemo") for p in positions]
    control = "pretrained + adapter block 8" if 8 in positions else rows[1][0]
    acoustic_tables(lab, n, rows, control, battery, failed)
    lab.say("Seeds run in order (17 for every block, then 18, then 19), so a stopped run still leaves a whole position curve at seed 17.")
    lab.say("Every adapter uses stage P's 6e-4 and 15 epochs; positions other than 8 are not tuned.")
    return acoustic_finish(lab, n, "34", failed)


SILENCE35 = "reports/lab-35-silence-preference"
SILENCE35_FORMS = ("stock", "normalized")


def silence_rows(lab: Lab, n: str) -> list[dict]:
    rows = []
    for path in sorted((lab.root / n).glob("*.json")):
        if path.name == "table.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        summary = data.get("summary", {})
        group, system, form = path.stem.split("-", 2)[0], path.stem.split("-", 1)[1].rsplit("-", 1)[0], path.stem.rsplit("-", 1)[1]
        rows.append({"speaker": group, "system": system, "form": form, "clips": summary.get("clips"),
                     "median_margin": summary.get("median_margin"),
                     "share_preferring_silence": summary.get("share_margin_below", {}).get("0.0"),
                     "empty_answers": summary.get("hypotheses", {}).get("empty")})
    return rows


def stage_35(lab: Lab) -> bool:
    n = SILENCE35
    lab.say(f"######## 35: does the model prefer silence, and does the best fine-tune remove that? started {now()}")
    lab.say("Lab test 6 again with three changes: the true words scored in the stock model's own form as well as scoring form,")
    lab.say("stage P's pretrained adapter added, and the typical-speech control speakers scored with stock as the comparison.")
    for s in lab.spk:
        if not lab.need_have(lab.wb_m(s), lab.wb_h(s) + "/hypotheses.jsonl", lab.stock_h(s) + "/hypotheses.jsonl",
                             f"{EX}/{s}/train.jsonl", f"{EX}/{s}/test.jsonl"):
            return False
    if not lab.need_free(f"{n}/table.json") or not lab.ensure_controls() or not lab.act():
        return False
    if not ensure_pretrained_models(lab):
        return False
    failed: list[str] = []

    def score(label: str, model: list[str], manifest_dir: str, hyps: str | None, out_stem: str) -> None:
        for form in SILENCE35_FORMS:
            out = f"{n}/{out_stem}-{form}.json"
            if not lab.dry and lab.exists(out):
                lab.say(f"reusing {out}")
                continue
            lab.banner(f"{label}, true words in {form} form")
            cmd = [lab.gpu_py, "scripts/asr/reference_score.py", *model, "--manifest", f"{manifest_dir}/test.jsonl",
                   "--audio-root", manifest_dir, "--reference", form, "--out", out]
            if hyps:
                cmd += ["--hyps", hyps]
            if not lab.run(cmd):
                failed.append(f"{out_stem}:{form}")

    for s in lab.spk:
        common = f"{EX}/{s}"
        score(f"{s}, stock", ["--stock"], common, lab.stock_h(s), f"{s}-stock")
        score(f"{s}, whole-block", ["--nemo", lab.wb_m(s)], common, lab.wb_h(s), f"{s}-whole-block")
        fresh = fresh_pretrained_adapter(lab, s)
        if fresh is None:
            failed.append(f"adapter:{s}")
            continue
        name, hyps = fresh
        score(f"{s}, stage P's pretrained adapter", lab.adapter_flags(name, pretrained_model(s)), common, hyps, f"{s}-pretrained-adapter")
        lab.trim_model(name)
        lab.trim_model(f"big-c-pre-{s}")
    score("control speakers (typical speech), stock", ["--stock"], f"{EX}/controls", None, "controls-stock")
    summary_lines(lab, failed)
    if lab.dry:
        return True
    rows = silence_rows(lab, n)
    lab.say("")
    lab.say("speaker   system              form        clips  median margin  share preferring silence  empty answers")
    for r in rows:
        share = "-" if r["share_preferring_silence"] is None else f"{100 * r['share_preferring_silence']:.0f}%"
        margin = "-" if r["median_margin"] is None else f"{r['median_margin']:.1f}"
        lab.say(f"{r['speaker']:<9} {r['system']:<19} {r['form']:<11} {r['clips']!s:>5}  {margin:>13}  {share:>24}  {r['empty_answers'] if r['empty_answers'] is not None else '-':>13}")
    lab.say("Margin = cost of saying nothing minus cost of the true words, in nats; below zero means the model prefers silence.")
    if failed:
        lab.say("not writing table.json because something failed; rerun 35 to fill the gaps")
        return False
    (lab.root / n / "table.json").write_text(json.dumps({"finished": now(), "rows": rows}, indent=1) + "\n", encoding="utf-8")
    lab.say(f"35 finished {now()}")
    return True


LM36 = "reports/lab-36-lm-rescoring"
LM_SYSTEMS = ("stock", "prototype", "folds")
SEVERE_SENTENCES = "data/public/torgo/manifests/severe-sentences.jsonl"


def lm_tags(lab: Lab) -> list[str]:
    import lm_score

    return lm_score.expand(str(lab.cfg["lm_models"]))


def lm_python(lab: Lab, tag: str) -> str:
    import lm_score

    return lab.gpu_py if lm_score.get(tag).backend == "hf" else "python" if lab.dry else str(lab.cfg["lmpy"])


def lm_pool(lab: Lab, model_flags: list[str], manifest: str, audio_root: str, out: str, extra: list[str] | None = None) -> bool:
    if not lab.dry and lab.exists(out):
        lab.say(f"reusing {out}")
        return True
    return lab.run([lab.gpu_py, "scripts/asr/lm_pool.py", *model_flags, "--manifest", manifest, "--audio-root", audio_root,
                    "--views", str(lab.cfg["lm_views"]), *(extra or []), "--out", out])


def stage_36(lab: Lab) -> bool:
    n = LM36
    wanted = str(lab.cfg["lm_systems"]).split()
    if not wanted or any(w not in LM_SYSTEMS for w in wanted):
        raise ValueError(f"lm_systems is one or more of: {' '.join(LM_SYSTEMS)}")
    tags = lm_tags(lab)
    folds, threads = int(lab.cfg["cv_folds"]), int(lab.cfg["lm_threads"])
    personal = [w for w in wanted if w != "stock"]
    lab.say(f"######## 36: language-model rescoring ({' '.join(wanted)}; {len(tags)} language models), started {now()}")
    lab.say("Each clip is decoded several ways; every distinct answer is scored against the audio by the recogniser and as English by a text")
    lab.say("model that never hears the audio; the answer with the best combined score is kept. The weights are chosen on other speakers.")
    need = [SEVERE_SENTENCES, "data/public/torgo/TORGO"] if "stock" in wanted else []
    if personal:
        need += [f"{EX}/{s}/{split}.jsonl" for s in lab.spk for split in ("train", "dev", "test")]
    if not lab.need_have(*need) or not lab.need_free(f"{n}/table.json") or not lab.act():
        return False
    lmpy = Path(str(lab.cfg["lmpy"]))
    if not lab.dry and any(lm_python(lab, t) == str(lmpy) for t in tags) and not lmpy.exists():
        lab.say(f"{lmpy} is not here yet; building the language-model environment")
        if not lab.run(["bash", "willspeak.sh", "lm-setup"]) or not lmpy.exists():
            lab.say(f"MISSING {lmpy}: ./willspeak.sh lm-setup builds it and needs the network")
            return False
    failed: list[str] = []
    systems: list[tuple[str, list[str]]] = []

    def pooled(label: str, flags: list[str], manifest: str, audio_root: str, out: str) -> str | None:
        lab.banner(f"{label}: candidate pools")
        if lm_pool(lab, flags, manifest, audio_root, out):
            return out
        lab.say(f"FAILED {label}")
        failed.append(f"pool:{Path(out).stem}")
        return None

    if "stock" in wanted:
        out = pooled("stock on every severe speaker's sentences", ["--stock"], SEVERE_SENTENCES, "data/public/torgo/TORGO", f"{n}/pools/stock-severe.json")
        if out:
            systems.append(("stock, every severe sentence", [out]))
    if personal:
        if not ensure_pretrained_models(lab):
            return False
        splits = ("train", "dev", "test") if "folds" in wanted else ("test",)
        stock: dict[str, list[str]] = {split: [] for split in splits}
        tuned: dict[str, list[str]] = {"test": [], "folds": []}
        for s in lab.spk:
            for split in splits:
                out = pooled(f"{s}, stock, {split} clips", ["--stock"], f"{EX}/{s}/{split}.jsonl", f"{EX}/{s}", f"{n}/pools/stock-{s}-{split}.json")
                stock[split] += [out] if out else []
            fresh = fresh_pretrained_adapter(lab, s)
            if fresh is None:
                failed.append(f"adapter:{s}")
                continue
            name, _hyps = fresh
            out = pooled(f"{s}, stage P's adapter, test clips", lab.adapter_flags(name, pretrained_model(s)), f"{EX}/{s}/test.jsonl", f"{EX}/{s}",
                         f"{n}/pools/prototype-{s}-test.json")
            tuned["test"] += [out] if out else []
            lab.trim_model(name)
            if "folds" in wanted:
                if not (lab.run([lab.sc, CV_PY, "make", "--export", f"{EX}/{s}", "--speaker", s, "--folds", str(folds)])
                        and lab.run([lab.sc, CV_PY, "check", "--speaker", s, "--folds", str(folds)])):
                    lab.say(f"FAILED building the folds for {s}; python lab.py doctor --fix repairs missing audio")
                    failed.append(f"folds:{s}")
                    continue
                for k in range(folds):
                    fold, export, out = f"lab-36-pre-ad8-{s}-f{k}", f"{EX}/cv-{s}-f{k}", f"{n}/pools/prototype-{s}-f{k}.json"
                    if not lab.dry and lab.exists(out):
                        lab.say(f"reusing {out}")
                        tuned["folds"].append(out)
                        continue
                    lab.banner(f"{s}, stage P's adapter trained without fold {k + 1} of {folds}")
                    if not train_adapter_arm(lab, export, fold, 8, 15, pretrained_model(s), seed=17, extra=["--lr", TTA25_LR]):
                        lab.say(f"FAILED {fold}")
                        failed.append(f"adapter:{s}:f{k}")
                        continue
                    out = pooled(f"{s}, stage P's adapter, held-out fold {k + 1}", lab.adapter_flags(fold, pretrained_model(s)), f"{export}/test.jsonl",
                                 export, out)
                    tuned["folds"] += [out] if out else []
                    lab.trim_model(fold)
            lab.trim_model(f"big-c-pre-{s}")
        every = len(lab.spk)
        if "prototype" in wanted and len(stock["test"]) == every and len(tuned["test"]) == every:
            systems += [("stock, the test clips", stock["test"]), ("stage P's adapter, the test clips", tuned["test"])]
        if "folds" in wanted and all(len(stock[x]) == every for x in splits) and len(tuned["test"]) == every and len(tuned["folds"]) == every * folds:
            systems += [("stock, every labelled clip", [p for x in splits for p in stock[x]]),
                        ("stage P's recipe, every labelled clip held out", tuned["test"] + tuned["folds"])]
    if not systems and not lab.dry:
        summary_lines(lab, failed)
        lab.say("no system has a complete set of pools; nothing to score")
        return False
    pools = list(dict.fromkeys(p for _label, paths in systems for p in paths))
    lab.banner("the language models, one at a time")
    import lm_score

    scored: list[str] = []
    for tag in tags:
        py, out, done = lm_python(lab, tag), f"{n}/scores/{tag}.json", f"{n}/completions/{tag}.json"
        if not lab.dry and not lm_score.pending([str(lab.root / p) for p in pools], lab.root / out, lab.root / done):
            lab.say(f"reusing {out}: it already covers these pools")
            scored.append(out)
            continue
        kept = not lab.dry and lm_score.present(lm_score.get(tag))
        common = ["--model", tag, "--pool", *pools, "--threads", str(threads), "--extend"]
        if not lab.run([py, "scripts/asr/lm_score.py", "fetch", "--models", tag]):
            lab.say(f"FAILED downloading {tag}; the other models are still scored")
            failed.append(f"fetch:{tag}")
            continue
        if lab.run([py, "scripts/asr/lm_score.py", "score", *common, "--out", out]):
            scored.append(out)
            if not lab.run([py, "scripts/asr/lm_score.py", "complete", *common, "--out", done]):
                failed.append(f"complete:{tag}")
        else:
            lab.say(f"FAILED scoring with {tag}; the other models are still scored")
            failed.append(f"score:{tag}")
        if lab.trim_on() and not kept:
            lab.say(f"TRIM_MODELS: removing the files of {tag} (its scores are kept)")
            lab.run([py, "scripts/asr/lm_score.py", "release", "--models", tag])
    summary_lines(lab, failed)
    if not scored and not lab.dry:
        lab.say("no language model produced scores; no table")
        return False
    args = [x for label, paths in systems for x in ("--system", label + "=" + ",".join(paths))]
    if not lab.run([lab.sc, "scripts/asr/lm_report.py", *args, "--scores", *scored, "--completions", f"{n}/completions", "--out", f"{n}/table.json"]):
        failed.append("report")
    else:
        lab.track_table(f"{n}/table.json")
    lab.say("Read the stock table first (the most clips), then the two tables on stage P's adapter. The memory and speed columns are this")
    lab.say("computer's; the Pi's come from running lm_score.py there on the same pools (docs/pi5.md).")
    lab.say(f"36 finished {now()}")
    return not failed


ROUTE37 = "reports/lab-37-kind-routing"
ROUTE37_ARMS = ("word", "sentence")
ROUTE37_RECIPE_EPOCHS = 15
ROUTE37_CONTROL = "control: adapter on every clip"
ROUTE37_WHOLE_BLOCK = "big-p-pre-wb-lr8e-5-ep20"
ROUTE37_DETECTORS = (("words", "routed, first decode has one word"), ("duration", "routed, short clip"), ("oracle", "oracle, true kind"))


def route37_arms(lab: Lab) -> list[str]:
    arms = str(lab.cfg["routing_arms"]).split()
    if "word" not in arms or any(a not in ROUTE37_ARMS for a in arms) or len(set(arms)) != len(arms):
        raise ValueError(f"routing_arms is word, or word and sentence (the control always trains): got {' '.join(arms) or 'nothing'}")
    if not 4 <= int(lab.cfg["routing_kind_epochs"]) <= 100:
        raise ValueError("routing_kind_epochs is from 4 (the adapter averages its last four epochs) to 100")
    return arms


def route37_name(arm: str, s: str, seed, epochs: int = ROUTE37_RECIPE_EPOCHS) -> str:
    tag = {"control": "all", "word": "word", "sentence": "sentence"}[arm]
    return f"lab-37-{tag}-{s}-s{seed}" + (f"-e{epochs}" if arm != "control" and epochs != ROUTE37_RECIPE_EPOCHS else "")


def route37_routed(arms: list[str]) -> list[tuple[str, str, str, str, str]]:
    others = [("ctl", "control", "control")] + ([("sent", "sentence", "sentence adapter")] if "sentence" in arms else [])
    return [(f"{detector}-{tag}", detector, arm, name, f"{text}: word adapter + {name}") for detector, text in ROUTE37_DETECTORS for tag, arm, name in others]


def route37_clip_counts(lab: Lab, s: str) -> dict[str, dict]:
    counts: dict[str, dict] = {"train": {}, "dev": {}}
    for split in counts:
        for line in (lab.root / EX / s / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                kind = json.loads(line).get("kind")
                counts[split][kind] = counts[split].get(kind, 0) + 1
    return counts


def route37_preflight(lab: Lab, arms: list[str]) -> bool:
    if lab.dry:
        return True
    for s in lab.spk:
        if not all(lab.exists(f"{EX}/{s}/{split}.jsonl") for split in ("train", "dev")):
            lab.say(f"MISSING the train and dev manifests of {EX}/{s}; python lab.py doctor --fix builds the exports")
            return False
        counts = route37_clip_counts(lab, s)
        lab.say(f"{s}: train {counts['train']}, dev {counts['dev']} (clips by kind; train.py --kinds keeps the train and dev clips of one kind)")
        for kind in ("word", *(["sentence"] if "sentence" in arms else [])):
            if None in counts["train"] or None in counts["dev"] or counts["train"].get(kind, 0) < 1 or counts["dev"].get(kind, 0) < 1:
                lab.say(f"{s} has no {kind} clips in train or in dev (train.py --kinds needs both) or a clip without a kind; python lab.py doctor --fix rebuilds the exports"
                        + ("; routing_arms=word leaves out the sentence adapter" if kind == "sentence" else ""))
                return False
    return True


def route37_adapter(lab: Lab, n: str, s: str, arm: str, seed: int, epochs: int, failed: list[str]) -> None:
    name = route37_name(arm, s, seed, epochs)
    if scored(lab, name):
        lab.say(f"reusing {FT}/{name}-test-nemo")
        return
    kind = {"control": None, "word": "word", "sentence": "sentence"}[arm]
    base, folder = pretrained_model(s), f"models/parakeet-ft-{name}"
    lab.banner(f"{s}, seed {seed}: block-8 adapter on {'every clip' if kind is None else kind + ' clips only'}")
    started = time.monotonic()
    fields = {"kind": "adapter", "arm": arm, "speaker": s, "seed": seed, "base": base, "epochs": ROUTE37_RECIPE_EPOCHS if arm == "control" else epochs}
    if not train_adapter_arm(lab, f"{EX}/{s}", name, 8, fields["epochs"], base, seed=seed, extra=["--lr", TTA25_LR, *(["--kinds", kind] if kind else [])]):
        failed.append(name)
        note_run(lab, n, name, **fields, status="training failed", minutes=round((time.monotonic() - started) / 60, 1))
        lab.trim_model(name)
        return
    mb = None if lab.dry or not lab.exists(f"{folder}/adapter.pt") else round((lab.root / folder / "adapter.pt").stat().st_size / 2**20, 2)
    ok = lab.decode(s, name, lab.adapter_flags(name, base))
    if not ok:
        failed.append(name)
    note_run(lab, n, name, **fields, status="scored" if ok else "decode failed", adapter_mb=mb, base_checkpoint=None if lab.dry else base_id(lab, base),
             wrong=None if lab.dry else wrong_words(lab, f"{FT}/{name}-test-nemo"), minutes=round((time.monotonic() - started) / 60, 1))
    lab.trim_model(name)


def route37_route(lab: Lab, n: str, s: str, seed: int, variant: str, detector: str, word: str, other: str, control: str, failed: list[str]) -> bool:
    name = f"lab-37-routed-{variant}-{s}-s{seed}"
    out, result, stats = f"{FT}/{name}-test-nemo.jsonl", f"{FT}/{name}-test-nemo", f"{n}/routing/{variant}-{s}-s{seed}.json"
    inputs = [f"{FT}/{x}-test-nemo/hypotheses.jsonl" for x in {word, other, control}]
    saved = [out, stats, f"{result}/results.json"]
    if not lab.dry and all(lab.exists(p) for p in saved) and min((lab.root / p).stat().st_mtime for p in saved) >= max((lab.root / p).stat().st_mtime for p in inputs):
        lab.say(f"reusing {result}")
        return True
    for stale in (out, result, stats):
        if not lab.dry and lab.exists(stale):
            lab.aside(stale)
    cmd = [lab.sc, "scripts/asr/kind_route.py", "--manifest", f"{EX}/{s}/test.jsonl", "--detector", detector,
           "--word-hyp", f"{FT}/{word}-test-nemo", "--other-hyp", f"{FT}/{other}-test-nemo", "--out", out, "--stats", stats]
    if detector == "words":
        cmd += ["--detect-hyp", f"{FT}/{control}-test-nemo"]
    if detector == "duration":
        cmd += ["--cut-from", f"{EX}/{s}/train.jsonl", f"{EX}/{s}/dev.jsonl"]
    if lab.run(cmd) and lab.score(s, name):
        return True
    failed.append(f"route:{variant}:{s}:s{seed}")
    return False


def route37_tables(lab: Lab, n: str, arms: list[str], epochs: int, failed: list[str]) -> None:
    def folder(name: str) -> str:
        return f"{FT}/{name}-test-nemo"

    def complete(pattern: str) -> bool:
        return lab.dry or all(lab.has_hyp(pattern.format(speaker=s, seed=seed)) for s in lab.spk for seed in ACOUSTIC_SEEDS)

    plain = [(ROUTE37_CONTROL, folder(route37_name("control", "{speaker}", "{seed}"))),
             ("word adapter: single-word clips only", folder(route37_name("word", "{speaker}", "{seed}", epochs)))]
    if "sentence" in arms:
        plain.append(("sentence adapter: sentence clips only", folder(route37_name("sentence", "{speaker}", "{seed}", epochs))))
    whole_block = folder(f"{ROUTE37_WHOLE_BLOCK}-{{speaker}}-s{{seed}}")
    plain += [(label, pattern) for label, pattern in (("stage P's adapter (older base; reference)", folder(f"{PRE_AD8}-{{speaker}}-s{{seed}}")),
                                                       ("whole-block from pretrained, 8e-5, 20 epochs (older base; reference)", whole_block)) if complete(pattern)]
    routed = [(variant, label, folder(f"lab-37-routed-{variant}-{{speaker}}-s{{seed}}")) for variant, _d, _a, _o, label in route37_routed(arms)]
    if complete(whole_block):
        routed += [(f"{detector}-wb", f"{text}: whole-block (older base; reference) + control", folder(f"lab-37-routed-{detector}-wb-{{speaker}}-s{{seed}}"))
                   for detector, text in ROUTE37_DETECTORS if detector != "duration"]
    for name in ("table.json", "table-words.json", "kind-table.json"):
        if not lab.dry and lab.exists(f"{n}/{name}"):
            lab.aside(f"{n}/{name}")
    systems = ["--system", lab.sysfn("stock", lab.stock_h)]
    for label, pattern in [*plain, *[(label, pattern) for _v, label, pattern in routed]]:
        for seed in ACOUSTIC_SEEDS:
            folders = {s: pattern.format(speaker=s, seed=seed) for s in lab.spk}
            if lab.dry or all(lab.has_hyp(f) for f in folders.values()):
                systems += ["--system", f"{label} s{seed}=" + ",".join(f"{s}:{f}" for s, f in folders.items())]
    if not lab.pooled(*systems, "--baseline", "stock", "--out", f"{n}/table.json"):
        failed.append("table")
    if not lab.pooled(*systems, "--baseline", "stock", "--kind", "word", "--out", f"{n}/table-words.json"):
        failed.append("table-words")
    report = [lab.sc, "scripts/asr/kind_route_report.py", "--control", ROUTE37_CONTROL, "--out", f"{n}/kind-table.json"]
    for label, pattern in [*plain, *[(label, pattern) for _v, label, pattern in routed]]:
        report += ["--system", f"{label}={pattern}"]
    for variant, label, _pattern in routed:
        report += ["--routing", f"{label}={n}/routing/{variant}-{{speaker}}-s{{seed}}.json"]
    if not lab.run(report):
        failed.append("kind-table")


def stage_37(lab: Lab) -> bool:
    n = ROUTE37
    arms = route37_arms(lab)
    epochs = int(lab.cfg["routing_kind_epochs"])
    lab.say(f"######## 37: routing single words and sentences to different adapters, started {now()}")
    lab.say("Adapters per speaker and seed on stage P's base: every clip (the control), single-word clips only, and (default) sentence clips only. The routed")
    lab.say("systems are computed afterwards from the saved decodes: a detector picks which adapter's answer each test clip gets. Nothing is retrained for them.")
    for s in lab.spk:
        if not lab.need_have(f"{EX}/{s}/train.jsonl", f"{EX}/{s}/dev.jsonl", f"{EX}/{s}/test.jsonl", lab.stock_h(s) + "/hypotheses.jsonl"):
            return False
    if not lab.need_free(f"{n}/finished.json") or not lab.act() or not route37_preflight(lab, arms):
        return False
    lab.out_of_disk = False
    note_plan(lab, n, {"test": "37", "base": "stage C phase 1 (models/parakeet-ft-big-c-pre-<speaker>)", "arms": ["control", *arms],
                       "adapter": {"block": 8, "epochs": ROUTE37_RECIPE_EPOCHS, "kind_epochs": epochs, "lr": TTA25_LR, "filter": "train.py --kinds"},
                       "seeds": list(ACOUSTIC_SEEDS), "speakers": lab.spk})
    lab.say("The word and sentence adapters train with train.py --kinds: the train and dev clips of that kind only, so dev and test stay prompt-disjoint exactly as in the exports.")
    lab.say(f"Checkpoints are not picked on dev (--select last); the last four of {epochs} epochs are averaged. Train and dev hold roughly 90 word and 25 sentence clips a speaker (counts printed above).")
    if not ensure_pretrained_models(lab):
        return False
    failed: list[str] = []
    for seed in ACOUSTIC_SEEDS:
        for s in lab.spk:
            for arm in ("control", *arms):
                route37_adapter(lab, n, s, arm, seed, epochs, failed)
    for s in lab.spk:
        release_base(lab, f"big-c-pre-{s}", [route37_name(arm, s, seed, epochs) for arm in ("control", *arms) for seed in ACOUSTIC_SEEDS])
    summary_lines(lab, failed)

    def decoded(name: str) -> bool:
        return lab.dry or lab.has_hyp(f"{FT}/{name}-test-nemo")

    for seed in ACOUSTIC_SEEDS:
        for s in lab.spk:
            word, control = route37_name("word", s, seed, epochs), route37_name("control", s, seed)
            for variant, detector, arm, _label, _text in route37_routed(arms):
                other = route37_name(arm, s, seed, epochs)
                if decoded(word) and decoded(other) and decoded(control):
                    route37_route(lab, n, s, seed, variant, detector, word, other, control, failed)
            wb = f"{ROUTE37_WHOLE_BLOCK}-{s}-s{seed}"
            for detector in ("words", "oracle"):
                if decoded(wb) and decoded(control):
                    route37_route(lab, n, s, seed, f"{detector}-wb", detector, wb, control, control, failed)
    route37_tables(lab, n, arms, epochs, failed)
    lab.say("Read table-words.json first (single words: the hypothesis), then the sentences table at the end of table.json (harm), then kind-table.json (three-seed")
    lab.say("means by kind, the rules, the detector's mistakes). The oracle rows are not device systems; they show what a perfect detector would give.")
    lab.say("Stage P's and the whole-block rows come from older bases (reference only); the control here is trained on the same base as the word adapter.")
    return acoustic_finish(lab, n, "37", failed)


WIDE39 = "reports/lab-39-wider-pools"
WIDE_BASELINE = "baseline"
WIDE_ARMS = ("seeds", "positions", "both")
WIDE_GROUPS = {"seeds": ("seeds",), "positions": ("positions",), "both": ("seeds", "positions")}
WIDE_SEEDS = (18, 19)
WIDE_POSITIONS = (12, 16)
WIDE_SYSTEMS = ("prototype", "folds")


def wide_choice(lab: Lab, name: str, allowed: tuple[str, ...]) -> list[str]:
    wanted = str(lab.cfg[name]).split()
    if not wanted or len(set(wanted)) != len(wanted) or any(w not in allowed for w in wanted):
        raise ValueError(f"{name} is one or more of: {' '.join(allowed)}")
    return [a for a in allowed if a in wanted]


def wide_unit(s: str, fold: int | None) -> dict:
    suffix = "" if fold is None else f"-f{fold}"
    extras = [{"name": f"seed{seed}", "adapter": f"lab-39-pre-ad8-{s}{suffix}-s{seed}", "blocks": 8, "seed": seed, "group": "seeds"} for seed in WIDE_SEEDS]
    extras += [{"name": f"block{p}", "adapter": f"lab-39-pre-ad{p}-{s}{suffix}", "blocks": p, "seed": 17, "group": "positions"} for p in WIDE_POSITIONS]
    export = f"{EX}/{s}" if fold is None else f"{EX}/cv-{s}-f{fold}"
    return {"speaker": s, "fold": fold, "key": "test" if fold is None else f"f{fold}", "export": export, "held_out": f"{export}/test.jsonl",
            "primary": f"lab-pre-ad8-{s}" if fold is None else f"lab-36-pre-ad8-{s}-f{fold}", "extras": extras}


def wide_recipe_ok(lab: Lab) -> bool:
    flags = Lab.RECIPES["ad8"] + shlex.split(str(lab.cfg["xflags"]))
    chosen = split_flags(flags)
    return chosen.get("--select") == ["last"] and "--no-keep-stock" in chosen and "--keep-stock" not in chosen


def wide_leaks(unit: dict) -> list[str]:
    s, fold = unit["speaker"], unit["fold"]
    export = f"{EX}/{s}" if fold is None else f"{EX}/cv-{s}-f{fold}"
    problems = []
    if unit["export"] != export or unit["held_out"] != f"{unit['export']}/test.jsonl":
        problems.append(f"{s} {unit['key']}: trains on {unit['export']} but its clips are held out of {export}")
    for name in [unit["primary"], *[e["adapter"] for e in unit["extras"]]]:
        m = re.search(r"-f(\d+)(?:-s\d+)?$", name)
        if (int(m.group(1)) if m else None) != fold:
            problems.append(f"{name} was trained for {'the test clips' if not m else 'fold ' + m.group(1)}, not for {unit['key']} of {s}")
    return problems


def wide_unseen(root: Path, unit: dict) -> tuple[list[str], list[str]]:
    import cv_folds

    def rows(path: Path) -> list[dict]:
        return cv_folds.read_rows(path)

    export = root / unit["export"]
    held = rows(root / unit["held_out"])
    train = rows(export / "train.jsonl") + (rows(export / "dev.jsonl") if unit["fold"] is None else [])
    problems, notes = [], []
    shared = {str(r["id"]) for r in held} & {str(r["id"]) for r in train}
    if shared:
        problems.append(f"{unit['speaker']} {unit['key']}: {len(shared)} held-out clips are in the training clips, e.g. {sorted(shared)[0]}")
    prompts = {cv_folds.prompt_key(r) for r in held} & {cv_folds.prompt_key(r) for r in train}
    if prompts and unit["fold"] is not None:
        problems.append(f"{unit['speaker']} {unit['key']}: {len(prompts)} held-out prompts are in the training clips, e.g. {sorted(prompts)[0]!r}")
    elif prompts:
        notes.append(f"{unit['speaker']}: {len(prompts)} test prompts also appear in train and dev (same as lab 36's adapter)")
    return problems, notes


def wide_extra_arg(name: str, flags: list[str]) -> str:
    base, adapter, i = "stock", None, 0
    while i < len(flags):
        if flags[i] == "--nemo":
            base, i = flags[i + 1], i + 2
        elif flags[i] == "--pretrained":
            base, i = "pretrained:" + flags[i + 1], i + 2
        elif flags[i] == "--adapter":
            adapter, i = flags[i + 1], i + 2
        else:
            i += 1
    return f"{name}={base}" + (f",{adapter}" if adapter else "")


def drop_adapter_checkpoints(lab: Lab, name: str) -> None:
    folder = lab.root / f"models/parakeet-ft-{name}"
    ckpt = folder / "ckpt"
    if lab.dry or not (folder / "adapter.pt").is_file() or not ckpt.is_dir():
        return
    size = sum(f.stat().st_size for f in ckpt.rglob("*") if f.is_file())
    shutil.rmtree(ckpt, ignore_errors=True)
    lab.say(f"freed {size / 2**30:.1f} GB: models/parakeet-ft-{name}/ckpt (adapter.pt is the model)")


def wide_pool_fits(lab: Lab, pool: str, flags: list[str]) -> bool:
    if lab.dry or not lab.exists(pool) or "--adapter" not in flags or "--nemo" not in flags:
        return False
    adapter, nemo = lab.root / flags[flags.index("--adapter") + 1], lab.root / flags[flags.index("--nemo") + 1]
    try:
        recorded = json.loads((lab.root / pool).read_text(encoding="utf-8")).get("model_files") or {}
        return (hashlib.sha256(adapter.read_bytes()).hexdigest() == recorded["adapter"]["sha256"]
                and nemo.stat().st_size == recorded["nemo"]["size"])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def lm_environment(lab: Lab, tags: list[str]) -> bool:
    lmpy = Path(str(lab.cfg["lmpy"]))
    if not lab.dry and any(lm_python(lab, t) == str(lmpy) for t in tags) and not lmpy.exists():
        lab.say(f"{lmpy} is not here yet; building the language-model environment")
        if not lab.run(["bash", "willspeak.sh", "lm-setup"]) or not lmpy.exists():
            lab.say(f"MISSING {lmpy}: ./willspeak.sh lm-setup builds it and needs the network")
            return False
    return True


def wide_pools(lab: Lab, unit: dict, arms: list[str], groups: list[str], extra_views: str, failed: list[str]) -> dict[str, str]:
    n, s, key, fold = WIDE39, unit["speaker"], unit["key"], unit["fold"]
    where = "the test clips" if fold is None else f"held-out fold {fold + 1}"
    lab.banner(f"{s}, {where}: stage P's adapter and the extra models")
    problems = wide_leaks(unit)
    if not lab.dry:
        found, notes = wide_unseen(lab.root, unit)
        problems += found
        for note in notes:
            lab.say(note)
    if problems:
        for problem in problems:
            lab.say(f"LEAK: {problem}")
        failed.append(f"leak:{s}:{key}")
        return {}
    base36 = f"{LM36}/pools/prototype-{s}-{key}.json"
    base39 = f"{n}/pools/base-{s}-{key}.json"
    flags = lab.adapter_flags(unit["primary"], pretrained_model(s))
    fits = wide_pool_fits(lab, base36, flags)
    base = base36 if fits else base39
    outs = {arm: f"{n}/pools/{arm}-{s}-{key}.json" for arm in arms}
    wide = f"{n}/pools/wide-{s}-{key}.json"
    target = {arm: wide if set(WIDE_GROUPS[arm]) == set(groups) else outs[arm] for arm in arms}
    if not lab.dry and lab.exists(base) and all(lab.exists(p) for p in target.values()):
        lab.say(f"reusing every pool of {s} {key}")
        return {WIDE_BASELINE: base, **target}
    if fold is None:
        ok = fresh_pretrained_adapter(lab, s) is not None
    else:
        ok = train_adapter_arm(lab, unit["export"], unit["primary"], 8, 15, pretrained_model(s), seed=17, extra=["--lr", TTA25_LR])
    drop_adapter_checkpoints(lab, unit["primary"])
    if not ok:
        lab.say(f"FAILED {unit['primary']}")
        failed.append(f"adapter:{s}:{key}")
        return {}
    extras = [e for e in unit["extras"] if e["group"] in groups]
    usable = []
    for e in extras:
        trained = train_adapter_arm(lab, unit["export"], e["adapter"], e["blocks"], 15, pretrained_model(s), seed=e["seed"], extra=["--lr", TTA25_LR])
        drop_adapter_checkpoints(lab, e["adapter"])
        if trained:
            usable.append(e)
        else:
            lab.say(f"FAILED {e['adapter']}")
            failed.append(f"adapter:{e['adapter']}")
    good = [g for g in groups if all(e in usable for e in extras if e["group"] == g)]
    feasible = [a for a in arms if set(WIDE_GROUPS[a]) <= set(good)]
    made: dict[str, str] = {}
    if fits:
        lab.say(f"reusing {base36}: the primary model on disk is the one that scored it")
    elif not lm_pool(lab, flags, unit["held_out"], unit["export"], base39):
        lab.say(f"FAILED the baseline pool of {s} {key}")
        failed.append(f"pool:{s}:{key}:baseline")
        return {}
    made[WIDE_BASELINE] = base
    if feasible:
        used = [g for g in good if any(g in WIDE_GROUPS[a] for a in feasible)]
        names = {g: [e["name"] for e in usable if e["group"] == g] for g in used}
        every = [name for g in used for name in names[g]]
        if not lab.dry and lab.exists(wide):
            lab.say(f"reusing {wide}")
        else:
            cmd = [lab.gpu_py, "scripts/asr/lm_pool.py", *flags, "--manifest", unit["held_out"], "--audio-root", unit["export"],
                   "--base-pool", base, "--extra-views", extra_views]
            for e in usable:
                if e["group"] in used:
                    cmd += ["--extra-model", wide_extra_arg(e["name"], lab.adapter_flags(e["adapter"], pretrained_model(s)))]
            if not lab.run(cmd + ["--out", wide]):
                lab.say(f"FAILED the wider pool of {s} {key}")
                failed.append(f"pool:{s}:{key}:wide")
                feasible = []
        for arm in feasible:
            keep = [name for g in WIDE_GROUPS[arm] for name in names[g]]
            if keep == every:
                made[arm] = wide
            elif (not lab.dry and lab.exists(outs[arm])) or lab.run([lab.sc, "scripts/asr/lm_pool.py", "--subset-of", wide,
                                                                     "--keep-extras", ",".join(keep), "--out", outs[arm]]):
                made[arm] = outs[arm]
            else:
                failed.append(f"pool:{s}:{key}:{arm}")
    for name in (unit["primary"], *[e["adapter"] for e in extras]):
        lab.trim_model(name)
    return made


def stage_39(lab: Lab) -> bool:
    import lm_pool as pool_module
    import lm_score

    n = WIDE39
    arms = wide_choice(lab, "wide_extras", WIDE_ARMS)
    wanted = wide_choice(lab, "wide_systems", WIDE_SYSTEMS)
    extra_views = str(lab.cfg["wide_extra_views"])
    pool_module.parse_extra_views(extra_views)
    if not str(lab.cfg["wide_lm_models"]).split():
        raise ValueError("wide_lm_models names at least one language model (a tag or a group; python scripts/asr/lm_score.py list)")
    tags = lm_score.expand(str(lab.cfg["wide_lm_models"]))
    if not wide_recipe_ok(lab):
        raise ValueError("the adapter recipe lets the dev clips choose weights (--select last and --no-keep-stock are required); in a fold's export "
                         "dev.jsonl holds the held-out clips, so a model that chose on dev would have seen them")
    folds, threads = int(lab.cfg["cv_folds"]), int(lab.cfg["lm_threads"])
    groups = [g for g in ("seeds", "positions") if any(g in WIDE_GROUPS[a] for a in arms)]
    units = [None] + (list(range(folds)) if "folds" in wanted else [])
    extra_count = sum(len(WIDE_SEEDS) if g == "seeds" else len(WIDE_POSITIONS) for g in groups)
    views_per_extra = len(pool_module.parse_extra_views(extra_views))
    lab.say(f"######## 39: wider candidate pools ({' '.join(arms)}; {' '.join(wanted)}; {len(tags)} language model), started {now()}")
    lab.say("Lab 36's pools hold nine decodes of each clip by one model, stage P's adapter. Here the same clips are also decoded by other models:")
    lab.say(f"{extra_count} extra model(s) ({' '.join(groups)}), {views_per_extra} decode(s) each, so {extra_count * views_per_extra} extra decode(s) a clip on top of nine. Every answer,")
    lab.say("whichever model proposed it, is scored against the unchanged audio by stage P's adapter and as English by the same text model.")
    need = [f"{EX}/{s}/{split}.jsonl" for s in lab.spk for split in ("train", "dev", "test")]
    if not lab.need_have(*need) or not lab.need_free(f"{n}/table.json") or not lab.act() or not lm_environment(lab, tags):
        return False
    if not ensure_pretrained_models(lab):
        return False
    failed: list[str] = []
    pools: dict[str, dict[str, list[str]]] = {a: {"test": [], "folds": []} for a in (WIDE_BASELINE, *arms)}
    for s in lab.spk:
        if "folds" in wanted and not (lab.run([lab.sc, CV_PY, "make", "--export", f"{EX}/{s}", "--speaker", s, "--folds", str(folds)])
                                      and lab.run([lab.sc, CV_PY, "check", "--speaker", s, "--folds", str(folds)])):
            lab.say(f"FAILED building the folds for {s}; python lab.py doctor --fix repairs missing audio")
            failed.append(f"folds:{s}")
            continue
        for fold in units:
            for arm, path in wide_pools(lab, wide_unit(s, fold), arms, groups, extra_views, failed).items():
                pools[arm]["test" if fold is None else "folds"].append(path)
        lab.trim_model(f"big-c-pre-{s}")
    every = len(lab.spk)
    systems: list[tuple[str, str, list[str]]] = []
    for arm in (WIDE_BASELINE, *arms):
        have = pools[arm]
        if "prototype" in wanted and len(have["test"]) == every:
            systems.append((arm, "stage P's adapter, the test clips", have["test"]))
        if "folds" in wanted and len(have["test"]) == every and len(have["folds"]) == every * folds:
            systems.append((arm, "stage P's recipe, every labelled clip held out", have["test"] + have["folds"]))
    if not systems and not lab.dry:
        summary_lines(lab, failed)
        lab.say("no arm has a complete set of pools; nothing to score")
        return False
    names = {(arm, kind): f"{arm}: {kind}" for arm, kind, _paths in systems}
    every_pool = list(dict.fromkeys(p for _a, _k, paths in systems for p in paths))
    lab.banner("the language model(s), one at a time")
    scored: list[str] = []
    for tag in tags:
        py, out = lm_python(lab, tag), f"{n}/scores/{tag}.json"
        if not lab.dry:
            earlier = lab.root / LM36 / "scores" / f"{tag}.json"
            if earlier.is_file() and not lab.exists(out):
                (lab.root / out).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(earlier, lab.root / out)
                lab.say(f"started {out} from lab 36's scores of {tag}: answers it already scored are not scored again")
            try:
                have_texts = json.loads((lab.root / out).read_text(encoding="utf-8"))["texts"]
                rows = lm_score.read_pools([str(lab.root / p) for p in every_pool])
                if all(t in have_texts for t in lm_score.pool_texts(rows)):
                    lab.say(f"reusing {out}: it already covers these pools")
                    scored.append(out)
                    continue
            except (OSError, ValueError, KeyError):
                pass
        kept = not lab.dry and lm_score.present(lm_score.get(tag))
        if not lab.run([py, "scripts/asr/lm_score.py", "fetch", "--models", tag]):
            lab.say(f"FAILED downloading {tag}")
            failed.append(f"fetch:{tag}")
            continue
        if lab.run([py, "scripts/asr/lm_score.py", "score", "--model", tag, "--pool", *every_pool, "--threads", str(threads), "--extend", "--out", out]):
            scored.append(out)
        else:
            lab.say(f"FAILED scoring with {tag}")
            failed.append(f"score:{tag}")
        if lab.trim_on() and not kept:
            lab.say(f"TRIM_MODELS: removing the files of {tag} (its scores are kept)")
            lab.run([py, "scripts/asr/lm_score.py", "release", "--models", tag])
    summary_lines(lab, failed)
    if not scored and not lab.dry:
        lab.say("no language model produced scores; no table")
        return False
    args = [x for arm, kind, paths in systems for x in ("--system", names[(arm, kind)] + "=" + ",".join(paths))]
    for arm, kind, _paths in systems:
        if arm != WIDE_BASELINE and (WIDE_BASELINE, kind) in names:
            args += ["--compare", f"{names[(arm, kind)]}={names[(WIDE_BASELINE, kind)]}"]
    if not lab.run([lab.sc, "scripts/asr/lm_report.py", *args, "--scores", *scored, "--out", f"{n}/table.json"]):
        failed.append("report")
    else:
        lab.track_table(f"{n}/table.json")
    lab.say("Read the baseline table first, then each arm's table and the 'paired change' block under them (the second interval resamples prompts).")
    lab.say("An arm counts only by the rules in the notes: ceiling and chooser both up against the baseline, with the cost per clip beside them.")
    lab.say(f"39 finished {now()}")
    return not failed


def stage_23(lab: Lab) -> bool:
    n = SEL23
    folds = int(lab.cfg["selector_folds"])
    lab.say(f"######## 23: choosing among the systems' answers for each clip, started {now()}")
    recorded = (("adapter-on-F-1", "adf1", lab.adf1_h), ("adapter", "ad", lab.ad_h))
    for s in lab.spk:
        if not lab.need_have(lab.f1_m(s), f"{EX}/{s}/test.jsonl", lab.stock_h(s) + "/hypotheses.jsonl", lab.f1_h(s) + "/hypotheses.jsonl",
                             *[fn(s) + "/hypotheses.jsonl" for _label, _arm, fn in recorded]):
            return False
        for _label, arm, _fn in recorded:
            if not lab.need_have(f"{CV20}/oof/{arm}-{s}.jsonl"):
                lab.say("Lab test 20 makes these out-of-fold answers; run it first (its cv_arms must include adf1 and ad).")
                return False
    if not lab.need_free(f"{n}/results.json", f"{n}/table.json") or not lab.act():
        return False
    failed: list[str] = []
    labelled: dict[str, dict[str, str]] = {"stock": {}, "F-1": {}}
    for s in lab.spk:
        if not lab.run([lab.sc, CV_PY, "make", "--export", f"{EX}/{s}", "--speaker", s, "--folds", str(int(lab.cfg["cv_folds"]))]) and not lab.dry:
            lab.say(f"FAILED the folders for {s}")
            return False
        for label, flags in (("stock", ["--stock"]), ("F-1", ["--nemo", lab.f1_m(s)])):
            out = f"{n}/labelled/{safe(label)}-{s}.jsonl"
            labelled[label][s] = out
            lab.banner(f"{s}: {label} model on his labelled clips (it never saw them)")
            if not decode_manifest(lab, flags, f"{EX}/cv-{s}-all/test.jsonl", f"{EX}/cv-{s}-all", out):
                failed.append(f"decode:{label}:{s}")
    if failed:
        summary_lines(lab, failed)
        return False
    spec = {"stock": {"labelled": labelled["stock"], "test": {s: lab.stock_h(s) for s in lab.spk}},
            "F-1": {"labelled": labelled["F-1"], "test": {s: lab.f1_h(s) for s in lab.spk}}}
    for label, arm, fn in recorded:
        spec[label] = {"labelled": {s: f"{CV20}/oof/{arm}-{s}.jsonl" for s in lab.spk}, "test": {s: fn(s) for s in lab.spk}}
    if not lab.dry:
        (lab.root / n).mkdir(parents=True, exist_ok=True)
        (lab.root / n / "systems.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    cmd = [lab.sc, SEL_PY, "--speaker", *lab.spk, "--systems-file", f"{n}/systems.json", "--folds", str(folds),
           "--phrases-from", f"{EX}/loso-{{speaker}}/train.jsonl", f"{EX}/loso-{{speaker}}/dev.jsonl", "--out", n]
    if not lab.run(cmd):
        failed.append("selector")
        summary_lines(lab, failed)
        return False
    systems = ["--system", lab.sysfn("stock", lab.stock_h), "--system", lab.sysfn("F-1", lab.f1_h), "--system", lab.sysfn("adapter-on-F-1", lab.adf1_h)]
    for method in ("vote", "wvote", "medoid", "learned"):
        systems += ["--system", lab.sysfmt(f"selector-{method}", f"{n}/hyps/{method}-@.jsonl")]
    summary_lines(lab, failed)
    if not lab.pooled(*systems, "--baseline", "stock", "--out", f"{n}/table.json"):
        failed.append("table")
    lab.track_table(f"{n}/results.json")
    lab.say("The learned selector was fitted on out-of-fold answers for his labelled clips; its labelled-clip score is leave-prompt-out. The test clips were read once, to count.")
    lab.say(f"23 finished {now()}")
    return not failed


LONG_RUNS = {
    "long-adapter": ("adapter", "f2-long-adapter", "long adapter training: epochs, text form and learning rate", 33),
    "acoustic-only": ("acoustic", "f2-acoustic-only", "100-epoch acoustic adaptation: depth and frozen text head", 39),
    "speaker-replay": ("replay", "f2-speaker-replay", "100-epoch personal adaptation with other-speaker replay", 24),
}
PENDING_QUEUE = ["L", "T", "E", "S", "C", "A", "N", "F", "W", *LONG_RUNS]
NEW_QUEUE = ["16", "17", "18", "19", "20", "23", "21", "22", "25", "29", "28", "30", "27", "26"]
FOLLOWUP_QUEUE = ["24", "P", "G2", *NEW_QUEUE, "B"]
IDEAS_QUEUE = ["P", "U", "R", "K", "G"]
ACOUSTIC_QUEUE = ["31", "32", "33"]
TONIGHT_QUEUE = ["34", "speaker-replay"]
FINAL_QUEUE = ["35"]
LM_QUEUE = ["36"]
ROUTING_QUEUE = ["37"]
SELFTRAIN_LM_QUEUE = ["38"]
REMAINING_QUEUE = ["5", "24", "28", "30", "19", "20", "23", "25", "27", "26", "B", "16", "17", "18", "29",
                   "L", "T", "E", "S", "C", "acoustic-only", "speaker-replay"]
AUTO_REBUILD = [*PENDING_QUEUE, *NEW_QUEUE, *IDEAS_QUEUE, *FOLLOWUP_QUEUE, *REMAINING_QUEUE, *ACOUSTIC_QUEUE, *TONIGHT_QUEUE, *FINAL_QUEUE, *LM_QUEUE, *ROUTING_QUEUE]
AUTO_REBUILD += SELFTRAIN_LM_QUEUE

WIDE_QUEUE = ["39"]
AUTO_REBUILD += WIDE_QUEUE


def long_stage(study: str):
    def run(lab: Lab) -> bool:
        flags = shlex.split(str(lab.cfg["long_flags"]))
        allowed = {"--batch", "--grad-accum", "--offload-optimizer", "--keep-all-models", "--min-free-gb"}
        if any(not key.startswith("--") or key not in allowed for key in flags if key.startswith("-")):
            raise ValueError("long_flags accepts only batch, grad-accum, offload-optimizer, keep-all-models and min-free-gb")
        lab.say("Long experiments use fixed M01/M02/M04 speakers and seeds 17/18/19; sweep overrides do not change them.")
        return lab.run([lab.sc, "scripts/asr/long_runs.py", study, lab.stamp,
                        "--resume", "--restart-changed", "--train-python", str(TRAIN_PY), "--cpu-python", lab.sc,
                        *flags, *(["--dry-run"] if lab.dry else [])])
    return run


class Exp:
    def __init__(self, id: str, title: str, kind: str, group: str, fn, markers: list[str], est: str = "", notes: str = "", runs: int = 0):
        self.id, self.title, self.kind, self.group, self.fn = id, title, kind, group, fn
        self.markers, self.est, self.notes, self.runs = markers, est, notes, runs


LAB_TESTS = {
    "1": ("weight interpolation, stock to whole-block at 25, 50, 75%", "decode", "~20 min", "lab-01-weight-interpolation", ["table.json"]),
    "2": ("merging the speakers' weight changes", "decode", "~20 min", "lab-02-weight-merging", ["table.json"]),
    "3": ("best-of-N and vote bounds over existing hypotheses", "cpu", "seconds", "lab-03-hypothesis-bounds", ["bounds"]),
    "4": ("stock severe-speech screen of candidate models", "decode", "~1 h", "lab-04-stock-model-screen", ["table.json"]),
    "5": ("which encoder layer matches words best", "decode", "~1 h", "lab-05-layer-screen", ["results.json", "results-last-session.json"]),
    "6": ("how likely the model finds the true words and silence", "decode", "~15 min", "lab-06-reference-score", None),
    "7": ("does stock WER predict fine-tuned WER (4 models, M02 and M04)", "train", "~4 h", "lab-07-stock-vs-fine-tuned", ["rank.json"]),
    "8": ("feed-forward-only fine-tuning in blocks 0-7", "train", "~1.5 h", "lab-08-feed-forward-only", ["table.json"]),
    "8b": ("bisect: feed_forward1 only, then feed_forward2 only (after 8)", "train", "~3 h", "lab-08-feed-forward-only", ["table-bisect.json"]),
    "9": ("depth sweep, 5 arms x 3 speakers", "train", "~7 h", "lab-09-depth-sweep", ["table.json"]),
    "10": ("anti-silence margin loss and non-speech clips", "train", "~5 h", "lab-10-silence-margin", ["table.json", "battery.json"]),
    "11": ("fine-tune on the other severe speakers, then on his own sessions", "train", "~4 h", "lab-11-severe-pretraining", ["table.json"]),
    "12": ("label-free pseudo-label loop, 3 rounds x 3 speakers", "train", "~5 h", "lab-12-pseudo-label-loop", ["table.json"]),
    "13": ("false inserts on clips with no speech", "decode", "~15 min", "lab-13-silence-false-inserts", ["score.json"]),
    "14": ("words to the matcher, sentences to Parakeet", "cpu", "~5 min", "lab-14-route-scoring", ["table.json"]),
    "15": ("int8 screen of checkpoints and exported files", "cpu", "~5 min", "lab-15-int8-screen", ["screen.json"]),
    "16": ("healthy recordings made to sound dysarthric: how hard for the stock model", "decode", "~1 h", "lab-16-synthetic-realism", ["summary.json"]),
    "17": ("training on synthetic dysarthric speech, three ways", "train", "~8 h", "lab-17-synthetic-training", ["table.json"]),
    "18": ("adapter on F-1 with silence protection and synthetic speech", "train", "~4 h", "lab-18-best-system-hardened", ["table.json", "battery.json"]),
    "19": ("what int8 costs: three export recipes, two models, three speakers", "train", "~3 h", "lab-19-int8-cost",
           ["table.json", *[f"{m}-{s}-{v}/results.json" for m in ("whole-block", "adapter-on-F-1") for s in SEVERE for v in INT8_VARIANTS]]),
    "20": ("grouped cross-validation: which adapter recipe wins on held-out prompts", "train", "~6 h", "lab-20-grouped-cv", ["selection.json", "table.json"]),
    "21": ("training on text-to-speech made to sound dysarthric", "train", "~3 h", "lab-21-tts-synthetic", ["table.json", "battery.json"]),
    "22": ("test-time adaptation on his own recordings, no transcripts", "train", "~3 h", "lab-22-test-time-adaptation", ["table.json"]),
    "23": ("a selector over the systems' answers for each clip", "decode", "~30 min", "lab-23-hypothesis-selector", ["results.json", "table.json"]),
    "24": ("which listed phrase the audio supports: acoustic scoring against the phrase list, settings chosen on other speakers, coverage", "decode", "~3 h with the F-1 baselines retrained",
           "lab-24-phrase-scoring", ["report/results.json"]),
    "25": ("test-time adaptation on the pretrained adapter at 6e-4, three seeds: do the two gains add?", "train", "~3 h with the pretrained models retrained",
           "lab-25-tta-on-pretrained", ["table.json"]),
    "26": ("the pretraining recipe under the adapter: bottom 8 blocks, a higher rate, a longer schedule, all dysarthric then severe", "train", "~3 h",
           "lab-26-pretraining-recipe", ["table.json"]),
    "27": ("test-time adaptation settings, one at a time: 8 chunks, recorded order, gate 0.3 / 0.7 / none, half cap", "train", "~2 h",
           "lab-27-tta-settings", ["table.json"]),
    "28": ("the deployable export: the pretrained adapter as int8 default and matmul, decoded by sherpa-onnx", "train", "~30 min",
           "lab-28-deployable-export", ["table.json"]),
    "29": ("the hard core: clips and words every saved system gets wrong, by kind and duration", "cpu", "seconds",
           "lab-29-hard-core", ["hard_core.json"]),
    "30": ("acoustic phrase scoring on the pretrained adapter (lab test 24 on the best raw system)", "train", "~30 min",
           "lab-30-phrase-scoring-pretrained", ["report/results.json"]),
    "31": ("an adapter on each speaker's own acoustic-only model (all 24 blocks, head frozen, 100 epochs), three seeds", "train", "~2 h",
           "lab-31-adapter-on-acoustic-only", ["finished.json"]),
    "32": ("the acoustic recipe on the other severe speakers, then stage P's adapter, three seeds", "train", "~2.5 h",
           "lab-32-acoustic-severe-pretraining", ["finished.json"]),
    "33": ("one acoustic model on every dysarthric speaker, then a per-speaker adapter at five positions, three seeds", "train", "~4.5 h",
           "lab-33-pooled-acoustic-adapters", ["finished.json"]),
    "34": ("stage P's adapter after blocks 8, 12, 16 and 24 on the severe-pretrained model: accuracy and false words, three seeds", "train", "~2.5 h",
           "lab-34-pretrained-adapter-positions", ["finished.json"]),
    "35": ("silence preference: stock, whole-block and stage P's adapter on the severe speakers, stock on the controls, true words in both forms", "train",
           "~45 min", "lab-35-silence-preference", ["table.json"]),
    "36": ("language-model rescoring: several decodes per clip, the answer chosen by audio score plus a small text model, on stock and on stage P's adapter",
           "train", "~6-8 h with the pretrained models, fold adapters and the pi group; ~1.5 h for stock alone", "lab-36-lm-rescoring", ["table.json"]),
    "37": ("kind routing: adapters trained on single words only and on sentences only beside the control, answers routed by a detector the device can run",
           "train", "~1.5-2 h with the pretrained models present, about 25 min more if they are rebuilt (estimate)", "lab-37-kind-routing", ["finished.json"]),
    "38": ("self-training on language-model-chosen pseudo-labels: lab 25's loop on stage P's adapter with the sentences' labels picked by the recogniser plus a small text model, three arms, three seeds",
           "train", "~4 h with the pretrained models and static adapters on hand; ~6 h on a fresh checkout", "lab-38-lm-pseudo-labels", ["table.json"]),
    "39": ("wider candidate pools: answers from stage P's other seeds and the block-12 and block-16 adapters added to lab 36's pools, scored by the same audio model and language model",
           "train", "~4-5 h with the folds (about 75 adapter trainings); ~1 h for the test clips alone", "lab-39-wider-pools", ["table.json"]),
}
BIG_RUNS = {
    "W": ("phrasebook matching: single words, and the whole test set", "cpu", "~1 min", "phrasebook-matching", None),
    "B": ("larger and other base models with the adapter on top", "train", "~2 h", "f2-base-model-ceiling", None),
    "A": ("augmentation variants that change what runs already do (3 seeds)", "train", "~1.5 h", "f2-augmentation-variants", None),
    "N": ("tempo stretch and white-noise augmentation (3 seeds)", "train", "~1.5 h", "f2-augmentation-variants", None),
    "F": ("data-quantity curve on M02, two schedules, three seeds", "train", "~1.5 h", "f3-m02-data-quantity", None),
    "U": ("adapter soup: average the weights of five seeds per speaker", "train", "~1 h", "f2-adapter-soup", None),
}
BIG_MARKERS = {
    "W": ["reports/phrasebook-matching/words", "reports/phrasebook-matching/device-both"],
    "B": [B_DONE],
    "A": [f"{FT}/f2-M04-adapter8-aug-nospec-seed19-test-nemo"],
    "N": [f"{FT}/f2-M04-adapter8-aug-tempo-noise-seed19-test-nemo"],
    "F": [f"{FT}/f3-M02-min7-updates-seed19-test-nemo"],
    "U": ["reports/lab-big-u/summary.json"],
}
LAB_MARKERS_EXTRA = {"6": [f"reports/lab-06-reference-score/{s}-whole-block.json" for s in SEVERE]}
DEFAULT_CHEAP = "14 3 15 1 2 6 13 4 5"


def build_catalog(lab: Lab) -> dict[str, Exp]:
    cat: dict[str, Exp] = {}
    fns = {k[len("stage_"):]: v for k, v in globals().items() if k.startswith("stage_") and callable(v) and not k.startswith("stage_grid")}
    for sid, (title, kind, est, folder, markers) in LAB_TESTS.items():
        marks = [f"reports/{folder}/{m}" for m in markers] if markers else LAB_MARKERS_EXTRA.get(sid, [])
        cat[sid] = Exp(sid, title, kind, "lab test", fns[sid], marks, est, f"reports/{folder}/notes.md")
    for sid, (title, kind, est, folder, _m) in BIG_RUNS.items():
        cat[sid] = Exp(sid, title, kind, "2 Oct big run", fns[sid], BIG_MARKERS[sid], est, f"reports/{folder}/notes.md")
    for gid, (title, arms, seeds, note) in GRIDS.items():
        runs = grid_run_names(lab, gid, arms(), seeds)
        marks = [f"{FT}/{name}-test-nemo.jsonl" for name in runs] + [f"reports/lab-big-{gid.lower()}/summary.json"]
        cat[gid] = Exp(gid, title, "train", "sweep", stage_grid(gid), marks, f"{len(runs)} runs", note, len(runs))
    for sid, (study, folder, title, count) in LONG_RUNS.items():
        cat[sid] = Exp(sid, title, "train", "long experiment", long_stage(study), long_markers(lab, folder),
                       f"up to {count} runs", f"reports/{folder}/notes.md", count)
    for spec in custom_experiments(lab.root):
        gid = "c:" + spec["name"]
        n = len(custom_arms(spec)) * len(spec.get("seeds", [17])) * len(spec.get("speakers") or lab.spk)
        cat[gid] = Exp(gid, spec.get("title") or spec["name"], "train", "yours", custom_stage(spec),
                       [f"reports/lab-big-c-{safe(spec['name']).lower()}/summary.json"], f"{n} runs", "defined in lab_experiments.json", n)
    return cat


def grid_run_names(lab: Lab, gid: str, arms: list, seeds: list[int], speakers: list[str] | None = None) -> list[str]:
    seeds = [int(x) for x in str(lab.cfg["seeds"]).split()] or list(seeds)
    return [f"big-{run_tag(gid)}-{safe(arm.label)}-{s}-s{seed}" for seed in seeds for arm in arms for s in (speakers or lab.spk)]


def long_markers(lab: Lab, folder: str) -> list[str]:
    base = lab.root / "reports" / folder
    for d in sorted(base.glob("*/summary.json")) if base.is_dir() else []:
        if (d.parent / "battery.json").is_file():
            return [str(p.relative_to(lab.root)) for p in (d, d.parent / "battery.json")]
    return [f"reports/{folder}/{lab.stamp}/{name}" for name in ("summary.json", "battery.json")]


def done_state(lab: Lab, exp: Exp) -> str:
    if not exp.markers:
        return "?"
    present = [lab.exists(m) for m in exp.markers]
    if all(present):
        return "done"
    return "partial" if any(present) else "new"


def code_version(lab: Lab) -> str:
    done = subprocess.run(["git", "log", "-1", "--format=%h %ad", "--date=format:%H:%M"], cwd=lab.root, capture_output=True, text=True, check=False)
    return done.stdout.strip() or "no git history here"


def sha12(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12] if path.is_file() else "missing"


def busy_processes() -> list[str]:
    done = subprocess.run(["pgrep", "-af", r"scripts/asr/(train|nemo_hyp|long_runs)\.py|lab\.py (run|queue)"], capture_output=True, text=True, check=False)
    me = str(os.getpid())
    return [ln[:200] for ln in done.stdout.splitlines() if ln.split(" ", 1)[0] != me and "pgrep" not in ln]


def resolve(lab: Lab, tokens: list[str]) -> list[str]:
    ids: list[str] = []
    for t in tokens:
        for part in t.replace(",", " ").split():
            key = part if part in lab.catalog else part.upper() if part.upper() in lab.catalog else part
            if key == "cheap":
                ids += DEFAULT_CHEAP.split()
            elif key == "all-lab":
                ids += [k for k in lab.catalog if k in LAB_TESTS]
            elif key == "sweeps":
                ids += list(GRIDS)
            elif key == "pending":
                ids += PENDING_QUEUE
            elif key in ("new", "synthetic"):
                ids += NEW_QUEUE
            elif key == "ideas":
                ids += IDEAS_QUEUE
            elif key == "followup":
                ids += FOLLOWUP_QUEUE
            elif key == "acoustic":
                ids += ACOUSTIC_QUEUE
            elif key == "positions":
                ids += TONIGHT_QUEUE
            elif key == "routing":
                ids += ROUTING_QUEUE
            elif key == "selftrain-lm":
                ids += SELFTRAIN_LM_QUEUE
            elif key == "widepool":
                ids += WIDE_QUEUE
            elif key == "remaining":
                ids += REMAINING_QUEUE + [k for k in lab.catalog if k not in REMAINING_QUEUE and done_state(lab, lab.catalog[k]) != "done"
                                          and done_state(lab, lab.catalog[k]) != "?"]
            else:
                ids.append(key)
    return ids


STORAGE_FOLDERS = ("models", "reports", "logs", f"{EX}", "data/public/torgo/TORGO", "data/emb")


def ram_available_gb() -> float | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            info = {line.split(":")[0]: int(line.split()[1]) for line in f if line.split()}
        return round(info["MemAvailable"] / 2**20, 1)
    except (OSError, ValueError, KeyError, IndexError):
        return None


def model_folder_parts(folder: Path, tree_size) -> dict:
    parts = {"bytes": tree_size(folder)}
    for key, name in (("ckpt", "ckpt"), ("nemo", "parakeet-ft.nemo"), ("adapter", "adapter.pt"), ("onnx", "onnx-int8")):
        path = folder / name
        if path.exists():
            parts[key] = tree_size(path)
    return parts


def storage_snapshot(lab: Lab, label: str, biggest: int = 6) -> dict | None:
    if lab.dry:
        return None
    try:
        import lab_space

        gb = 2**30
        usage = shutil.disk_usage(lab.root)
        tmp = Path(tempfile.gettempdir())
        tmp_free = shutil.disk_usage(tmp).free / gb if tmp.is_dir() else None
        folders = {name: lab_space.tree_size(lab.root / name) for name in STORAGE_FOLDERS if (lab.root / name).exists()}
        models = []
        for root in (lab.root / "models", lab.root / "models/lab-baselines"):
            for d in sorted(root.iterdir()) if root.is_dir() else []:
                if d.is_dir() and d.name != "lab-baselines":
                    models.append({"path": d.relative_to(lab.root).as_posix(), **model_folder_parts(d, lab_space.tree_size)})
        models.sort(key=lambda m: m["bytes"], reverse=True)
        record = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "stamp": lab.stamp, "stage": lab.stage, "label": label,
                  "disk": str(lab.root), "free_gb": round(usage.free / gb, 1), "total_gb": round(usage.total / gb, 1),
                  "used_pct": round(100 * usage.used / usage.total, 1), "tmp": str(tmp),
                  "tmp_free_gb": None if tmp_free is None else round(tmp_free, 1), "ram_available_gb": ram_available_gb(),
                  "folders_gb": {k: round(v / gb, 2) for k, v in folders.items()}, "model_folders": len(models),
                  "models_with_ckpt": sum(1 for m in models if m.get("ckpt")),
                  "biggest_models": [{k: (round(v / gb, 2) if isinstance(v, int) else v) for k, v in m.items()} for m in models[:biggest]]}
        tmp_text = "?" if tmp_free is None else f"{tmp_free:.1f}"
        lab.say(f"STORAGE {label}: {record['free_gb']} GB free of {record['total_gb']} GB ({record['used_pct']}% used) at {lab.root}; "
                f"temp {tmp} {tmp_text} GB free; RAM available {record['ram_available_gb']} GB")
        lab.say("  folders: " + ", ".join(f"{k} {v:.2f} GB" for k, v in record["folders_gb"].items()))
        lab.say(f"  {record['model_folders']} model folders, {record['models_with_ckpt']} still holding ckpt/; biggest:")
        for m in record["biggest_models"]:
            inside = ", ".join(f"{k} {m[k]:.2f} GB" for k in ("ckpt", "nemo", "adapter", "onnx") if k in m)
            lab.say(f"    {m['bytes']:>7.2f} GB  {m['path']}" + (f"  ({inside})" if inside else ""))
        lab.logdir.mkdir(parents=True, exist_ok=True)
        with (lab.logdir / f"lab-{lab.stamp}-storage.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        return record
    except Exception as exc:  # noqa: BLE001
        lab.say(f"STORAGE {label}: snapshot failed ({type(exc).__name__}: {exc}); the stage continues")
        return None


def run_stages(lab: Lab, tokens: list[str]) -> bool:
    ids = resolve(lab, tokens) or DEFAULT_CHEAP.split()
    if any(st in AUTO_REBUILD for st in ids):
        lab.cfg.overrides["rebuild_missing"] = True
    lab.logdir.mkdir(parents=True, exist_ok=True)
    lab.say(f"lab runs {lab.stamp}, stages: {' '.join(ids)}, started {time.strftime('%Y-%m-%d %H:%M:%S')}"
            + ("  (DRY RUN: nothing is executed)" if lab.dry else ""))
    lab.say(f"code: {code_version(lab)}")
    lab.say(f"train.py sha256: {sha12(lab.root / 'scripts/asr/train.py')}   labkit.py sha256: {sha12(lab.root / 'scripts/asr/labkit.py')}")
    if not lab.dry:
        busy = busy_processes()
        if busy:
            lab.say("a lab run is already going, stopping:")
            for line in busy:
                lab.say("  " + line)
            return False
        done = smi("name,memory.used,memory.total")
        if done.stdout.strip():
            lab.say(done.stdout.strip())
    lab.say(f"disk free here: {shutil.disk_usage(lab.root).free / 2**30:.0f} GB")
    lab.say(f"Weights & Biases: {lab.cfg['wandb']} (project {lab.cfg['wandb_project']})")
    failed_any = False
    stage_lines: dict[str, list[str]] = {}
    for st in ids:
        lab.stage = st
        exp = lab.catalog.get(st)
        if exp is None:
            lab.say(f"unknown stage {st}, skipping")
            continue
        log_path = lab.logdir / f"lab-{lab.stamp}-{safe(st)}.txt"
        lab.log = log_path.open("w", encoding="utf-8")
        lab.keep_weights = st in KEEP_BASELINE_WEIGHTS
        start = len(lab.lines)
        lab.say("")
        lab.say(f"================ stage {st}, {now()} ================")
        storage_snapshot(lab, f"before stage {st}")
        ok = True
        try:
            if lab.rebuild and not lab.dry and st in Lab.BASELINE_NEEDS:
                if not (lab.act() and lab.prepare_baselines(st)):
                    lab.say(f"FAILED prerequisite preparation for stage {st}; see {log_path}")
                    ok = False
            if ok:
                ok = bool(exp.fn(lab))
        except KeyboardInterrupt:
            lab.say(f"interrupted during stage {st}")
            lab.log.close()
            lab.log = None
            raise
        except Exception as exc:  # noqa: BLE001
            lab.say(f"FAILED stage {st}: {type(exc).__name__}: {exc}")
            ok = False
        storage_snapshot(lab, f"after stage {st}" if ok else f"after stage {st} failed")
        lab.say(f"stage {st} done {now()}" if ok else f"FAILED stage {st}; see {log_path}")
        stage_lines[st] = lab.lines[start:]
        lab.log.close()
        lab.log = None
        if not ok:
            failed_any = True
            lab.failed_stages.append(st)
    summaries = lab.logdir / f"lab-{lab.stamp}-summaries.txt"
    with summaries.open("w", encoding="utf-8") as handle:
        for st, lines in stage_lines.items():
            handle.write(f"\n######## {st}\n")
            handle.write("\n".join(lines[:3]) + "\n")
            for n, line in enumerate(lines, 1):
                if re.search(r"FAILED|ALREADY|MISSING", line):
                    handle.write(f"{n}:{line}\n")
            if "######## summary" in lines:
                handle.write("\n".join(lines[lines.index("######## summary"):]) + "\n")
    lab.say("")
    lab.say(f"SUMMARY FILE: {summaries}")
    lab.say(f"code at the end: {code_version(lab)}")
    lab.say(f"lab runs finished {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return not failed_any


DEFAULT_QUEUE = ["8", "8b", "10", "5", "15", "T", "L", "C", "E", "S", "B", "A", "N", "F", "W"]


def free_gb(root: Path) -> int:
    return int(shutil.disk_usage(root).free / 2**30)


SYNTHETIC_STAGES = ("16", "17", "18")


def doctor_gate(lab: Lab, st: str, minimum: int) -> list[str]:
    if lab.dry:
        return []
    need_controls = st in SYNTHETIC_STAGES
    cpu, train = sys.executable, str(TRAIN_PY)

    def failures() -> tuple[list[lab_doctor.Finding], list[str]]:
        found = lab_doctor.run_checks(lab.root, cpu, train, need_controls=need_controls, need_gb=minimum, need_gpu=st not in ("14",))
        return found, [f"[{f.area}] {f.message}" for f in found if f.level == "fail" and f.area in ("runtime", "disk", "data")]

    try:
        found, bad = failures()
        if bad and lab.cfg["auto_fix"] and any(f.fix for f in found if f.level != "ok"):
            lab.say(f"DOCTOR: {len(bad)} problem(s) before {st}; repairing")
            for line in lab_doctor.apply_fixes(found, lab.root, cpu, train, bool(lab.cfg["allow_download"])):
                lab.say("  " + line)
            found, bad = failures()
        for line in bad:
            lab.say(f"DOCTOR: still failing: {line}")
        return bad
    except Exception as exc:  # noqa: BLE001
        lab.say(f"DOCTOR: skipped, it failed itself: {type(exc).__name__}: {exc}")
        return []


def export_models(lab: Lab, target_free_gb: int) -> bool:
    place = str(lab.cfg["export_to"]).strip()
    if not place:
        return False
    lab.say(f"EXPORT: sending finished models to {place} until {target_free_gb} GB is free")
    return lab.run([sys.executable, "scripts/asr/model_export.py", "export", "--to", place, "--apply", "--free-after-verify",
                    "--until-free-gb", str(target_free_gb)])


def write_status(lab: Lab, state: dict) -> None:
    if lab.dry:
        return
    path = lab.logdir / f"lab-queue-{lab.stamp}.status.json"
    state = {**state, "updated": time.strftime("%Y-%m-%d %H:%M:%S"), "free_gb": free_gb(lab.root)}
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        pass


def stage_log_text(lab: Lab, st: str) -> str:
    path = lab.logdir / f"lab-{lab.stamp}-{safe(st)}.txt"
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-300:])
    except OSError:
        return ""


def run_stage_with_retries(lab: Lab, st: str, minimum: int) -> bool:
    tries = int(lab.cfg["stage_retries"])
    for attempt in range(tries + 1):
        if run_stages(lab, [st]):
            return True
        lab.failed_stages = [x for x in lab.failed_stages if x != st]
        why = lab_doctor.classify_failure(stage_log_text(lab, st))
        if attempt >= tries or not why.retryable:
            lab.failed_stages.append(st)
            lab.say(f"QUEUE: {st} failed ({why.category}); not retrying. {why.remedy}")
            return False
        lab.say(f"QUEUE: {st} failed ({why.category}); fixing and retrying ({attempt + 1} of {tries}). {why.remedy}")
        if why.category == "disk":
            lab.run([sys.executable, "scripts/asr/lab_space.py", "clean", "--apply", "--lab-models", "--long-models"])
            if free_gb(lab.root) < minimum:
                export_models(lab, minimum + 20)
        elif why.category == "network":
            time.sleep(60)
        elif why.category in ("oom", "killed"):
            time.sleep(20)
        doctor_gate(lab, st, minimum)
    return False


def remaining_defaults(lab: Lab, tokens: list[str]) -> bool:
    if not any(str(t).lower() == "remaining" for t in tokens):
        return False
    lab.cfg.overrides["rebuild_missing"] = True
    for name in ("allow_download", "push_results", "wait_busy", "rerun"):
        if lab.cfg.source(name)[1] == "default":
            lab.cfg.overrides[name] = True
    return True


def push_results(lab: Lab) -> bool:
    if not lab.cfg["push_results"] or lab.dry:
        return False
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    try:
        done = subprocess.run(["git", "push", "origin", "refs/heads/lab-results-*:refs/heads/lab-results-*"], cwd=lab.root,
                              capture_output=True, text=True, env=env, timeout=900, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        lab.say(f"PUSH: failed ({type(exc).__name__}); results stay on the local lab-results-* branches")
        return False
    if done.returncode == 0:
        lab.say("PUSH: lab-results-* branches are on origin")
        return True
    lab.say("PUSH: failed; results stay on the local lab-results-* branches. Git said:")
    for line in (done.stderr or done.stdout).strip().splitlines()[-4:]:
        lab.say("  " + line)
    return False


def run_queue(lab: Lab, tokens: list[str], until: str | None = None, rerun: bool = False) -> bool:
    ids = resolve(lab, tokens) or list(DEFAULT_QUEUE)
    if any(st in AUTO_REBUILD for st in ids):
        lab.cfg.overrides["rebuild_missing"] = True
    if remaining_defaults(lab, tokens):
        seen: set[str] = set()
        ids = [st for st in ids if not (st in seen or seen.add(st))]
        todo = [st for st in ids if st in lab.catalog and done_state(lab, lab.catalog[st]) != "done"]
        lab.say(f"remaining: {len(todo)} of {len(ids)} experiments are not finished on this computer: {' '.join(todo) or '(none)'}")
    deadline = None
    if until:
        hour, minute = (int(x) for x in until.split(":"))
        deadline = time.mktime(time.strptime(time.strftime("%Y-%m-%d ") + f"{hour:02d}:{minute:02d}", "%Y-%m-%d %H:%M"))
        if deadline < time.time():
            deadline += 86400
    if rerun:
        lab.cfg.overrides.update({"rerun": True, "rebuild_missing": True})
    if str(lab.cfg["trim_models"]) == "auto" and free_gb(lab.root) < 100 and "trim_models" not in lab.cfg.overrides:
        lab.cfg.overrides["trim_models"] = "on"
    minimum = int(lab.cfg["min_free_gb"]) or (14 if lab.trim_on() else 40)
    lab.say(f"queue {lab.stamp}: {' '.join(ids)}; keep {minimum} GB free; trimming {'on' if lab.trim_on() else 'off'}"
            + (f"; no new experiment after {until}" if until else ""))
    space = [sys.executable, "scripts/asr/lab_space.py"]
    failed: list[str] = []
    blocked: list[str] = []
    finished: list[str] = []
    reasons: dict[str, str] = {}

    def status(running: str = "") -> None:
        write_status(lab, {"stamp": lab.stamp, "planned": ids, "running": running, "done": finished, "failed": failed, "blocked": blocked,
                           "reasons": reasons, "export_to": str(lab.cfg["export_to"]) or None})

    status()
    if lab.cfg["wait_busy"] and not lab.dry:
        busy, since = busy_processes(), time.time()
        reported = since
        if busy:
            lab.say("another lab run is going; waiting for it to finish (checked every 5 minutes):")
            for line in busy:
                lab.say("  " + line)
        while busy:
            if deadline and time.time() > deadline:
                lab.say(f"past {until} while waiting; not starting anything")
                return False
            time.sleep(300)
            busy = busy_processes()
            if busy and time.time() - reported >= 3600:
                reported = time.time()
                lab.say(f"  still waiting ({(reported - since) / 3600:.1f} h)")
        lab.say("no other lab run is going; starting")
    for st in ids:
        exp = lab.catalog.get(st)
        if exp is None:
            lab.say(f"unknown experiment {st}, skipping")
            continue
        if deadline and time.time() > deadline:
            lab.say(f"past {until}: not starting {st} or anything after it")
            break
        state = done_state(lab, exp)
        if state == "done" and not rerun:
            lab.say(f"-- {st} already done ({exp.title}); skipping")
            finished.append(st)
            continue
        lab.banner(f"queue: before {st} ({exp.title}), {free_gb(lab.root)} GB free")
        if not lab.dry:
            lab.run([*space, "git-results", "--commit", "--stamp", time.strftime("%m%d-%H%M%S") + f"-{time.time_ns() % 1000000:06d}"])
            push_results(lab)
            lab.run([*space, "clean", "--apply"])
            lab.clean_baseline_weights()
            lab.run([*space, "clean", "--apply", "--long-models"])
            if free_gb(lab.root) < minimum:
                lab.say(f"under {minimum} GB free: removing lab-test models whose decodes are saved")
                lab.run([*space, "clean", "--apply", "--lab-models", "--long-models"])
            if free_gb(lab.root) < minimum and str(lab.cfg["export_to"]).strip():
                export_models(lab, minimum + 20)
            remaining = free_gb(lab.root)
            if remaining < minimum:
                lab.say(f"only {remaining} GB free after cleaning (need {minimum}); skipping {st} and continuing the queue")
                storage_snapshot(lab, f"queue skipped {st} for disk")
                if not str(lab.cfg["export_to"]).strip():
                    lab.say("  to free more: python lab.py set export_to rclone:gdrive:willspeak-models   (see scripts/asr/model_export.py)")
                blocked.append(f"disk-before-{st}")
                reasons[st] = f"only {remaining} GB free"
                status()
                continue
            bad = doctor_gate(lab, st, minimum)
            if bad:
                lab.say(f"SKIPPED {st}: the doctor found problems it could not repair")
                blocked.append(st)
                reasons[st] = "; ".join(bad)[:300]
                status()
                continue
        if not lab.dry and not (lab.rebuild and st in Lab.BASELINE_NEEDS):
            dry = Lab(lab.root, lab.cfg, dry=True, stamp=lab.stamp, quiet=True, strict=True)
            _commands, notes = plan(lambda: dry, [st])
            blockers = [line for line in notes.get(st, [])
                        if any(marker in line for marker in ("MISSING", "FAILED", "ALREADY EXISTS"))
                        and "skipping" not in line.lower()]
            if blockers:
                log_path = lab.logdir / f"lab-{lab.stamp}-{safe(st)}-preflight.txt"
                log_path.write_text("\n".join([f"SKIPPED {st}: preflight found unavailable inputs or existing outputs.", *blockers]) + "\n",
                                    encoding="utf-8")
                lab.say(f"SKIPPED {st}: preflight found unavailable inputs or existing outputs; see {log_path}")
                for line in blockers:
                    lab.say(f"  {line}")
                blocked.append(st)
                reasons[st] = "; ".join(blockers)[:300]
                status()
                continue
        status(st)
        if run_stage_with_retries(lab, st, minimum):
            finished.append(st)
        else:
            failed.append(st)
            reasons[st] = "failed; see " + str(lab.logdir / f"lab-{lab.stamp}-{safe(st)}.txt")
        lab.clean_baseline_weights()
        status()
    if not lab.dry:
        lab.run([lab.sc, "scripts/asr/sentence_rescore.py", "--out", f"reports/sentence-only/table-{lab.stamp}.json"])
        lab.run([*space, "git-results", "--commit", "--stamp", time.strftime("%m%d-%H%M%S") + f"-{time.time_ns() % 1000000:06d}"])
        pushed = push_results(lab)
        lab.run([*space, "clean", "--apply"])
    else:
        pushed = False
    lab.say("")
    if pushed:
        lab.say("Results are pushed (lab-results-* branches on origin). On the other computer:")
    else:
        lab.say("NEXT: push the results (the branch names are printed above):")
        lab.say("    git push origin 'refs/heads/lab-results-*:refs/heads/lab-results-*'")
        lab.say("then on the other computer:")
    lab.say("    git fetch && python lab.py space fetch-results origin/<newest lab-results branch>")
    status()
    if blocked:
        lab.say(f"BLOCKED: {' '.join(blocked)}; restore the listed inputs or clear the stale outputs, then rerun")
    if failed:
        lab.say(f"FAILED:{' ' + ' '.join(failed)}; inspect their logs before treating the queue as complete")
    return not failed and not blocked


def unique_train_commands(commands: list[list[str]]) -> list[list[str]]:
    seen, out = set(), []
    for cmd in commands:
        if "scripts/asr/train.py" not in cmd:
            continue
        flags = split_flags(cmd[cmd.index("scripts/asr/train.py") + 1:])
        key = tuple(sorted((k, tuple(v)) for k, v in flags.items() if k not in ("--export", "--out", "--seed", "--base")))
        if key not in seen:
            seen.add(key)
            out.append(cmd)
    return out


def help_text(root: Path, script: str, sub: str | None, cache: dict) -> str | None:
    key = (script, sub)
    if key not in cache:
        done = subprocess.run([sys.executable, str(root / script), *([sub] if sub else []), "--help"], capture_output=True, text=True,
                              check=False, cwd=root)
        cache[key] = done.stdout if done.returncode == 0 else None
    return cache[key]


def flag_problems(root: Path, commands: list[list[str]]) -> tuple[list[str], int]:
    cache: dict = {}
    problems, checked = [], set()
    for cmd in commands:
        scripts = [w for w in cmd if re.fullmatch(r"scripts/(asr|corpus|matcher)/\w+\.py", w)]
        if not scripts:
            continue
        script = scripts[0]
        rest = cmd[cmd.index(script) + 1:]
        sub = rest[0] if script.endswith(("battery.py", "synth_dysarthria.py", "cv_folds.py", "tts_synth.py", "tta.py", "lm_score.py", "lm_pick.py")) else None
        text = help_text(root, script, sub, cache)
        if text is None:
            continue
        checked.add(script)
        for flag in sorted({w.split("=")[0] for w in rest if re.fullmatch(r"--[\w-]+(=.*)?", w)}):
            if not re.search(rf"(?<![\w-]){re.escape(flag)}(?![\w-])", text):
                problems.append(f"{script} {sub or ''} has no {flag}")
    return problems, len(checked)


def plan(lab_factory, ids: list[str]) -> tuple[list[list[str]], dict[str, list[str]]]:
    commands, notes = [], {}
    for st in ids:
        dry = lab_factory()
        dry.stage = st
        exp = dry.catalog.get(st)
        if exp is None:
            notes[st] = [f"unknown stage {st}"]
            continue
        try:
            exp.fn(dry)
        except Exception as exc:  # noqa: BLE001
            dry.say(f"FAILED while planning: {type(exc).__name__}: {exc}")
        commands += dry.commands
        notes[st] = [ln for ln in dry.lines if re.search(r"MISSING|ALREADY EXISTS|FAILED|unknown", ln)]
    return commands, notes


def run_check(lab: Lab, tokens: list[str], smoke: bool, tests: bool) -> bool:
    remaining_defaults(lab, tokens)
    ids = resolve(lab, tokens) or list(DEFAULT_QUEUE)
    if any(str(t).lower() == "remaining" for t in tokens):
        ids = [st for st in dict.fromkeys(ids) if st in lab.catalog and done_state(lab, lab.catalog[st]) != "done"]
        print(f"remaining on this computer: {' '.join(ids) or '(nothing)'}")
    results: list[tuple[str, str, str]] = []

    def note(status: str, what: str, detail: str = "") -> None:
        results.append((status, what, detail))
        print(f"  {status:<4} {what}" + (f"\n         {detail}" if detail else ""), flush=True)

    print("== 1. machine")
    note("PASS" if TRAIN_PY.is_file() else "FAIL", f"training environment {TRAIN_PY}", "" if TRAIN_PY.is_file() else "run ./willspeak.sh lab-setup")
    hf = Path(str(lab.cfg["hfpy"]))
    note("PASS" if hf.is_file() else "WARN", f"Hugging Face environment {hf}", "" if hf.is_file() else "only lab test 4's non-NeMo models need it")
    gpu = smi("name,memory.total")
    note("PASS" if gpu.stdout.strip() else "FAIL", "GPU: " + (gpu.stdout.strip() or "nvidia-smi found none"))
    gb = free_gb(lab.root)
    note("PASS" if gb >= 25 else "WARN", f"disk free: {gb} GB", "" if gb >= 25 else "python lab.py space clean --apply frees checkpoints of finished runs")
    busy = busy_processes()
    if busy and lab.cfg["wait_busy"]:
        note("WARN", "a lab run is going; the queue will wait for it to finish", "; ".join(busy)[:300])
    else:
        note("FAIL" if busy else "PASS", "no lab run going" if not busy else "a lab run is going", "; ".join(busy)[:300])
    tracking = wandb_status(lab)
    note("PASS" if tracking["mode"] != "off" else "WARN", f"Weights & Biases mode: {tracking['mode']}", tracking["advice"])

    print("== 2. plan (a dry pass of each experiment: every missing input or existing output that would stop it)")
    make = lambda: Lab(lab.root, lab.cfg, dry=True, stamp="CHECK", quiet=True, strict=True)  # noqa: E731
    commands, notes = plan(make, ids)
    for st in ids:
        missing = [x for x in notes.get(st, []) if "MISSING" in x or "unknown" in x or "FAILED" in x]
        existing = [x for x in notes.get(st, []) if "ALREADY EXISTS" in x]
        if missing:
            note("WARN", f"experiment {st}: inputs missing", "; ".join(missing)[:300])
        elif existing:
            note("PASS", f"experiment {st}: already has results ({existing[0].split(' ')[2]})")
        else:
            note("PASS", f"experiment {st}: ready")
    problems, checked = flag_problems(lab.root, commands)
    note("FAIL" if problems else "PASS", f"flags accepted by {checked} scripts", "; ".join(problems[:6]))

    print("== 2b. doctor (runtime, disk, data files; `python lab.py doctor --fix` repairs what it safely can)")
    try:
        found = lab_doctor.run_checks(lab.root, sys.executable, str(TRAIN_PY), need_controls=any(x in SYNTHETIC_STAGES for x in ids),
                                      need_gb=int(lab.cfg["min_free_gb"]) or 14, need_gpu=bool(gpu.stdout.strip()))
        for f in found:
            if f.level == "fail" and lab.cfg["allow_download"] and "control" in f.message:
                note("WARN", f"[{f.area}] {f.message}", "the queue's doctor downloads them before lab tests 16 to 18")
            elif f.level == "fail":
                note("FAIL", f"[{f.area}] {f.message}", (f.advice + (f"  (--fix: {f.fix[0]})" if f.fix else ""))[:300])
            elif f.level == "warn":
                note("WARN", f"[{f.area}] {f.message}", f.advice[:300])
        note("PASS", f"doctor: {sum(f.level == 'ok' for f in found)} checks passed")
    except Exception as exc:  # noqa: BLE001
        note("WARN", "doctor could not run", f"{type(exc).__name__}: {exc}")
    place = str(lab.cfg["export_to"]).strip()
    note("PASS" if place else "WARN", f"model export target: {place or 'not set'}",
         "" if place else "set one so the queue can free disk: python lab.py set export_to rclone:gdrive:willspeak-models")

    if smoke:
        print("== 3. smoke: two optimizer steps per distinct recipe (temporary outputs, removed afterwards)")
        if not TRAIN_PY.is_file() or not gpu.stdout.strip():
            note("WARN", "smoke runs skipped: no training environment or GPU here")
        elif busy:
            note("WARN", "smoke runs skipped: another lab run is using the GPU")
        else:
            import tempfile
            for cmd in unique_train_commands(commands):
                flags = merge_flags(cmd[cmd.index("scripts/asr/train.py") + 1:], ["--max-steps", "2", "--epochs", "1", "--select", "dev"])
                given = split_flags(flags)
                export = (given.get("--export") or ["data/export"])[0]
                base = (given.get("--base") or [""])[0]
                pending = [x for x in ([str(Path(export) / "train.jsonl")] + ([base] if base.endswith(".nemo") else []))
                           if not (lab.root / x).is_file()]
                if pending:
                    note("WARN", "smoke run skipped: its input is built earlier in the queue", ", ".join(pending)[:300])
                    continue
                with tempfile.TemporaryDirectory(dir=lab.root / "models", prefix="smoke-") as tmp:
                    flags = merge_flags(flags, ["--out", str(Path(tmp) / "run"), "--wandb", "off"])
                    start = time.time()
                    done = subprocess.run([str(TRAIN_PY), "scripts/asr/train.py", *flags], cwd=lab.root, capture_output=True, text=True, check=False)
                    label = " ".join(t for t in flags if t.startswith("--") and t not in ("--export", "--out", "--max-steps", "--epochs", "--wandb"))[:90]
                    if done.returncode == 0:
                        note("PASS", f"recipe trains ({time.time() - start:.0f}s): {label}")
                    else:
                        tail = " | ".join((done.stdout + done.stderr).strip().splitlines()[-3:])[:300]
                        note("FAIL", f"recipe does not train: {label}", tail)
    if tests:
        print("== 4. unit tests")
        done = subprocess.run([sys.executable, "-m", "pytest", "-rfE", "--tb=line", "-p", "no:cacheprovider", "tests/unit/test_lab.py", "tests/unit/test_lab_cli.py",
                               "tests/unit/test_training_lab.py", "tests/unit/test_wandb_log.py", "tests/unit/test_wandb_backfill.py"],
                              cwd=lab.root, capture_output=True, text=True, check=False)
        lines = (done.stdout + done.stderr).strip().splitlines()
        failed = [x for x in lines if x.startswith(("FAILED", "ERROR"))]
        tail = " | ".join(failed[:4] or lines[-3:])[:600]
        note("PASS" if done.returncode == 0 else "FAIL", "lab unit tests", tail)
    bad = [r for r in results if r[0] == "FAIL"]
    warn = [r for r in results if r[0] == "WARN"]
    print(f"\n== verdict: {sum(r[0] == 'PASS' for r in results)} pass, {len(warn)} warn, {len(bad)} FAIL")
    if bad:
        print("NOT SAFE TO LEAVE. Fix these first:")
        for status, what, detail in bad:
            print(f"  - {what}: {detail}")
    return not bad


def wandb_status(lab: Lab) -> dict:
    os.environ["WILLSPEAK_WANDB"] = str(lab.cfg["wandb"])
    try:
        import wandb_log
        mode = wandb_log.mode()
        installed = wandb_log.available()
        creds = wandb_log._has_credentials()
    except Exception:  # noqa: BLE001
        mode, installed, creds = "off", False, False
    offline = sorted((lab.logdir / "wandb").glob("offline-run-*")) if (lab.logdir / "wandb").is_dir() else []
    advice = ""
    if not installed:
        advice = "wandb is not installed in this Python: python lab.py setup (or pip install wandb)"
    elif not creds and lab.cfg["wandb"] in ("auto", "online"):
        advice = "not logged in: run `python lab.py wandb login`; until then runs are saved offline or skipped"
    return {"mode": mode, "installed": installed, "logged_in": creds, "offline_runs": len(offline), "advice": advice,
            "project": lab.cfg["wandb_project"], "setting": lab.cfg["wandb"]}


def cmd_wandb(lab: Lab, args) -> int:
    action = args.action
    if action == "status":
        st = wandb_status(lab)
        print(f"setting {st['setting']}  ->  mode {st['mode']}   project {st['project']}")
        print(f"wandb installed: {st['installed']}   logged in: {st['logged_in']}   offline runs waiting to upload: {st['offline_runs']}")
        if st["advice"]:
            print(st["advice"])
        print("Runs get a group per lab run (STAMP-STAGE); `python lab.py wandb backfill --dry-run` lists the old runs it can upload.")
        return 0
    env = {"WILLSPEAK_WANDB": str(lab.cfg["wandb"]), "WANDB_PROJECT": str(lab.cfg["wandb_project"])}
    if lab.cfg["wandb_entity"]:
        env["WANDB_ENTITY"] = str(lab.cfg["wandb_entity"])
    if action == "login":
        print("Paste your key at the prompt (it is stored by wandb in ~/.netrc, never in this repository).")
        return subprocess.call([lab.sc, "-m", "wandb", "login"], cwd=lab.root)
    if action == "backfill":
        return subprocess.call([lab.sc, "scripts/asr/wandb_backfill.py", *args.rest], cwd=lab.root, env={**os.environ, **env})
    if action == "sync":
        runs = sorted(str(p) for p in (lab.logdir / "wandb").glob("offline-run-*")) if (lab.logdir / "wandb").is_dir() else []
        if not runs:
            print("no offline runs to upload")
            return 0
        return subprocess.call([lab.sc, "-m", "wandb", "sync", *runs], cwd=lab.root, env={**os.environ, **env})
    return 2


def prepare_exports(lab: Lab) -> None:
    from willspeak.torgo import DYSARTHRIC

    corpus = lab.root / "data/public/torgo"
    if any(not (corpus / "TORGO" / s).is_dir() for s in DYSARTHRIC):
        lab.run([sys.executable, "scripts/corpus/fetch_torgo.py"])
    expected = [corpus / "export" / n for s in SEVERE for n in (s, f"loso-{s}", f"loso-severe-{s}")]
    missing = [f for f in expected if any(not (f / f"{sp}.jsonl").is_file() for sp in ("train", "dev", "test"))]
    if missing:
        partial = [f for f in missing if f.exists() and any(f.iterdir())]
        if partial:
            raise RuntimeError("Existing TORGO exports are incomplete for the lab. Copy the matching full data/public/torgo/export/ from the "
                               "original training computer; lab setup will not replace research splits. Partial: " + ", ".join(str(p) for p in partial))
        lab.run([sys.executable, "scripts/corpus/torgo_manifest.py", "--export", "--missing-only", "--speakers", *SEVERE, "--severe-loso"])
    elif not (corpus / "manifests/words.csv").is_file():
        lab.run([sys.executable, "scripts/corpus/torgo_manifest.py"])


def cmd_setup(lab: Lab) -> int:
    try:
        prepare_exports(lab)
    except RuntimeError as exc:
        print(exc)
        return 1
    from huggingface_hub import snapshot_download

    from fetch_models import check_parakeet, check_wavlm
    if not check_parakeet(download=True) or not check_wavlm(download=True):
        return 1
    snapshot_download("nvidia/parakeet-tdt-0.6b-v2", allow_patterns=["*.nemo"])
    for py in {str(TRAIN_PY) if TRAIN_PY.is_file() else "", lab.sc}:
        if py:
            lab.run([py, "-m", "pip", "install", "-q", "wandb>=0.19"])
    print("\nPublic lab assets ready. WavLM word embeddings are built on first use.")
    print("Earlier fine-tunes and hypotheses are local research artifacts: copy models/parakeet-ft-*/ and reports/finetune-torgo/ from Yack,")
    print("or let `python lab.py run ... --rebuild` train the baselines it needs. Then: python lab.py wandb login")
    return 0


def cmd_status(lab: Lab, as_list: bool = False) -> int:
    if not as_list:
        gpu = smi("name,memory.used,memory.total,utilization.gpu")
        print(f"GPU:   {gpu.stdout.strip() or 'none found'}")
        print(f"Disk:  {free_gb(lab.root)} GB free")
        busy = busy_processes()
        print("Busy:  " + ("; ".join(busy)[:200] if busy else "nothing running"))
        st = wandb_status(lab)
        print(f"W&B:   {st['mode']} (project {st['project']}), logged in: {st['logged_in']}, offline runs waiting: {st['offline_runs']}")
        print(f"Where: {'rebuilding baselines' if lab.rebuild else 'using Yack-era baselines'}; speakers {' '.join(lab.spk)}")
        print()
    groups: dict[str, list[Exp]] = {}
    for exp in lab.catalog.values():
        groups.setdefault(exp.group, []).append(exp)
    for group, exps in groups.items():
        print(f"{group}s")
        for exp in exps:
            print(f"  {exp.id:<10} {done_state(lab, exp):<8} {exp.kind:<7} {exp.est:<9} {exp.title}")
    print("\nstate: done = the result files exist; partial = some do; new = none; ? = no single marker.  Estimates, not measurements.")
    return 0


def validate_spec(root: Path, spec: dict) -> list[str]:
    text = help_text(root, "scripts/asr/train.py", None, {})
    if text is None:
        return []
    problems = []
    for arm in custom_arms(spec):
        for flag in split_flags(arm.flags):
            if not re.search(rf"(?<![\w-]){re.escape(flag)}(?![\w-])", text):
                problems.append(f"train.py has no {flag} (arm {arm.label})")
    return sorted(set(problems))


def make_spec(args) -> dict:
    axes = {}
    for item in args.axis or []:
        if "=" not in item:
            raise SystemExit(f"--axis wants NAME=V1,V2,..., not {item!r}")
        key, values = item.split("=", 1)
        axes[key.strip().replace("-", "_")] = [v.strip() for v in values.split(",") if v.strip()]
    return {"name": safe(args.name), "title": args.title or args.name, "recipe": args.recipe, "init": args.init, "axes": axes,
            "flags": args.flags or "", "seeds": [int(x) for x in args.seeds], "speakers": args.speakers or None, "default": not args.no_default}


def describe_spec(spec: dict) -> str:
    arms = custom_arms(spec)
    runs = len(arms) * len(spec["seeds"]) * len(spec["speakers"] or SEVERE)
    return (f"{spec['name']}: recipe {spec['recipe']} on {spec['init']}, {len(arms)} arms "
            f"({', '.join(a.label for a in arms)}) x seeds {spec['seeds']} x speakers {' '.join(spec['speakers'] or SEVERE)} = {runs} runs")


def cmd_new(lab: Lab, args) -> int:
    spec = make_spec(args)
    problems = validate_spec(lab.root, spec)
    if problems:
        print("Not saved:\n  " + "\n  ".join(problems))
        return 2
    existing = [e for e in custom_experiments(lab.root) if e["name"] != spec["name"]]
    save_custom(lab.root, existing + [spec])
    print("Saved to lab_experiments.json (commit it to take it to the lab computer):\n  " + describe_spec(spec))
    print(f"Run it:  python lab.py run c:{spec['name']}      preview:  python lab.py run c:{spec['name']} --dry")
    return 0


def wizard_new(lab: Lab) -> None:
    print("\nNew sweep. A recipe trained over axes (settings that vary), seeds and speakers; every arm is compared with the default arm.")
    print("  wb   the whole-block recipe (blocks 0-7, 5 epochs)        ad8  the adapter after block 8 (15 epochs)")
    name = ask("name (letters, digits, dashes)")
    if not name:
        return
    recipe = ask("recipe wb or ad8", "wb")
    init = ask("start from stock or f1 (the F-1 model)", "stock")
    print("axes: a train.py option and the values to try, e.g.  lr=1e-5,4e-5   epochs=5,8   target-form=scoring,stock   spec-augment=on,off")
    axes = []
    while True:
        axis = ask("axis (empty to finish)")
        if not axis:
            break
        axes.append(axis)
    seeds = ask("seeds", "17 18 19").split()
    speakers = ask("speakers", " ".join(lab.spk)).split()
    ns = argparse.Namespace(name=name, title=name, recipe=recipe, init=init, axis=axes, flags="", seeds=seeds, speakers=speakers, no_default=False)
    cmd_new(lab, ns)


COLORS = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def paint(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if COLORS else text


def ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        raise SystemExit(0) from None
    return answer or (default or "")


def header(lab: Lab) -> None:
    gpu = smi("name").stdout.strip()
    st = wandb_status(lab)
    print(paint("\n willspeak lab ", "1;7"))
    print(f" GPU {gpu or 'none'}   disk {free_gb(lab.root)} GB free   W&B {st['mode']}   speakers {' '.join(lab.spk)}"
          f"   {'rebuilding baselines' if lab.rebuild else ''}")


def show_experiments(lab: Lab) -> None:
    colour = {"done": "32", "partial": "33", "new": "0", "?": "90"}
    for group in ("lab test", "2 Oct big run", "sweep", "yours"):
        exps = [e for e in lab.catalog.values() if e.group == group]
        if not exps:
            continue
        print(paint(f"\n {group}s", "1"))
        for e in exps:
            state = done_state(lab, e)
            print(f"  {e.id:<10} {paint(f'{state:<8}', colour[state])} {e.kind:<7} {e.est:<9} {e.title}")
    print("\n shortcuts: cheap (decode-only lab tests)   all-lab   sweeps   acoustic (lab tests 31-33)   positions (34, then speaker-replay)   widepool (39)   or any ids separated by spaces")


def do_run(lab_args: argparse.Namespace, ids: list[str], dry: bool) -> None:
    lab = make_lab(lab_args, dry=dry)
    run_stages(lab, ids)


def menu_run(args) -> None:
    lab = make_lab(args)
    show_experiments(lab)
    chosen = ask("\n experiments to run (empty to go back)")
    if not chosen:
        return
    ids = resolve(lab, [chosen])
    unknown = [i for i in ids if i not in lab.catalog]
    if unknown:
        print(f" unknown: {' '.join(unknown)}")
        return
    print(paint(f"\n will run: {' '.join(ids)}", "1"))
    print(" Enter = run   d = dry run (print commands only)   v = edit variables first   r = resume mode (rerun + rebuild)   b = back")
    answer = ask(" >", "").lower()
    if answer == "b":
        return
    if answer == "v":
        menu_vars(args)
    if answer == "r":
        args.set = (args.set or []) + ["rerun=1", "rebuild_missing=1"]
    do_run(args, ids, dry=(answer == "d"))


def menu_vars(args) -> None:
    while True:
        lab = make_lab(args)
        print(paint("\n variables", "1") + "   (source: default | saved | env | --set)")
        names = list(VARIABLES)
        for i, name in enumerate(names, 1):
            value, source = lab.cfg.source(name)
            print(f"  {i:>2}  {name:<16} {str(value)!s:<34} {source:<8} {VARIABLES[name][2]}")
        pick = ask("\n number to change, u<number> to reset to default, empty to go back")
        if not pick:
            return
        reset = pick.startswith("u")
        try:
            name = names[int(pick.lstrip("u")) - 1]
        except (ValueError, IndexError):
            print(" not a number in the list")
            continue
        if reset:
            lab.cfg.unset(name)
            print(f" {name} back to its default")
            continue
        try:
            lab.cfg.save(name, ask(f" new value for {name}", str(lab.cfg[name])))
        except ValueError as exc:
            print(f" {exc}")


def menu_queue(args) -> None:
    lab = make_lab(args)
    show_experiments(lab)
    print(f"\n default queue: {' '.join(DEFAULT_QUEUE)}  (finished experiments are skipped)")
    chosen = ask(" experiments in order (Enter = default queue, b = back)")
    if chosen.lower() == "b":
        return
    until = ask(" stop starting new experiments after HH:MM (Enter = no limit)")
    rerun = ask(" resume half-finished earlier attempts and rebuild missing baselines? y/n", "y").lower().startswith("y")
    mode = ask(" Enter = run in the background, f = foreground, d = dry run", "").lower()
    cmd = [sys.executable, str(CODE / "lab.py"), "queue", *(chosen.split() if chosen else [])]
    if until:
        cmd += ["--until", until]
    if rerun:
        cmd.append("--rerun")
    if mode == "d":
        cmd.append("--dry")
    elif mode != "f":
        cmd.append("--detach")
    subprocess.call(cmd, cwd=CODE)


def menu_wandb(args) -> None:
    lab = make_lab(args)
    cmd_wandb(lab, argparse.Namespace(action="status", rest=[]))
    print("\n  1 log in    2 upload offline runs    3 list old runs a backfill would upload    4 backfill old runs now    b back")
    pick = ask(" >", "b")
    actions = {"1": ("login", []), "2": ("sync", []), "3": ("backfill", ["--dry-run"]), "4": ("backfill", [])}
    if pick in actions:
        action, rest = actions[pick]
        cmd_wandb(lab, argparse.Namespace(action=action, rest=rest))


def menu_space(args) -> None:
    print("\n  1 report    2 save results on a git side branch (git-results)    3 clean checkpoints of finished runs (dry)    4 clean (apply)")
    print("  5 doctor (check the data and results)    6 doctor --fix    7 list models    8 export models, dry run    9 export models to the export_to place    b back")
    pick = ask(" >", "b")
    table = {"1": ["report"], "2": ["git-results", "--commit"], "3": ["clean"], "4": ["clean", "--apply"]}
    if pick in table:
        subprocess.call([sys.executable, str(CODE / "scripts/asr/lab_space.py"), *table[pick]], cwd=CODE)
    elif pick in ("5", "6"):
        subprocess.call([sys.executable, str(CODE / "scripts/asr/lab_doctor.py"), *(["--fix"] if pick == "6" else [])], cwd=CODE)
    elif pick in ("7", "8", "9"):
        place = str(make_lab(args).cfg["export_to"]).strip()
        if pick == "7":
            cmd = ["list"]
        elif not place:
            print(" no place set: python lab.py set export_to rclone:gdrive:willspeak-models (steps are at the top of scripts/asr/model_export.py)")
            return
        else:
            cmd = ["export", "--to", place, *(["--apply", "--free-after-verify"] if pick == "9" else [])]
        subprocess.call([sys.executable, str(CODE / "scripts/asr/model_export.py"), *cmd], cwd=CODE)


def menu(args) -> int:
    while True:
        header(make_lab(args))
        print("""
  1  Run experiments      lab tests, big runs, sweeps, your own
  2  Overnight queue      ordered, resumable, saves results between experiments
  3  Variables            see and change every setting
  4  New sweep           define your own: recipe x settings x seeds x speakers
  5  Status               what is done, what is running
  6  Weights & Biases     log in, upload, backfill old runs
  7  Disk & transfer      report, save results to git, clean up
  8  Setup & checks       prepare this machine, verify before leaving
  q  Quit""")
        pick = ask("\n >").lower()
        if pick in ("q", ""):
            return 0
        if pick == "1":
            menu_run(args)
        elif pick == "2":
            menu_queue(args)
        elif pick == "3":
            menu_vars(args)
        elif pick == "4":
            wizard_new(make_lab(args))
        elif pick == "5":
            cmd_status(make_lab(args))
        elif pick == "6":
            menu_wandb(args)
        elif pick == "7":
            menu_space(args)
        elif pick == "8":
            sub = ask(" s = setup this machine, c = check (verify before leaving), b = back", "c").lower()
            if sub == "s":
                cmd_setup(make_lab(args))
            elif sub == "c":
                run_check(make_lab(args), [], smoke=ask(" include the GPU smoke runs? y/n", "y").lower().startswith("y"), tests=True)


def make_lab(args, dry: bool | None = None) -> Lab:
    root = Path(getattr(args, "root", None) or CODE)
    overrides = parse_sets(getattr(args, "set", None))
    if getattr(args, "rerun", False):
        overrides["rerun"] = True
    if getattr(args, "rebuild", False):
        overrides["rebuild_missing"] = True
    return Lab(root, Settings(root, overrides), dry=bool(getattr(args, "dry", False) if dry is None else dry), stamp=getattr(args, "stamp", None))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lab.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", help=argparse.SUPPRESS)
    p.add_argument("--set", action="append", metavar="NAME=VALUE", help="override a variable for this command (see `vars`)")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("menu", help="the interactive menu (also what plain `python lab.py` opens)")
    sub.add_parser("list", help="every experiment and whether it is done")
    s = sub.add_parser("status", help="GPU, disk, running jobs, and what is done")
    s.set_defaults(list=False)

    s = sub.add_parser("run", help="run experiments: ids such as 14 3 9 L, or cheap | all-lab | sweeps | pending | acoustic")
    s.add_argument("ids", nargs="*")
    s.add_argument("--dry", action="store_true", help="print every command, run none")
    s.add_argument("--stamp", help="name for this run's logs (default: date and time)")
    s.add_argument("--rerun", action="store_true", help="move half-finished earlier attempts aside (NAME-first-run) instead of refusing")
    s.add_argument("--rebuild", action="store_true", help="train missing baselines into models/lab-baselines/")

    s = sub.add_parser("queue", help="experiments one after another with saving and disk checks; pending = all pending sweeps and long experiments except B")
    s.add_argument("ids", nargs="*")
    s.add_argument("--until", metavar="HH:MM", help="do not start a new experiment after this time")
    s.add_argument("--rerun", action="store_true", help="resume half-finished attempts and rebuild missing baselines")
    s.add_argument("--dry", action="store_true")
    s.add_argument("--detach", action="store_true", help="run in the background; output goes to logs/lab-queue-STAMP.out")
    s.add_argument("--stamp")

    sub.add_parser("vars", help="every variable, its value and where the value comes from")
    s = sub.add_parser("set", help="save a variable on this machine: set margin 3")
    s.add_argument("name")
    s.add_argument("value")
    s = sub.add_parser("unset", help="forget a saved variable")
    s.add_argument("name")

    s = sub.add_parser("new", help="define a sweep: python lab.py new lr-wide --recipe wb --axis lr=1e-5,3e-5 --seeds 17 18")
    s.add_argument("name")
    s.add_argument("--recipe", choices=["wb", "ad8"], default="wb", help="wb = whole-block (blocks 0-7), ad8 = adapter after block 8")
    s.add_argument("--init", choices=["stock", "f1"], default="stock")
    s.add_argument("--axis", action="append", metavar="NAME=V1,V2", help="a train.py option and the values to try (repeatable)")
    s.add_argument("--flags", help="extra train.py flags for every arm")
    s.add_argument("--seeds", nargs="+", default=["17", "18", "19"])
    s.add_argument("--speakers", nargs="+")
    s.add_argument("--title")
    s.add_argument("--no-default", action="store_true", help="leave out the default-flags arm (comparisons need it)")
    s = sub.add_parser("drop", help="remove one of your sweeps from lab_experiments.json")
    s.add_argument("name")

    s = sub.add_parser("wandb", help="Weights & Biases: status | login | sync | backfill [backfill options]")
    s.add_argument("action", choices=["status", "login", "sync", "backfill"])
    s.add_argument("rest", nargs=argparse.REMAINDER)

    s = sub.add_parser("space", help="disk and transfer tools (lab_space.py): report, git-results, fetch-results, clean, pack, unpack, archive-models")
    s.add_argument("rest", nargs=argparse.REMAINDER)
    s = sub.add_parser("doctor", help="check the data, the environment and the saved results, and repair what is safe (lab_doctor.py): [--fix] [--controls] [--deep]")
    s.add_argument("rest", nargs=argparse.REMAINDER)
    s = sub.add_parser("export-models", help="zip finished models and send them to Google Drive (rclone) or a folder, verify, free the disk (model_export.py): list | export | check | import")
    s.add_argument("rest", nargs=argparse.REMAINDER)

    sub.add_parser("setup", help="prepare public lab assets on this machine")
    s = sub.add_parser("check", help="verify the machine and the plan before leaving a run unattended")
    s.add_argument("ids", nargs="*")
    s.add_argument("--no-smoke", action="store_true", help="skip the GPU smoke runs")
    s.add_argument("--no-tests", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    if extra:
        if getattr(args, "cmd", None) in ("space", "doctor", "export-models", "wandb") and hasattr(args, "rest"):
            args.rest = [*extra, *args.rest]
        else:
            parser.error("unrecognized arguments: " + " ".join(extra))
    cmd = args.cmd or "menu"
    if cmd == "menu":
        return menu(args)
    lab = make_lab(args)
    if cmd == "list":
        show_experiments(lab)
        return 0
    if cmd == "status":
        return cmd_status(lab)
    if cmd == "run":
        try:
            return 0 if run_stages(lab, args.ids) else 1
        except KeyboardInterrupt:
            print("\ninterrupted; finished runs are kept and a rerun skips them where the experiment supports it")
            return 130
    if cmd == "queue":
        if args.detach and not args.dry:
            lab.logdir.mkdir(parents=True, exist_ok=True)
            out = lab.logdir / f"lab-queue-{lab.stamp}.out"
            child = [sys.executable, str(CODE / "lab.py"), "queue", *args.ids, "--stamp", lab.stamp]
            for item in args.set or []:
                child[2:2] = ["--set", item]
            if args.until:
                child += ["--until", args.until]
            if args.rerun:
                child.append("--rerun")
            with out.open("w", encoding="utf-8") as handle:
                proc = subprocess.Popen(child, cwd=lab.root, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            print(f"queue started in the background (pid {proc.pid}); it survives closing this terminal.")
            print(f"  watch it:  tail -f {out}")
            print(f"  stop it:   kill {proc.pid}   (finished runs are kept; start the queue again to resume)")
            return 0
        try:
            return 0 if run_queue(lab, args.ids, args.until, args.rerun) else 1
        except KeyboardInterrupt:
            print("\nqueue interrupted")
            return 130
    if cmd == "vars":
        for name, (_default, _env, help_text_) in VARIABLES.items():
            value, source = lab.cfg.source(name)
            print(f"{name:<16} {str(value)!s:<36} {source:<8} {help_text_}")
        return 0
    if cmd == "set":
        key = args.name.replace("-", "_")
        if key not in VARIABLES:
            print(f"No variable {key!r}; `python lab.py vars` lists them")
            return 2
        lab.cfg.save(key, args.value)
        print(f"{key} = {lab.cfg.file[key]}   (saved in {SETTINGS_FILE} on this machine)")
        return 0
    if cmd == "unset":
        lab.cfg.unset(args.name.replace("-", "_"))
        return 0
    if cmd == "new":
        return cmd_new(lab, args)
    if cmd == "drop":
        keep = [e for e in custom_experiments(lab.root) if e["name"] != args.name]
        save_custom(lab.root, keep)
        return 0
    if cmd == "wandb":
        return cmd_wandb(lab, args)
    if cmd == "space":
        return subprocess.call([sys.executable, str(CODE / "scripts/asr/lab_space.py"), *args.rest], cwd=lab.root)
    if cmd == "doctor":
        return subprocess.call([sys.executable, str(CODE / "scripts/asr/lab_doctor.py"), "--root", str(lab.root), *args.rest], cwd=lab.root)
    if cmd == "export-models":
        rest = list(args.rest) or ["list"]
        place = str(lab.cfg["export_to"]).strip()
        if rest[0] in ("export", "check") and place and "--to" not in rest:
            rest += ["--to", place]
        if rest[0] == "import" and place and "--from" not in rest:
            rest += ["--from", place]
        if rest[0] in ("export", "check") and "--to" not in rest:
            print("Say where: python lab.py set export_to rclone:gdrive:willspeak-models   (a folder path works too; setup steps are at the top of scripts/asr/model_export.py)")
            return 2
        return subprocess.call([sys.executable, str(CODE / "scripts/asr/model_export.py"), *rest], cwd=lab.root)
    if cmd == "setup":
        return cmd_setup(lab)
    if cmd == "check":
        return 0 if run_check(lab, args.ids, smoke=not args.no_smoke, tests=not args.no_tests) else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
