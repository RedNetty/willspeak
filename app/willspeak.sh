#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PY="${VIRTUAL_ENV}/bin/python"
elif [[ -x .venv/bin/python ]]; then
  PY=".venv/bin/python"
elif [[ -x ../.venv/bin/python ]]; then
  PY="../.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
else
  PY=""
fi

usage() {
  cat <<'USAGE'
usage: willspeak.sh [command] [args...]

  setup          first-time install: .venv + requirements + Parakeet (add --no-models to skip the download)
  run            start the recorder + Speak page server   (run.py)
  check          load the model and decode one clip       (scripts/asr/check.py)
  fetch-models   download Parakeet (+ test recordings)    (scripts/fetch_models.py)
  bench          speed benchmark on this box              (scripts/asr/bench.py)
  evaluate       WER of a model on an export              (scripts/asr/evaluate.py)
  train          fine-tune Parakeet (training machine)    (scripts/asr/train.py, uses ~/venvs/willspeak-train)
  export         export a fine-tune to int8 ONNX          (scripts/asr/export.py, uses ~/venvs/willspeak-train)
  personal       a personal device model, no scoring      (scripts/asr/personal.py, uses ~/venvs/willspeak-train)
  quality-gate   rebuild the Record page's quality gate    (scripts/quality/gate.py, uses ~/venvs/willspeak-train)
  train-setup    build/repair the GPU training env (WSL2) (~/venvs/willspeak-train)
  lab            the lab: run/queue experiments, sweeps, variables, wandb (python lab.py)
  lab-setup      prepare lab environments, TORGO exports, WavLM and stock NeMo (Linux/WSL2)
  lm-setup       language-model environment for lab test 36 (~/venvs/willspeak-lm); lab test 36 fetches each model when it scores it
                 (add --models to download the pi group now, about 17 GB)
  lab-lm         lab test 36 on the lab computer: pull, environments, a download check, then the queue in the background
                 options: --models "pi large" (default), --quick (stock and the named model only, no training), --dry (print the plan only)
  lab-remaining  pull, repair, check, then queue every unfinished lab in the background (Linux/WSL2)
                 options: --quick (no GPU smoke runs), --force (start even if the check fails), --until HH:MM
  torgo          fetch TORGO / build manifests            (scripts/corpus/fetch_torgo.py, torgo_manifest.py)
  match-eval     matcher evaluation                       (scripts/matcher/evaluate.py)
  test           run the test suite                       (pytest)
  help           this text

No command: an interactive menu of the same list.
USAGE
}

setup() {
  local base="" v
  if [[ ! -x .venv/bin/python ]]; then
    for v in python3.12 python3.13 python3.11 python3; do
      if command -v "$v" >/dev/null 2>&1 && \
         "$v" -c 'import sys, venv, ensurepip; sys.exit(sys.version_info < (3, 11))' >/dev/null 2>&1; then
        base="$v"; break
      fi
    done
    if [[ -n "$base" ]]; then
      echo "== creating .venv with $("$base" --version)"
      "$base" -m venv .venv
    else
      if ! command -v uv >/dev/null 2>&1; then
        if [[ ! -x "$HOME/.local/bin/uv" ]]; then
          echo "== no Python 3.11+ with venv found; installing uv to supply Python 3.12"
          curl -LsSf https://astral.sh/uv/install.sh | sh
        fi
        export PATH="$HOME/.local/bin:$PATH"
      fi
      echo "== creating .venv on Python 3.12 (uv)"
      uv venv --python 3.12 --seed .venv
    fi
  fi
  local py=".venv/bin/python"
  echo "== installing requirements (runtime + tests) into .venv"
  "$py" -m pip install --upgrade pip
  "$py" -m pip install -r requirements-dev.txt
  if [[ "${1:-}" != "--no-models" ]]; then
    echo "== downloading Parakeet (once; a few hundred MB)"
    "$py" scripts/fetch_models.py --parakeet-only
  fi
  cat <<'NEXT'

app environment ready (.venv). Next:
  ./willspeak.sh test          run the tests
  ./willspeak.sh run           start the app, then open http://127.0.0.1:8000/live
  ./willspeak.sh train-setup   training only (Linux/WSL2 with an NVIDIA GPU): builds ~/venvs/willspeak-train
NEXT
}

use_train_env() {
  local venv="$HOME/venvs/willspeak-train"
  if [[ "${VIRTUAL_ENV:-}" == "$venv" ]]; then
    return 0
  fi
  if [[ -x "$venv/bin/python" ]]; then
    set +u
    # shellcheck disable=SC1091
    source "$venv/bin/activate"
    set -u
    PY="$venv/bin/python"
  elif [[ "$(uname -s)" == "Linux" ]]; then
    echo "note: ~/venvs/willspeak-train not found; run ./willspeak.sh train-setup first if this fails" >&2
  fi
}

need_python() {
  if [[ -z "$PY" ]]; then
    echo "Python was not found. Run ./willspeak.sh setup first." >&2
    exit 1
  fi
}

train_setup() {
  local venv="$HOME/venvs/willspeak-train"
  local act="$venv/bin/activate"
  if [[ "$(uname -s)" != "Linux" ]]; then
    echo "train-setup runs on Linux or WSL2 only (NeMo does not run on Windows or macOS)." >&2; exit 1
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi does not see a GPU. Under WSL2 update the Windows NVIDIA driver first." >&2; exit 1
  fi
  if ! dpkg -s libsndfile1 ffmpeg >/dev/null 2>&1; then
    echo "== system audio libraries (asks for your password)"
    sudo apt-get install -y libsndfile1 ffmpeg
  fi
  if ! command -v uv >/dev/null 2>&1; then
    if [[ ! -x "$HOME/.local/bin/uv" ]]; then
      echo "== installing uv (it supplies Python 3.12; the system Python may be too new)"
      curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    export PATH="$HOME/.local/bin:$PATH"
  fi
  if [[ ! -x "$venv/bin/python" ]]; then
    echo "== creating $venv on Python 3.12"
    uv venv --python 3.12 --seed "$venv"
  fi
  grep -q 'CUDA_HOME=' "$act" || \
    echo 'export CUDA_HOME=$HOME/venvs/willspeak-train/lib/python3.12/site-packages/nvidia/cuda_nvcc' >> "$act"
  grep -q 'nvvm/lib64' "$act" || \
    echo 'export LD_LIBRARY_PATH=$CUDA_HOME/nvvm/lib64:$LD_LIBRARY_PATH' >> "$act"
  mkdir -p "$HOME/.cache/willspeak-tmp"
  grep -q 'TMPDIR=' "$act" || \
    echo 'export TMPDIR=$HOME/.cache/willspeak-tmp' >> "$act"
  local py="$venv/bin/python"
  echo "== installing torch (CUDA 12.6 build) and the training extras"
  "$py" -m pip install --upgrade pip
  "$py" -m pip install torch --index-url https://download.pytorch.org/whl/cu126
  "$py" -m pip install -e ".[train]"
  echo "== verifying"
  (
    set +u
    # shellcheck disable=SC1090
    source "$act"
    python - <<'PY'
import sys
sys.path.insert(0, "scripts/asr")
import torch
from train import gpu_environment_problems, visible_ram_gb
ok = sys.version_info[:2] == (3, 12) and torch.cuda.is_available()
print(f"python {sys.version.split()[0]} · CUDA {torch.cuda.is_available()}"
      + (f" · {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
problems = gpu_environment_problems() if torch.cuda.is_available() else ["torch cannot see the GPU"]
for p in problems:
    print("  problem:", p)
ram = visible_ram_gb()
if ram is not None:
    print(f"RAM + swap visible to Linux: {ram:.1f} GB" + ("  (low: add swap, see .wslconfig in docs/training.md §2)" if ram < 14 else ""))
if ok and not problems:
    print("training environment ready. Each new terminal: source ~/venvs/willspeak-train/bin/activate")
sys.exit(0 if ok and not problems else 1)
PY
  )
}

lab_setup() {
  if [[ $# -gt 0 ]]; then
    echo "usage: ./willspeak.sh lab-setup" >&2; return 2
  fi
  if [[ "$(uname -s)" != "Linux" ]]; then
    echo "lab-setup runs on Linux/WSL2 on the GPU computer." >&2; return 1
  fi
  train_setup
  setup --no-models
  local py=".venv/bin/python"
  "$py" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  local hf="$HOME/venvs/willspeak-hf" qwen="$HOME/venvs/willspeak-qwen"
  for env in "$hf" "$qwen"; do
    [[ -x "$env/bin/python" ]] || uv venv --python 3.12 --seed "$env"
    "$env/bin/python" -m pip install torch --index-url https://download.pytorch.org/whl/cu126
  done
  "$hf/bin/python" -m pip install -e ".[dev]" 'transformers>=5.4' accelerate
  local tv
  tv=$("$hf/bin/python" -c 'import torch; print(torch.__version__.split("+")[0])')
  "$hf/bin/python" -m pip install "torchaudio==$tv" --index-url https://download.pytorch.org/whl/cu126
  "$hf/bin/python" -c 'import torchaudio'
  "$qwen/bin/python" -m pip install -e ".[dev]" qwen-asr accelerate
  "$hf/bin/python" -c 'from transformers import CohereAsrForConditionalGeneration, AutoModelForSpeechSeq2Seq'
  "$qwen/bin/python" -c 'from qwen_asr import Qwen3ASRModel'
  "$py" lab.py setup
}

lm_setup() {
  local venv="$HOME/venvs/willspeak-lm" base="" v
  if [[ ! -x "$venv/bin/python" ]]; then
    for v in python3.12 python3.13 python3.11; do
      if command -v "$v" >/dev/null 2>&1 && "$v" -c 'import venv, ensurepip' >/dev/null 2>&1; then
        base="$v"; break
      fi
    done
    if [[ -n "$base" ]]; then
      echo "== creating $venv with $("$base" --version)"
      "$base" -m venv "$venv"
    else
      if ! command -v uv >/dev/null 2>&1; then
        if [[ ! -x "$HOME/.local/bin/uv" ]]; then
          echo "== installing uv (it supplies Python 3.12)"
          curl -LsSf https://astral.sh/uv/install.sh | sh
        fi
        export PATH="$HOME/.local/bin:$PATH"
      fi
      echo "== creating $venv on Python 3.12 (uv)"
      uv venv --python 3.12 --seed "$venv"
    fi
  fi
  local py="$venv/bin/python"
  echo "== installing llama-cpp-python (CPU build), numpy and huggingface_hub"
  "$py" -m pip install --upgrade pip
  "$py" -m pip install "llama-cpp-python>=0.3.32" numpy huggingface_hub \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
  "$py" -c 'import llama_cpp; print("llama-cpp-python", llama_cpp.__version__)'
  if [[ "${1:-}" == "--models" ]]; then
    echo "== downloading the pi group of language models (about 17 GB in all; finished files are skipped)"
    "$py" scripts/asr/lm_score.py fetch --models pi || echo "   some downloads failed; lab test 36 fetches what it still needs when it runs"
  fi
  echo
  echo "language-model environment ready ($venv). Next: python lab.py run 36   (python scripts/asr/lm_score.py list shows the models)"
}

lab_lm() {
  local models="pi large" quick="" dry=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --models) [[ $# -ge 2 ]] || { echo "--models needs tags or groups, for example \"pi large\"" >&2; return 2; }; models="$2"; shift ;;
      --quick) quick=1 ;;
      --dry) dry=1 ;;
      *) echo "usage: ./willspeak.sh lab-lm [--models \"pi large\"] [--quick] [--dry]" >&2; return 2 ;;
    esac
    shift
  done
  if [[ "$(uname -s)" != "Linux" ]]; then
    echo "lab-lm runs on Linux/WSL2 on the GPU computer." >&2; return 1
  fi
  local sets=(--set "lm_models=$models" --set push_results=on --set wait_busy=on)
  [[ -n "$quick" ]] && sets=(--set lm_systems=stock --set lm_models=qwen25-05b-q8 --set push_results=on --set wait_busy=on)
  if [[ -n "$dry" ]]; then
    "$PY" lab.py "${sets[@]}" queue 36 --dry
    return $?
  fi
  echo "== 1. code"
  if git diff --quiet HEAD -- 2>/dev/null; then
    git pull --ff-only || echo "   git pull failed; using the code already here ($(git log -1 --format=%h))"
  else
    echo "   tracked files have local changes, so the code is not pulled; using $(git log -1 --format=%h)"
  fi
  if ! grep -q '"36"' lab.py; then
    echo "This checkout has no lab test 36. Check out the branch that has it (git branch -a), then run this again." >&2; return 1
  fi
  echo "== 2. training environment and TORGO"
  if [[ ! -x "$HOME/venvs/willspeak-train/bin/python" || ! -f data/public/torgo/manifests/severe-sentences.jsonl ]]; then
    echo "   missing; running lab-setup first (environments, TORGO, exports, stock Parakeet)"
    lab_setup || return 1
  else
    echo "   present"
  fi
  echo "== 3. language-model environment"
  lm_setup || return 1
  echo "== 4. download check (the named model, $(du -sh models/lm 2>/dev/null | cut -f1 || echo 'nothing') here so far)"
  local lmpy="$HOME/venvs/willspeak-lm/bin/python"
  "$lmpy" scripts/asr/lm_score.py fetch --models qwen25-05b-q8 && "$lmpy" scripts/asr/lm_score.py check --models qwen25-05b-q8 || {
    echo "The language model did not download or load; fix that first (network, disk: df -h .)." >&2; return 1; }
  echo "   the other models download one at a time while the test runs"
  echo "== 5. check"
  "$PY" lab.py "${sets[@]}" check --no-smoke --no-tests 36 || echo "   the check reported problems; the queue skips what it cannot run and says why"
  echo "== 6. queue"
  local again=()
  if [[ -f reports/lab-36-lm-rescoring/table.json ]]; then
    echo "   an earlier table is here; it is moved aside and remade (pools, models and scores already made are reused)"
    again=(--rerun)
  fi
  "$PY" lab.py "${sets[@]}" queue 36 --detach ${again[@]+"${again[@]}"}
}

lab_remaining() {
  local quick="" force="" until=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --quick) quick="--no-smoke" ;;
      --force) force=1 ;;
      --until) [[ $# -ge 2 ]] || { echo "--until needs HH:MM" >&2; return 2; }; until=(--until "$2"); shift ;;
      *) echo "usage: ./willspeak.sh lab-remaining [--quick] [--force] [--until HH:MM]" >&2; return 2 ;;
    esac
    shift
  done
  if [[ "$(uname -s)" != "Linux" ]]; then
    echo "lab-remaining runs on Linux/WSL2 on the GPU computer." >&2; return 1
  fi
  echo "== 1. code"
  if git diff --quiet HEAD -- 2>/dev/null; then
    git pull --ff-only || echo "   git pull failed; using the code already here ($(git log -1 --format=%h))"
  else
    echo "   tracked files have local changes, so the code is not pulled; using $(git log -1 --format=%h)"
  fi
  echo "== 2. doctor (repairs and downloads; never deletes)"
  "$PY" lab.py doctor --fix --allow-download --controls || echo "   the doctor reported problems; the check below decides"
  echo "== 3. check"
  if ! "$PY" lab.py check ${quick} remaining; then
    if [[ -z "$force" ]]; then
      echo "Not starting. Fix the FAIL lines above and run ./willspeak.sh lab-remaining again (--force starts anyway)." >&2
      return 1
    fi
    echo "   starting anyway (--force)"
  fi
  echo "== 4. queue"
  "$PY" lab.py queue remaining --detach ${until[@]+"${until[@]}"}
}

run_cmd() {
  local cmd="$1"; shift || true
  case "$cmd" in
    setup|train-setup|lab-setup|lm-setup|help|-h|--help) ;;
    *) need_python ;;
  esac
  case "$cmd" in
    setup)        setup "$@"; exit $? ;;
    run)          exec "$PY" run.py "$@" ;;
    check)        exec "$PY" scripts/asr/check.py "$@" ;;
    fetch-models) exec "$PY" scripts/fetch_models.py "$@" ;;
    bench)        exec "$PY" scripts/asr/bench.py "$@" ;;
    evaluate)     exec "$PY" scripts/asr/evaluate.py "$@" ;;
    train)        use_train_env; exec "$PY" scripts/asr/train.py "$@" ;;
    export)       use_train_env; exec "$PY" scripts/asr/export.py "$@" ;;
    personal)     use_train_env; exec "$PY" scripts/asr/personal.py "$@" ;;
    quality-gate) use_train_env; exec "$PY" scripts/quality/gate.py "$@" ;;
    train-setup)  train_setup; exit $? ;;
    lab-setup)    lab_setup "$@"; exit $? ;;
    lab-remaining) lab_remaining "$@"; exit $? ;;
    lm-setup)     lm_setup "$@"; exit $? ;;
    lab-lm)       lab_lm "$@"; exit $? ;;
    lab)          exec "$PY" lab.py "$@" ;;
    torgo)        if [[ "${1:-}" == "manifest" ]]; then shift; exec "$PY" scripts/corpus/torgo_manifest.py "$@"
                  else exec "$PY" scripts/corpus/fetch_torgo.py "$@"; fi ;;
    match-eval)   exec "$PY" scripts/matcher/evaluate.py "$@" ;;
    test)         exec "$PY" -m pytest "${@:-tests}" ;;
    help|-h|--help) usage; exit 0 ;;
    *)            echo "unknown command: $cmd" >&2; usage >&2; exit 2 ;;
  esac
}

if [[ $# -gt 0 ]]; then
  run_cmd "$@"
fi

echo "willspeak"
echo "  0) setup        first-time install (.venv, requirements, Parakeet)"
echo "  1) run          start the server (then open http://127.0.0.1:8000/live)"
echo "  2) check        load the model, decode one clip"
echo "  3) fetch-models download Parakeet"
echo "  4) bench        speed benchmark on this box"
echo "  5) evaluate     WER on an export"
echo "  6) train        fine-tune (training machine)"
echo "  7) export       fine-tune -> int8 ONNX"
echo "  8) torgo        fetch TORGO ('torgo manifest' builds manifests)"
echo "  9) match-eval   matcher evaluation"
echo " 10) test         run the tests"
echo " 11) train-setup  build/repair the GPU training environment"
echo " 12) lab-setup    prepare the lab computer (environments, corpus, models)"
echo " 13) lab-remaining queue every unfinished lab, fully prepared (lab computer)"
echo " 14) lm-setup     language-model environment for lab test 36"
echo " 15) lab-lm       lab test 36 on the lab computer, in the background"
echo "  q) quit"
read -r -p "> " choice
case "$choice" in
  0) run_cmd setup ;;       1) run_cmd run ;;         2) run_cmd check ;;      3) run_cmd fetch-models ;;
  4) run_cmd bench ;;       5) run_cmd evaluate ;;   6) run_cmd train ;;
  7) run_cmd export ;;      8) run_cmd torgo ;;      9) run_cmd match-eval ;;
  10) run_cmd test ;;       11) run_cmd train-setup ;; 12) run_cmd lab-setup ;; 13) run_cmd lab-remaining ;;
  14) run_cmd lm-setup ;;   15) run_cmd lab-lm ;;     q|Q|"") exit 0 ;;
  *) echo "not an option: $choice" >&2; exit 2 ;;
esac
