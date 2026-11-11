#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$ROOT/.venv"
PY="${PYTHON:-python3}"

WITH_MATCHER=0
INSTALL_SERVICE=0
SKIP_APT=0
SKIP_TESTS=0
STRICT_TESTS=0
MODEL_DIR=""
KIOSK_PAGE="/live"

while [ $# -gt 0 ]; do
  case "$1" in
    --with-matcher) WITH_MATCHER=1 ;;
    --service)      INSTALL_SERVICE=1 ;;
    --skip-apt)     SKIP_APT=1 ;;
    --skip-tests)   SKIP_TESTS=1 ;;
    --strict-tests) STRICT_TESTS=1 ;;
    --model-dir)    shift; MODEL_DIR="${1:-}"; [ -n "$MODEL_DIR" ] || { echo "--model-dir needs a folder" >&2; exit 2; } ;;
    --kiosk-page)   shift; KIOSK_PAGE="${1:-}"; [ -n "$KIOSK_PAGE" ] || { echo "--kiosk-page needs a page such as /compare" >&2; exit 2; } ;;
    -h|--help)
      echo "usage: setup.sh [--with-matcher] [--service] [--model-dir DIR] [--kiosk-page /live|/compare] [--skip-apt] [--skip-tests] [--strict-tests]"
      exit 0 ;;
    *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
  shift
done
if [ -n "$MODEL_DIR" ]; then
  case "$MODEL_DIR" in /*) ;; *) MODEL_DIR="$ROOT/$MODEL_DIR" ;; esac
  for f in encoder decoder joiner; do
    [ -s "$MODEL_DIR/$f.int8.onnx" ] || [ -s "$MODEL_DIR/$f.onnx" ] || { echo "--model-dir: no $f.onnx in $MODEL_DIR" >&2; exit 2; }
  done
  [ -s "$MODEL_DIR/tokens.txt" ] || { echo "--model-dir: no tokens.txt in $MODEL_DIR" >&2; exit 2; }
else
  MODEL_DIR="$ROOT/models/parakeet-tdt-0.6b-v2-int8"
fi

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
note() { printf '   %s\n' "$*"; }

say "willspeak setup"
note "checkout : $ROOT"
note "machine  : $(uname -m)  $( [ -r /proc/device-tree/model ] && tr -d '\0' < /proc/device-tree/model )"
note "python   : $($PY -V 2>&1)"

if ! "$PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
  echo "willspeak needs Python 3.11 or newer; this is $($PY -V 2>&1)." >&2
  echo "On Raspberry Pi OS Bookworm the system python is 3.11 and is fine." >&2
  exit 1
fi

if [ "$SKIP_APT" -eq 0 ]; then
  say "OS packages"
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends \
    python3-venv python3-dev build-essential pkg-config \
    libsndfile1 ffmpeg alsa-utils \
    speech-dispatcher espeak-ng \
    chromium-browser curl bzip2 || \
  sudo apt-get install -y --no-install-recommends \
    python3-venv python3-dev build-essential pkg-config \
    libsndfile1 ffmpeg alsa-utils \
    speech-dispatcher espeak-ng \
    chromium curl bzip2
else
  note "skipping apt (--skip-apt)"
fi

say "virtual environment"
if [ ! -x "$VENV/bin/python" ]; then
  "$PY" -m venv "$VENV"
  note "created $VENV"
else
  note "reusing $VENV"
fi
"$VENV/bin/python" -m pip install --upgrade pip wheel setuptools

say "willspeak and its dependencies"
"$VENV/bin/python" -m pip install -e "$ROOT"

if [ "$WITH_MATCHER" -eq 1 ]; then
  say "PyTorch (CPU) for the exemplar matcher"
  note "about 200 MB; the sentence recogniser does not need it"
  "$VENV/bin/python" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
else
  note "skipping PyTorch. The Speak page runs; the matcher reports itself unavailable."
  note "Add it later with: bash deploy/pi5/setup.sh --with-matcher --skip-apt"
fi

say "models"
if [ "$WITH_MATCHER" -eq 1 ]; then
  "$VENV/bin/python" "$ROOT/scripts/fetch_models.py"
else
  "$VENV/bin/python" "$ROOT/scripts/fetch_models.py" --parakeet-only
fi

if [ "$SKIP_TESTS" -eq 0 ]; then
  say "test suite"
  "$VENV/bin/python" -m pip install -q pytest jiwer httpx
  if ( cd "$ROOT" && "$VENV/bin/python" -m pytest tests -q ); then
    note "all tests passed"
  elif [ "$STRICT_TESTS" -eq 1 ]; then
    echo "Tests failed and --strict-tests was given." >&2
    exit 1
  else
    cat >&2 <<'WARN'

   TESTS FAILED. Read the failures above before trusting anything.

   Failures in tests/unit/test_repository_layout.py are about documentation
   links and script paths, not about whether the recogniser works here.

   Anything else in that list is real. Continuing; re-run with --strict-tests
   to make a failure stop the install.

WARN
  fi
  say "one real decode"
  ( cd "$ROOT" && WILLSPEAK_PARAKEET_DIR="$MODEL_DIR" "$VENV/bin/python" scripts/asr/check.py )
fi

if [ "$INSTALL_SERVICE" -eq 1 ]; then
  say "boot service"
  UNIT=/etc/systemd/system/willspeak.service
  sed -e "s|@ROOT@|$ROOT|g" -e "s|@USER@|$USER|g" -e "s|@MODEL@|$MODEL_DIR|g" "$ROOT/deploy/pi5/willspeak.service" \
    | sudo tee "$UNIT" > /dev/null
  sudo systemctl daemon-reload
  sudo systemctl enable willspeak
  sudo systemctl restart willspeak
  note "installed $UNIT"
  note "model folder: $MODEL_DIR (change it with setup.sh --service --model-dir <folder>)"

  say "kiosk autostart"
  mkdir -p "$HOME/.config/autostart"
  sed -e "s|@ROOT@|$ROOT|g" -e "s|@PAGE@|$KIOSK_PAGE|g" "$ROOT/deploy/pi5/willspeak-kiosk.desktop" \
    > "$HOME/.config/autostart/willspeak-kiosk.desktop"
  chmod +x "$ROOT/deploy/pi5/kiosk.sh"
  note "installed $HOME/.config/autostart/willspeak-kiosk.desktop (opens $KIOSK_PAGE; change it with --kiosk-page)"
  note "reboot, then the acceptance test: power-cycle -> it comes up speaking, no keyboard."
fi

say "done"
cat <<EOF
   Next, in this order:

   1. bash deploy/pi5/check-audio.sh        settle the microphone and speaker
   2. .venv/bin/python run.py               then open http://127.0.0.1:8000/live
   3. bash deploy/pi5/bench.sh              measurement P-A: how fast is this box
   4. bash deploy/pi5/setup.sh --service    only once 1-3 look right

   The destination-machine quickstart is in README.md.
EOF
