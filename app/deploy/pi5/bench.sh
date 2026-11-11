#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"

if pgrep -x chromium-browser > /dev/null 2>&1 || pgrep -x chromium > /dev/null 2>&1; then
  echo "NOTE: Chromium is running. Close it (or expect a slower, noisier number)." >&2
fi
if [ -r /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
  gov="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
  if [ "$gov" != "performance" ]; then
    echo "NOTE: cpu governor is '$gov'. 'performance' is typically worth 20-30%:" >&2
    echo "      sudo apt install linux-cpupower && sudo cpupower frequency-set -g performance" >&2
  fi
fi

cd "$ROOT"
exec "$PY" scripts/asr/bench.py "$@"
