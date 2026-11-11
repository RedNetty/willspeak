#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"

SECONDS_TO_RECORD="${1:-5}"
DEVICE="${2:-}"
OUT="$(mktemp -t willspeak-audio-XXXX.wav)"
trap 'rm -f "$OUT"' EXIT

echo "== capture devices =="
arecord -l || echo "(arecord found nothing -- is the USB microphone plugged in?)"
echo
echo "== playback devices =="
aplay -l || echo "(aplay found nothing)"
echo
if command -v wpctl > /dev/null 2>&1; then
  echo "== PipeWire defaults (this is what Chromium uses) =="
  wpctl status | sed -n '/Audio/,/Video/p' | head -40
  echo "   change with:  wpctl set-default <id>"
  echo
fi

echo "== recording ${SECONDS_TO_RECORD}s -- say a sentence now =="
if [ -n "$DEVICE" ]; then
  arecord -D "$DEVICE" -f S16_LE -c 1 -r 16000 -d "$SECONDS_TO_RECORD" "$OUT"
else
  arecord -f S16_LE -c 1 -r 16000 -d "$SECONDS_TO_RECORD" "$OUT"
fi

echo
echo "== level =="
"$PY" - "$OUT" <<'PY'
import sys, wave, array, math

with wave.open(sys.argv[1], "rb") as w:
    frames = w.getnframes()
    rate = w.getframerate()
    samples = array.array("h", w.readframes(frames))

peak = max((abs(s) for s in samples), default=0) / 32768.0
rms = math.sqrt(sum(s * s for s in samples) / max(len(samples), 1)) / 32768.0
db = lambda x: -99.0 if x <= 0 else 20 * math.log10(x)

window = max(1, rate // 20)
blocks = [math.sqrt(sum(s * s for s in samples[i:i + window]) / window) / 32768.0
          for i in range(0, len(samples) - window, window)]
blocks.sort()
floor = blocks[len(blocks) // 10] if blocks else 0.0

print(f"  duration   {frames / rate:.2f} s at {rate} Hz")
print(f"  peak       {db(peak):6.1f} dBFS")
print(f"  rms        {db(rms):6.1f} dBFS")
print(f"  noise      {db(floor):6.1f} dBFS  (quietest tenth)")
print(f"  snr        {db(rms) - db(floor):6.1f} dB")
print()
if peak >= 0.99:
    print("  CLIPPING -- turn the input gain down (alsamixer, or the volume icon).")
elif db(peak) < -30:
    print("  Very quiet. Move the microphone closer or raise the gain in alsamixer.")
elif db(rms) - db(floor) < 15:
    print("  Low signal-to-noise. Quieter room, or a closer / more directional mic.")
else:
    print("  Levels look usable.")
PY

echo
echo "== playing it back =="
aplay "$OUT" || echo "(playback failed -- check the speaker and the default output above)"
