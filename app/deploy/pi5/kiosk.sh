#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-${WILLSPEAK_URL:-/live}}"
case "$TARGET" in
  http://*|https://*) URL="$TARGET" ;;
  /*)                 URL="http://127.0.0.1:8000$TARGET" ;;
  *)                  URL="http://127.0.0.1:8000/$TARGET" ;;
esac
ORIGIN="$(printf '%s' "$URL" | sed -E 's#^(https?://[^/]+).*#\1#')"
DEADLINE=$((SECONDS + 180))

while ! curl -sf -o /dev/null "$ORIGIN/api/state"; do
  if [ "$SECONDS" -ge "$DEADLINE" ]; then
    echo "willspeak did not come up within 180s; not starting the kiosk" >&2
    exit 1
  fi
  sleep 2
done

BROWSER="$(command -v chromium-browser || command -v chromium || true)"
[ -n "$BROWSER" ] || { echo "no chromium found" >&2; exit 1; }

exec "$BROWSER" \
  --kiosk \
  --noerrdialogs \
  --disable-infobars \
  --disable-session-crashed-bubble \
  --password-store=basic \
  --check-for-update-interval=31536000 \
  --autoplay-policy=no-user-gesture-required \
  --use-fake-ui-for-media-stream \
  --unsafely-treat-insecure-origin-as-secure="$ORIGIN" \
  "$URL"
