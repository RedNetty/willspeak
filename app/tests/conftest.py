import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("WILLSPEAK_LOG_DIR", tempfile.mkdtemp(prefix="willspeak-test-logs-"))
os.environ.setdefault("WILLSPEAK_QUALITY_GATE", "0")
