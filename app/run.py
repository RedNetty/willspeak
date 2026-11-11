#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WillSpeak locally with a stock or candidate Parakeet model.")
    parser.add_argument("--model-dir", type=Path, help="model folder; does not change your saved/default model")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--host", default=os.environ.get("WILLSPEAK_HOST", "127.0.0.1"),
                        help="address to listen on; 0.0.0.0 lets other computers on the network open the pages")
    args = parser.parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "src"))
    if args.model_dir:
        selected = (ROOT / args.model_dir).resolve()
        os.environ["WILLSPEAK_PARAKEET_DIR"] = str(selected)
        from willspeak.asr.parakeet import model_files

        if model_files(selected) is None:
            parser.error(f"Incomplete Parakeet model: {selected}")
    try:
        import uvicorn
    except ImportError:
        sys.exit("uvicorn is not installed. Run: python -m pip install -r requirements.txt")
    uvicorn.run("willspeak.server:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
