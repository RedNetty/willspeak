#!/usr/bin/env python3
"""Build a portable application ZIP without personal data or development caches."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent
FILES = ("README.md", "pyproject.toml", "requirements.txt", "requirements-dev.txt",
         "run.py", "package.py", "willspeak.sh", "willspeak.bat")
DIRECTORIES = ("src", "scripts", "static", "prompts", "tests", "deploy")


def package(destination: Path, *, with_models: bool = False) -> None:
    paths = [ROOT / name for name in FILES]
    for name in (*DIRECTORIES, *(("models",) if with_models else ())):
        directory = ROOT / name
        if directory.is_symlink():
            raise ValueError(f"Cannot package a symlink: {directory}")
        for path in sorted(directory.rglob("*")):
            relative = path.relative_to(directory)
            if any(part.startswith(".") or part == "__pycache__" or part.endswith(".egg-info")
                   for part in relative.parts):
                continue
            if path.is_symlink():
                raise ValueError(f"Cannot package a symlink: {path}")
            if path.is_file() and path.suffix not in (".pyc", ".pyo"):
                paths.append(path)
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or unsafe application file: {path}")
    with ZipFile(destination, "x", compression=ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, Path("app") / path.relative_to(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new ZIP path")
    parser.add_argument("--with-models", action="store_true", help="include local model files")
    args = parser.parse_args()
    try:
        package(args.out, with_models=args.with_models)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Cannot package application: {exc}\n")
    print(f"Created {args.out.resolve()}")


if __name__ == "__main__":
    main()
