#!/usr/bin/env python3
"""Route each clip to one of two saved decodes by a detector the device could run, and write the routed decode."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import labkit as lab

from willspeak.normalize import normalize

DETECTORS = ("words", "duration", "oracle")


def word_count(text: str) -> int:
    return len(normalize(text).split())


def says_one_word(text: str, max_words: int = 1) -> bool:
    return 1 <= word_count(text) <= max_words


def duration_cut(rows: list[dict]) -> float:
    pairs = sorted((float(r["duration_s"]), lab.clip_kind(r) == "word") for r in rows if r.get("duration_s") is not None)
    if not pairs or not any(w for _, w in pairs) or all(w for _, w in pairs):
        raise ValueError("a duration cut needs labelled clips of both kinds with durations")
    best_cut, best_right = 0.0, sum(1 for _, w in pairs if not w)
    right = best_right
    for i, (duration, is_word) in enumerate(pairs):
        right += 1 if is_word else -1
        if i + 1 < len(pairs) and pairs[i + 1][0] == duration:
            continue
        if right > best_right:
            best_cut, best_right = duration, right
    return best_cut


def detect(detector: str, manifest: dict[str, dict], first_decode: dict[str, str] | None = None, cut: float | None = None,
           max_words: int = 1) -> dict[str, bool]:
    if detector not in DETECTORS:
        raise ValueError(f"detector is one of {', '.join(DETECTORS)}, not {detector!r}")
    called: dict[str, bool] = {}
    for clip_id, row in manifest.items():
        if detector == "oracle":
            called[clip_id] = lab.clip_kind(row) == "word"
        elif detector == "duration":
            if cut is None:
                raise ValueError("the duration detector needs a cut")
            called[clip_id] = float(row["duration_s"]) <= cut
        else:
            if first_decode is None or clip_id not in first_decode:
                raise KeyError(f"no first decode for clip {clip_id}")
            called[clip_id] = says_one_word(first_decode[clip_id], max_words)
    return called


def route(called: dict[str, bool], word: dict[str, str], other: dict[str, str]) -> dict[str, str]:
    missing = [c for c in called if c not in word or c not in other]
    if missing:
        raise KeyError(f"{len(missing)} clips lack a decode from one of the two models, e.g. {missing[0]}")
    return {c: word[c] if is_word else other[c] for c, is_word in called.items()}


def confusion(manifest: dict[str, dict], called: dict[str, bool]) -> dict:
    tp = sum(1 for c, r in manifest.items() if called[c] and lab.clip_kind(r) == "word")
    fn = sum(1 for c, r in manifest.items() if not called[c] and lab.clip_kind(r) == "word")
    fp = sum(1 for c, r in manifest.items() if called[c] and lab.clip_kind(r) != "word")
    tn = sum(1 for c, r in manifest.items() if not called[c] and lab.clip_kind(r) != "word")
    clips = tp + fn + fp + tn
    return {"clips": clips, "word_clips": tp + fn, "sentence_clips": fp + tn,
            "word_called_word": tp, "word_called_sentence": fn,
            "sentence_called_word": fp, "sentence_called_sentence": tn,
            "routed_wrong": fn + fp, "accuracy": (tp + tn) / clips if clips else None}


def read_rows(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with lab.here(path).open(encoding="utf-8") as handle:
            rows += [json.loads(line) for line in handle if line.strip()]
    return rows


def write_hyps(path: str | Path, hyps: dict[str, str]) -> None:
    path = lab.here(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"id": clip_id, "hyp": text}) + "\n" for clip_id, text in hyps.items()), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="the test manifest both decodes cover (test.jsonl)")
    ap.add_argument("--detector", required=True, choices=DETECTORS)
    ap.add_argument("--word-hyp", required=True, help="decode of the model trained on single words only (folder or hypotheses file)")
    ap.add_argument("--other-hyp", required=True, help="decode of the model that answers the clips the detector does not call single words")
    ap.add_argument("--detect-hyp", help="the first decode the words detector reads (default: --other-hyp); the device's own model")
    ap.add_argument("--max-words", type=int, default=1, help="words detector: most words in the first decode that still counts as a single word")
    ap.add_argument("--cut-from", nargs="+", metavar="MANIFEST",
                    help="duration detector: labelled manifests (the speaker's train and dev, never test) that choose the cut")
    ap.add_argument("--cut", type=float, help="duration detector: use this cut in seconds instead of choosing one")
    ap.add_argument("--out", required=True, help="routed decode, {id, hyp} rows (refuses to overwrite)")
    ap.add_argument("--stats", help="detector counts as .json (refuses to overwrite)")
    a = ap.parse_args()
    if a.max_words < 1:
        ap.error("--max-words must be 1 or more")
    if a.detector == "duration" and a.cut is None and not a.cut_from:
        ap.error("--detector duration needs --cut-from (labelled train and dev manifests) or --cut")
    lab.refuse_existing(a.out, *([a.stats] if a.stats else []))
    manifest = lab.read_manifest(a.manifest)
    other, word = lab.read_hyps(a.other_hyp), lab.read_hyps(a.word_hyp)
    first = lab.read_hyps(a.detect_hyp) if a.detect_hyp else other
    cut = a.cut
    if a.detector == "duration" and cut is None:
        cut = duration_cut(read_rows(a.cut_from))
    called = detect(a.detector, manifest, first, cut, a.max_words)
    routed = route(called, word, other)
    counts = confusion(manifest, called)
    write_hyps(a.out, routed)
    print(f"{a.detector}: {sum(called.values())} of {counts['clips']} clips sent to the word model; "
          f"{counts['routed_wrong']} routed against their true kind "
          f"({counts['sentence_called_word']} sentences sent to the word model, {counts['word_called_sentence']} single words kept from it); "
          f"accuracy {100 * counts['accuracy']:.1f}%" + (f"; duration cut {cut:.3f} s" if a.detector == "duration" else ""))
    if a.stats:
        inputs = {"manifest": a.manifest, "word": a.word_hyp, "other": a.other_hyp, **({"detect": a.detect_hyp} if a.detect_hyp else {})}
        lab.save_json(a.stats, {"format": "willspeak-kind-route-v1", "detector": a.detector, "max_words": a.max_words, "duration_cut_s": cut,
                                "counts": counts, "called_word": sorted(c for c, v in called.items() if v),
                                "provenance": lab.provenance(__file__, {k: v for k, v in inputs.items() if lab.here(v).exists()})})
    print(f"wrote {lab.here(a.out)}")
    return 0


if __name__ == "__main__":
    import _cli

    sys.exit(_cli.run(main, "kind_route"))
