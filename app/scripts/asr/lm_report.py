#!/usr/bin/env python3
"""Choose one answer per clip from its candidate pool, with the audio score plus a language-model score, and count whole sentences right. Lab test 36."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import ci
import labkit as lab
from willspeak.evaluate import OK, align

GRID_WEIGHT = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0)
GRID_BONUS = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
FIXED = (0.7, 3.0)
GRID_WORD_WEIGHT = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
GRID_WORD_VOTE = (0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0)
GRID_WORD_BONUS = (-2.0, -1.0, 0.0, 1.0)
FIRST, AUDIO, BEST = "first decode", "audio score only", "best answer in the pool"


@dataclass
class Clip:
    id: str
    speaker: str
    kind: str
    ref: str
    words: int
    texts: list[str]
    audio: np.ndarray
    length: np.ndarray
    right: np.ndarray
    wrong: np.ndarray
    votes: np.ndarray | None = None
    extra_only: np.ndarray | None = None


def only_extra(views: list[str], names: set[str]) -> bool:
    return bool(names) and bool(views) and all(v.split("/")[0] in names for v in views)


def word_errors(ref: list[str], hyp: list[str]) -> int:
    return sum(op != OK for op, _, _ in align(ref, hyp))


def read_clips(paths: list[str]) -> tuple[list[Clip], dict]:
    clips: list[Clip] = []
    views: list[str] = []
    decodes: list[int] = []
    passes: list[int] = []
    extras: list[str] = []
    for path in paths:
        data = json.loads(lab.here(path).read_text(encoding="utf-8"))
        if data.get("format") != "willspeak-lm-pool-v1":
            raise ValueError(f"{path} is not a candidate pool written by lm_pool.py")
        views = views or list(data.get("views", []))
        names = {str(m["name"]) for m in data.get("extra_models", [])}
        extras += sorted(names - set(extras))
        for row in data["rows"]:
            decodes.append(int(data.get("decodes_per_clip", len(data.get("views", [])))))
            passes.append(int(data.get("encoder_passes_per_clip", 0)))
            ref = row["ref"].split()
            cands = row["candidates"]
            if not ref or not cands:
                raise ValueError(f"Clip {row['id']} in {path} has no reference words or no candidates")
            clips.append(Clip(
                id=str(row["id"]), speaker=str(row.get("speaker") or "?"), kind=str(row["kind"]), ref=row["ref"], words=len(ref),
                texts=[c["lm_text"] for c in cands],
                audio=np.array([-c["nll"] if c["nll"] is not None else -np.inf for c in cands], dtype=float),
                length=np.array([c["words"] for c in cands], dtype=float),
                right=np.array([c["norm"] == row["ref"] for c in cands], dtype=bool),
                wrong=np.array([word_errors(ref, c["norm"].split()) for c in cands], dtype=int),
                votes=np.array([len(c.get("views") or []) for c in cands], dtype=float),
                extra_only=np.array([only_extra(c.get("views", []), names) for c in cands], dtype=bool)))
    if len({c.id for c in clips}) != len(clips):
        raise ValueError("A clip appears in two pools of the same system")
    info = {"views": views, "answers_per_clip": round(float(np.mean([len(c.texts) for c in clips])), 2),
            "decodes_per_clip": round(float(np.mean(decodes)), 2) if decodes else 0.0, "extra_models": extras}
    if any(passes):
        info["encoder_passes_per_clip"] = round(float(np.mean(passes)), 2)
    return clips, info


def read_scores(items: list[str]) -> dict[str, dict]:
    files: list[Path] = []
    for item in items:
        path = lab.here(item)
        files += sorted(path.glob("*.json")) if path.is_dir() else [path]
    out: dict[str, dict] = {}
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("format") != "willspeak-lm-scores-v1":
            raise ValueError(f"{path} is not a score file written by lm_score.py")
        tag = data["model"]["tag"]
        if tag in out:
            raise ValueError(f"Two score files for {tag}")
        out[tag] = data
    if not out:
        raise ValueError("No language-model score files found")
    return out


def lm_values(clips: list[Clip], texts: dict[str, list]) -> list[np.ndarray]:
    missing = sorted({t for c in clips for t in c.texts if t and t not in texts})
    if missing:
        raise ValueError(f"{len(missing)} candidate sentences have no language-model score, for example {missing[0]!r}; score the same pools")
    return [np.array([texts[t][0] if t else 0.0 for t in c.texts], dtype=float) for c in clips]


def choose(clip: Clip, lm: np.ndarray, weight: float, bonus: float) -> int:
    return int(np.argmax(clip.audio + weight * lm + bonus * clip.length))


def fit(clips: list[Clip], lms: list[np.ndarray], weights: tuple, bonuses: tuple) -> tuple[float, float]:
    best, key = (weights[0], bonuses[0]), None
    for weight in weights:
        for bonus in bonuses:
            picks = [choose(c, lm, weight, bonus) for c, lm in zip(clips, lms)]
            score = (sum(int(c.right[p]) for c, p in zip(clips, picks)), -sum(int(c.wrong[p]) for c, p in zip(clips, picks)), -weight, -bonus)
            if key is None or score > key:
                best, key = (weight, bonus), score
    return best


def fold_of(clips: list[Clip]) -> tuple[list[str], str]:
    speakers = sorted({c.speaker for c in clips})
    if len(speakers) >= 2:
        return [c.speaker for c in clips], "speaker"
    return [f"half {int(hashlib.sha256(c.ref.encode()).hexdigest(), 16) % 2 + 1}" for c in clips], "half of the prompts"


def cross_fit(clips: list[Clip], lms: list[np.ndarray], folds: list[str], weights: tuple, bonuses: tuple) -> tuple[list[int], dict]:
    picks = [0] * len(clips)
    chosen: dict[str, list[float]] = {}
    for fold in sorted(set(folds)):
        train = [i for i, f in enumerate(folds) if f != fold and clips[i].kind == "sentence"]
        if not train:
            raise ValueError("Choosing the weights needs sentence clips outside the held-out group")
        weight, bonus = fit([clips[i] for i in train], [lms[i] for i in train], weights, bonuses)
        chosen[fold] = [weight, bonus]
        for i, f in enumerate(folds):
            if f == fold:
                picks[i] = choose(clips[i], lms[i], weight, bonus)
    return picks, chosen


def oracle(clips: list[Clip]) -> list[int]:
    return [int(np.argmax(c.right)) if c.right.any() else int(np.argmin(c.wrong)) for c in clips]


def tally(clips: list[Clip], picks: list[int], kind: str) -> dict:
    rows = [(c, p) for c, p in zip(clips, picks) if kind in ("all", c.kind)]
    return {"clips": len(rows), "right": sum(int(c.right[p]) for c, p in rows), "wrong_words": sum(int(c.wrong[p]) for c, p in rows),
            "words": sum(c.words for c, _ in rows)}


def picked_extra_only(clip: Clip, pick: int) -> bool:
    return clip.extra_only is not None and bool(clip.extra_only[pick])


def paired(clips: list[Clip], base: list[int], picks: list[int], kind: str = "sentence") -> dict:
    rows = [(c, b, p) for c, b, p in zip(clips, base, picks) if c.kind == kind]
    if len(rows) < 2:
        return {}
    ones = np.ones(len(rows))
    miss_base = np.array([1.0 - c.right[b] for c, b, _ in rows])
    miss_new = np.array([1.0 - c.right[p] for c, _, p in rows])
    wrong_base = np.array([float(c.wrong[b]) for c, b, _ in rows])
    wrong_new = np.array([float(c.wrong[p]) for c, _, p in rows])
    words = np.array([float(c.words) for c, _, _ in rows])
    prompts = [c.ref for c, _, _ in rows]
    out = {"gained": int(sum(1 for c, b, p in rows if c.right[p] and not c.right[b])),
           "gained_from_extra_models_only": int(sum(1 for c, b, p in rows if c.right[p] and not c.right[b] and picked_extra_only(c, p))),
           "lost": int(sum(1 for c, b, p in rows if c.right[b] and not c.right[p])),
           "right_gain": float(miss_base.mean() - miss_new.mean()),
           "right_gain_ci": [float(x) for x in ci.interval(ci.bootstrap(miss_base, ones, miss_new)["delta"])],
           "wer_gain": float((wrong_base.sum() - wrong_new.sum()) / words.sum()),
           "wer_gain_ci": [float(x) for x in ci.interval(ci.bootstrap(wrong_base, words, wrong_new)["delta"])]}
    if len(set(prompts)) >= 2:
        out["right_gain_ci_by_prompt"] = [float(x) for x in ci.interval(ci.bootstrap(miss_base, ones, miss_new, clusters=prompts)["delta"])]
    return out


def describe(clips: list[Clip], base: list[int], picks: list[int], label: str, **extra) -> dict:
    return {"label": label, "sentence": tally(clips, picks, "sentence"), "word": tally(clips, picks, "word"), "all": tally(clips, picks, "all"),
            "against_first_decode": paired(clips, base, picks) if label != FIRST else {}, **extra}


def ceiling_from_extras_only(clips: list[Clip]) -> int:
    return sum(1 for c in clips if c.kind == "sentence" and c.right.any() and c.extra_only is not None and not (c.right & ~c.extra_only).any())


def system_rows(clips: list[Clip], scores: dict[str, dict], weights: tuple, bonuses: tuple, fixed: tuple,
                picks_out: dict[str, list[int]] | None = None) -> tuple[list[dict], str]:
    folds, unit = fold_of(clips)
    base = [0] * len(clips)
    zeros = [np.zeros(len(c.texts)) for c in clips]
    keep = picks_out if picks_out is not None else {}
    keep[FIRST] = base
    rows = [describe(clips, base, base, FIRST)]
    picks, chosen = cross_fit(clips, zeros, folds, (0.0,), bonuses)
    keep[AUDIO] = picks
    rows.append(describe(clips, base, picks, AUDIO, weights=chosen))
    for tag, data in scores.items():
        lms = lm_values(clips, data["texts"])
        picks, chosen = cross_fit(clips, lms, folds, weights, bonuses)
        keep[tag] = picks
        preset = [choose(c, lm, *fixed) for c, lm in zip(clips, lms)]
        rows.append(describe(clips, base, picks, tag, weights=chosen,
                             preset={"weights": list(fixed), "sentence": tally(clips, preset, "sentence"), "word": tally(clips, preset, "word")}))
    keep[BEST] = oracle(clips)
    extra = {"right_only_from_extra_models": ceiling_from_extras_only(clips)} if any(c.extra_only is not None and c.extra_only.any() for c in clips) else {}
    rows.append(describe(clips, base, keep[BEST], BEST, **extra))
    return rows, unit


def looks_single(clip: Clip) -> bool:
    return bool(clip.length[0] <= 1)


def choose_word(clip: Clip, lm: np.ndarray, weight: float, vote: float, bonus: float) -> int:
    return int(np.argmax(clip.audio + weight * lm + vote * clip.votes + bonus * clip.length))


def fit_word(clips: list[Clip], lms: list[np.ndarray]) -> tuple[float, float, float]:
    best, key = (GRID_WORD_WEIGHT[0], GRID_WORD_VOTE[0], 0.0), None
    for weight in GRID_WORD_WEIGHT:
        for vote in GRID_WORD_VOTE:
            for bonus in GRID_WORD_BONUS:
                picks = [choose_word(c, lm, weight, vote, bonus) for c, lm in zip(clips, lms)]
                score = (sum(int(c.right[p]) for c, p in zip(clips, picks)), -sum(int(c.wrong[p]) for c, p in zip(clips, picks)),
                         -(weight + vote + abs(bonus)))
                if key is None or score > key:
                    best, key = (weight, vote, bonus), score
    return best


def cross_fit_words(clips: list[Clip], lms: list[np.ndarray], folds: list[str]) -> tuple[list[int], dict]:
    picks = [0] * len(clips)
    chosen: dict[str, list[float]] = {}
    for fold in sorted(set(folds)):
        train = [i for i, f in enumerate(folds) if f != fold and clips[i].kind == "word"]
        if not train:
            raise ValueError("Choosing the single-word weights needs single-word clips outside the held-out group")
        params = fit_word([clips[i] for i in train], [lms[i] for i in train])
        chosen[fold] = list(params)
        for i, f in enumerate(folds):
            if f == fold:
                picks[i] = choose_word(clips[i], lms[i], *params)
    return picks, chosen


def routed(clips: list[Clip], sentence_picks: list[int], word_picks: list[int]) -> list[int]:
    return [w if looks_single(c) else s for c, s, w in zip(clips, sentence_picks, word_picks)]


def routing_counts(clips: list[Clip]) -> dict:
    word, sentence = [c for c in clips if c.kind == "word"], [c for c in clips if c.kind == "sentence"]
    return {"single_words": len(word), "single_words_called_single": sum(looks_single(c) for c in word),
            "sentences": len(sentence), "sentences_called_single": sum(looks_single(c) for c in sentence)}


def split_rows(clips: list[Clip], scores: dict[str, dict], weights: tuple, bonuses: tuple) -> list[dict]:
    folds, _unit = fold_of(clips)
    base = [0] * len(clips)
    zeros = [np.zeros(len(c.texts)) for c in clips]
    sentence_audio, _ = cross_fit(clips, zeros, folds, (0.0,), bonuses)
    word_audio, chosen_audio = cross_fit_words(clips, zeros, folds)

    def row(picks: list[int], label: str, **extra) -> dict:
        out = describe(clips, base, picks, label, **extra)
        out["words_against_first_decode"] = paired(clips, base, picks, "word") if label != FIRST else {}
        return out

    rows = [row(base, FIRST), row(routed(clips, sentence_audio, word_audio), "audio + votes, no language model", weights=chosen_audio)]
    for tag, data in scores.items():
        lms = lm_values(clips, data["texts"])
        sentence, chosen_sentence = cross_fit(clips, lms, folds, weights, bonuses)
        word, chosen_word = cross_fit_words(clips, lms, folds)
        rows.append(row(routed(clips, sentence, word), tag,
                        weights={f: {"sentence": chosen_sentence[f], "word": chosen_word[f]} for f in chosen_sentence}))
        alone = routed(clips, sentence, base)
        rows[-1]["single_words_left_alone"] = {kind: tally(clips, alone, kind) for kind in ("sentence", "word", "all")}
    return rows


def compare_systems(new: list[Clip], new_picks: list[int], old: list[Clip], old_picks: list[int]) -> dict:
    if sorted(c.id for c in new) != sorted(c.id for c in old):
        raise ValueError("Two systems can be compared only when they hold the same clips")
    by_id = {c.id: (c, p) for c, p in zip(old, old_picks)}
    rows = [(c, p, *by_id[c.id]) for c, p in zip(new, new_picks) if c.kind == "sentence"]
    if len(rows) < 2:
        return {}
    ones = np.ones(len(rows))
    miss_old = np.array([1.0 - o.right[q] for _, _, o, q in rows])
    miss_new = np.array([1.0 - c.right[p] for c, p, _, _ in rows])
    wrong_old = np.array([float(o.wrong[q]) for _, _, o, q in rows])
    wrong_new = np.array([float(c.wrong[p]) for c, p, _, _ in rows])
    words = np.array([float(c.words) for c, _, _, _ in rows])
    prompts = [c.ref for c, _, _, _ in rows]
    out = {"sentences": len(rows), "right_old": int((1 - miss_old).sum()), "right_new": int((1 - miss_new).sum()),
           "gained": int(((miss_old == 1) & (miss_new == 0)).sum()), "lost": int(((miss_old == 0) & (miss_new == 1)).sum()),
           "gained_from_extra_models_only": int(sum(1 for c, p, o, q in rows if c.right[p] and not o.right[q] and picked_extra_only(c, p))),
           "same_first_decode": int(sum(1 for c, _, o, _ in rows if c.texts[0] == o.texts[0])),
           "right_gain": float(miss_old.mean() - miss_new.mean()),
           "right_gain_ci": [float(x) for x in ci.interval(ci.bootstrap(miss_old, ones, miss_new)["delta"])],
           "wer_gain": float((wrong_old.sum() - wrong_new.sum()) / words.sum()),
           "wer_gain_ci": [float(x) for x in ci.interval(ci.bootstrap(wrong_old, words, wrong_new)["delta"])]}
    if len(set(prompts)) >= 2:
        out["right_gain_ci_by_prompt"] = [float(x) for x in ci.interval(ci.bootstrap(miss_old, ones, miss_new, clusters=prompts)["delta"])]
    return out


def show_comparison(label: str, rows: dict[str, dict]) -> None:
    print(f"\n== {label}: paired change in sentences right (clips matched by id; 95% intervals, 2000 resamples; the second is by prompt)")
    print(f"   {'':<24} {'right (old -> new)':>20}  {'change [interval]':>24}  {'by prompt':>20}  {'gained/lost':>11}  {'extra-only':>10}  {'WER change':>10}")
    for name, d in rows.items():
        if not d:
            continue
        by_prompt = d.get("right_gain_ci_by_prompt")
        spread = f"[{100 * by_prompt[0]:+.1f}, {100 * by_prompt[1]:+.1f}]" if by_prompt else ""
        print(f"   {name:<24} {d['right_old']:>9} -> {d['right_new']:<7}  {100 * d['right_gain']:+.1f} [{100 * d['right_gain_ci'][0]:+.1f}, {100 * d['right_gain_ci'][1]:+.1f}]"
              f"  {spread:>20}  {'+' + str(d['gained']) + ' / -' + str(d['lost']):>11}  {d['gained_from_extra_models_only']:>10}  {100 * d['wer_gain']:>+9.1f}%")


def model_rows(scores: dict[str, dict], systems: dict[str, list[Clip]], completions: dict[str, dict]) -> list[dict]:
    clips = [c for group in systems.values() for c in group if c.kind == "sentence"]
    out = []
    for tag, data in scores.items():
        texts = data["texts"]
        per_clip = [sum(texts[t][2] for t in c.texts if t in texts) for c in clips]
        done = completions.get(tag)
        out.append({"tag": tag, "name": data["model"]["name"], "released": data["model"]["released"], "params": data["model"]["params"],
                    "kind": data["model"]["kind"], "runtime": data["runtime"], "file_mb": data["file_mb"], "peak_rss_mb": data.get("peak_rss_mb"),
                    "threads": data["threads"], "tokens_per_second": data.get("tokens_per_second"), "load_seconds": data.get("load_seconds"),
                    "seconds_per_clip_median": round(statistics.median(per_clip), 3) if per_clip else None,
                    "seconds_per_clip_p90": round(float(np.percentile(per_clip, 90)), 3) if per_clip else None,
                    "machine": data.get("machine", {}).get("node"),
                    "before_the_first_word": data.get("start", {}).get("before_the_first_word"),
                    "finished_test_sentences": None if done is None else [done["finished"], done["prompts"]]})
    return out


def read_completions(folder: str | None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if folder and lab.here(folder).is_dir():
        for path in sorted(lab.here(folder).glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("format") == "willspeak-lm-completions-v1":
                out[data["model"]["tag"]] = data
    return out


def pct(part: float, whole: float) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "-"


def show(label: str, clips: list[Clip], info: dict, rows: list[dict], unit: str) -> None:
    n = rows[0]["sentence"]
    speakers = sorted({c.speaker for c in clips})
    print(f"\n== {label}: {n['clips']} sentence clips ({n['words']} words) and {rows[0]['word']['clips']} single-word clips, "
          f"speakers {' '.join(speakers)}; {info['answers_per_clip']} answers per clip from {' '.join(info['views'])}")
    if info.get("extra_models"):
        own = len(info["views"])
        print(f"   {info['decodes_per_clip']:g} decodes per clip: {own} views of the primary model and {info['decodes_per_clip'] - own:g} from the extra models "
              f"({' '.join(info['extra_models'])}); every answer is scored against the audio by the primary model")
    print(f"   weights chosen on the other clips, held out by {unit}; intervals are 95%, paired, 2000 resamples of clips")
    print(f"   {'':<24} {'sentences right':>17}  {'change [interval]':>24}  {'gained/lost':>11}  {'sentence WER':>14}  {'single words right':>18}  "
          f"{'at 0.7 / 3':>10}")
    ranked = sorted(rows[2:-1], key=lambda r: (-r["sentence"]["right"], r["sentence"]["wrong_words"]))
    for r in [rows[0], rows[1], *ranked, rows[-1]]:
        s, w, d = r["sentence"], r["word"], r["against_first_decode"]
        change = f"{100 * d['right_gain']:+.1f} [{100 * d['right_gain_ci'][0]:+.1f}, {100 * d['right_gain_ci'][1]:+.1f}]" if d else ""
        moved = f"+{d['gained']} / -{d['lost']}" if d else ""
        if d and info.get("extra_models") and r["label"] != BEST:
            moved += f" ({d['gained_from_extra_models_only']} extra-only)"
        preset = f"{r['preset']['sentence']['right']}" if "preset" in r else ""
        print(f"   {r['label']:<24} {s['right']:>4}/{s['clips']:<4} {pct(s['right'], s['clips']):>7}  {change:>24}  {moved:>11}  "
              f"{s['wrong_words']:>4} {pct(s['wrong_words'], s['words']):>7}  {str(w['right']) + '/' + str(w['clips']) if w['clips'] else '-':>18}  {preset:>10}")
    print("   'best answer in the pool' needs a perfect chooser; it is the ceiling for any re-ranking of these pools, not a method.")
    if "right_only_from_extra_models" in rows[-1]:
        print(f"   in the pool only because an extra model proposed it: {rows[-1]['right_only_from_extra_models']} of the {n['clips']} sentence clips "
              "(extra-only: the chosen answer came from no view of the primary model)")


def show_split(label: str, rows: list[dict], counts: dict) -> None:
    print(f"\n== {label}: single words with their own chooser (weights fitted on other speakers' single-word clips: language model, votes, per-word term)")
    print(f"   a clip counts as a single word when its first decode has one word: {counts['single_words_called_single']} of "
          f"{counts['single_words']} single-word clips and {counts['sentences_called_single']} of {counts['sentences']} sentence clips are called single")
    print(f"   {'':<34} {'single words right':>18}  {'change [interval]':>24}  {'gained/lost':>11}  {'sentences right':>15}  {'wrong words, all clips':>22}  {'if single words left alone':>26}")
    ranked = sorted(rows[2:], key=lambda r: (-r["word"]["right"], r["word"]["wrong_words"]))
    for r in [rows[0], rows[1], *ranked]:
        w, s, a, d = r["word"], r["sentence"], r["all"], r["words_against_first_decode"]
        change = f"{100 * d['right_gain']:+.1f} [{100 * d['right_gain_ci'][0]:+.1f}, {100 * d['right_gain_ci'][1]:+.1f}]" if d else ""
        moved = f"+{d['gained']} / -{d['lost']}" if d else ""
        left = r.get("single_words_left_alone")
        alone = f"{left['all']['wrong_words']} {pct(left['all']['wrong_words'], left['all']['words'])}" if left else ""
        print(f"   {r['label']:<34} {w['right']:>4}/{w['clips']:<4} {pct(w['right'], w['clips']):>7}  {change:>24}  {moved:>11}  "
              f"{s['right']:>5}/{s['clips']:<4}   {a['wrong_words']:>9} {pct(a['wrong_words'], a['words']):>7}  {alone:>26}")


def show_models(rows: list[dict]) -> None:
    print("\n== the language models: size, memory and speed on this computer (not the Pi)")
    print(f"   {'':<21} {'released':<9} {'kind':<9} {'file MB':>8} {'peak MB':>8} {'tokens/s':>9} {'s per clip':>11} {'p90':>6}  finished test sentences")
    for r in rows:
        done = "-" if r["finished_test_sentences"] is None else f"{r['finished_test_sentences'][0]} of {r['finished_test_sentences'][1]}"
        print(f"   {r['tag']:<21} {r['released']:<9} {r['kind']:<9} {r['file_mb']:>8.0f} {r['peak_rss_mb'] or 0:>8.0f} {r['tokens_per_second'] or 0:>9.0f} "
              f"{r['seconds_per_clip_median'] or 0:>11.2f} {r['seconds_per_clip_p90'] or 0:>6.2f}  {done}")
    print("   peak MB is the scoring process alone, with the language model loaded and nothing else; s per clip is the time to score")
    print("   every answer in one sentence clip's pool; 'finished' counts test sentences the model completed from their first half.")
    print("   kind: base models continue text; instruct models are tuned for chat, with no base file published in this format.")


def parse_system(text: str) -> tuple[str, list[str]]:
    label, sep, rest = text.partition("=")
    paths = [p for p in rest.split(",") if p]
    if not sep or not label.strip() or not paths:
        raise ValueError(f"A system is label=pool.json,pool.json ... not {text!r}")
    return label.strip(), paths


def parse_compare(text: str) -> tuple[str, str]:
    new, sep, old = text.partition("=")
    if not sep or not new.strip() or not old.strip() or new.strip() == old.strip():
        raise ValueError(f"--compare is NEW=OLD with two different system labels, not {text!r}")
    return new.strip(), old.strip()


def numbers(text: str) -> tuple[float, ...]:
    values = tuple(float(x) for x in text.replace(",", " ").split())
    if not values or any(v < 0 for v in values):
        raise ValueError("A grid is a list of numbers that are zero or more")
    return values


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", action="append", required=True, metavar="LABEL=POOL,POOL", help="one recogniser's pools (repeatable)")
    ap.add_argument("--scores", nargs="+", required=True, help="score files from lm_score.py score, or the folder that holds them")
    ap.add_argument("--completions", help="folder of lm_score.py complete results (the memorisation check)")
    ap.add_argument("--weights", default=" ".join(str(x) for x in GRID_WEIGHT), help="language-model weights to choose from")
    ap.add_argument("--bonuses", default=" ".join(str(x) for x in GRID_BONUS), help="per-word bonuses to choose from")
    ap.add_argument("--preset", default=" ".join(str(x) for x in FIXED), help="one weight and bonus also reported as they are, not chosen here")
    ap.add_argument("--single-word-chooser", action="store_true",
                    help="also choose single-word clips (first decode of one word) with weights fitted on single-word clips, using audio, language model and votes")
    ap.add_argument("--compare", action="append", default=[], metavar="NEW=OLD",
                    help="also print and save the paired change in sentences right of system NEW against system OLD (two --system labels that "
                         "hold the same clips; repeatable). Each row uses that system's own weights, chosen on other speakers")
    ap.add_argument("--out", required=True, help="table .json to write (refuses to overwrite)")
    a = ap.parse_args()
    weights, bonuses, preset = numbers(a.weights), numbers(a.bonuses), numbers(a.preset)
    if len(preset) != 2:
        ap.error("--preset is two numbers: the language-model weight and the per-word bonus")
    lab.refuse_existing(a.out)
    scores = read_scores(a.scores)
    systems: dict[str, list[Clip]] = {}
    table: dict[str, dict] = {}
    inputs: dict[str, str] = {}
    chosen: dict[str, dict[str, list[int]]] = {}
    pairs = [parse_compare(item) for item in a.compare]
    for item in a.system:
        label, paths = parse_system(item)
        if label in systems:
            raise ValueError(f"System {label!r} is listed twice")
        clips, info = read_clips(paths)
        chosen[label] = {}
        rows, unit = system_rows(clips, scores, weights, bonuses, preset, chosen[label])
        systems[label] = clips
        table[label] = {"pools": paths, "speakers": sorted({c.speaker for c in clips}), "held_out_by": unit, **info, "rows": rows}
        inputs.update({f"{label}/{Path(p).name}": p for p in paths})
        show(label, clips, info, rows, unit)
        if a.single_word_chooser:
            counts = routing_counts(clips)
            if not counts["single_words"]:
                print(f"   (single-word chooser skipped for {label}: it has no single-word clips)")
            else:
                try:
                    split = split_rows(clips, scores, weights, bonuses)
                except ValueError as err:
                    print(f"   (single-word chooser skipped for {label}: {err})")
                else:
                    show_split(label, split, counts)
                    table[label]["single_word_chooser"] = {"detector": "first decode has at most one word", "counts": counts, "rows": split}
    comparisons: dict[str, dict] = {}
    for new, old in pairs:
        if new not in systems or old not in systems:
            raise ValueError(f"--compare {new}={old}: both must be --system labels")
        key = f"{new} vs {old}"
        comparisons[key] = {name: compare_systems(systems[new], picks, systems[old], chosen[old][name]) for name, picks in chosen[new].items()}
        show_comparison(key, comparisons[key])
    models = model_rows(scores, systems, read_completions(a.completions))
    show_models(models)
    lab.save_json(a.out, {"format": "willspeak-lm-report-v1", "weights": list(weights), "bonuses": list(bonuses), "preset": list(preset),
                          "resamples": ci.N_RESAMPLES, "seed": ci.SEED, "systems": table, "comparisons": comparisons, "models": models,
                          "provenance": lab.provenance(__file__, inputs)})
    print(f"\nwrote {lab.here(a.out)}")
    return 0


if __name__ == "__main__":
    import _cli

    sys.exit(_cli.run(main, "lm_report"))
