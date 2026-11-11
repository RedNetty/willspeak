#!/usr/bin/env python3
"""Choose a pseudo-label for each clip that has no transcript, from its candidate pool and a language model's scores. Lab test 38."""
from __future__ import annotations

import argparse
import json
import math
import sys

import labkit as lab
from willspeak.evaluate import OK, align
from willspeak.normalize import normalize

POOL_FORMAT = "willspeak-lm-pool-v1"
SCORES_FORMAT = "willspeak-lm-scores-v1"
WEIGHT, BONUS = 0.7, 3.0
RULES = ("all", "top", "margin")
SKIP_REASON = "margin filter"
INF = float("inf")


def words(text: str) -> list[str]:
    return normalize(text).split()


def word_errors(ref: list[str], hyp: list[str]) -> int:
    return sum(op != OK for op, _, _ in align(ref, hyp))


def read_pools(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for item in paths:
        data = json.loads(lab.here(item).read_text(encoding="utf-8"))
        if data.get("format") != POOL_FORMAT:
            raise ValueError(f"{item} is not a candidate pool written by lm_pool.py")
        rows += data["rows"]
    if not rows:
        raise ValueError("The pools hold no clips")
    if len({str(r["id"]) for r in rows}) != len(rows):
        raise ValueError("A clip appears twice in the pools")
    return rows


def read_texts(path: str) -> tuple[str, dict[str, list]]:
    data = json.loads(lab.here(path).read_text(encoding="utf-8"))
    if data.get("format") != SCORES_FORMAT:
        raise ValueError(f"{path} is not a score file written by lm_score.py")
    return data["model"]["tag"], data["texts"]


def totals(row: dict, texts: dict[str, list], weight: float, bonus: float) -> list[float]:
    cands = row["candidates"]
    if not cands:
        raise ValueError(f"Clip {row['id']} has no candidates")
    out = []
    for c in cands:
        if c["lm_text"] and c["lm_text"] not in texts:
            raise ValueError(f"Clip {row['id']}: {c['lm_text']!r} has no language-model score; score the same pools")
        audio = -c["nll"] if c["nll"] is not None else -INF
        lm = texts[c["lm_text"]][0] if c["lm_text"] else 0.0
        out.append(audio + weight * lm + bonus * c["words"])
    return out


def best_and_margin(values: list[float]) -> tuple[int, float | None]:
    best = max(range(len(values)), key=lambda i: (values[i], -i))
    rest = [v for i, v in enumerate(values) if i != best]
    if values[best] == -INF:
        return best, 0.0
    if not rest or max(rest) == -INF:
        return best, None
    return best, values[best] - max(rest)


def pick_row(row: dict, texts: dict[str, list], plain: str | None, weight: float = WEIGHT, bonus: float = BONUS) -> dict:
    cands = row["candidates"]
    pool_first = cands[0]["text"]
    plain_text = pool_first if plain is None else plain
    single = len(words(plain_text)) == 1
    index, margin = best_and_margin(totals(row, texts, weight, bonus))
    chosen = cands[index]
    changed = (not single) and normalize(chosen["text"]) != normalize(plain_text)
    return {"id": str(row["id"]), "speaker": str(row.get("speaker", "")), "hyp": chosen["text"] if changed else plain_text, "plain": plain_text,
            "single": single, "changed": changed, "margin": None if single else margin, "candidates": len(cands),
            "pool_first_differs": normalize(pool_first) != normalize(plain_text)}


def pick(rows: list[dict], texts: dict[str, list], plain: dict[str, str] | None, weight: float = WEIGHT, bonus: float = BONUS) -> list[dict]:
    if plain is not None:
        missing = [str(r["id"]) for r in rows if str(r["id"]) not in plain]
        if missing:
            raise ValueError(f"The plain answers miss {len(missing)} clips of the pools, first {missing[:3]}")
    return [pick_row(r, texts, None if plain is None else plain[str(r["id"])], weight, bonus) for r in rows]


def margin_key(record: dict) -> tuple:
    margin = record["margin"]
    return (-(INF if margin is None else margin), record["id"])


def keep_top(records: list[dict], share: float) -> set[str]:
    if not 0 < share <= 1:
        raise ValueError("--share is above 0 and at most 1")
    sentence = sorted((r for r in records if not r["single"]), key=margin_key)
    return {r["id"] for r in sentence[: max(1, math.ceil(share * len(sentence) - 1e-9))]} if sentence else set()


def apply_rule(records: list[dict], rule: str, share: float = 0.5, min_margin: float = 3.0) -> list[dict]:
    if rule not in RULES:
        raise ValueError(f"The rule is one of {', '.join(RULES)}")
    if rule == "top":
        kept = keep_top(records, share)
    elif rule == "margin":
        if min_margin < 0:
            raise ValueError("--min-margin cannot be negative")
        kept = {r["id"] for r in records if not r["single"] and (r["margin"] is None or r["margin"] >= min_margin)}
    else:
        kept = {r["id"] for r in records}
    out = []
    for r in records:
        r = dict(r)
        r.pop("skip", None)
        if not r["single"] and r["id"] not in kept:
            r["skip"] = SKIP_REASON
        out.append(r)
    return out


def deciles(values: list[float]) -> list[float]:
    ordered = sorted(values)
    if not ordered:
        return []
    return [round(ordered[min(len(ordered) - 1, int(len(ordered) * q / 10))], 2) for q in range(11)]


def describe(records: list[dict]) -> dict:
    sentence = [r for r in records if not r["single"]]
    margins = [r["margin"] for r in sentence if r["margin"] is not None]
    return {"clips": len(records), "single_words": len(records) - len(sentence), "sentence_like": len(sentence),
            "changed": sum(r["changed"] for r in records), "uncontested": sum(r["margin"] is None for r in sentence),
            "pool_first_differs_from_plain": sum(r["pool_first_differs"] for r in records),
            "margin_deciles": deciles(margins), "kept": sum("skip" not in r for r in records)}


def read_records(path: str) -> list[dict]:
    rows = []
    with lab.here(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_records(path: str, records: list[dict]) -> None:
    target = lab.here(path)
    lab.refuse_existing(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    scratch = target.with_name(target.name + ".partial")
    scratch.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    scratch.replace(target)


def read_labels(path: str) -> dict:
    view = path.endswith("@plain")
    path = path[: -len("@plain")] if view else path
    target = lab.here(path)
    if target.is_dir():
        found: dict[str, str] = {}
        for row in read_records(f"{path}/train.jsonl"):
            if str(row["id"]).startswith("tta-"):
                found[str(row["id"])[len("tta-"):]] = str(row["text"])
        return {"labels": found, "single": {}, "skipped": set(), "plain": {}}
    out: dict = {"labels": {}, "single": {}, "skipped": set(), "plain": {}}
    for r in read_records(path):
        key = str(r["id"])
        if view:
            out["labels"][key] = str(r["plain"])
            continue
        if "skip" in r:
            out["skipped"].add(key)
        out["labels"][key] = str(r["hyp"])
        if "single" in r:
            out["single"][key] = bool(r["single"])
        if "plain" in r:
            out["plain"][key] = str(r["plain"])
    return out


def audit_one(truth: dict[str, list[str]], held: dict) -> dict:
    labels, single, skipped, plain = held["labels"], held["single"], held["skipped"], held["plain"]

    def tally(ids: list[str], text: dict[str, str]) -> dict:
        wrong = sum(word_errors(truth[i], words(text[i])) for i in ids)
        total = sum(len(truth[i]) for i in ids)
        exact = sum(words(text[i]) == truth[i] for i in ids)
        return {"clips": len(ids), "words": total, "wrong": wrong, "wer": round(wrong / total, 4) if total else None, "exactly_right": exact}

    unknown = sorted(set(labels) - set(truth))
    if unknown:
        raise ValueError(f"{len(unknown)} labelled clips are not in the truth manifest, first {unknown[:3]}")
    kept = [i for i in labels if i not in skipped]
    sentence = [i for i in kept if not single.get(i, len(words(labels[i])) == 1)]
    alone = [i for i in kept if i not in sentence]
    out = {"labels_used": tally(kept, labels), "sentence_like": tally(sentence, labels), "single_word": tally(alone, labels)}
    if skipped:
        out["dropped_by_the_filter"] = tally(sorted(skipped & set(labels)), labels)
    if plain and len(plain) == len(labels):
        out["plain_answers_on_the_same_clips"] = tally(kept, plain)
        out["plain_answers_on_every_clip"] = tally(list(labels), plain)
    return out


def read_truth(paths: list[str]) -> dict[str, list[str]]:
    truth: dict[str, list[str]] = {}
    for path in paths:
        for i, row in lab.read_manifest(path).items():
            if i in truth:
                raise ValueError(f"Clip {i} is in two truth manifests")
            truth[i] = words(row["text"])
    return truth


AUDIT_PARTS = (("labels_used", "all labels used"), ("sentence_like", "sentence-like"), ("single_word", "single-word"),
               ("dropped_by_the_filter", "dropped by the filter"), ("plain_answers_on_the_same_clips", "plain, same clips"),
               ("plain_answers_on_every_clip", "plain, every clip"))


def show_audit(name: str, result: dict) -> None:
    for part, label in AUDIT_PARTS:
        r = result.get(part)
        if not r:
            continue
        wer = "-" if r["wer"] is None else f"{100 * r['wer']:.1f}%"
        print(f"   {name:<30} {label:<20} {r['clips']:>4} clips {r['words']:>5} words  {r['wrong']:>4} wrong ({wer:>6})  {r['exactly_right']:>4} exactly right")


def cmd_pick(a) -> int:
    rows = read_pools(a.pool)
    tag, texts = read_texts(a.scores)
    plain = None
    if a.plain:
        plain = {}
        for item in a.plain:
            plain.update(lab.read_hyps(item))
    records = pick(rows, texts, plain, a.weight, a.bonus)
    write_records(a.out, records)
    info = describe(records)
    print(f"{tag}: {info['clips']} clips, {info['single_words']} single words left at their plain answer, {info['sentence_like']} sentences of which "
          f"{info['changed']} changed by the language model; {info['uncontested']} had no competing answer; the pool's first decode differs "
          f"from the plain answer on {info['pool_first_differs_from_plain']}")
    print(f"   margin deciles (nats, sentence-like clips with a competing answer): {info['margin_deciles']}")
    print(f"wrote {lab.here(a.out)}")
    return 0


def cmd_filter(a) -> int:
    records: list[dict] = []
    for item in a.picks:
        records += read_records(item)
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("A clip appears in two pick files")
    out = apply_rule(records, a.rule, a.share, a.min_margin)
    write_records(a.out, out)
    info = describe(out)
    print(f"rule {a.rule}: {info['kept']} of {info['clips']} clips kept ({info['single_words']} single words always; "
          f"{info['kept'] - info['single_words']} of {info['sentence_like']} sentences)")
    print(f"wrote {lab.here(a.out)}")
    return 0


def cmd_audit(a) -> int:
    truth = read_truth(a.truth)
    results: dict[str, dict] = {}
    print("   word error of the labels against the true text (evaluation only; none of this reaches training)")
    for item in a.system:
        name, sep, rest = item.partition("=")
        paths = [p for p in rest.split(",") if p]
        if not sep or not name or not paths:
            raise ValueError(f"--system wants NAME=PATH,PATH ..., not {item!r}")
        held: dict = {"labels": {}, "single": {}, "skipped": set(), "plain": {}}
        for path in paths:
            if not lab.here(path.removesuffix("@plain")).exists():
                print(f"   {name:<30} not available: {path} is not here")
                break
            got = read_labels(path)
            for key in ("labels", "single", "plain"):
                held[key].update(got[key])
            held["skipped"] |= got["skipped"]
        else:
            results[name] = audit_one(truth, held)
            show_audit(name, results[name])
    if a.out:
        target = lab.here(a.out)
        lab.refuse_existing(target)
        lab.save_json(target, {"format": "willspeak-lm-label-audit-v1", "truth": a.truth, "systems": results})
        print(f"wrote {target}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pick", help="one answer per clip from the pool, the audio score plus the language model's, with the margin")
    p.add_argument("--pool", nargs="+", required=True, help="candidate pools written by lm_pool.py (--no-reference pools are fine)")
    p.add_argument("--scores", required=True, help="the language model's score file written by lm_score.py score")
    p.add_argument("--plain", nargs="*", default=[], help="the ordinary decode's answers (hypotheses folder or .jsonl): what single words and "
                   "unchanged picks keep (default: the pool's first decode)")
    p.add_argument("--weight", type=float, default=WEIGHT, help="weight of the language-model score (default: lm_report.py's preset)")
    p.add_argument("--bonus", type=float, default=BONUS, help="bonus per word (default: lm_report.py's preset)")
    p.add_argument("--out", required=True, help="picks .jsonl to write (id and hyp as a hypotheses file, plus the margin; refuses to overwrite)")
    p = sub.add_parser("filter", help="keep all sentence picks, the top share by margin, or those above a margin")
    p.add_argument("--picks", nargs="+", required=True, help="pick files; the share is taken over every clip in them together")
    p.add_argument("--rule", required=True, choices=RULES, help="all: keep every pick; top: the --share of sentence-like clips with the widest margins; "
                   "margin: sentence-like clips whose margin is at least --min-margin nats")
    p.add_argument("--share", type=float, default=0.5, help="top: the share of sentence-like clips kept (0 to 1)")
    p.add_argument("--min-margin", type=float, default=3.0, help="margin: the smallest margin in nats that is kept (an uncontested answer always is)")
    p.add_argument("--out", required=True, help="labels .jsonl to write: every clip, with skip set on those dropped (refuses to overwrite)")
    p = sub.add_parser("audit", help="word error of labels against the true text: evaluation only, never an input to training")
    p.add_argument("--truth", nargs="+", required=True, help="manifests with the true text (clip ids must differ between them)")
    p.add_argument("--system", action="append", required=True, metavar="NAME=PATH,PATH",
                   help="labels: picks or labels .jsonl files, or training exports (the clips in them that are not his own); PATH@plain reads the "
                   "plain answers a picks file carries")
    p.add_argument("--out", help="audit .json to write (refuses to overwrite)")
    a = ap.parse_args(argv)
    return {"pick": cmd_pick, "filter": cmd_filter, "audit": cmd_audit}[a.cmd](a)


if __name__ == "__main__":
    import _cli

    sys.exit(_cli.run(main, "lm_pick"))
