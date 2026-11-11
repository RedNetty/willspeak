#!/usr/bin/env python3
"""Kind-aware tallies for lab test 37: wrong words on sentences and on single words, by seed, against a control."""
from __future__ import annotations

import argparse
import json
import sys

import labkit as lab
import pool_table

WORD_GAIN_MIN = 10.0
SENTENCE_HARM_MAX = 5.0
KINDS = ("all", "sentence", "word")


def parse_pattern(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise ValueError(f"a system is LABEL=PATTERN with {{speaker}} and {{seed}} in the pattern, not {text!r}")
    label, pattern = text.split("=", 1)
    if "{speaker}" not in pattern or "{seed}" not in pattern:
        raise ValueError(f"the pattern of {label!r} must contain {{speaker}} and {{seed}}")
    return label.strip(), pattern


def find_hyps(pattern: str, speakers: list[str], seed: int) -> dict[str, dict[str, str]] | None:
    out: dict[str, dict[str, str]] = {}
    for s in speakers:
        path = lab.here(pattern.format(speaker=s, seed=seed))
        if not (path.is_file() or (path / "hypotheses.jsonl").is_file()):
            return None
        out[s] = lab.read_hyps(path)
    return out


def tally(manifests: dict[str, dict[str, dict]], hyps: dict[str, dict[str, str]]) -> dict:
    result: dict = {}
    for kind in KINDS:
        subset = {s: lab.only_kind(m, kind) for s, m in manifests.items()}
        pooled = lab.Pooled(kind, subset, hyps)
        result[kind] = {"wrong": pooled.pooled_wrong(), "words": pooled.pooled_words(),
                        "by_speaker": {s: pooled.wrong(s) for s in pooled.speakers},
                        "words_by_speaker": {s: pooled.words(s) for s in pooled.speakers}, "_pooled": pooled}
    return result


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def compare(control: dict[int, dict], system: dict[int, dict], speakers: list[str]) -> dict:
    seeds = sorted(set(control) & set(system))
    if not seeds:
        return {}
    out: dict = {"seeds": seeds}
    for kind in KINDS:
        by_seed = [control[x][kind]["wrong"] - system[x][kind]["wrong"] for x in seeds]
        by_speaker = {s: mean([control[x][kind]["by_speaker"][s] - system[x][kind]["by_speaker"][s] for x in seeds]) for s in speakers}
        out[kind] = {"gain_by_seed": by_seed, "gain": mean(by_seed), "gain_by_speaker": by_speaker,
                     "speakers_improved": sum(1 for v in by_speaker.values() if v > 0),
                     "speakers_worse": sum(1 for v in by_speaker.values() if v < 0)}
    words = control[seeds[0]]["all"]["words"]
    points = {s: 100.0 * out["all"]["gain_by_speaker"][s] / control[seeds[0]]["all"]["words_by_speaker"][s] for s in speakers}
    out["project_verdict"] = pool_table.verdict(100.0 * out["all"]["gain"] / words, points)
    intervals = []
    for x in seeds:
        interval = lab.pooled_interval(system[x]["word"]["_pooled"], control[x]["word"]["_pooled"])
        n = control[x]["word"]["words"]
        intervals.append([round(interval["gain_ci"][0] * n, 1), round(interval["gain_ci"][1] * n, 1)])
    out["word_gain_ci_words_by_seed"] = intervals
    out["single_word_rule"] = bool(out["word"]["gain"] >= WORD_GAIN_MIN and out["word"]["speakers_improved"] >= 2
                                   and all(low > 0 for low, _ in intervals))
    out["sentence_harm"] = bool(-out["sentence"]["gain"] > SENTENCE_HARM_MAX)
    return out


def routing_counts(pattern: str, speakers: list[str], seeds: list[int]) -> dict | None:
    total = {"clips": 0, "routed_wrong": 0, "sentence_called_word": 0, "word_called_sentence": 0, "word_clips": 0, "sentence_clips": 0}
    runs = 0
    for seed in seeds:
        for s in speakers:
            path = lab.here(pattern.format(speaker=s, seed=seed))
            if not path.is_file():
                return None
            counts = json.loads(path.read_text(encoding="utf-8"))["counts"]
            for key in total:
                total[key] += counts[key]
            runs += 1
    if not runs:
        return None
    per = {key: value / len(seeds) for key, value in total.items()}
    per["accuracy"] = 1 - total["routed_wrong"] / total["clips"] if total["clips"] else None
    return per


def render(rows: list[dict], control: str, speakers: list[str]) -> str:
    out = [f"wrong words, means over the seeds each system has; speakers {', '.join(speakers)}; lower is better; differences are {control} minus the system (positive = fewer wrong)",
           f"{'system':<52} {'seeds':>5} {'all':>13} {'sentences':>15} {'single words':>16} {'vs control: all / sent. / words':>34} {'spk':>4}  rules"]
    for r in rows:
        a, s, w = r["mean"]["all"], r["mean"]["sentence"], r["mean"]["word"]
        line = (f"{r['system']:<52} {len(r['seeds']):>5} {a:>6.1f}/{r['words']['all']:<5} {s:>7.1f}/{r['words']['sentence']:<5} {w:>8.1f}/{r['words']['word']:<5}")
        c = r.get("against_control")
        if c:
            line += (f" {c['all']['gain']:>+9.1f} {c['sentence']['gain']:>+9.1f} {c['word']['gain']:>+9.1f} {c['all']['speakers_improved']:>3}/{len(speakers)}"
                     f"  project {c['project_verdict']}; single-word {'met' if c['single_word_rule'] else 'not met'}; sentences {'HARMED' if c['sentence_harm'] else 'not harmed'}")
        else:
            line += "  (control)" if r["system"] == control else ""
        out.append(line)
        if r.get("routing"):
            q = r["routing"]
            out.append(f"{'':<52}   detector: {100 * q['accuracy']:.1f}% right; per seed {q['routed_wrong']:.1f} of {q['clips']:.0f} clips routed against their kind "
                       f"({q['sentence_called_word']:.1f} sentences sent to the word model, {q['word_called_sentence']:.1f} of {q['word_clips']:.0f} single words kept from it)")
    out.append(f"rules: project = at least 2 of {len(speakers)} speakers the same way and more than {pool_table.NOISE_POINTS} points ({round(pool_table.NOISE_POINTS / 100 * rows[0]['words']['all']) if rows else '?'} words) on all words; "
               f"single-word = at least {WORD_GAIN_MIN:g} words fewer wrong on single words, at least 2 speakers the same way, and the paired 95% interval above zero at every seed; "
               f"sentences harmed = more than {SENTENCE_HARM_MAX:g} words more wrong on sentences")
    return "\n".join(out)


def build(manifests: dict[str, dict[str, dict]], systems: dict[str, str], control: str, seeds: list[int], routing: dict[str, str]) -> list[dict]:
    speakers = list(manifests)
    scored: dict[str, dict[int, dict]] = {}
    for label, pattern in systems.items():
        for seed in seeds:
            hyps = find_hyps(pattern, speakers, seed)
            if hyps is not None:
                scored.setdefault(label, {})[seed] = tally(manifests, hyps)
    if control not in scored:
        raise ValueError(f"the control {control!r} has no complete decode at any seed")
    rows = []
    for label in systems:
        if label not in scored:
            continue
        per = scored[label]
        first = per[min(per)]
        row = {"system": label, "seeds": sorted(per),
               "words": {k: first[k]["words"] for k in KINDS},
               "by_seed": {k: [per[x][k]["wrong"] for x in sorted(per)] for k in KINDS},
               "mean": {k: mean([per[x][k]["wrong"] for x in sorted(per)]) for k in KINDS},
               "mean_by_speaker": {k: {s: mean([per[x][k]["by_speaker"][s] for x in sorted(per)]) for s in speakers} for k in KINDS}}
        if label != control:
            row["against_control"] = compare(scored[control], per, speakers)
        if label in routing:
            row["routing"] = routing_counts(routing[label], speakers, sorted(per))
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", action="append", required=True, metavar="LABEL=PATTERN", help="a system's decode folders, with {speaker} and {seed} in the pattern; repeat for each")
    ap.add_argument("--control", required=True, help="label of the control system")
    ap.add_argument("--routing", action="append", default=[], metavar="LABEL=PATTERN", help="a routed system's kind_route.py --stats files, with {speaker} and {seed}")
    ap.add_argument("--seeds", type=int, nargs="+", default=[17, 18, 19])
    ap.add_argument("--speakers", default=",".join(lab.SEVERE_TEST))
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--export", default=lab.EXPORT)
    ap.add_argument("--out", help="also write the table as .json (refuses to overwrite)")
    a = ap.parse_args()
    if a.out:
        lab.refuse_existing(a.out)
    speakers = [s.strip() for s in a.speakers.split(",") if s.strip()]
    systems = dict(parse_pattern(t) for t in a.system)
    if len(systems) != len(a.system):
        raise ValueError("a system label is listed twice")
    routing = dict(parse_pattern(t) for t in a.routing)
    manifests = {s: lab.read_manifest(f"{a.export}/{s}/{a.split}.jsonl") for s in speakers}
    rows = build(manifests, systems, a.control, a.seeds, routing)
    print(render(rows, a.control, speakers))
    if a.out:
        lab.save_json(a.out, {"format": "willspeak-kind-route-report-v1", "control": a.control, "speakers": speakers, "seeds": a.seeds, "split": a.split,
                              "rules": {"word_gain_min": WORD_GAIN_MIN, "sentence_harm_max": SENTENCE_HARM_MAX, "noise_points": pool_table.NOISE_POINTS},
                              "rows": rows, "provenance": lab.provenance(__file__, {})})
        print(f"wrote {lab.here(a.out)}")
    return 0


if __name__ == "__main__":
    import _cli

    sys.exit(_cli.run(main, "kind_route_report"))
