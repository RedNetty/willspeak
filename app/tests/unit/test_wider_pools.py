import copy
import hashlib
import importlib.util
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "asr"))
import lm_pool  # noqa: E402
import lm_report  # noqa: E402
import lm_score  # noqa: E402


def cand(norm: str, nll: float | None, views=("greedy",)) -> dict:
    return {"norm": norm, "text": norm, "nll": nll, "views": list(views), "words": len(norm.split()), "lm_text": lm_pool.lm_text(norm)}


def scorer(table: dict[str, float], empty: float = 9.0, asked: list | None = None):
    def score(texts):
        if asked is not None:
            asked.append(list(texts))
        return [table[t] for t in texts], empty
    return score


BASE = [cand("the boy ran to the stone sore", 10.0), cand("the boy ran to the store", 11.0, ("beam4",)), cand("the boy ran", 12.0, ("tempo1.1",))]


def test_an_extra_answer_that_matches_a_pool_answer_only_adds_its_tag():
    asked: list = []
    merged, dropped, added = lm_pool.merge_extras(BASE, {"The boy ran to the store.": ["seed18"], "the boy ran to the stone sore": ["seed19", "seed18"]},
                                                  scorer({}, asked=asked), 15.0)
    assert asked == [] and added == 0 and dropped == 0
    assert [c["norm"] for c in merged] == [c["norm"] for c in BASE] and [c["nll"] for c in merged] == [10.0, 11.0, 12.0]
    assert merged[0]["views"] == ["greedy", "seed19", "seed18"] and merged[1]["views"] == ["beam4", "seed18"] and merged[2]["views"] == ["tempo1.1"]
    assert BASE[0]["views"] == ["greedy"], "the baseline pool is copied, never edited"


def test_new_extra_answers_are_scored_once_by_the_primary_and_appended_after_the_first_decode():
    asked: list = []
    found = {"The boy ran to the shore.": ["seed18"], "the boy ran to the shore": ["block12"], "boy ran to the store": ["block12/beam4"], "": ["seed19"]}
    table = {"The boy ran to the shore.": 10.5, "the boy ran to the shore": 10.2, "boy ran to the store": 13.0, "": 99.0}
    merged, dropped, added = lm_pool.merge_extras(BASE, found, scorer(table, empty=8.0, asked=asked), 15.0)
    assert asked == [["The boy ran to the shore.", "the boy ran to the shore", "boy ran to the store", ""]], "one scoring call, for the answers no baseline candidate has"
    assert added == 3 and dropped == 0 and len(merged) == 6
    assert [c["norm"] for c in merged[:3]] == [c["norm"] for c in BASE], "the first decode stays first, baseline order is kept"
    shore = merged[3]
    assert shore["norm"] == "the boy ran to the shore" and shore["views"] == ["seed18", "block12"], "same answer from two extra models is one candidate"
    assert shore["nll"] == 10.2 and shore["text"] == "the boy ran to the shore" and shore["lm_text"] == "The boy ran to the shore."
    assert [f["nll"] for f in shore["forms"]] == [10.5, 10.2] and "forms" not in merged[4]
    assert merged[4]["views"] == ["block12/beam4"] and merged[5]["norm"] == "" and merged[5]["nll"] == 8.0 and merged[5]["words"] == 0


def test_the_fifteen_nat_rule_is_held_against_the_primary_models_own_best_so_the_baseline_pool_survives_whole():
    found = {"the boy ran to the store maybe": ["seed18"], "a far worse one": ["seed18"]}
    table = {"the boy ran to the store maybe": -40.0, "a far worse one": 30.0}
    merged, dropped, added = lm_pool.merge_extras(BASE, found, scorer(table), 15.0)
    assert [c["norm"] for c in merged[:3]] == [c["norm"] for c in BASE], "no baseline answer is dropped by an extra answer"
    assert "a far worse one" not in [c["norm"] for c in merged] and dropped == 1 and added == 1
    assert any(c["norm"] == "the boy ran to the store maybe" for c in merged)
    kept, _, _ = lm_pool.merge_extras(BASE, {"edge": ["a"], "over": ["a"]}, scorer({"edge": 25.0, "over": 25.5}), 15.0)
    assert [c["norm"] for c in kept][3:] == ["edge"]
    bad_first = [cand("x y", 90.0), cand("z", 1.0, ("beam4",))]
    kept, _, _ = lm_pool.merge_extras(bad_first, {"new": ["a"]}, scorer({"new": 2.0}), 15.0)
    assert [c["norm"] for c in kept][0] == "x y"
    kept, dropped, _ = lm_pool.merge_extras(BASE, {"way off": ["a"]}, scorer({"way off": 500.0}), 0.0)
    assert dropped == 0 and len(kept) == 4, "keep_within 0 keeps everything, as for the views"


def test_without_extras_prune_behaves_as_before():
    cands = [cand("a", 20.0), cand("b", 3.0), cand("c", 18.5), cand("d", 25.0)]
    kept, dropped = lm_pool.prune(cands, 15.0)
    assert [c["norm"] for c in kept] == ["a", "b"] and dropped == 2, "the first decode stays; 18.5 is more than 15 behind the best (3.0)"
    kept, dropped = lm_pool.prune(cands, 15.0, line=20.0)
    assert [c["norm"] for c in kept] == ["a", "b", "c", "d"] and dropped == 0


def random_clip(rng: random.Random) -> tuple[list[dict], dict[str, list[str]], dict[str, float], str]:
    words = ["the", "boy", "ran", "to", "store", "shore", "door", "open", "a"]
    sentence = lambda: " ".join(rng.choice(words) for _ in range(rng.randint(1, 5)))  # noqa: E731
    ref = sentence()
    base, seen = [], set()
    for i in range(rng.randint(1, 6)):
        text = ref if i and rng.random() < 0.3 else sentence()
        if text not in seen:
            seen.add(text)
            base.append(cand(text, rng.uniform(0, 30), (rng.choice(["greedy", "beam4", "tempo1.2"]),)))
    found = {}
    for _ in range(rng.randint(0, 6)):
        text = ref if rng.random() < 0.3 else sentence()
        found[text.capitalize() + rng.choice(["", "."])] = [rng.choice(["seed18", "block12"])]
    table = {text: rng.uniform(-30, 60) for text in found}
    base, _ = lm_pool.prune(base, 15.0)
    return base, found, table, ref


def test_the_pool_ceiling_can_only_rise_or_stay():
    rng = random.Random(7)
    gained = 0
    for _ in range(400):
        base, found, table, ref = random_clip(rng)
        merged, _dropped, added = lm_pool.merge_extras(base, found, scorer(table), 15.0)
        before, after = any(c["norm"] == ref for c in base), any(c["norm"] == ref for c in merged)
        assert after or not before, "an extra model removed the right answer from a pool"
        assert merged[0]["norm"] == base[0]["norm"] and {c["norm"] for c in base} <= {c["norm"] for c in merged}
        assert [c["nll"] for c in merged[: len(base)]] == [c["nll"] for c in base]
        assert len(merged) == len(base) + added
        gained += after and not before
    assert gained > 20, "the random pools should sometimes gain the right answer, or the test checks nothing"


def test_extra_models_parse_in_the_base_and_adapter_form_the_model_flags_use():
    parsed = lm_pool.parse_extra_models(["seed18=models/a/parakeet-ft.nemo,models/b/adapter.pt", "plain=stock", "named=pretrained:nvidia/x,adapter.pt"])
    assert [m.name for m in parsed] == ["seed18", "plain", "named"]
    ns = parsed[0].namespace()
    assert (ns.stock, ns.pretrained, ns.nemo, ns.adapter, ns.mix) == (False, None, "models/a/parakeet-ft.nemo", "models/b/adapter.pt", [])
    ns = parsed[1].namespace()
    assert (ns.stock, ns.nemo, ns.adapter) == (True, None, None)
    ns = parsed[2].namespace()
    assert (ns.stock, ns.pretrained, ns.nemo) == (False, "nvidia/x", None)
    assert parsed[0].tag("greedy") == "seed18" and parsed[0].tag("beam4") == "seed18/beam4"
    for bad in ("noequals", "x=", "Seed=stock", "greedy=stock", "beam4=stock", "tempo1.1=stock", "9x=stock", "=stock"):
        with pytest.raises(ValueError):
            lm_pool.parse_extra_models([bad])
    with pytest.raises(ValueError, match="twice"):
        lm_pool.parse_extra_models(["a=stock", "a=stock"])
    assert lm_pool.parse_extra_models([]) == []


def test_extra_models_propose_greedy_and_optionally_beams_only():
    assert [v[0] for v in lm_pool.parse_extra_views("greedy beam4")] == ["greedy", "beam4"]
    assert [v[0] for v in lm_pool.parse_extra_views("beam4")] == ["greedy", "beam4"]
    for bad in ("tempo1.1", "greedy pitch0.9", "beam1"):
        with pytest.raises(ValueError):
            lm_pool.parse_extra_views(bad)


def test_an_earlier_pool_is_reused_only_for_the_model_that_made_it():
    files = {"nemo": {"sha256": "n1", "size": 5}, "adapter": {"sha256": "a1", "size": 7}}
    data = {"format": "willspeak-lm-pool-v1", "model_files": files, "sigma": 0.05, "durations": [0, 1, 2], "keep_within": 15.0, "kinds": ["sentence", "word"]}
    lm_pool.check_base_pool(data, files, 0.05, [0, 1, 2], 15.0, ["word", "sentence"])
    retrained = {**files, "adapter": {"sha256": "a2", "size": 7}}
    with pytest.raises(ValueError, match="different primary model"):
        lm_pool.check_base_pool(data, retrained, 0.05, [0, 1, 2], 15.0, ["sentence", "word"])
    with pytest.raises(ValueError, match="different primary model"):
        lm_pool.check_base_pool({**data, "model_files": {}}, files, 0.05, [0, 1, 2], 15.0, ["sentence", "word"])
    with pytest.raises(ValueError, match="keep-within"):
        lm_pool.check_base_pool(data, files, 0.05, [0, 1, 2], 10.0, ["sentence", "word"])
    with pytest.raises(ValueError, match="TDT"):
        lm_pool.check_base_pool(data, files, 0.05, [0, 1], 15.0, ["sentence", "word"])
    with pytest.raises(ValueError, match="kinds"):
        lm_pool.check_base_pool(data, files, 0.05, [0, 1, 2], 15.0, ["sentence"])
    with pytest.raises(ValueError, match="not a candidate pool"):
        lm_pool.check_base_pool({"format": "other"}, files, 0.05, [0, 1, 2], 15.0, ["sentence", "word"])


def wide_pool(rows: list[dict], extras=("seed18", "seed19", "block12", "block16")) -> dict:
    return {"format": "willspeak-lm-pool-v1", "views": ["greedy", "beam4"], "extra_views": ["greedy"], "decodes_per_clip": 2 + len(extras),
            "encoder_passes_per_clip": 1 + len(extras), "extra_models": [{"name": n} for n in extras], "rows": rows}


def test_a_narrower_arm_is_cut_from_the_wider_pool_exactly_as_if_only_its_models_had_run():
    rng = random.Random(3)
    for _ in range(100):
        base, found, table, _ref = random_clip(rng)
        names = ["seed18", "seed19", "block12", "block16"]
        everything = {text: [t for t in tags] + ([rng.choice(names)] if rng.random() < 0.5 else []) for text, tags in found.items()}
        everything = {text: list(dict.fromkeys(tags)) for text, tags in everything.items()}
        table = {text: table.get(text, rng.uniform(0, 30)) for text in everything}
        wide, _, _ = lm_pool.merge_extras(base, everything, scorer(table), 15.0)
        pool = wide_pool([{"id": "c", "candidates": wide}])
        keep = ["seed18", "block16"]
        cut = lm_pool.subset_pool(pool, keep)["rows"][0]["candidates"]
        narrow_found = {text: [t for t in tags if t in keep] for text, tags in everything.items() if any(t in keep for t in tags)}
        direct, _, _ = lm_pool.merge_extras(base, narrow_found, scorer(table), 15.0)
        assert [(c["norm"], c["nll"], c["views"]) for c in cut] == [(c["norm"], c["nll"], c["views"]) for c in direct]


def test_cutting_a_pool_refuses_models_it_does_not_have_and_updates_the_cost():
    pool = wide_pool([{"id": "c", "candidates": [cand("a b", 1.0), cand("c d", 2.0, ("seed18",))]}])
    cut = lm_pool.subset_pool(pool, ["seed19"])
    assert [c["norm"] for c in cut["rows"][0]["candidates"]] == ["a b"] and cut["rows"][0]["extra_answers"] == 0
    assert cut["decodes_per_clip"] == 3 and [m["name"] for m in cut["extra_models"]] == ["seed19"] and cut["encoder_passes_per_clip"] == 2
    with pytest.raises(ValueError, match="no seed20"):
        lm_pool.subset_pool(pool, ["seed20"])
    with pytest.raises(ValueError):
        lm_pool.subset_pool({"format": "willspeak-lm-pool-v1", "rows": []}, ["seed18"])
    assert pool["rows"][0]["candidates"][1]["views"] == ["seed18"], "the wider pool is not edited"


def test_the_subset_command_needs_no_model_and_refuses_to_overwrite(tmp_path, monkeypatch, capsys):
    pool = tmp_path / "wide.json"
    pool.write_text(json.dumps(wide_pool([{"id": "c", "candidates": [cand("a b", 1.0), cand("c d", 2.0, ("seed18",)), cand("e f", 3.0, ("block12",))]}])), encoding="utf-8")
    out = tmp_path / "narrow.json"
    monkeypatch.setattr(sys, "argv", ["lm_pool.py", "--subset-of", str(pool), "--keep-extras", "seed18", "--out", str(out)])
    assert lm_pool.main() == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert [c["norm"] for c in data["rows"][0]["candidates"]] == ["a b", "c d"] and data["subset_of"]["kept"] == ["seed18"]
    assert "kept extra models seed18" in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        lm_pool.main()
    monkeypatch.setattr(sys, "argv", ["lm_pool.py", "--subset-of", str(pool), "--out", str(tmp_path / "x.json")])
    with pytest.raises(ValueError, match="keep-extras"):
        lm_pool.main()


def test_scripts_print_help_with_the_new_options():
    for script, flags in (("lm_pool.py", ("--extra-model", "--extra-views", "--base-pool", "--subset-of", "--keep-extras")), ("lm_report.py", ("--compare",))):
        done = subprocess.run([sys.executable, str(ROOT / "scripts" / "asr" / script), "--help"], capture_output=True, text=True, check=False)
        assert done.returncode == 0, done.stderr[-400:]
        assert all(flag in done.stdout for flag in flags)


def row(clip_id: str, speaker: str, ref: str, cands: list[dict], kind: str = "sentence") -> dict:
    return {"id": clip_id, "speaker": speaker, "kind": kind, "ref": ref, "duration_s": 3.0, "frames": 40, "nll_empty": 9.0, "candidates": cands}


def write(path: Path, rows: list[dict], **header) -> str:
    path.write_text(json.dumps({"format": "willspeak-lm-pool-v1", "views": ["greedy", "beam4"], "rows": rows, **header}), encoding="utf-8")
    return str(path)


BASE_ROWS = [
    row("a1", "M01", "the boy ran to the store", [cand("the boy ran to the stone sore", 10.0), cand("the boy ran to the store", 11.0, ("beam4",))]),
    row("a2", "M01", "i want a glass of water", [cand("i want a grass of water", 4.0), cand("i want a glass of water", 4.5, ("beam4",))]),
    row("b1", "M02", "we went home early", [cand("we went home curly", 6.0), cand("we went home early", 6.5, ("tempo1.15",))]),
    row("b2", "M02", "she is at school", [cand("sheep is at school", 3.0), cand("she is at school", 3.4)]),
    row("c1", "M04", "open the door", [cand("open a door", 5.0)]),
    row("c2", "M04", "tomato", [cand("potato", 2.0), cand("tomato", 2.2, ("beam4",))], kind="word"),
]
WIDE_ROWS = copy.deepcopy(BASE_ROWS)
WIDE_ROWS[2]["candidates"][1]["views"].append("seed18")
WIDE_ROWS[3]["candidates"].append(cand("she is at the school", 3.6, ("block12",)))
WIDE_ROWS[4]["candidates"].append(cand("open the door", 5.2, ("seed18", "block12/beam4")))
LOGP = {
    "The boy ran to the stone sore.": -45.0, "The boy ran to the store.": -25.0, "I want a grass of water.": -33.0, "I want a glass of water.": -20.0,
    "We went home curly.": -30.0, "We went home early.": -18.0, "Sheep is at school.": -22.0, "She is at school.": -12.0, "She is at the school.": -14.0,
    "Open a door.": -14.0, "Open the door.": -9.0, "Potato.": -9.0, "Tomato.": -9.5,
}


def write_scores(path: Path, logp=LOGP) -> str:
    lm = lm_score.get("qwen25-05b-q4")
    data = {"format": "willspeak-lm-scores-v1", "model": lm.__dict__, "runtime": "test", "file_mb": lm.mb, "peak_rss_mb": 600.0, "threads": 4,
            "tokens_per_second": 100.0, "load_seconds": 0.5, "machine": {"node": "test"},
            "texts": {text: [value, len(text.split()) + 1, 0.01] for text, value in logp.items()}}
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_pools_made_before_lab_39_still_read(tmp_path):
    old = write(tmp_path / "old.json", BASE_ROWS)
    clips, info = lm_report.read_clips([old])
    assert info["decodes_per_clip"] == 2 and info["extra_models"] == [] and "encoder_passes_per_clip" not in info
    assert all(c.extra_only is not None and not c.extra_only.any() for c in clips)
    no_views = copy.deepcopy(BASE_ROWS)
    for r in no_views:
        for c in r["candidates"]:
            del c["views"]
    clips, _info = lm_report.read_clips([write(tmp_path / "bare.json", no_views)])
    assert len(clips) == 6
    assert len(lm_score.read_pools([old])) == 6 and len(lm_score.pool_texts(lm_score.read_pools([old]))) == len(LOGP) - 2


def test_a_wider_pool_keeps_the_format_and_the_report_reads_it(tmp_path):
    wide = write(tmp_path / "wide.json", WIDE_ROWS, decodes_per_clip=4, encoder_passes_per_clip=3, extra_models=[{"name": "seed18"}, {"name": "block12"}])
    clips, info = lm_report.read_clips([wide])
    assert info["decodes_per_clip"] == 4 and info["encoder_passes_per_clip"] == 3 and info["extra_models"] == ["block12", "seed18"]
    by = {c.id: c for c in clips}
    assert by["b1"].extra_only.tolist() == [False, False], "found by a view of the primary model as well"
    assert by["b2"].extra_only.tolist() == [False, False, True] and by["c1"].extra_only.tolist() == [False, True]
    assert len(lm_score.read_pools([wide])) == 6


def test_the_ceiling_is_higher_with_extras_and_the_report_counts_gains_only_extras_made_possible(tmp_path):
    base_clips, _ = lm_report.read_clips([write(tmp_path / "b.json", BASE_ROWS, decodes_per_clip=2)])
    wide_clips, _ = lm_report.read_clips([write(tmp_path / "w.json", WIDE_ROWS, decodes_per_clip=4, extra_models=[{"name": "seed18"}, {"name": "block12"}])])
    scores = {"qwen25-05b-q4": json.loads(Path(write_scores(tmp_path / "s.json")).read_text(encoding="utf-8"))}
    base_picks, wide_picks = {}, {}
    base_rows, _ = lm_report.system_rows(base_clips, scores, lm_report.GRID_WEIGHT, lm_report.GRID_BONUS, lm_report.FIXED, base_picks)
    wide_rows, _ = lm_report.system_rows(wide_clips, scores, lm_report.GRID_WEIGHT, lm_report.GRID_BONUS, lm_report.FIXED, wide_picks)
    ceiling = lambda rows: next(r for r in rows if r["label"] == lm_report.BEST)  # noqa: E731
    assert ceiling(base_rows)["sentence"]["right"] == 4 and ceiling(wide_rows)["sentence"]["right"] == 5
    assert "right_only_from_extra_models" not in ceiling(base_rows) and ceiling(wide_rows)["right_only_from_extra_models"] == 1
    first = next(r for r in wide_rows if r["label"] == lm_report.FIRST)["sentence"]["right"]
    assert first == next(r for r in base_rows if r["label"] == lm_report.FIRST)["sentence"]["right"] == 0, "the first decode is the same in both arms"
    chooser = next(r for r in wide_rows if r["label"] == "qwen25-05b-q4")
    assert chooser["sentence"]["right"] == 5 and chooser["against_first_decode"]["gained"] == 5
    assert chooser["against_first_decode"]["gained_from_extra_models_only"] == 1
    comparison = lm_report.compare_systems(wide_clips, wide_picks["qwen25-05b-q4"], base_clips, base_picks["qwen25-05b-q4"])
    assert (comparison["right_old"], comparison["right_new"], comparison["gained"], comparison["lost"]) == (4, 5, 1, 0)
    assert comparison["gained_from_extra_models_only"] == 1 and comparison["same_first_decode"] == 5 and comparison["sentences"] == 5
    assert comparison["right_gain_ci"][0] <= 0.2 <= comparison["right_gain_ci"][1] and comparison["right_gain_ci_by_prompt"]
    ceilings = lm_report.compare_systems(wide_clips, wide_picks[lm_report.BEST], base_clips, base_picks[lm_report.BEST])
    assert ceilings["right_gain"] == pytest.approx(0.2) and ceilings["lost"] == 0, "the ceiling can only rise or stay"


def test_two_systems_with_different_clips_are_not_compared(tmp_path):
    a, _ = lm_report.read_clips([write(tmp_path / "a.json", BASE_ROWS)])
    b, _ = lm_report.read_clips([write(tmp_path / "b.json", BASE_ROWS[:-1])])
    with pytest.raises(ValueError, match="same clips"):
        lm_report.compare_systems(a, [0] * len(a), b, [0] * len(b))
    with pytest.raises(ValueError):
        lm_report.parse_compare("one=one")
    with pytest.raises(ValueError):
        lm_report.parse_compare("one")


def test_the_report_command_prints_a_table_per_arm_and_the_paired_change(tmp_path, monkeypatch, capsys):
    base = write(tmp_path / "b.json", BASE_ROWS, decodes_per_clip=9)
    wide = write(tmp_path / "w.json", WIDE_ROWS, decodes_per_clip=11, encoder_passes_per_clip=9,
                 extra_models=[{"name": "seed18"}, {"name": "block12"}])
    scores = write_scores(tmp_path / "qwen25-05b-q4.json")
    out = tmp_path / "table.json"
    monkeypatch.setattr(sys, "argv", ["lm_report.py", "--system", "baseline: adapter=" + base, "--system", "both: adapter=" + wide,
                                      "--compare", "both: adapter=baseline: adapter", "--scores", scores, "--out", str(out)])
    assert lm_report.main() == 0
    table = json.loads(out.read_text(encoding="utf-8"))
    assert set(table["systems"]) == {"baseline: adapter", "both: adapter"}
    assert table["systems"]["both: adapter"]["decodes_per_clip"] == 11 and table["systems"]["baseline: adapter"]["extra_models"] == []
    rows = table["comparisons"]["both: adapter vs baseline: adapter"]
    assert set(rows) == {lm_report.FIRST, lm_report.AUDIO, "qwen25-05b-q4", lm_report.BEST}
    assert rows[lm_report.BEST]["gained"] == 1 and rows["qwen25-05b-q4"]["gained_from_extra_models_only"] == 1
    shown = capsys.readouterr().out
    assert "11 decodes per clip" in shown and "paired change in sentences right" in shown and "extra-only" in shown
    assert "in the pool only because an extra model proposed it: 1" in shown
    monkeypatch.setattr(sys, "argv", ["lm_report.py", "--system", "a=" + base, "--compare", "b=a", "--scores", scores, "--out", str(tmp_path / "t2.json")])
    with pytest.raises(ValueError, match="both must be"):
        lm_report.main()


def load_lab():
    spec = importlib.util.spec_from_file_location("lab_cli_wide", ROOT / "lab.py")
    lab = importlib.util.module_from_spec(spec)
    sys.modules["lab_cli_wide"] = lab
    spec.loader.exec_module(lab)
    return lab


def make_app(tmp_path) -> Path:
    app = tmp_path / "app"
    app.mkdir(parents=True)
    for name in ("scripts", "src"):
        (app / name).symlink_to(ROOT / name)
    return app


def plan(tmp_path, **overrides):
    lab = load_lab()
    app = make_app(tmp_path)
    commands, notes = lab.plan(lambda: lab.Lab(app, lab.Settings(app, overrides, env={}), dry=True, stamp="TEST", quiet=True), ["39"])
    assert notes == {"39": []}
    return lab, app, commands


def option(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def options(cmd: list[str], flag: str) -> list[str]:
    return [cmd[i + 1] for i, word in enumerate(cmd) if word == flag]


def test_stage_39_pools_every_arm_from_one_decode_of_the_extra_models_and_reports_once(tmp_path):
    lab, app, commands = plan(tmp_path, trim_models="off")
    pools = [c for c in commands if "scripts/asr/lm_pool.py" in c]
    bases = [c for c in pools if "--base-pool" not in c and "--subset-of" not in c]
    wide = [c for c in pools if "--base-pool" in c]
    cuts = [c for c in pools if "--subset-of" in c]
    assert len(bases) == 15 and len(wide) == 15 and len(cuts) == 30
    assert all("--views" in c for c in bases) and all(len(options(c, "--extra-model")) == 4 for c in wide)
    assert all(option(c, "--base-pool").endswith("/" + option(c, "--out").rsplit("/", 1)[-1].replace("wide-", "base-")) for c in wide)
    assert {option(c, "--keep-extras") for c in cuts} == {"seed18,seed19", "block12,block16"}
    outs = [option(c, "--out") for c in pools]
    assert len(set(outs)) == len(outs) and all(o.startswith("reports/lab-39-wider-pools/pools/") for o in outs), "nothing is written into lab 36's folder"
    trained = [option(c, "--out") for c in commands if "scripts/asr/train.py" in c]
    assert sum("big-c-pre-" in t for t in trained) == 3
    assert sum("lab-39-pre-ad" in t for t in trained) == 3 * 5 * 4 and sum("lab-36-pre-ad8-" in t for t in trained) == 12 and sum("lab-pre-ad8-" in t for t in trained) == 3
    assert [t for t in trained if "lab-39-pre-ad12-M01-f2" in t] and [t for t in trained if t.endswith("lab-39-pre-ad8-M04-s19")]
    lm = [c for c in commands if "scripts/asr/lm_score.py" in c]
    assert [c[2:5] for c in lm] == [["fetch", "--models", "qwen25-05b-q4"], ["score", "--model", "qwen25-05b-q4"]]
    score = lm[1]
    assert option(score, "--out") == "reports/lab-39-wider-pools/scores/qwen25-05b-q4.json" and "--extend" in score and "release" not in score
    scored_pools = score[score.index("--pool") + 1: score.index("--threads")]
    assert len(scored_pools) == 15 + 15 + 30 and set(scored_pools) == set(outs)
    assert commands.index(lm[0]) > max(commands.index(c) for c in pools)
    reports = [c for c in commands if "scripts/asr/lm_report.py" in c]
    assert len(reports) == 1 and reports[0].count("--system") == 8 and reports[0].count("--compare") == 6
    assert option(reports[0], "--out") == "reports/lab-39-wider-pools/table.json"
    systems = options(reports[0], "--system")
    assert [s.split("=")[0] for s in systems][:2] == ["baseline: stage P's adapter, the test clips", "baseline: stage P's recipe, every labelled clip held out"]
    folded = next(s for s in systems if s.startswith("both: stage P's recipe"))
    assert len(folded.split("=", 1)[1].split(",")) == 3 * 5
    problems, checked = lab.flag_problems(app, commands)
    assert not problems, problems
    assert checked >= 5


def test_the_baseline_pool_is_lab_36s_only_when_the_same_primary_model_is_still_on_disk(tmp_path):
    lab = load_lab()
    app = tmp_path / "app"
    app.mkdir()
    session = lab.Lab(app, lab.Settings(app, {"trim_models": "off"}, env={}), dry=False, stamp="TEST", quiet=True)
    base = "models/parakeet-ft-big-c-pre-M01/parakeet-ft.nemo"
    (app / base).parent.mkdir(parents=True)
    (app / base).write_bytes(b"12345")
    adapter = app / "models/parakeet-ft-lab-pre-ad8-M01/adapter.pt"
    adapter.parent.mkdir(parents=True)
    adapter.write_bytes(b"adapter-one")
    pool = app / "reports/lab-36-lm-rescoring/pools/prototype-M01-test.json"
    pool.parent.mkdir(parents=True)
    flags = session.adapter_flags("lab-pre-ad8-M01", base)
    recorded = {"nemo": {"sha256": "x", "size": 5}, "adapter": {"sha256": hashlib.sha256(b"adapter-one").hexdigest(), "size": 11}}
    pool.write_text(json.dumps({"model_files": recorded}), encoding="utf-8")
    pool_path = "reports/lab-36-lm-rescoring/pools/prototype-M01-test.json"
    assert lab.wide_pool_fits(session, pool_path, flags)
    adapter.write_bytes(b"adapter-two")
    assert not lab.wide_pool_fits(session, pool_path, flags)
    adapter.write_bytes(b"adapter-one")
    (app / base).write_bytes(b"123456")
    assert not lab.wide_pool_fits(session, pool_path, flags)
    (app / base).write_bytes(b"12345")
    adapter.unlink()
    assert not lab.wide_pool_fits(session, pool_path, flags)
    pool.unlink()
    assert not lab.wide_pool_fits(session, pool_path, flags)


def run_unit(tmp_path, fits: bool):
    lab = load_lab()
    app = tmp_path / "app"
    app.mkdir()
    session = lab.Lab(app, lab.Settings(app, {"trim_models": "off"}, env={}), dry=False, stamp="TEST", quiet=True)
    export = app / lab.EX / "M01"
    export.mkdir(parents=True)
    for split, rows in (("train", [("t1", "the cat sat")]), ("dev", [("d1", "a red door")]), ("test", [("x1", "open the window")])):
        (export / f"{split}.jsonl").write_text("".join(json.dumps({"id": i, "text": t, "audio": f"{split}/{i}.wav"}) + "\n" for i, t in rows), encoding="utf-8")
    base = app / "models/parakeet-ft-big-c-pre-M01/parakeet-ft.nemo"
    base.parent.mkdir(parents=True)
    base.write_bytes(b"12345")
    made: list[str] = []

    def adapter_for(name):
        path = app / f"models/parakeet-ft-{name}/adapter.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        (path.parent / "ckpt").mkdir(exist_ok=True)
        (path.parent / "ckpt" / "epoch=14.ckpt").write_bytes(b"x" * 1000)
        made.append(name)
        return True

    lab.fresh_pretrained_adapter = lambda lab_, s: (adapter_for(f"lab-pre-ad8-{s}") and (f"lab-pre-ad8-{s}", "hyps"))
    lab.train_adapter_arm = lambda lab_, exp, name, blocks, epochs, base_model, seed=17, extra=None: adapter_for(name)
    if fits:
        adapter_for("lab-pre-ad8-M01")
        made.clear()
        pool = app / "reports/lab-36-lm-rescoring/pools/prototype-M01-test.json"
        pool.parent.mkdir(parents=True)
        pool.write_text(json.dumps({"model_files": {"nemo": {"sha256": "x", "size": 5}, "adapter": {"sha256": hashlib.sha256(b"lab-pre-ad8-M01").hexdigest()}}}),
                        encoding="utf-8")
    commands: list[list[str]] = []

    def run(cmd, env=None):
        commands.append([str(c) for c in cmd])
        for out in (option(commands[-1], "--out"),) if "--out" in commands[-1] else ():
            (app / out).parent.mkdir(parents=True, exist_ok=True)
            (app / out).write_text("{}", encoding="utf-8")
        return True

    session.run = run
    failed: list[str] = []
    arms = ["seeds", "positions", "both"]
    got = lab.wide_pools(session, lab.wide_unit("M01", None), arms, ["seeds", "positions"], "greedy", failed)
    return lab, app, commands, got, failed, made


@pytest.mark.parametrize("fits", [True, False])
def test_one_unit_builds_the_baseline_only_when_lab_36s_pool_does_not_fit_and_cuts_the_narrow_arms(tmp_path, fits):
    lab, app, commands, got, failed, made = run_unit(tmp_path, fits)
    assert not failed and set(got) == {"baseline", "seeds", "positions", "both"}
    pools = [c for c in commands if "scripts/asr/lm_pool.py" in c]
    builds = [c for c in pools if "--views" in c]
    wide = next(c for c in pools if "--base-pool" in c)
    if fits:
        assert not builds and option(wide, "--base-pool") == "reports/lab-36-lm-rescoring/pools/prototype-M01-test.json"
        assert got["baseline"] == "reports/lab-36-lm-rescoring/pools/prototype-M01-test.json"
    else:
        assert len(builds) == 1 and option(wide, "--base-pool") == option(builds[0], "--out") == "reports/lab-39-wider-pools/pools/base-M01-test.json"
        assert got["baseline"] == option(builds[0], "--out")
    assert got["both"] == option(wide, "--out") == "reports/lab-39-wider-pools/pools/wide-M01-test.json"
    assert got["seeds"].endswith("seeds-M01-test.json") and got["positions"].endswith("positions-M01-test.json")
    assert sorted(made) == sorted(["lab-pre-ad8-M01", "lab-39-pre-ad8-M01-s18", "lab-39-pre-ad8-M01-s19", "lab-39-pre-ad12-M01", "lab-39-pre-ad16-M01"])
    assert len(options(wide, "--extra-model")) == 4 and option(wide, "--nemo") == "models/parakeet-ft-big-c-pre-M01/parakeet-ft.nemo"
    folders = sorted((app / "models").glob("parakeet-ft-lab-*"))
    assert len(folders) == 5 and all((f / "adapter.pt").is_file() and not (f / "ckpt").exists() for f in folders), "checkpoints are freed, adapters kept"
    assert options(wide, "--extra-model")[0] == ("seed18=models/parakeet-ft-big-c-pre-M01/parakeet-ft.nemo,"
                                                 "models/parakeet-ft-lab-39-pre-ad8-M01-s18/adapter.pt")


def test_a_unit_whose_pools_are_all_made_is_not_trained_again(tmp_path):
    lab, app, commands, got, failed, made = run_unit(tmp_path, False)
    again = lab.Lab(app, lab.Settings(app, {"trim_models": "off"}, env={}), dry=False, stamp="TEST2", quiet=True)
    again.run = lambda cmd, env=None: pytest.fail(f"should not run {cmd}")
    lab.fresh_pretrained_adapter = lambda *a, **k: pytest.fail("trained again")
    lab.train_adapter_arm = lambda *a, **k: pytest.fail("trained again")
    assert lab.wide_pools(again, lab.wide_unit("M01", None), ["seeds", "positions", "both"], ["seeds", "positions"], "greedy", []) == got


def test_a_failed_extra_adapter_leaves_the_arms_that_do_not_need_it(tmp_path):
    lab = load_lab()
    app = tmp_path / "app"
    app.mkdir()
    session = lab.Lab(app, lab.Settings(app, {"trim_models": "off"}, env={}), dry=True, stamp="TEST", quiet=True)
    lab.fresh_pretrained_adapter = lambda lab_, s: (f"lab-pre-ad8-{s}", "hyps")
    lab.train_adapter_arm = lambda lab_, exp, name, blocks, epochs, base_model, seed=17, extra=None: "-s19" not in name
    failed: list[str] = []
    got = lab.wide_pools(session, lab.wide_unit("M01", None), ["seeds", "positions", "both"], ["seeds", "positions"], "greedy", failed)
    assert set(got) == {"baseline", "positions"} and failed == ["adapter:lab-39-pre-ad8-M01-s19"]
    cmds = [c for c in session.commands if "--base-pool" in c]
    assert len(cmds) == 1 and [x.split("=")[0] for x in options(cmds[0], "--extra-model")] == ["block12", "block16"]
    assert got["positions"] == option(cmds[0], "--out"), "the one group that trained is the whole wide pool; no cut is needed"


def test_stage_39_arms_follow_the_setting(tmp_path):
    lab, app, commands = plan(tmp_path, trim_models="off", wide_extras="seeds", wide_systems="prototype")
    pools = [c for c in commands if "scripts/asr/lm_pool.py" in c]
    assert [bool("--base-pool" in c) for c in pools] == [False, True] * 3 and not [c for c in pools if "--subset-of" in c]
    assert all(len(options(c, "--extra-model")) == 2 for c in pools if "--base-pool" in c)
    trained = [option(c, "--out") for c in commands if "scripts/asr/train.py" in c]
    assert sum("lab-39-pre-ad" in t for t in trained) == 3 * 2 and not [t for t in trained if "lab-36-pre-ad8" in t or "ad12" in t]
    report = next(c for c in commands if "scripts/asr/lm_report.py" in c)
    assert report.count("--system") == 2 and report.count("--compare") == 1
    assert [s.split("=")[0] for s in options(report, "--system")] == ["baseline: stage P's adapter, the test clips", "seeds: stage P's adapter, the test clips"]
    _lab, _app, commands = plan(tmp_path / "second", trim_models="off", wide_extras="both", wide_extra_views="greedy beam4", wide_lm_models="qwen25-05b-q4 qwen25-05b-q8")
    wide = [c for c in commands if "--base-pool" in c]
    assert len(wide) == 15 and not [c for c in commands if "--subset-of" in c] and all(option(c, "--extra-views") == "greedy beam4" for c in wide)
    assert [option(c, "--model") for c in commands if "scripts/asr/lm_score.py" in c and "score" in c[:4]] == ["qwen25-05b-q4", "qwen25-05b-q8"]
    report = next(c for c in commands if "scripts/asr/lm_report.py" in c)
    assert report.count("--system") == 4 and report.count("--compare") == 2


def test_stage_39_frees_a_language_model_it_downloaded_when_disk_is_short(tmp_path):
    lab, app, commands = plan(tmp_path, trim_models="on", wide_systems="prototype", wide_extras="seeds")
    lm = [c[2:] for c in commands if "scripts/asr/lm_score.py" in c]
    assert [c[0] for c in lm] == ["fetch", "score", "release"]


def test_stage_39_is_queued_under_widepool_and_reuses_the_catalog(tmp_path):
    lab = load_lab()
    app = tmp_path / "app"
    app.mkdir()
    session = lab.Lab(app, lab.Settings(app, {}, env={}), dry=True, stamp="TEST", quiet=True)
    assert lab.resolve(session, ["widepool"]) == ["39"] and lab.WIDE_QUEUE == ["39"]
    assert "39" in session.catalog and "39" in lab.AUTO_REBUILD and "39" in lab.Lab.BASELINE_NEEDS and "39" in lab.LAB_TESTS
    assert session.catalog["39"].markers == ["reports/lab-39-wider-pools/table.json"]
    assert lab.LAB_TESTS["39"][3] == "lab-39-wider-pools"
    for name in ("wide_extras", "wide_systems", "wide_extra_views", "wide_lm_models"):
        assert name in lab.VARIABLES and "lab test 39" in lab.VARIABLES[name][2]


def test_stage_39_refuses_bad_settings(tmp_path):
    lab = load_lab()
    app = make_app(tmp_path)
    bad_settings = ({"wide_extras": "everything"}, {"wide_extras": ""}, {"wide_extras": "seeds seeds"}, {"wide_extras": "seeds baseline"},
                    {"wide_systems": "stock"}, {"wide_systems": ""}, {"wide_extra_views": "tempo1.1"}, {"wide_extra_views": "beam1"},
                    {"wide_lm_models": ""}, {"wide_lm_models": "gpt-9"}, {"wide_lm_models": "pi huge"}, {"xflags": "--select dev"},
                    {"xflags": "--keep-stock"})
    for bad in bad_settings:
        session = lab.Lab(app, lab.Settings(app, bad, env={}), dry=True, stamp="TEST", quiet=True)
        with pytest.raises((ValueError, KeyError)):
            lab.stage_39(session)
        assert not [c for c in session.commands if "train.py" in " ".join(c) or "lm_pool.py" in " ".join(c)], f"{bad} started work before refusing"


def test_no_extra_model_of_a_fold_is_trained_on_that_fold_or_on_anything_that_includes_it(tmp_path):
    lab = load_lab()
    for s in ("M01", "M02", "M04"):
        test = lab.wide_unit(s, None)
        assert lab.wide_leaks(test) == [] and test["export"] == f"{lab.EX}/{s}"
        for k in range(4):
            unit = lab.wide_unit(s, k)
            assert lab.wide_leaks(unit) == []
            assert unit["export"] == f"{lab.EX}/cv-{s}-f{k}" and unit["held_out"] == f"{lab.EX}/cv-{s}-f{k}/test.jsonl"
            assert all(e["adapter"].endswith(f"-f{k}") or e["adapter"].endswith(f"-f{k}-s{e['seed']}") for e in unit["extras"])
            assert unit["primary"] == f"lab-36-pre-ad8-{s}-f{k}", "the primary is lab 36's fold adapter"
            assert not {e["adapter"] for e in unit["extras"]} & {e["adapter"] for e in test["extras"]}
    unit = lab.wide_unit("M01", 2)
    unit["extras"][0] = lab.wide_unit("M01", None)["extras"][0]
    assert any("lab-39-pre-ad8-M01-s18" in p for p in lab.wide_leaks(unit))
    other_fold = lab.wide_unit("M01", 2)
    other_fold["extras"][1] = lab.wide_unit("M01", 1)["extras"][1]
    assert any("fold 1" in p for p in lab.wide_leaks(other_fold))
    wrong_export = lab.wide_unit("M01", 2)
    wrong_export["export"] = f"{lab.EX}/M01"
    assert any("held out" in p for p in lab.wide_leaks(wrong_export))
    primary_of_test = lab.wide_unit("M01", 3)
    primary_of_test["primary"] = "lab-pre-ad8-M01"
    assert lab.wide_leaks(primary_of_test)
    test_unit_with_fold_extra = lab.wide_unit("M01", None)
    test_unit_with_fold_extra["extras"][2]["adapter"] = "lab-39-pre-ad12-M01-f0"
    assert lab.wide_leaks(test_unit_with_fold_extra)


def test_the_planned_commands_train_each_folds_extras_on_that_fold_and_pool_them_with_that_folds_primary(tmp_path):
    lab, app, commands = plan(tmp_path, trim_models="off")
    import re

    def fold_of(text: str):
        m = re.search(r"-f(\d)(?:-s\d+)?(?:/|$|\.json)", text)
        return m.group(1) if m else None

    trains = [c for c in commands if "scripts/asr/train.py" in c and "lab-39-pre-ad" in option(c, "--out")]
    assert len(trains) == 60
    for c in trains:
        name, export = option(c, "--out"), option(c, "--export")
        speaker = re.search(r"-(M0\d)", name).group(1)
        k = fold_of(name)
        assert export == (f"data/public/torgo/export/{speaker}" if k is None else f"data/public/torgo/export/cv-{speaker}-f{k}"), name
        assert option(c, "--select") == "last" and "--no-keep-stock" in c, "the dev clips (a fold's held-out clips) choose nothing"
        assert option(c, "--base") == f"models/parakeet-ft-big-c-pre-{speaker}/parakeet-ft.nemo"
    pools = [c for c in commands if "scripts/asr/lm_pool.py" in c and "--nemo" in c]
    assert len(pools) == 30
    for c in pools:
        k = fold_of(option(c, "--adapter"))
        assert k == fold_of(option(c, "--manifest")) == fold_of(option(c, "--audio-root")), "the primary model, its clips and"
        for spec in options(c, "--extra-model"):
            assert fold_of(spec.split(",")[1]) == k, f"{spec} was not trained for the clips of {option(c, '--manifest')}"
            assert spec.split("=")[1].split(",")[0] == option(c, "--nemo"), "every model starts from the same pretrained base, never one that saw the speaker"


def test_the_manifests_are_checked_too_when_the_stage_runs(tmp_path):
    lab = load_lab()
    root = tmp_path / "app"

    def export(folder: str, split_rows: dict[str, list[tuple[str, str]]]):
        path = root / lab.EX / folder
        path.mkdir(parents=True)
        for split, rows in split_rows.items():
            (path / f"{split}.jsonl").write_text("".join(json.dumps({"id": i, "text": t, "audio": f"{split}/{i}.wav"}) + "\n" for i, t in rows), encoding="utf-8")

    export("cv-M01-f1", {"train": [("a", "the cat sat"), ("b", "open the door")], "dev": [("c", "red car")], "test": [("c", "red car")]})
    problems, notes = lab.wide_unseen(root, lab.wide_unit("M01", 1))
    assert problems == [] and notes == [], "dev is the held-out fold in a cv export; it is not training data, and the recipe never selects on it"
    export("cv-M01-f2", {"train": [("a", "the cat sat"), ("c", "red car")], "dev": [("c", "red car")], "test": [("c", "red car")]})
    problems, _ = lab.wide_unseen(root, lab.wide_unit("M01", 2))
    assert len(problems) == 2 and "clips are in the training clips" in problems[0] and "prompts" in problems[1]
    export("cv-M01-f3", {"train": [("a", "the cat sat"), ("z", "Red car.")], "dev": [("d", "x y")], "test": [("d", "red car")]})
    problems, _ = lab.wide_unseen(root, lab.wide_unit("M01", 3))
    assert len(problems) == 1 and "prompts" in problems[0], "the same prompt on another clip id is still a leak for a fold"
    export("M01", {"train": [("a", "the cat sat")], "dev": [("b", "a red door")], "test": [("b", "open it")]})
    problems, notes = lab.wide_unseen(root, lab.wide_unit("M01", None))
    assert len(problems) == 1 and "clips are in the training clips" in problems[0], "a test clip that is in dev counts: the test-clip extras train on train and dev"
    export("M02", {"train": [("a", "the cat sat")], "dev": [("b", "a red door")], "test": [("t", "the cat sat")]})
    problems, notes = lab.wide_unseen(root, lab.wide_unit("M02", None))
    assert problems == [] and len(notes) == 1, "a shared test prompt is reported, as it applies to lab 36's adapter too"


def test_a_unit_that_would_leak_stops_before_anything_trains(tmp_path):
    lab = load_lab()
    app = tmp_path / "app"
    app.mkdir()
    session = lab.Lab(app, lab.Settings(app, {"trim_models": "off"}, env={}), dry=True, stamp="TEST", quiet=True)
    unit = lab.wide_unit("M01", 1)
    unit["extras"][0]["adapter"] = "lab-39-pre-ad8-M01-s18"
    failed: list[str] = []
    assert lab.wide_pools(session, unit, ["seeds"], ["seeds"], "greedy", failed) == {}
    assert failed == ["leak:M01:f1"] and not session.commands
    assert any("LEAK" in line for line in session.lines)


def test_the_adapter_recipe_never_lets_dev_clips_choose():
    lab = load_lab()
    flags = lab.split_flags(lab.Lab.RECIPES["ad8"])
    assert flags["--select"] == ["last"] and "--no-keep-stock" in flags
    session = lab.Lab(ROOT, lab.Settings(ROOT, {}, env={}), dry=True, stamp="T", quiet=True)
    assert lab.wide_recipe_ok(session)
    for xflags in ("--select dev", "--keep-stock"):
        assert not lab.wide_recipe_ok(lab.Lab(ROOT, lab.Settings(ROOT, {"xflags": xflags}, env={}), dry=True, stamp="T", quiet=True))


def test_extra_model_arguments_follow_the_flags_the_lab_already_uses():
    lab = load_lab()
    session = lab.Lab(ROOT, lab.Settings(ROOT, {}, env={}), dry=True, stamp="T", quiet=True)
    assert lab.wide_extra_arg("seed18", session.adapter_flags("x", "models/p/parakeet-ft.nemo")) == "seed18=models/p/parakeet-ft.nemo,models/parakeet-ft-x/adapter.pt"
    assert lab.wide_extra_arg("a", session.adapter_flags("x", None)) == "a=stock,models/parakeet-ft-x/adapter.pt"
    assert lab.wide_extra_arg("b", session.adapter_flags("x", "nvidia/model")) == "b=pretrained:nvidia/model,models/parakeet-ft-x/adapter.pt"
