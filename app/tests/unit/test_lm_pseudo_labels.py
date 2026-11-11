import ast
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "asr"))
sys.path.insert(0, str(ROOT / "src"))
import lm_pick  # noqa: E402
import lm_pool  # noqa: E402
import lm_report  # noqa: E402
import lm_score  # noqa: E402
import tta  # noqa: E402
from tests.unit.test_lm_rescoring import LOGP, ROWS, cand, write_scores  # noqa: E402

SECRET = "zebra quantum"


def unlabelled(clip_id: str, cands: list[dict], speaker: str = "M01") -> dict:
    return {"id": clip_id, "speaker": speaker, "kind": "unknown", "ref": "", "duration_s": 3.0, "frames": 40, "nll_empty": 9.0, "candidates": cands}


def blind(rows: list[dict]) -> list[dict]:
    return [unlabelled(r["id"], r["candidates"], r["speaker"]) for r in rows]


def write_pool(path: Path, rows: list[dict]) -> str:
    path.write_text(json.dumps({"format": "willspeak-lm-pool-v1", "views": ["greedy", "beam4"], "labelled": False, "rows": rows}), encoding="utf-8")
    return str(path)


def texts_of(tmp_path: Path, logp: dict) -> dict:
    return lm_pick.read_texts(write_scores(tmp_path / "scores.json", logp=logp))[1]


def test_the_default_weights_are_the_ones_lab_36_reports_as_its_preset():
    assert (lm_pick.WEIGHT, lm_pick.BONUS) == lm_report.FIXED == (0.7, 3.0)


def test_a_pick_is_exactly_what_lab_36s_chooser_picks_on_the_same_pools(tmp_path):
    scores = write_scores(tmp_path / "s.json")
    clips, _ = lm_report.read_clips([_labelled_pool(tmp_path)])
    lms = lm_report.lm_values(clips, lm_report.read_scores([scores])["qwen25-05b-q8"]["texts"])
    texts = lm_pick.read_texts(scores)[1]
    records = {r["id"]: r for r in lm_pick.pick(blind(ROWS), texts, None)}
    for clip, lm in zip(clips, lms):
        mine = records[clip.id]
        theirs = clip.texts[lm_report.choose(clip, lm, 0.7, 3.0)]
        if mine["single"]:
            assert mine["hyp"] == ROW_BY_ID[clip.id]["candidates"][0]["text"]
        else:
            assert lm_pool.lm_text(lm_pick.normalize(mine["hyp"])) == theirs


ROW_BY_ID = {r["id"]: r for r in ROWS}


def _labelled_pool(tmp_path: Path) -> str:
    from tests.unit.test_lm_rescoring import write_pool as write_labelled

    return write_labelled(tmp_path / "labelled.json")


def test_margin_is_the_lead_of_the_best_total_over_the_runner_up(tmp_path):
    texts = texts_of(tmp_path, LOGP)
    a1 = lm_pick.pick([unlabelled("a1", ROWS[0]["candidates"])], texts, None)[0]
    assert a1["hyp"] == "the boy ran to the store" and a1["changed"] and a1["margin"] == pytest.approx(10.0)
    assert lm_pick.totals(unlabelled("a1", ROWS[0]["candidates"]), texts, 0.7, 3.0) == pytest.approx([-20.5, -10.5])


def test_weight_zero_and_bonus_zero_give_the_audio_choice(tmp_path):
    texts = texts_of(tmp_path, LOGP)
    row = unlabelled("a1", ROWS[0]["candidates"])
    assert lm_pick.pick([row], texts, None, weight=0.0, bonus=0.0)[0]["hyp"] == "the boy ran to the stone sore"


def test_a_clip_whose_plain_answer_is_one_word_keeps_it_even_when_the_language_model_would_change_it(tmp_path):
    texts = texts_of(tmp_path, {**LOGP, "Tomato.": -1.0, "Potato.": -20.0})
    row = unlabelled("c2", ROWS[5]["candidates"], "M04")
    got = lm_pick.pick([row], texts, None)[0]
    assert got["single"] and got["hyp"] == "potato" and not got["changed"] and got["margin"] is None
    chooser = lm_pick.totals(row, texts, 0.7, 3.0)
    assert lm_pick.best_and_margin(chooser)[0] == 1


def test_the_single_word_rule_follows_the_plain_answer_not_the_pool(tmp_path):
    texts = texts_of(tmp_path, LOGP)
    row = unlabelled("a2", ROWS[1]["candidates"])
    got = lm_pick.pick([row], texts, {"a2": "Water"})[0]
    assert got["single"] and got["hyp"] == "Water" and got["pool_first_differs"]
    sentence = lm_pick.pick([row], texts, {"a2": "I want a glass of water."})[0]
    assert not sentence["single"] and not sentence["pool_first_differs"] and sentence["hyp"] == "I want a glass of water."


def test_a_pick_that_only_differs_in_form_keeps_the_plain_text_and_counts_as_unchanged(tmp_path):
    texts = texts_of(tmp_path, LOGP)
    row = unlabelled("a2", ROWS[1]["candidates"])
    got = lm_pick.pick([row], texts, {"a2": "I want a glass of water."})[0]
    assert not got["changed"] and got["hyp"] == "I want a glass of water."


def test_a_pool_of_one_has_no_margin_and_counts_as_uncontested(tmp_path):
    texts = texts_of(tmp_path, LOGP)
    only = unlabelled("x", [cand("open the door", 5.0)])
    got = lm_pick.pick([only], texts, None)[0]
    assert got["margin"] is None and not got["single"] and got["hyp"] == "open the door"
    assert lm_pick.best_and_margin([-3.0]) == (0, None)
    assert lm_pick.best_and_margin([-3.0, -lm_pick.INF]) == (0, None)
    assert lm_pick.best_and_margin([-lm_pick.INF, -lm_pick.INF]) == (0, 0.0)
    assert lm_pick.best_and_margin([-5.0, -2.0, -2.0]) == (1, 0.0)


def test_a_candidate_without_a_language_model_score_is_named(tmp_path):
    texts = texts_of(tmp_path, {"The boy ran to the store.": -25.0})
    with pytest.raises(ValueError, match="no language-model score"):
        lm_pick.pick([unlabelled("a1", ROWS[0]["candidates"])], texts, None)


def test_plain_answers_must_cover_every_clip(tmp_path):
    texts = texts_of(tmp_path, LOGP)
    with pytest.raises(ValueError, match="miss 1 clips"):
        lm_pick.pick(blind(ROWS[:2]), texts, {"a1": "the boy ran"})


def record(clip_id: str, margin, single: bool = False) -> dict:
    return {"id": clip_id, "hyp": "a b c", "plain": "a b c", "single": single, "changed": False, "margin": None if single else margin,
            "candidates": 3, "pool_first_differs": False, "speaker": "M01"}


RECORDS = [record("s1", 0.5), record("s2", 9.0), record("s3", 3.0), record("s4", 3.0), record("s5", None), record("w1", None, single=True)]


def dropped(rows: list[dict]) -> set[str]:
    return {r["id"] for r in rows if "skip" in r}


def test_the_all_rule_drops_nothing_and_never_marks_single_words():
    assert dropped(lm_pick.apply_rule(RECORDS, "all")) == set()


def test_top_share_keeps_the_widest_margins_of_the_sentences_and_ignores_single_words():
    rows = lm_pick.apply_rule(RECORDS, "top", share=0.5)
    assert dropped(rows) == {"s1", "s4"}
    assert all(r["skip"] == lm_pick.SKIP_REASON for r in rows if "skip" in r)
    assert dropped(lm_pick.apply_rule(RECORDS, "top", share=1.0)) == set()
    assert len(dropped(lm_pick.apply_rule(RECORDS, "top", share=0.01))) == 4


def test_the_fixed_margin_is_inclusive_and_an_uncontested_answer_always_passes():
    assert dropped(lm_pick.apply_rule(RECORDS, "margin", min_margin=3.0)) == {"s1"}
    assert dropped(lm_pick.apply_rule(RECORDS, "margin", min_margin=9.5)) == {"s1", "s2", "s3", "s4"}
    assert dropped(lm_pick.apply_rule(RECORDS, "margin", min_margin=0.0)) == set()


def test_applying_a_rule_again_starts_from_the_picks_not_from_an_earlier_filter():
    first = lm_pick.apply_rule(RECORDS, "margin", min_margin=9.5)
    assert dropped(lm_pick.apply_rule(first, "all")) == set()


def test_bad_rules_and_shares_are_refused():
    for call in (lambda: lm_pick.apply_rule(RECORDS, "best"), lambda: lm_pick.apply_rule(RECORDS, "top", share=0.0),
                 lambda: lm_pick.apply_rule(RECORDS, "top", share=1.5), lambda: lm_pick.apply_rule(RECORDS, "margin", min_margin=-1.0)):
        with pytest.raises(ValueError):
            call()


def test_deciles_and_counts_describe_the_picks():
    info = lm_pick.describe(lm_pick.apply_rule(RECORDS, "top", share=0.5))
    assert info["clips"] == 6 and info["single_words"] == 1 and info["sentence_like"] == 5 and info["uncontested"] == 1 and info["kept"] == 4
    assert lm_pick.deciles([1.0, 2.0, 3.0]) == [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0] and lm_pick.deciles([]) == []


def test_command_line_pick_filter_and_audit_round_trip(tmp_path, capsys):
    pool = write_pool(tmp_path / "pool.json", blind(ROWS))
    scores = write_scores(tmp_path / "scores.json")
    plain = tmp_path / "plain.jsonl"
    plain.write_text("".join(json.dumps({"id": r["id"], "hyp": r["candidates"][0]["text"]}) + "\n" for r in ROWS), encoding="utf-8")
    picks, labels = tmp_path / "picks.jsonl", tmp_path / "labels.jsonl"
    assert lm_pick.main(["pick", "--pool", pool, "--scores", scores, "--plain", str(plain), "--out", str(picks)]) == 0
    assert lm_pick.main(["filter", "--picks", str(picks), "--rule", "top", "--share", "0.5", "--out", str(labels)]) == 0
    rows = [json.loads(line) for line in labels.read_text().splitlines()]
    assert len(rows) == len(ROWS) and {r["id"] for r in rows if "skip" in r}
    truth = tmp_path / "truth.jsonl"
    truth.write_text("".join(json.dumps({"id": r["id"], "text": r["ref"], "audio": "x.wav", "speaker": r["speaker"], "kind": r["kind"]}) + "\n"
                             for r in ROWS), encoding="utf-8")
    out = tmp_path / "audit.json"
    assert lm_pick.main(["audit", "--truth", str(truth), "--system", f"top={labels}", "--system", f"gone={tmp_path / 'missing'}", "--out", str(out)]) == 0
    shown = capsys.readouterr().out
    assert "not available" in shown and "plain, same clips" in shown and "dropped by the filter" in shown
    data = json.loads(out.read_text())["systems"]["top"]
    assert data["labels_used"]["clips"] + data["dropped_by_the_filter"]["clips"] == len(ROWS)
    with pytest.raises(FileExistsError):
        lm_pick.main(["pick", "--pool", pool, "--scores", scores, "--out", str(picks)])


def test_label_quality_counts_wrong_words_exact_clips_and_the_plain_answers_on_the_same_clips():
    truth = {"a": "open the door".split(), "b": "water".split(), "c": "i am cold".split()}
    held = {"labels": {"a": "Open the door.", "b": "water", "c": "i am old"}, "single": {"a": False, "b": True, "c": False},
            "skipped": {"c"}, "plain": {"a": "open a door", "b": "water", "c": "i am cold"}}
    got = lm_pick.audit_one(truth, held)
    assert got["labels_used"] == {"clips": 2, "words": 4, "wrong": 0, "wer": 0.0, "exactly_right": 2}
    assert got["sentence_like"]["clips"] == 1 and got["single_word"]["clips"] == 1
    assert got["dropped_by_the_filter"]["wrong"] == 1 and got["plain_answers_on_the_same_clips"]["wrong"] == 1
    assert got["plain_answers_on_every_clip"]["wrong"] == 1
    with pytest.raises(ValueError, match="not in the truth"):
        lm_pick.audit_one({"a": ["x"]}, {"labels": {"zz": "x"}, "single": {}, "skipped": set(), "plain": {}})


def test_an_export_is_read_for_the_pseudo_labelled_clips_only(tmp_path):
    export = tmp_path / "r3"
    export.mkdir()
    rows = [{"id": "M01-1", "text": "his own"}, {"id": "tta-T001", "text": "open the door"}, {"id": "tta-T002", "text": "water"}]
    (export / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    held = lm_pick.read_labels(str(export))
    assert held["labels"] == {"T001": "open the door", "T002": "water"} and not held["skipped"]


def test_the_plain_answers_in_a_picks_file_can_be_audited_like_labels(tmp_path):
    picks = tmp_path / "picks.jsonl"
    picks.write_text(json.dumps({"id": "T1", "hyp": "open the door", "plain": "open a door", "single": False, "skip": "margin filter"}) + "\n",
                     encoding="utf-8")
    plain = lm_pick.read_labels(f"{picks}@plain")
    assert plain["labels"] == {"T1": "open a door"} and not plain["skipped"]
    assert lm_pick.read_labels(str(picks))["labels"] == {"T1": "open the door"} and lm_pick.read_labels(str(picks))["skipped"] == {"T1"}


def test_a_manifest_is_read_for_id_audio_and_speaker_only(tmp_path):
    with_text = tmp_path / "with.jsonl"
    with_text.write_text(json.dumps({"id": "T1", "audio": "test/T1.wav", "text": SECRET, "kind": "sentence", "prompt_set": "torgo-sentence",
                                     "speaker": "M01", "duration_s": 1.0}) + "\n", encoding="utf-8")
    bare = tmp_path / "bare.jsonl"
    bare.write_text(json.dumps({"id": "T1", "audio": "test/T1.wav", "speaker": "M01"}) + "\n", encoding="utf-8")
    one, two = lm_pool.read_unlabelled(with_text), lm_pool.read_unlabelled(bare)
    assert one == two == {"T1": {"id": "T1", "audio": "test/T1.wav", "speaker": "M01"}}
    assert SECRET not in json.dumps(one)
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"id": "T1"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="audio path"):
        lm_pool.read_unlabelled(bad)
    bad.write_text(json.dumps({"id": "T1", "audio": "a.wav"}) + "\n" + json.dumps({"id": "T1", "audio": "b.wav"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        lm_pool.read_unlabelled(bad)


def test_a_pool_row_is_assembled_without_the_text_for_an_unlabelled_clip_and_as_before_for_a_labelled_one(tmp_path):
    manifest = tmp_path / "chunk.jsonl"
    manifest.write_text(json.dumps({"id": "T1", "audio": "test/T1.wav", "text": SECRET, "kind": "sentence", "speaker": "M01", "duration_s": 2.0}) + "\n",
                        encoding="utf-8")
    clip = lm_pool.read_unlabelled(manifest)["T1"]
    assert "text" not in clip and "kind" not in clip
    cands = [cand("open the door", 5.0)]
    got = lm_pool.pool_row("T1", clip, False, 32000, 40, 9.0, cands, 0, {"decode": 0.1, "score": 0.1})
    assert got["ref"] == "" and got["kind"] == "unknown" and got["speaker"] == "M01" and got["duration_s"] == 2.0 and SECRET not in json.dumps(got)
    old = lm_pool.pool_row("T1", {"text": "Open the door.", "kind": "sentence", "speaker": "M01"}, True, 32000, 40, 9.0, cands, 0, {})
    assert old["ref"] == "open the door" and old["kind"] == "sentence" and list(old) == list(got)
    with pytest.raises(KeyError):
        lm_pool.pool_row("T1", clip, True, 32000, 40, 9.0, cands, 0, {})


def test_pool_builder_has_a_no_reference_switch_and_other_tools_treat_such_pools_correctly(tmp_path):
    done = subprocess.run([sys.executable, str(ROOT / "scripts/asr/lm_pool.py"), "--help"], capture_output=True, text=True, check=False)
    assert done.returncode == 0 and "--no-reference" in done.stdout
    pool = write_pool(tmp_path / "blind.json", blind(ROWS))
    assert lm_score.pool_texts(lm_score.read_pools([pool])) and lm_score.halves(lm_score.read_pools([pool])) == []
    with pytest.raises(ValueError, match="no reference words"):
        lm_report.read_clips([pool])


sf = pytest.importorskip("soundfile")
PROMPTS = ["open the door", "water", "please turn on the light", "good morning", "I am cold", "call my brother", "yes", "thank you",
           "bring me a blanket", "what time is it", "my head hurts", "close the window"]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def make_speaker(tmp_path: Path, with_text: bool = True) -> tuple[Path, Path]:
    speaker = tmp_path / "M01"
    split_rows = {"train": [], "dev": []}
    for i, text in enumerate(PROMPTS):
        split = "dev" if i in (3, 9) else "train"
        (speaker / split).mkdir(parents=True, exist_ok=True)
        sf.write(speaker / split / f"M01-{i}.wav", np.random.default_rng(i).normal(0, 0.05, 1600), 16000)
        split_rows[split].append({"id": f"M01-{i}", "audio": f"{split}/M01-{i}.wav", "text": text, "duration_s": 0.1, "speaker": "M01", "kind": "word"})
    write_rows(speaker / "train.jsonl", split_rows["train"])
    write_rows(speaker / "dev.jsonl", split_rows["dev"])
    tests = []
    for i in range(24):
        (speaker / "test").mkdir(exist_ok=True)
        sf.write(speaker / "test" / f"T{i:03d}.wav", np.random.default_rng(100 + i).normal(0, 0.05, 1600), 16000)
        row = {"id": f"T{i:03d}", "audio": f"test/T{i:03d}.wav", "duration_s": 0.1, "speaker": "M01", "kind": "sentence"}
        tests.append({**row, "text": SECRET} if with_text else row)
    write_rows(speaker / "test.jsonl", tests)
    return speaker, speaker / "test.jsonl"


def run_chain(tmp_path: Path, test: Path, speaker: Path) -> dict:
    clips = lm_pool.read_unlabelled(test)
    ids = sorted(clips)
    chunk = tta.chunk_ids(ids, 4)[0]
    rows = [unlabelled(i, [cand("open the window", 4.0), cand("open the door", 4.4 if n % 2 else 3.0, ("beam4",))], clips[i]["speaker"])
            for n, i in enumerate(chunk)]
    texts = {"Open the window.": -12.0, "Open the door.": -9.0}
    pool = write_pool(tmp_path / "pool.json", rows)
    scores = write_scores(tmp_path / "scores.json", logp=texts)
    plain = tmp_path / "plain.jsonl"
    plain.write_text("".join(json.dumps({"id": i, "hyp": "open the window"}) + "\n" for i in ids), encoding="utf-8")
    picks, labels = tmp_path / "picks.jsonl", tmp_path / "labels.jsonl"
    assert lm_pick.main(["pick", "--pool", pool, "--scores", scores, "--plain", str(plain), "--out", str(picks)]) == 0
    assert lm_pick.main(["filter", "--picks", str(picks), "--rule", "top", "--share", "0.5", "--out", str(labels)]) == 0
    other = tmp_path / "other.jsonl"
    other.write_text("".join(json.dumps({"id": i, "hyp": "open the door"}) + "\n" for i in ids), encoding="utf-8")
    out = tmp_path / "r1"
    info = tta.build(speaker, test, 4, 1, {0: str(labels)}, [str(other)], out, withhold=tta.read_withheld([str(labels)]))
    return {"info": info, "chunk": chunk, "labels": labels, "out": out}


def test_a_transcript_in_the_test_manifest_reaches_no_file_of_the_loop(tmp_path):
    speaker, test = make_speaker(tmp_path)
    made = run_chain(tmp_path, test, speaker)
    assert made["info"]["test_text_read"] is False and made["info"]["gate"].get("margin filter")
    for path in tmp_path.rglob("*"):
        if path.suffix in (".json", ".jsonl") and path != test:
            assert SECRET not in path.read_text(encoding="utf-8"), path
        assert SECRET.split()[0] not in path.name


def test_the_whole_chain_runs_when_the_manifest_has_no_text_column_at_all(tmp_path):
    speaker, test = make_speaker(tmp_path, with_text=False)
    assert all("text" not in json.loads(line) for line in test.read_text().splitlines())
    made = run_chain(tmp_path, test, speaker)
    assert made["info"]["pseudo_clips"] > 0


def test_clips_the_filter_drops_are_not_trained_on_and_are_counted_by_reason(tmp_path):
    speaker, test = make_speaker(tmp_path)
    made = run_chain(tmp_path, test, speaker)
    rows = [json.loads(line) for line in made["labels"].read_text().splitlines()]
    skipped = {r["id"] for r in rows if "skip" in r and r["id"] in made["chunk"]}
    trained = {json.loads(line)["id"][len("tta-"):] for line in (made["out"] / "train.jsonl").read_text().splitlines()
               if json.loads(line)["id"].startswith("tta-")}
    assert skipped and not skipped & trained and trained <= set(made["chunk"])
    assert made["info"]["gate"]["margin filter"] == len(skipped) == made["info"]["withheld_by_the_caller"]


def test_without_a_withhold_list_the_build_is_what_it_was_before_this_test(tmp_path):
    speaker, test = make_speaker(tmp_path)
    ids = sorted(lm_pool.read_unlabelled(test))
    base = tmp_path / "base.jsonl"
    base.write_text("".join(json.dumps({"id": i, "hyp": "open the door"}) + "\n" for i in ids), encoding="utf-8")
    info = tta.build(speaker, test, 4, 1, {0: str(base)}, [str(base)], tmp_path / "r1")
    assert "withheld_by_the_caller" not in info and "margin filter" not in info["gate"]
    done = subprocess.run([sys.executable, str(ROOT / "scripts/asr/tta.py"), "build", "--help"], capture_output=True, text=True, check=False)
    assert done.returncode == 0 and "--withhold" in done.stdout


def functions_naming(source: Path, names: set[str]) -> set[str]:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            used = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)} | {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            if used & names:
                found.add(node.name)
    return found


def test_only_the_audit_can_reach_the_true_text_in_the_picking_script():
    readers = functions_naming(ROOT / "scripts/asr/lm_pick.py", {"read_manifest", "read_truth"})
    assert readers == {"read_truth", "cmd_audit"}
    tree = ast.parse((ROOT / "scripts/asr/lm_pick.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in ("pick", "pick_row", "totals", "apply_rule", "keep_top", "cmd_pick", "cmd_filter"):
            subscripts = {s.slice.value for s in ast.walk(node) if isinstance(s, ast.Subscript) and isinstance(s.slice, ast.Constant)}
            assert "ref" not in subscripts, node.name


def load_lab():
    spec = importlib.util.spec_from_file_location("lab_cli_lm38", ROOT / "lab.py")
    lab = importlib.util.module_from_spec(spec)
    sys.modules["lab_cli_lm38"] = lab
    spec.loader.exec_module(lab)
    return lab


def mirror(tmp_path: Path) -> Path:
    app = tmp_path / "app"
    app.mkdir()
    for name in ("scripts", "src"):
        (app / name).symlink_to(ROOT / name)
    return app


def plan(tmp_path: Path, **overrides):
    lab = load_lab()
    app = mirror(tmp_path)
    commands, notes = lab.plan(lambda: lab.Lab(app, lab.Settings(app, overrides, env={}), dry=True, stamp="TEST", quiet=True), ["38"])
    assert notes == {"38": []}, notes
    return lab, app, commands


def flag(cmd: list[str], name: str) -> str:
    return cmd[cmd.index(name) + 1]


def test_stage_38_plans_lab_25s_loop_with_three_label_arms_and_never_gives_the_pools_a_transcript(tmp_path):
    lab, app, commands = plan(tmp_path, trim_models="off")
    pools = [c for c in commands if "scripts/asr/lm_pool.py" in c]
    assert len(pools) == 9 * (1 + 3 * 2) and all("--no-reference" in c for c in pools)
    assert len({flag(c, "--out") for c in pools}) == len(pools)
    assert sum("static-" in flag(c, "--out") for c in pools) == 9 and all("--adapter" in c and "--views" in c for c in pools)
    picks = [c for c in commands if "lm_pick.py" in " ".join(c) and "pick" in c[2:3]]
    assert len(picks) == len(pools) and all("--plain" in c and "--weight" in c and flag(c, "--weight") == "0.7" and flag(c, "--bonus") == "3" for c in picks)
    scored = [c for c in commands if "scripts/asr/lm_score.py" in c and "score" in c]
    assert len(scored) == len(pools) and all(flag(c, "--model") == "qwen25-05b-q4" and "--extend" in c for c in scored)
    assert len({flag(c, "--out") for c in scored}) == 1 and all(flag(c, "--out").endswith("scores/qwen25-05b-q4.json") for c in scored)
    sub = [c[2] for c in commands if "scripts/asr/lm_score.py" in c]
    assert sub.count("fetch") == 1 and "complete" not in sub and "release" not in sub
    filters = [c for c in commands if "lm_pick.py" in " ".join(c) and "filter" in c[2:3]]
    assert len(filters) == 9 * 3 * 3
    assert {flag(c, "--rule") for c in filters} == {"all", "top", "margin"}
    assert all(flag(c, "--share") == "0.5" for c in filters if flag(c, "--rule") == "top")
    assert all(flag(c, "--min-margin") == "3" for c in filters if flag(c, "--rule") == "margin")
    builds = [c for c in commands if "scripts/asr/tta.py" in c and "build" in c]
    mine = [c for c in builds if "lab-38-" in flag(c, "--out")]
    assert len(builds) == 27 + 81 and len(mine) == 81 and all("--withhold" in c for c in mine) and not any("--withhold" in c for c in builds if c not in mine)
    assert all(flag(c, "--chunk-hyps").split("=")[1].endswith(".jsonl") and "/labels/" in flag(c, "--chunk-hyps") for c in mine)
    trains = [flag(c, "--out") for c in commands if "scripts/asr/train.py" in c]
    assert sum("lab-38-" in t for t in trains) == 81 and sum("lab-25-" in t for t in trains) == 27
    problems, checked = lab.flag_problems(app, commands)
    assert not problems, problems
    assert checked


def test_stage_38_gives_only_the_audit_a_truth_flag_and_runs_it_after_the_last_training_step(tmp_path):
    _lab, _app, commands = plan(tmp_path, trim_models="off")
    truth = [i for i, c in enumerate(commands) if "--truth" in c]
    assert len(truth) == 1 and "lm_pick.py" in " ".join(commands[truth[0]]) and "audit" in commands[truth[0]]
    last_train = max(i for i, c in enumerate(commands) if "scripts/asr/train.py" in c)
    assert truth[0] > last_train
    for c in commands:
        if "lm_pick.py" in " ".join(c) and ("pick" in c[2:3] or "filter" in c[2:3]):
            assert not any(x.endswith("test.jsonl") for x in c), c
    audit = commands[truth[0]]
    shown = " ".join(audit)
    assert "control (lab 25) s17, trained on=" in shown and "control (lab 25) s17, before the gate=" in shown and "@plain" in shown
    assert "lab-25-M01-s17-chunk2-answers.jsonl" in shown and "lab-25-M01-s17-chunk3-answers" not in shown and "before the gate=" in shown and flag(audit, "--out").endswith("label-quality.json")


def test_stage_38_orders_each_round_pool_after_the_decode_and_before_the_adapter_is_trimmed(tmp_path):
    lab = load_lab()
    app = mirror(tmp_path)
    name = "lab-38-lm-all-M01-s17-r1"
    (app / "models" / f"parakeet-ft-{name}").mkdir(parents=True)
    session = lab.Lab(app, lab.Settings(app, {"trim_models": "on", "lm_label_arms": "all"}, env={}), dry=True, stamp="TEST", quiet=True)
    session.stage = "38"
    lab.stage_38(session)
    lines = session.lines
    at = {"train": next(i for i, x in enumerate(lines) if x.startswith("+") and "train.py" in x and f"--out models/parakeet-ft-{name}" in x),
          "decode": next(i for i, x in enumerate(lines) if x.startswith("+") and "nemo_hyp.py" in x and f"parakeet-ft-{name}/adapter.pt" in x),
          "pool": next(i for i, x in enumerate(lines) if x.startswith("+") and "lm_pool.py" in x and f"parakeet-ft-{name}/adapter.pt" in x),
          "trim": next(i for i, x in enumerate(lines) if x.startswith("TRIM_MODELS") and f"parakeet-ft-{name} " in x),
          "next": next(i for i, x in enumerate(lines) if x.startswith("+") and "tta.py" in x and " build " in x and "lab-38-lm-all-M01-s17-r2" in x)}
    assert at["train"] < at["decode"] < at["pool"] < at["trim"] < at["next"]


def test_stage_38_arms_and_settings_follow_the_settings(tmp_path):
    _lab, _app, commands = plan(tmp_path, trim_models="off", lm_label_arms="top", lm_top_share="0.25", lm_weight="0.5", lm_bonus="2",
                                lm_label_model="qwen25-05b-q8", tta_chunks="3")
    filters = [c for c in commands if "lm_pick.py" in " ".join(c) and "filter" in c[2:3]]
    assert len(filters) == 9 * 2 and all(flag(c, "--share") == "0.25" and flag(c, "--rule") == "top" for c in filters)
    assert all("lm-top25" in flag(c, "--out") for c in filters)
    picks = [c for c in commands if "lm_pick.py" in " ".join(c) and "pick" in c[2:3]]
    assert all(flag(c, "--weight") == "0.5" and flag(c, "--bonus") == "2" for c in picks)
    assert len(picks) == 9 * (1 + 1)
    assert all(flag(c, "--model") == "qwen25-05b-q8" for c in commands if "scripts/asr/lm_score.py" in c and "score" in c)


def test_stage_38_frees_the_language_model_file_when_disk_is_short(tmp_path):
    lab, app, commands = plan(tmp_path, trim_models="on", lm_label_arms="all")
    sub = [c for c in commands if "scripts/asr/lm_score.py" in c]
    assert sub[0][2] == "fetch" and sub[-1][2] == "release" and flag(sub[-1], "--models") == "qwen25-05b-q4"
    problems, _checked = lab.flag_problems(app, commands)
    assert not problems, problems


BAD = ({"lm_label_arms": "everything"}, {"lm_label_arms": "all all"}, {"lm_label_arms": ""}, {"lm_label_arms": "all top50"},
       {"lm_label_model": "gpt-9"}, {"lm_label_model": "pi"}, {"lm_label_model": "qwen25-05b-q4 qwen25-05b-q8"}, {"lm_label_model": ""},
       {"lm_top_share": "0"}, {"lm_top_share": "1.5"}, {"lm_min_margin": "-1"}, {"lm_weight": "-0.5"}, {"lm_bonus": "-1"})


@pytest.mark.parametrize("bad", BAD, ids=lambda b: "-".join(f"{k}={v!r}" for k, v in b.items()))
def test_stage_38_refuses_bad_settings(tmp_path, bad):
    lab = load_lab()
    app = mirror(tmp_path)
    session = lab.Lab(app, lab.Settings(app, bad, env={}), dry=True, stamp="TEST", quiet=True)
    with pytest.raises((ValueError, KeyError)):
        lab.stage_38(session)
    assert not session.commands


def test_the_test_is_listed_queued_and_rebuilds_missing_weights(tmp_path):
    lab = load_lab()
    app = mirror(tmp_path)
    session = lab.Lab(app, lab.Settings(app, {}, env={}), dry=True, stamp="TEST", quiet=True)
    assert "38" in session.catalog and "38" in lab.AUTO_REBUILD and lab.SELFTRAIN_LM_QUEUE == ["38"]
    assert session.catalog["38"].markers == ["reports/lab-38-lm-pseudo-labels/table.json"]
    assert lab.resolve(session, ["selftrain-lm"]) == ["38"]
    assert lab.Lab.BASELINE_NEEDS["38"] == ("stock", "f1")
    for name in ("lm_label_model", "lm_label_arms", "lm_weight", "lm_bonus", "lm_top_share", "lm_min_margin"):
        _default, env, help_text = lab.VARIABLES[name]
        assert env and len(help_text) > 40 and "lab test 38" in help_text


def test_resuming_waits_for_the_chunks_picks_and_names_one_label_file_per_round(tmp_path):
    lab = load_lab()
    app = mirror(tmp_path)
    session = lab.Lab(app, lab.Settings(app, {}, env={}), dry=False, stamp="TEST", quiet=True)
    cfg = lab.lm38_settings(session)
    labels = lab.LmLabels(session, "M01", 17, "top", cfg, 4, "models/base.nemo", "reports/lab-38-lm-pseudo-labels/picks/static-M01-s17-c0.jsonl")
    assert labels.ready(3) and not labels.ready(1) and not labels.ready(2)
    picks = app / labels.picks(1)
    picks.parent.mkdir(parents=True)
    picks.write_text("")
    assert labels.ready(1) and not labels.ready(2)
    assert labels.picks(0).endswith("static-M01-s17-c0.jsonl") and labels.picks(2).endswith("lm-top50-M01-s17-c2.jsonl")
    assert labels.chunk_hyps(3) == ["--chunk-hyps", f"0={labels.round_file(3)}", "--chunk-hyps", f"1={labels.round_file(3)}",
                                    "--chunk-hyps", f"2={labels.round_file(3)}"]
    assert labels.withhold(2) == ["--withhold", labels.round_file(2)]
    assert math.isclose(cfg["share"], 0.5)


def test_scripts_print_help_without_torch_or_llama():
    for sub in ([], ["pick"], ["filter"], ["audit"]):
        done = subprocess.run([sys.executable, str(ROOT / "scripts" / "asr" / "lm_pick.py"), *sub, "--help"], capture_output=True, text=True, check=False)
        assert done.returncode == 0, done.stderr[-400:]
