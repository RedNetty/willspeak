import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "asr"))
import kind_route
import kind_route_report
import labkit


def clip(clip_id: str, text: str, kind: str, duration: float = 1.0, speaker: str = "M01") -> dict:
    return {"id": clip_id, "audio": f"test/{clip_id}.wav", "text": text, "prompt_text": text, "duration_s": duration, "kind": kind, "speaker": speaker}


ROWS = [clip("w1", "water", "word", 1.2), clip("w2", "tomato", "word", 2.0), clip("s1", "the boy ran home", "sentence", 3.0),
        clip("s2", "open the door", "sentence", 4.0)]
CONTROL = {"w1": "waiter", "w2": "tomato", "s1": "the", "s2": "open the door"}
WORD = {"w1": "water", "w2": "potato", "s1": "water", "s2": "water"}
SENTENCE = {"w1": "waiter", "w2": "tomato", "s1": "the boy ran home", "s2": "open the door"}


def manifest(rows=ROWS) -> dict[str, dict]:
    return {r["id"]: r for r in rows}


def wrong(hyps: dict[str, str], kind: str = "all") -> int:
    return labkit.totals(labkit.clip_counts(labkit.only_kind(manifest(), kind), {k: hyps[k] for k in labkit.only_kind(manifest(), kind)}))[0]


def test_detector_says_single_word_only_for_exactly_one_word():
    assert kind_route.says_one_word("water") and kind_route.says_one_word("Water.") and kind_route.says_one_word("  WATER  ")
    assert not kind_route.says_one_word("ice cream") and not kind_route.says_one_word("the boy ran home")
    assert kind_route.word_count("The boy, ran home!") == 4


def test_a_silent_first_decode_goes_to_the_other_model():
    assert not kind_route.says_one_word("") and not kind_route.says_one_word("   ") and not kind_route.says_one_word("...")


def test_detector_word_limit_is_adjustable_and_never_below_one():
    assert kind_route.says_one_word("ice cream", max_words=2) and not kind_route.says_one_word("a big dog", max_words=2)
    assert not kind_route.says_one_word("", max_words=5)


def test_words_detector_reads_the_first_decode_and_needs_one_for_every_clip():
    called = kind_route.detect("words", manifest(), CONTROL)
    assert called == {"w1": True, "w2": True, "s1": True, "s2": False}
    with pytest.raises(KeyError, match="s2"):
        kind_route.detect("words", manifest(), {k: v for k, v in CONTROL.items() if k != "s2"})
    with pytest.raises(ValueError, match="one of"):
        kind_route.detect("guess", manifest(), CONTROL)


def test_oracle_detector_uses_the_true_kind_and_ignores_decodes():
    assert kind_route.detect("oracle", manifest()) == {"w1": True, "w2": True, "s1": False, "s2": False}


def test_duration_cut_maximises_accuracy_on_labelled_clips_and_prefers_the_shorter_cut():
    rows = [clip("a", "x", "word", 1.0), clip("b", "x", "word", 1.5), clip("c", "x", "word", 2.5),
            clip("d", "a b c", "sentence", 2.4), clip("e", "a b c", "sentence", 3.0), clip("f", "a b", "sentence", 4.0)]
    cut = kind_route.duration_cut(rows)
    assert cut == 1.5
    called = kind_route.detect("duration", manifest(rows), cut=cut)
    assert called == {"a": True, "b": True, "c": False, "d": False, "e": False, "f": False}


def test_duration_cut_with_one_kind_only_is_refused():
    with pytest.raises(ValueError, match="both kinds"):
        kind_route.duration_cut([clip("a", "x", "word", 1.0), clip("b", "y", "word", 2.0)])
    with pytest.raises(ValueError, match="needs a cut"):
        kind_route.detect("duration", manifest())


def test_routing_takes_the_word_decode_where_called_and_the_other_decode_elsewhere():
    called = kind_route.detect("words", manifest(), CONTROL)
    routed = kind_route.route(called, WORD, CONTROL)
    assert routed == {"w1": "water", "w2": "potato", "s1": "water", "s2": "open the door"}
    with pytest.raises(KeyError, match="lack a decode"):
        kind_route.route(called, {"w1": "x"}, CONTROL)


def test_routing_arithmetic_oracle_against_the_detector():
    detected = kind_route.route(kind_route.detect("words", manifest(), CONTROL), WORD, SENTENCE)
    oracle = kind_route.route(kind_route.detect("oracle", manifest()), WORD, SENTENCE)
    assert wrong(CONTROL, "word") == 1 and wrong(CONTROL, "sentence") == 3
    assert wrong(oracle, "word") == 1 and wrong(oracle, "sentence") == 0 and wrong(oracle) == 1
    assert wrong(detected, "word") == 1 and wrong(detected, "sentence") == 4
    counts = kind_route.confusion(manifest(), kind_route.detect("words", manifest(), CONTROL))
    assert counts["sentence_called_word"] == 1 and counts["word_called_sentence"] == 0 and counts["routed_wrong"] == 1
    assert counts["accuracy"] == pytest.approx(0.75) and counts["clips"] == 4 and counts["word_clips"] == 2 and counts["sentence_clips"] == 2
    perfect = kind_route.confusion(manifest(), kind_route.detect("oracle", manifest()))
    assert perfect["routed_wrong"] == 0 and perfect["accuracy"] == 1.0


def test_routed_errors_equal_oracle_errors_plus_the_cost_of_each_misrouted_clip():
    called = kind_route.detect("words", manifest(), CONTROL)
    routed = kind_route.route(called, WORD, SENTENCE)
    oracle = kind_route.route(kind_route.detect("oracle", manifest()), WORD, SENTENCE)
    extra = sum(labkit.clip_counts({c: manifest()[c]}, {c: routed[c]})[c][0] - labkit.clip_counts({c: manifest()[c]}, {c: oracle[c]})[c][0]
                for c in manifest() if called[c] != (manifest()[c]["kind"] == "word"))
    assert wrong(routed) - wrong(oracle) == extra == 4


def write_manifest(folder: Path, rows) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "test.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def write_decode(folder: Path, hyps: dict[str, str]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "hypotheses.jsonl").write_text("".join(json.dumps({"id": k, "hyp": v}) + "\n" for k, v in hyps.items()), encoding="utf-8")
    return folder


def run_cli(monkeypatch, capsys, *args):
    monkeypatch.setattr(sys, "argv", ["kind_route.py", *map(str, args)])
    code = kind_route.main()
    return code, capsys.readouterr().out


def test_cli_writes_the_routed_decode_and_the_detector_counts(tmp_path, monkeypatch, capsys):
    test = write_manifest(tmp_path / "M01", ROWS)
    word, other = write_decode(tmp_path / "word", WORD), write_decode(tmp_path / "other", CONTROL)
    out, stats = tmp_path / "routed.jsonl", tmp_path / "stats.json"
    code, text = run_cli(monkeypatch, capsys, "--manifest", test, "--detector", "words", "--word-hyp", word, "--other-hyp", other, "--out", out, "--stats", stats)
    assert code == 0 and "1 routed against their true kind" in text and "1 sentences sent to the word model" in text
    rows = {json.loads(line)["id"]: json.loads(line)["hyp"] for line in out.read_text().splitlines()}
    assert rows == {"w1": "water", "w2": "potato", "s1": "water", "s2": "open the door"}
    data = json.loads(stats.read_text())
    assert data["detector"] == "words" and data["counts"]["routed_wrong"] == 1 and data["called_word"] == ["s1", "w1", "w2"]
    with pytest.raises(FileExistsError):
        run_cli(monkeypatch, capsys, "--manifest", test, "--detector", "words", "--word-hyp", word, "--other-hyp", other, "--out", out)


def test_cli_detects_from_a_separate_decode_and_chooses_the_duration_cut_from_labelled_clips(tmp_path, monkeypatch, capsys):
    test = write_manifest(tmp_path / "test", ROWS)
    labelled = write_manifest(tmp_path / "train", [clip("a", "x", "word", 1.0), clip("b", "x", "word", 2.0), clip("c", "a b c", "sentence", 3.5),
                                                    clip("d", "a b", "sentence", 4.5)])
    word, other = write_decode(tmp_path / "word", WORD), write_decode(tmp_path / "other", SENTENCE)
    stats = tmp_path / "stats.json"
    code, text = run_cli(monkeypatch, capsys, "--manifest", test, "--detector", "duration", "--cut-from", labelled, "--word-hyp", word, "--other-hyp", other,
                         "--out", tmp_path / "d.jsonl", "--stats", stats)
    assert code == 0 and "duration cut 2.000 s" in text
    assert json.loads(stats.read_text())["called_word"] == ["w1", "w2"] and json.loads(stats.read_text())["counts"]["routed_wrong"] == 0
    first = write_decode(tmp_path / "first", {"w1": "two words", "w2": "tomato", "s1": "the", "s2": "a b"})
    code, _ = run_cli(monkeypatch, capsys, "--manifest", test, "--detector", "words", "--detect-hyp", first, "--word-hyp", word, "--other-hyp", other,
                      "--out", tmp_path / "w.jsonl")
    assert code == 0
    assert {json.loads(x)["id"]: json.loads(x)["hyp"] for x in (tmp_path / "w.jsonl").read_text().splitlines()}["w1"] == "waiter"


def test_cli_refuses_a_duration_detector_with_nothing_to_choose_the_cut_from(tmp_path, monkeypatch, capsys):
    test = write_manifest(tmp_path / "t", ROWS)
    word, other = write_decode(tmp_path / "word", WORD), write_decode(tmp_path / "other", CONTROL)
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, capsys, "--manifest", test, "--detector", "duration", "--word-hyp", word, "--other-hyp", other, "--out", tmp_path / "x.jsonl")
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, capsys, "--manifest", test, "--detector", "words", "--max-words", "0", "--word-hyp", word, "--other-hyp", other, "--out", tmp_path / "y.jsonl")


def test_scripts_print_help():
    for script in ("kind_route.py", "kind_route_report.py"):
        done = subprocess.run([sys.executable, str(ROOT / "scripts" / "asr" / script), "--help"], capture_output=True, text=True, check=False)
        assert done.returncode == 0, done.stderr[-400:]


def write_export(folder: Path) -> None:
    sf = pytest.importorskip("soundfile")
    np = pytest.importorskip("numpy")
    for split, rows in (("train", [("t1", "water", "word"), ("t2", "tomato", "word"), ("t3", "open the door", "sentence"), ("t4", "the boy ran home", "sentence"),
                                   ("t5", "coffee", "word")]),
                        ("dev", [("d1", "bread", "word"), ("d2", "we went home early", "sentence")])):
        (folder / split).mkdir(parents=True, exist_ok=True)
        out = []
        for clip_id, text, kind in rows:
            noise = np.random.default_rng(sum(map(ord, clip_id))).standard_normal(8000).astype("float32") * 0.1
            sf.write(folder / split / f"{clip_id}.wav", noise, 16000)
            out.append({"id": clip_id, "audio": f"{split}/{clip_id}.wav", "text": text, "prompt_text": text, "duration_s": 0.5, "kind": kind,
                        "speaker": "M01", "prompt_set": f"torgo-{kind}"})
        (folder / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out), encoding="utf-8")
    (folder / "test.jsonl").write_text("must never be opened by training")


def check_kinds(folder: Path, kinds: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ROOT / "scripts/asr/train.py"), "--check", "--export", str(folder), "--out", str(folder / "run"), "--kinds", kinds],
                          capture_output=True, text=True, check=False)


def test_training_filters_train_and_dev_to_one_kind_without_a_new_export(tmp_path):
    write_export(tmp_path)
    done = check_kinds(tmp_path, "word")
    assert done.returncode == 0, done.stderr[-600:]
    assert "--kinds word: train 5 -> 3 clips, dev 2 -> 1 clips" in done.stdout + done.stderr
    done = check_kinds(tmp_path, "sentence")
    assert done.returncode == 0 and "--kinds sentence: train 5 -> 2 clips, dev 2 -> 1 clips" in done.stdout + done.stderr


def test_the_kind_filter_keeps_prompts_disjoint_between_train_and_dev(tmp_path):
    write_export(tmp_path)
    for split in ("train", "dev"):
        rows = [json.loads(x) for x in (tmp_path / f"{split}.jsonl").read_text().splitlines()]
        kept = {r["prompt_text"] for r in rows if r["kind"] == "word"}
        assert len(kept) == len([r for r in rows if r["kind"] == "word"])
    train = {json.loads(x)["prompt_text"] for x in (tmp_path / "train.jsonl").read_text().splitlines()}
    dev = {json.loads(x)["prompt_text"] for x in (tmp_path / "dev.jsonl").read_text().splitlines()}
    assert not train & dev


def test_the_kind_filter_refuses_an_empty_dev_set_and_unknown_kinds(tmp_path):
    write_export(tmp_path)
    dev = [json.loads(x) for x in (tmp_path / "dev.jsonl").read_text().splitlines()]
    (tmp_path / "dev.jsonl").write_text("".join(json.dumps(r) + "\n" for r in dev if r["kind"] == "sentence"), encoding="utf-8")
    done = check_kinds(tmp_path, "word")
    assert done.returncode != 0 and "no dev clips" in done.stderr
    done = check_kinds(tmp_path, "digits")
    assert done.returncode != 0 and "sentence and/or word" in done.stderr


def report_fixture(tmp_path: Path, seeds=(17, 18)):
    export = tmp_path / "export"
    for s in ("M01", "M02"):
        write_manifest(export / s, [{**r, "speaker": s} for r in ROWS])
    systems = {"control": CONTROL, "routed": kind_route.route(kind_route.detect("words", manifest(), CONTROL), WORD, SENTENCE), "word": WORD}
    for label, hyps in systems.items():
        for s in ("M01", "M02"):
            for seed in seeds:
                write_decode(tmp_path / label / f"{s}-s{seed}", hyps)
    patterns = {label: str(tmp_path / label / "{speaker}-s{seed}") for label in systems}
    return export, patterns


def test_report_splits_wrong_words_by_kind_and_compares_each_seed_with_the_control(tmp_path):
    export, patterns = report_fixture(tmp_path)
    manifests = {s: labkit.read_manifest(export / s / "test.jsonl") for s in ("M01", "M02")}
    rows = kind_route_report.build(manifests, patterns, "control", [17, 18], {})
    by = {r["system"]: r for r in rows}
    assert by["control"]["mean"] == {"all": 8.0, "sentence": 6.0, "word": 2.0}
    assert by["control"]["words"] == {"all": 18, "sentence": 14, "word": 4}
    assert by["routed"]["mean"] == {"all": 10.0, "sentence": 8.0, "word": 2.0}
    gain = by["routed"]["against_control"]
    assert gain["seeds"] == [17, 18] and gain["sentence"]["gain"] == -2.0 and gain["all"]["gain"] == -2.0 and gain["word"]["gain"] == 0.0
    assert gain["sentence_harm"] is False and gain["single_word_rule"] is False
    assert gain["project_verdict"] == "worse"


def test_report_rules_count_a_clear_single_word_gain_and_a_harmed_sentence_set():
    assert kind_route_report.WORD_GAIN_MIN == 10.0 and kind_route_report.SENTENCE_HARM_MAX == 5.0
    manifests = {s: {f"{s}{i}": clip(f"{s}{i}", "water", "word") for i in range(25)} | {f"{s}x{i}": clip(f"{s}x{i}", "one two three", "sentence") for i in range(25)}
                 for s in ("A", "B")}
    control = {s: {c: "wrong" for c in m} for s, m in manifests.items()}
    better = {s: {c: ("water" if m[c]["kind"] == "word" else "wrong") for c in m} for s, m in manifests.items()}
    rows = {}
    for label, hyps in (("control", control), ("better", better)):
        rows[label] = {17: kind_route_report.tally(manifests, hyps)}
    result = kind_route_report.compare(rows["control"], rows["better"], ["A", "B"])
    assert result["word"]["gain"] == 50 and result["single_word_rule"] is True and result["sentence_harm"] is False
    assert result["project_verdict"] == "better"
    control2 = {s: {c: ("wrong" if m[c]["kind"] == "word" else "one two three") for c in m} for s, m in manifests.items()}
    harmed = kind_route_report.compare({17: kind_route_report.tally(manifests, control2)}, rows["better"], ["A", "B"])
    assert harmed["word"]["gain"] == 50 and harmed["single_word_rule"] is True
    assert harmed["sentence"]["gain"] == -150 and harmed["sentence_harm"] is True and harmed["project_verdict"] == "worse"


def test_report_skips_a_system_with_no_complete_seed_and_needs_a_control(tmp_path):
    export, patterns = report_fixture(tmp_path, seeds=(17,))
    manifests = {s: labkit.read_manifest(export / s / "test.jsonl") for s in ("M01", "M02")}
    rows = kind_route_report.build(manifests, {**patterns, "missing": str(tmp_path / "nowhere" / "{speaker}-s{seed}")}, "control", [17, 18, 19], {})
    assert [r["system"] for r in rows] == ["control", "routed", "word"] and rows[0]["seeds"] == [17]
    with pytest.raises(ValueError, match="no complete decode"):
        kind_route_report.build(manifests, {"word": patterns["word"]}, "control", [17], {})
    with pytest.raises(ValueError, match="speaker"):
        kind_route_report.parse_pattern("x=reports/no-placeholders")


def test_report_prints_the_detector_mistakes_when_routing_files_exist(tmp_path):
    export, patterns = report_fixture(tmp_path, seeds=(17,))
    manifests = {s: labkit.read_manifest(export / s / "test.jsonl") for s in ("M01", "M02")}
    counts = kind_route.confusion(manifest(), kind_route.detect("words", manifest(), CONTROL))
    for s in ("M01", "M02"):
        path = tmp_path / "stats" / f"{s}-s17.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"counts": counts}), encoding="utf-8")
    rows = kind_route_report.build(manifests, patterns, "control", [17], {"routed": str(tmp_path / "stats" / "{speaker}-s17.json").replace("-s17.json", "-s{seed}.json")})
    routed = next(r for r in rows if r["system"] == "routed")["routing"]
    assert routed["routed_wrong"] == 2 and routed["sentence_called_word"] == 2 and routed["clips"] == 8 and routed["accuracy"] == pytest.approx(0.75)
    assert "detector: 75.0% right" in kind_route_report.render(rows, "control", ["M01", "M02"])


def load_lab():
    spec = importlib.util.spec_from_file_location("lab_cli_routing", ROOT / "lab.py")
    lab = importlib.util.module_from_spec(spec)
    sys.modules["lab_cli_routing"] = lab
    spec.loader.exec_module(lab)
    return lab


def mirror(tmp_path: Path) -> Path:
    app = tmp_path / "app"
    app.mkdir()
    for name in ("scripts", "src"):
        (app / name).symlink_to(ROOT / name)
    return app


def plan(tmp_path, **overrides):
    lab = load_lab()
    app = mirror(tmp_path)
    commands, notes = lab.plan(lambda: lab.Lab(app, lab.Settings(app, overrides, env={}), dry=True, stamp="TEST", quiet=True), ["37"])
    assert notes == {"37": []}
    return lab, app, commands


def out_of(command: list[str]) -> str:
    return command[command.index("--out") + 1]


def test_queue_names_stage_37_and_it_is_rebuilt_automatically():
    lab = load_lab()
    assert lab.ROUTING_QUEUE == ["37"] and "37" in lab.AUTO_REBUILD and "37" in lab.LAB_TESTS and "37" in lab.Lab.BASELINE_NEEDS
    session = lab.Lab(ROOT, lab.Settings(ROOT, {}, env={}), dry=True, stamp="TEST", quiet=True)
    assert lab.resolve(session, ["routing"]) == ["37"] and "37" in session.catalog
    assert session.catalog["37"].markers == ["reports/lab-37-kind-routing/finished.json"]


def test_dry_plan_trains_three_adapters_per_speaker_and_seed_and_routes_offline(tmp_path):
    lab, app, commands = plan(tmp_path, trim_models="off")
    trains = [c for c in commands if "scripts/asr/train.py" in c]
    adapters = [c for c in trains if "lab-37-" in out_of(c)]
    assert len(adapters) == 27 and len({out_of(c) for c in adapters}) == 27
    assert sum("big-c-pre-" in out_of(c) for c in trains) == 3
    for arm, kinds in (("all", None), ("word", "word"), ("sentence", "sentence")):
        mine = [c for c in adapters if f"lab-37-{arm}-" in out_of(c)]
        assert len(mine) == 9
        assert all(("--kinds" in c and c[c.index("--kinds") + 1] == kinds) if kinds else "--kinds" not in c for c in mine)
        assert all(c[c.index("--base") + 1].startswith("models/parakeet-ft-big-c-pre-") and c[c.index("--lr") + 1] == "6e-4"
                   and c[c.index("--adapter-after-blocks") + 1] == "8" and c[c.index("--epochs") + 1] == "15" and "--select" in c for c in mine)
    seeds = [c[c.index("--seed") + 1] for c in adapters]
    assert seeds == [str(s) for s in (17, 18, 19) for _ in range(9)]
    decodes = [c for c in commands if "scripts/asr/nemo_hyp.py" in c and "lab-37-" in out_of(c)]
    assert len(decodes) == 27 and all("--adapter" in c for c in decodes)
    routes = [c for c in commands if "scripts/asr/kind_route.py" in c]
    assert len(routes) == 3 * 3 * (6 + 2)
    words = [c for c in routes if c[c.index("--detector") + 1] == "words"]
    assert len(words) == 27 and all(c[c.index("--detect-hyp") + 1].split("/")[-1].startswith("lab-37-all-") for c in words)
    assert all("--detect-hyp" not in c for c in routes if c[c.index("--detector") + 1] != "words")
    assert sum(c[c.index("--detector") + 1] == "duration" for c in routes) == 18 and all("--cut-from" in c for c in routes if c[c.index("--detector") + 1] == "duration")
    assert all("train.jsonl" not in " ".join(c) or c[c.index("--detector") + 1] == "duration" for c in routes)
    scored = [c for c in commands if "scripts/asr/evaluate.py" in c and "lab-37-routed-" in " ".join(c)]
    assert len(scored) == len(routes) and all("--hyp" in c and "kind" in c for c in scored)
    pools = [c for c in commands if "scripts/asr/pool_table.py" in c]
    assert [out_of(c) for c in pools] == ["reports/lab-37-kind-routing/table.json", "reports/lab-37-kind-routing/table-words.json"]
    assert "--kind" in pools[1] and pools[1][pools[1].index("--kind") + 1] == "word"
    report = [c for c in commands if "scripts/asr/kind_route_report.py" in c]
    assert len(report) == 1 and report[0][report[0].index("--control") + 1] == lab.ROUTE37_CONTROL
    assert max(commands.index(c) for c in decodes) < min(commands.index(c) for c in routes) < commands.index(pools[0]) < commands.index(report[0])
    problems, _checked = lab.flag_problems(app, commands)
    assert not problems, problems


def test_the_word_arm_alone_leaves_out_the_sentence_adapter_and_its_routed_systems(tmp_path):
    lab, app, commands = plan(tmp_path, trim_models="off", routing_arms="word")
    adapters = [c for c in commands if "scripts/asr/train.py" in c and "lab-37-" in out_of(c)]
    assert len(adapters) == 18 and not [c for c in adapters if "lab-37-sentence-" in out_of(c)]
    assert len([c for c in commands if "scripts/asr/kind_route.py" in c]) == 3 * 3 * (3 + 2)
    problems, _checked = lab.flag_problems(app, commands)
    assert not problems, problems


def test_a_longer_schedule_for_the_kind_adapters_gets_its_own_run_names_and_leaves_the_control_alone(tmp_path):
    lab, app, commands = plan(tmp_path, trim_models="off", routing_kind_epochs=45)
    for arm in ("word", "sentence"):
        mine = [c for c in commands if "scripts/asr/train.py" in c and f"lab-37-{arm}-" in out_of(c)]
        assert len(mine) == 9 and all(out_of(c).endswith("-e45") and c[c.index("--epochs") + 1] == "45" for c in mine)
    control = [c for c in commands if "scripts/asr/train.py" in c and "lab-37-all-" in out_of(c)]
    assert len(control) == 9 and all(c[c.index("--epochs") + 1] == "15" and not out_of(c).endswith("-e45") for c in control)
    routes = [" ".join(c) for c in commands if "scripts/asr/kind_route.py" in c]
    assert any("lab-37-word-M01-s17-e45-test-nemo" in c for c in routes) and any("lab-37-sentence-M01-s17-e45-test-nemo" in c for c in routes)
    assert not [c for c in routes if "lab-37-word-M01-s17-test-nemo" in c]
    problems, _checked = lab.flag_problems(app, commands)
    assert not problems, problems


def test_stage_37_refuses_bad_settings(tmp_path):
    lab = load_lab()
    app = mirror(tmp_path)
    for bad in ({"routing_arms": "sentence"}, {"routing_arms": ""}, {"routing_arms": "word word"}, {"routing_arms": "word digits"},
                {"routing_kind_epochs": 3}, {"routing_kind_epochs": 101}):
        session = lab.Lab(app, lab.Settings(app, bad, env={}), dry=True, stamp="TEST", quiet=True)
        with pytest.raises(ValueError):
            lab.stage_37(session)
    assert not [c for c in session.commands if "train.py" in " ".join(c)]


def test_preflight_stops_only_where_training_would_refuse(tmp_path):
    lab = load_lab()
    app = mirror(tmp_path)

    def build(words_in_train: int = 25, words_in_dev: int = 3, sentences_in_dev: int = 3, drop_kind: bool = False) -> None:
        for s in ("M01", "M02", "M04"):
            (app / lab.EX / s).mkdir(parents=True, exist_ok=True)
            for split, n_word, n_sentence in (("train", words_in_train, 22), ("dev", words_in_dev, sentences_in_dev)):
                rows = [clip(f"{split}{i}", "water", "word") for i in range(n_word)] + [clip(f"{split}s{i}", "open the door", "sentence") for i in range(n_sentence)]
                if drop_kind:
                    rows[0].pop("kind")
                (app / lab.EX / s / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def session():
        return lab.Lab(app, lab.Settings(app, {}, env={}), dry=False, stamp="TEST", quiet=True)

    build()
    assert lab.route37_preflight(session(), ["word", "sentence"]) is True
    assert lab.route37_clip_counts(session(), "M01") == {"train": {"word": 25, "sentence": 22}, "dev": {"word": 3, "sentence": 3}}
    build(words_in_train=3, words_in_dev=1, sentences_in_dev=1)
    assert lab.route37_preflight(session(), ["word", "sentence"]) is True
    for broken, kind in (({"words_in_train": 0}, "word"), ({"words_in_dev": 0}, "word"), ({"sentences_in_dev": 0}, "sentence"), ({"drop_kind": True}, "word")):
        build(**broken)
        s = session()
        assert lab.route37_preflight(s, ["word", "sentence"]) is False and any(f"no {kind} clips in train or in dev" in line for line in s.lines), broken
    build(sentences_in_dev=0)
    assert lab.route37_preflight(session(), ["word"]) is True
    s = session()
    assert lab.route37_preflight(s, ["word", "sentence"]) is False and any("routing_arms=word leaves out" in line for line in s.lines)
    build()
    assert not (app / lab.EX / "M01" / "test.jsonl").exists()
    (app / lab.EX / "M04" / "dev.jsonl").unlink()
    s = session()
    assert lab.route37_preflight(s, ["word"]) is False and any("MISSING the train and dev manifests" in line for line in s.lines)
