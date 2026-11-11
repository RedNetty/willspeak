#!/usr/bin/env python3
"""Decode each clip several ways, keep every distinct answer, and score each answer against the audio. Lab tests 36 and 39."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import labkit as lab
import nemo_hyp
from willspeak.artifacts import contained_file, file_record, timestamp, versions
from willspeak.normalize import normalize, stock_form

DEFAULT_VIEWS = "greedy,beam4,beam8,tempo1.1,tempo1.2,tempo1.3,pitch0.9,pitch0.95,pitch1.05"
VIEW = re.compile(r"(greedy|beam|tempo|pitch)(\d+(?:\.\d+)?)?$")
GPU_NEEDS_GB = 3.5


def parse_views(text: str) -> list[tuple[str, str, float]]:
    views: list[tuple[str, str, float]] = []
    for name in [v for v in re.split(r"[,\s]+", text) if v]:
        m = VIEW.fullmatch(name)
        if not m:
            raise ValueError(f"View {name!r} is not greedy, beamN, tempoF or pitchF (for example beam4, tempo1.15, pitch0.95)")
        kind, number = m.group(1), m.group(2)
        if kind == "greedy":
            if number:
                raise ValueError("greedy takes no number")
            value = 0.0
        elif kind == "beam":
            if not number or "." in number or int(number) < 2:
                raise ValueError(f"{name!r}: the beam width is a whole number of at least 2")
            value = float(int(number))
        else:
            if not number or not 0.5 <= float(number) <= 2.0 or float(number) == 1.0:
                raise ValueError(f"{name!r}: the factor is between 0.5 and 2 and not 1")
            value = float(number)
        if name in [v[0] for v in views]:
            raise ValueError(f"View {name!r} is listed twice")
        views.append((name, kind, value))
    if "greedy" in [v[0] for v in views]:
        views.sort(key=lambda v: v[0] != "greedy")
    else:
        views.insert(0, ("greedy", "greedy", 0.0))
    return views


def read_unlabelled(path: str | Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with lab.here(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict) or raw.get("id") is None or not isinstance(raw.get("audio"), str):
                raise ValueError(f"Manifest rows need an id and an audio path: {path}")
            key = str(raw["id"])
            if key in out:
                raise ValueError(f"Duplicate manifest id {key}: {path}")
            out[key] = {"id": key, "audio": raw["audio"], "speaker": str(raw.get("speaker", ""))}
    return out


def pool_row(clip_id: str, row: dict, labelled: bool, samples: int, frames: int, empty: float, cands: list[dict], dropped: int, seconds: dict) -> dict:
    return {"id": clip_id, "speaker": str(row.get("speaker", "")), "kind": lab.clip_kind(row) if labelled else "unknown",
            "ref": normalize(row["text"]) if labelled else "", "duration_s": round(samples / 16000, 3), "frames": frames, "nll_empty": empty,
            "candidates": cands, "dropped": dropped, "seconds": seconds}


def lm_text(norm: str) -> str:
    return stock_form(" ".join("I" if w == "i" else w for w in norm.split()))


def stretch(audio, ratio: float):
    from scipy.signal import resample_poly

    frac = Fraction(ratio).limit_denominator(200)
    return resample_poly(audio.astype("float64"), frac.numerator, frac.denominator).astype("float32")


def read_clip(path):
    import soundfile as sf

    from willspeak.audio import to_mono_16k

    data, rate = sf.read(str(path), dtype="float32", always_2d=False)
    return to_mono_16k(data, rate)


def group_candidates(found: dict[str, list[str]], nlls: list[float], empty: float) -> list[dict]:
    groups: dict[str, dict] = {}
    for (text, views), nll in zip(found.items(), nlls):
        norm = normalize(text)
        value = empty if not norm else nll
        g = groups.setdefault(norm, {"norm": norm, "text": text, "nll": value, "views": []})
        if value < g["nll"]:
            g["text"], g["nll"] = text, value
        g["views"] += [v for v in views if v not in g["views"]]
    out = []
    for g in groups.values():
        out.append({"norm": g["norm"], "text": g["text"], "nll": None if g["nll"] == float("inf") else g["nll"], "views": g["views"],
                    "words": len(g["norm"].split()), "lm_text": lm_text(g["norm"])})
    return out


def prune(cands: list[dict], keep_within: float, line: float | None = None) -> tuple[list[dict], int]:
    if keep_within <= 0:
        return cands, 0
    best = min(c["nll"] for c in cands if c["nll"] is not None) if line is None else line
    kept = [c for i, c in enumerate(cands) if i == 0 or (c["nll"] is not None and c["nll"] <= best + keep_within)]
    return kept, len(cands) - len(kept)


RESERVED_NAMES = re.compile(r"(greedy|beam\d*|tempo[\d.]*|pitch[\d.]*)$")
EXTRA_NAME = re.compile(r"[a-z][a-z0-9-]{0,30}$")


@dataclass(frozen=True)
class ExtraModel:
    name: str
    base: str
    adapter: str | None = None

    def namespace(self) -> SimpleNamespace:
        pretrained = self.base.partition(":")[2] if self.base.startswith("pretrained:") else None
        return SimpleNamespace(stock=self.base == "stock", pretrained=pretrained,
                               nemo=None if self.base == "stock" or pretrained else self.base, adapter=self.adapter, mix=[])

    def tag(self, view: str) -> str:
        return self.name if view == "greedy" else f"{self.name}/{view}"


def parse_extra_models(items: list[str]) -> list[ExtraModel]:
    out: list[ExtraModel] = []
    for item in items:
        name, sep, rest = item.partition("=")
        base, _comma, adapter = rest.partition(",")
        name, base, adapter = name.strip(), base.strip(), adapter.strip()
        if not sep or not base:
            raise ValueError(f"--extra-model is NAME=BASE or NAME=BASE,ADAPTER (BASE is stock, pretrained:<NeMo name> or a .nemo path), not {item!r}")
        if not EXTRA_NAME.fullmatch(name) or RESERVED_NAMES.fullmatch(name):
            raise ValueError(f"Extra model name {name!r}: lower-case letters, digits and hyphens, starting with a letter, and not a view name")
        if name in [m.name for m in out]:
            raise ValueError(f"Extra model {name!r} is listed twice")
        out.append(ExtraModel(name, base, adapter or None))
    return out


def parse_extra_views(text: str) -> list[tuple[str, str, float]]:
    views = parse_views(text)
    if any(kind not in ("greedy", "beam") for _, kind, _ in views):
        raise ValueError("An extra model proposes its greedy decode and, optionally, beamN answers (for example greedy,beam4), not tempo or pitch views")
    return views


def merge_extras(base: list[dict], extra_found: dict[str, list[str]], score, keep_within: float) -> tuple[list[dict], int, int]:
    cands = copy.deepcopy(base)
    known = {c["norm"]: c for c in cands}
    forms: dict[str, list[tuple[str, list[str]]]] = {}
    for text, tags in extra_found.items():
        norm = normalize(text)
        if norm in known:
            known[norm]["views"] += [t for t in dict.fromkeys(tags) if t not in known[norm]["views"]]
        else:
            forms.setdefault(norm, []).append((text, list(dict.fromkeys(tags))))
    texts = [text for group in forms.values() for text, _tags in group]
    new: list[dict] = []
    if texts:
        nlls, empty = score(texts)
        scored = dict(zip(texts, nlls))
        for norm, group in forms.items():
            values = [empty if not norm else scored[text] for text, _tags in group]
            entry = {"norm": norm, "words": len(norm.split()), "lm_text": lm_text(norm)}
            entry["forms"] = [{"text": text, "nll": None if v == float("inf") else v, "views": tags} for (text, tags), v in zip(group, values)]
            new.append(entry)
            choose_form(entry)
            if len(group) == 1:
                del entry["forms"]
    line = min(c["nll"] for c in cands if c["nll"] is not None)
    kept, dropped = prune(cands + new, keep_within, line)
    return kept, dropped, len(kept) - len(cands)


def choose_form(entry: dict) -> None:
    best = min(entry["forms"], key=lambda f: float("inf") if f["nll"] is None else f["nll"])
    entry["text"], entry["nll"] = best["text"], best["nll"]
    entry["views"] = list(dict.fromkeys(v for f in entry["forms"] for v in f["views"]))


def check_base_pool(data: dict, model_files: dict, sigma: float, durations: list[int], keep_within: float, kinds: list[str]) -> None:
    if data.get("format") != "willspeak-lm-pool-v1":
        raise ValueError("--base-pool is not a candidate pool written by lm_pool.py")
    old = {k: v.get("sha256") for k, v in (data.get("model_files") or {}).items()}
    new = {k: v.get("sha256") for k, v in model_files.items()}
    if old != new:
        raise ValueError("--base-pool was made by a different primary model (its model_files differ from the model loaded now); "
                         "its audio scores are not comparable with new ones. Build the pool again with this model, or pass the pool this model made")
    if abs(float(data.get("sigma", sigma)) - sigma) > 1e-9 or [int(d) for d in data.get("durations", durations)] != list(durations):
        raise ValueError("--base-pool was scored with other TDT settings (sigma or durations)")
    if float(data.get("keep_within", keep_within)) != keep_within:
        raise ValueError(f"--base-pool kept answers within {data.get('keep_within')} nats; pass the same --keep-within")
    if sorted(data.get("kinds", kinds)) != sorted(kinds):
        raise ValueError("--base-pool covers other clip kinds than --kinds")


def subset_pool(data: dict, keep: list[str]) -> dict:
    if data.get("format") != "willspeak-lm-pool-v1":
        raise ValueError("Not a candidate pool written by lm_pool.py")
    have = [str(m["name"]) for m in data.get("extra_models", [])]
    unknown = sorted(set(keep) - set(have))
    if not have or unknown:
        raise ValueError(f"The pool has extra models {' '.join(have) or '(none)'}; it has no {' '.join(unknown) or 'extra model to keep'}")
    leave = set(have) - set(keep)
    out = copy.deepcopy(data)
    for row in out["rows"]:
        cands: list[dict] = []
        added = 0
        for c in row["candidates"]:
            if "forms" in c:
                c["forms"] = [f for f in c["forms"] if any(v.split("/")[0] in keep for v in f["views"])]
                if not c["forms"]:
                    continue
                choose_form(c)
                if len(c["forms"]) == 1:
                    del c["forms"]
            views = [v for v in c["views"] if v.split("/")[0] not in leave]
            if not views:
                continue
            c["views"] = views
            added += int(all(v.split("/")[0] in keep for v in views))
            cands.append(c)
        row["candidates"], row["extra_answers"] = cands, added
    out["extra_models"] = [m for m in out["extra_models"] if m["name"] in keep]
    out["decodes_per_clip"] = len(out["views"]) + len(keep) * len(out.get("extra_views", ["greedy"]))
    out["encoder_passes_per_clip"] = int(data.get("encoder_passes_per_clip", 0)) - len(leave)
    out["subset_of"] = {"extra_models": have, "kept": list(keep)}
    return out


def subset_main(subset_of: str, keep_extras: str | None, out: str) -> int:
    keep = [k for k in re.split(r"[,\s]+", keep_extras or "") if k]
    if not keep:
        raise ValueError("--keep-extras names the extra models to keep, for example seed18,seed19")
    lab.refuse_existing(out)
    data = json.loads(lab.here(subset_of).read_text(encoding="utf-8"))
    cut = subset_pool(data, keep)
    cut["created_at"] = timestamp()
    cut["provenance"] = lab.provenance(__file__, {"subset_of": subset_of})
    lab.save_json(out, cut)
    sizes = [len(r["candidates"]) for r in cut["rows"]]
    print(f"kept extra models {' '.join(keep)}: {sum(sizes) / len(sizes):.1f} answers per clip (the wider pool had "
          f"{sum(len(r['candidates']) for r in data['rows']) / len(data['rows']):.1f})")
    print(f"wrote {lab.here(out)}")
    return 0


class Decoder:
    def __init__(self, model, views: list[tuple[str, str, float]], device: str):
        import torch
        from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTBPEDecoding
        from omegaconf import open_dict

        self.torch, self.model, self.views, self.device = torch, model, views, device
        self.featurizer = model.preprocessor.featurizer
        self.hop = int(self.featurizer.hop_length)
        self.beams: dict[str, object] = {}
        for name, kind, value in views:
            if kind != "beam":
                continue
            cfg = copy.deepcopy(model.cfg.decoding)
            with open_dict(cfg):
                cfg.strategy = "malsd_batch"
                cfg.beam.beam_size = int(value)
                cfg.beam.return_best_hypothesis = False
                cfg.beam.allow_cuda_graphs = False
            self.beams[name] = RNNTBPEDecoding(decoding_cfg=cfg, decoder=model.decoder, joint=model.joint, tokenizer=model.tokenizer)

    def encode(self, audio, hop: int | None = None):
        torch = self.torch
        self.featurizer.hop_length = hop or self.hop
        try:
            with torch.no_grad():
                signal = torch.from_numpy(audio).unsqueeze(0).to(self.device)
                encoded, length = self.model.forward(input_signal=signal, input_signal_length=torch.tensor([len(audio)]).to(self.device))
        finally:
            self.featurizer.hop_length = self.hop
        return encoded, length

    def answers(self, decoding, encoded, length) -> list[str]:
        result = decoding.rnnt_decoder_predictions_tensor(encoder_output=encoded, encoded_lengths=length, return_hypotheses=True)
        first = result[0]
        return [nemo_hyp.text_of(h) for h in (first if isinstance(first, list) else [first])]

    def pool(self, audio) -> tuple[dict[str, list[str]], object, int]:
        encoded, length = self.encode(audio)
        found: dict[str, list[str]] = {}
        for name, kind, value in self.views:
            if kind == "greedy":
                texts = self.answers(self.model.decoding, encoded, length)
            elif kind == "beam":
                texts = self.answers(self.beams[name], encoded, length)
            elif kind == "tempo":
                texts = self.answers(self.model.decoding, *self.encode(audio, round(self.hop * value)))
            else:
                texts = self.answers(self.model.decoding, *self.encode(stretch(audio, 1.0 / value), round(self.hop / value)))
            for text in texts:
                found.setdefault(text.strip(), [])
                if name not in found[text.strip()]:
                    found[text.strip()].append(name)
        return found, encoded, int(length[0])


def free_gpu(torch) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def to_device(model, gpu: bool):
    model = model.cuda() if gpu else model.cpu()
    model.eval()
    if hasattr(model.joint, "set_fuse_loss_wer"):
        model.joint.set_fuse_loss_wer(False)
    return model


def decode_extras(extras: list[ExtraModel], views: list[tuple[str, str, float]], ids: list[str], audio_paths: dict, gpu: bool, device: str,
                  torch) -> tuple[dict[str, dict[str, list[str]]], list[dict]]:
    found: dict[str, dict[str, list[str]]] = {i: {} for i in ids}
    records: list[dict] = []
    for ex in extras:
        _, paths = nemo_hyp.model_inputs(ex.namespace())
        files = {k: file_record(p) for k, p in paths.items()}
        model, record = nemo_hyp.load_model(ex.namespace(), files)
        model = to_device(model, gpu)
        decoder = Decoder(model, views, device)
        started = time.monotonic()
        for clip_id in ids:
            answers, _encoded, _frames = decoder.pool(read_clip(audio_paths[clip_id]))
            for text, names in answers.items():
                tags = found[clip_id].setdefault(text, [])
                tags += [t for t in (ex.tag(n) for n in names) if t not in tags]
        print(f"  extra model {ex.name}: {len(ids)} clips decoded, {time.monotonic() - started:.0f} s", flush=True)
        records.append({"name": ex.name, "base": ex.base, "adapter": ex.adapter, "views": [v[0] for v in views], "model": record, "model_files": files})
        del decoder, model
        free_gpu(torch)
    return found, records


def main() -> int:
    cut = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    cut.add_argument("--subset-of")
    cut.add_argument("--keep-extras")
    cut.add_argument("--out")
    known, _rest = cut.parse_known_args()
    if known.subset_of:
        if not known.out:
            raise ValueError("--subset-of needs --out")
        return subset_main(known.subset_of, known.keep_extras, known.out)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    nemo_hyp.add_model_args(ap)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--views", default=DEFAULT_VIEWS,
                    help="the decodes that propose answers: greedy (always first), beamN (every answer of an N-wide beam search), "
                         "tempoF (the model hears the clip F times faster, pitch unchanged), pitchF (pitch times F, speed unchanged)")
    ap.add_argument("--extra-model", action="append", default=[], metavar="NAME=BASE[,ADAPTER]",
                    help="another recogniser whose answers join the pool (repeatable). BASE is stock, pretrained:<NeMo model name> or a .nemo path; "
                         "ADAPTER is an adapter.pt made on that base. Its answers are tagged NAME (greedy) or NAME/beamN in each candidate's views, "
                         "and are scored against the audio by the primary model, like every other answer")
    ap.add_argument("--extra-views", default="greedy",
                    help="what each extra model proposes: greedy, and optionally beamN (for example greedy,beam4)")
    ap.add_argument("--base-pool", help="an earlier pool made by this same primary model on these clips: its candidates and audio scores are copied, "
                                        "the views are not decoded again, and only the extra models' answers are added (needs --extra-model)")
    ap.add_argument("--subset-of", metavar="POOL",
                    help="no decoding: write the pool that only the extra models named by --keep-extras would have made, cut from this wider "
                         "pool (the model flags are not needed; --out is)")
    ap.add_argument("--keep-extras", metavar="NAME,NAME", help="with --subset-of: the extra models to keep")
    ap.add_argument("--keep-within", type=float, default=15.0,
                    help="drop answers whose audio score is more than this many nats worse than the best of the primary model's own answers for the "
                         "clip (the first decode always stays; 0 keeps everything)")
    ap.add_argument("--kinds", default="sentence,word", help="which clips are pooled: sentence, word or sentence,word (ignored with --no-reference)")
    ap.add_argument("--no-reference", action="store_true",
                    help="the clips have no transcript (lab test 38): read only id, audio and speaker, pool every clip, and write an empty reference "
                         "and kind 'unknown'; such pools can be picked from (lm_pick.py) but not tallied (lm_report.py)")
    ap.add_argument("--limit", type=int, default=0, help="pool only the first N clips (0 = all)")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads for torch (0 = its default)")
    ap.add_argument("--cpu", action="store_true", help="stay on the CPU even when a GPU is free")
    ap.add_argument("--out", required=True, help="pool .json to write (refuses to overwrite)")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    kinds = [k for k in a.kinds.split(",") if k]
    if not kinds or any(k not in lab.KINDS for k in kinds) or a.limit < 0 or a.threads < 0 or a.keep_within < 0:
        ap.error("--kinds must be sentence, word or sentence,word; --limit, --threads and --keep-within cannot be negative")
    views = parse_views(a.views)
    extras = parse_extra_models(a.extra_model)
    extra_views = parse_extra_views(a.extra_views)
    if a.base_pool and not extras:
        ap.error("--base-pool only adds other models' answers; give at least one --extra-model")
    if a.base_pool and a.limit:
        ap.error("--base-pool and --limit cannot be combined: the pool already fixes the clips")
    if a.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    lab.refuse_existing(a.out)
    labelled = not a.no_reference
    if a.base_pool and not labelled:
        ap.error("--base-pool compares reference words, so it cannot be combined with --no-reference")
    manifest = lab.read_manifest(a.manifest) if labelled else read_unlabelled(a.manifest)
    base_data = None
    base_rows: dict[str, dict] = {}
    if a.base_pool:
        base_data = json.loads(lab.here(a.base_pool).read_text(encoding="utf-8"))
        if base_data.get("format") != "willspeak-lm-pool-v1":
            raise ValueError(f"{a.base_pool} is not a candidate pool written by lm_pool.py")
        kinds = list(base_data.get("kinds", kinds))
        base_rows = {str(r["id"]): r for r in base_data.get("rows", [])}
        ids = list(base_rows)
        unknown = [i for i in ids if i not in manifest]
        if unknown:
            raise ValueError(f"--base-pool has {len(unknown)} clips that are not in {a.manifest}, for example {unknown[0]}")
        if any(normalize(manifest[i]["text"]) != base_rows[i]["ref"] for i in ids):
            raise ValueError(f"--base-pool and {a.manifest} give different reference words for some clips")
        views = parse_views(",".join(base_data.get("views") or [v[0] for v in views]))
    elif labelled:
        ids = [i for i, row in manifest.items() if lab.clip_kind(row) in kinds and normalize(row["text"])]
        ids = ids[: a.limit] if a.limit else ids
    else:
        ids = list(manifest)
        kinds = []
        ids = ids[: a.limit] if a.limit else ids
    if not ids:
        raise ValueError(f"No {' or '.join(kinds)} clips in {a.manifest}" if kinds else f"No clips in {a.manifest}")
    audio_root = lab.here(a.audio_root)
    audio_paths = {i: contained_file(audio_root, manifest[i]["audio"]) for i in ids}
    _, model_paths = nemo_hyp.model_inputs(a)
    model_files = {k: file_record(p) for k, p in model_paths.items()}

    import torch

    import phrase_score

    if a.threads:
        torch.set_num_threads(a.threads)
    gpu = torch.cuda.is_available() and not a.cpu
    device = "cuda" if gpu else "cpu"
    if torch.cuda.is_available():
        free = torch.cuda.mem_get_info()[0] / 2**30
        if free < GPU_NEEDS_GB:
            raise ValueError(f"The GPU has {free:.1f} GB free and the model needs about {GPU_NEEDS_GB:g}: close what is using it, or pass --cpu")
    extra_found: dict[str, dict[str, list[str]]] = {}
    extra_records: list[dict] = []
    if extras:
        print(f"on the {device}; extra models {' '.join(m.name for m in extras)} propose {' '.join(v[0] for v in extra_views)}; {len(ids)} clips")
        extra_found, extra_records = decode_extras(extras, extra_views, ids, audio_paths, gpu, device, torch)
    model, record = nemo_hyp.load_model(a, model_files)
    inner = getattr(getattr(model, "loss", None), "_loss", None)
    if not hasattr(inner, "durations") or not list(inner.durations):
        raise ValueError("This script needs a TDT model (its loss has no durations)")
    sigma, durations, blank = float(getattr(inner, "sigma", 0.0)), [int(d) for d in inner.durations], int(inner.blank)
    if base_data is not None:
        check_base_pool(base_data, model_files, sigma, durations, a.keep_within, kinds)
    model = to_device(model, gpu)
    decoder = Decoder(model, [("greedy", "greedy", 0.0)] if base_data is not None else views, device)
    print(f"on the {device}; views {' '.join(v[0] for v in views)}{' (copied from --base-pool)' if base_data is not None else ''}; {len(ids)} clips")

    rows = []
    started = time.monotonic()
    for n, clip_id in enumerate(ids):
        row = manifest[clip_id]
        audio = read_clip(audio_paths[clip_id])
        t0 = time.monotonic()
        dropped = 0
        if base_data is not None:
            old = base_rows[clip_id]
            encoded, length = decoder.encode(audio)
            frames, empty = int(length[0]), old["nll_empty"]
            if frames != old["frames"]:
                raise ValueError(f"Clip {clip_id} has {frames} frames now and {old['frames']} in --base-pool: it is not the same audio")
            cands, dropped = old["candidates"], int(old.get("dropped", 0))
            t1 = time.monotonic()
        else:
            found, encoded, frames = decoder.pool(audio)
            t1 = time.monotonic()
            nlls, empty = phrase_score.score_encoded(model, encoded, frames, list(found), sigma, durations, blank)
            cands = group_candidates(found, nlls, empty)
        added = 0
        if extras:
            def score(texts, encoded=encoded, frames=frames):
                return phrase_score.score_encoded(model, encoded, frames, texts, sigma, durations, blank)

            cands, more, added = merge_extras(cands, extra_found[clip_id], score, a.keep_within)
        else:
            cands, more = prune(cands, a.keep_within)
        dropped += more
        t2 = time.monotonic()
        out_row = pool_row(clip_id, row, labelled, len(audio), frames, empty, cands, dropped,
                           {"decode": round(t1 - t0, 3), "score": round(t2 - t1, 3)})
        out_row["extra_answers"] = added
        rows.append(out_row)
        if n % 25 == 24 or n + 1 == len(ids):
            print(f"  {n + 1}/{len(ids)} clips, {time.monotonic() - started:.0f} s", flush=True)
    sizes = [len(r["candidates"]) for r in rows]
    tail = ""
    if labelled:
        held = sum(1 for r in rows if any(c["norm"] == r["ref"] for c in r["candidates"]))
        first = sum(1 for r in rows if r["candidates"][0]["norm"] == r["ref"])
        tail = f"; the first decode is right on {first} of {len(rows)} clips, and some answer in the pool is right on {held}"
    print(f"{sum(sizes) / len(sizes):.1f} distinct answers kept per clip (most {max(sizes)}; {sum(r['dropped'] for r in rows)} dropped as more than "
          f"{a.keep_within:g} nats behind){tail}")
    if extras:
        new = sum(r["extra_answers"] for r in rows)
        print(f"the extra models added {new} answers that no view of the primary model had ({new / len(rows):.2f} per clip)")
    passes = 1 + sum(1 for v in views if v[1] in ("tempo", "pitch"))
    saved = {
        "format": "willspeak-lm-pool-v1", "created_at": timestamp(), "model": record, "views": [v[0] for v in views], "kinds": kinds, "labelled": labelled,
        "keep_within": a.keep_within, "device": device, "sigma": sigma, "durations": durations, "rows": rows, "model_files": model_files, "packages": versions(),
        "decodes_per_clip": len(views) + len(extras) * len(extra_views), "encoder_passes_per_clip": passes + len(extras),
        "provenance": lab.provenance(__file__, {"manifest": a.manifest, **({"base_pool": a.base_pool} if a.base_pool else {})})}
    if extras:
        saved["extra_models"] = extra_records
        saved["extra_views"] = [v[0] for v in extra_views]
        saved["base_pool"] = {"path": a.base_pool, "sha256": file_record(lab.here(a.base_pool))["sha256"]} if a.base_pool else None
    lab.save_json(a.out, saved)
    print(f"wrote {lab.here(a.out)}")
    return 0


if __name__ == "__main__":
    import _cli

    sys.exit(_cli.run(main, "lm_pool", console_filter_factory=None if "--verbose" in sys.argv else _cli.NemoConsole))
