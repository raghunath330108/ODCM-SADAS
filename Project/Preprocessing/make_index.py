"""Build splits.json (manifest split + scene-grouped alternative) and tracks.json (instance tracks for the LSTM).

Usage: python make_index.py [--metadata-dir ../../metadata_json] [--out ../Output] [--seed 42]
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from config import CLASSES, DEFAULT_METADATA_DIR, DEFAULT_OUT

TARGET = {"train": 0.70, "val": 0.15, "test": 0.15}


def load_all(meta_dir):
    metas = []
    metadata_files = sorted(Path(meta_dir).rglob("metadata.json"))
    if not metadata_files:
        raise FileNotFoundError(f"No sample metadata.json files found under {meta_dir}")
    for f in metadata_files:
        m = json.loads(f.read_text(encoding="utf8"))
        metas.append({"token": m["sample_token"], "scene": m["scene_token"], "timestamp": m["timestamp"],
                      "split": m["split"], "anns": m["annotations"]})
    return metas


def grouped_split(metas, seed):
    """Whole scenes go to one split (no temporal leakage); greedy fill toward 70/15/15 of samples, with class balance."""
    by_scene = defaultdict(list)
    for m in metas:
        by_scene[m["scene"]].append(m)
    scenes = sorted(by_scene)
    random.Random(seed).shuffle(scenes)
    scenes.sort(key=lambda s: -len(by_scene[s]))  # place big scenes first
    total = len(metas)
    cls_total = Counter(a["class_name"] for m in metas for a in m["anns"])
    fill = {k: 0 for k in TARGET}
    cls_fill = {k: Counter() for k in TARGET}
    assign = {}
    for s in scenes:
        n = len(by_scene[s])
        sc = Counter(a["class_name"] for m in by_scene[s] for a in m["anns"])

        def deficit(k):  # remaining share of the target, in samples and per class
            size = TARGET[k] - fill[k] / total
            cl = sum(TARGET[k] - cls_fill[k][c] / cls_total[c] for c in CLASSES if cls_total[c])
            return size + 0.5 * cl / len(CLASSES)
        k = max(TARGET, key=deficit)
        assign[s] = k
        fill[k] += n
        cls_fill[k].update(sc)
    return {m["token"]: assign[m["scene"]] for m in metas}


def build_tracks(metas):
    tr = defaultdict(list)
    for m in sorted(metas, key=lambda m: (m["scene"], m["timestamp"])):
        for a in m["anns"]:
            tr[a["instance_token"]].append({"sample": m["token"], "timestamp": m["timestamp"], "ann": a["token"],
                                            "class": a["class_name"], "scene": m["scene"]})
    return {k: v for k, v in tr.items()}


def summarize(name, assign, metas):
    sc = defaultdict(set)
    for m in metas:
        sc[m["scene"]].add(assign[m["token"]])
    leak = sum(len(v) > 1 for v in sc.values())
    cnt = Counter(assign.values())
    ann = {k: Counter() for k in TARGET}
    for m in metas:
        ann[assign[m["token"]]].update(a["class_name"] for a in m["anns"])
    print(f"{name}: samples={dict(cnt)} scenes_spanning_multiple_splits={leak}/{len(sc)}")
    for k in TARGET:
        print(f"   {k:5s} " + " ".join(f"{c}={ann[k][c]}" for c in CLASSES))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata-dir", default=str(DEFAULT_METADATA_DIR))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    metas = load_all(a.metadata_dir)
    manifest = {m["token"]: m["split"] for m in metas}
    grouped = grouped_split(metas, a.seed)
    tracks = build_tracks(metas)
    summarize("manifest (primary, sample-level)", manifest, metas)
    summarize("grouped  (scene-level, leakage-free)", grouped, metas)
    lens = Counter(min(len(v), 10) for v in tracks.values())
    print(f"tracks={len(tracks)}  length histogram (10 = 10+): {dict(sorted(lens.items()))}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "splits.json").write_text(json.dumps({"manifest": manifest, "scene_grouped": grouped}, indent=1))
    (out / "tracks.json").write_text(json.dumps(tracks))
