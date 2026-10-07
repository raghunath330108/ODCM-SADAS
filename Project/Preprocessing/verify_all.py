"""Full-set verification: aggregate checks over all processed samples, build_inputs test per split, overlays for a few samples.

Usage: python verify_all.py [--out ../Output] [--samples-root ../../Samples/senses] [--per-split 2]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from config import CAMERAS, DEFAULT_OUT, DEFAULT_SAMPLES_ROOT, OBJ_FEATURES
from io_utils import load_metadata, load_split
from normalization import build_inputs
from objects import box_in_vehicle, points_in_box
from rasterize import box_to_canvas
from verify import check_sample


def aggregate(out, root):
    ours, ref, rad, lid = [], [], [], []
    tot = hit = nlab = no_label = 0
    i_box = OBJ_FEATURES.index("n_lidar_in_box")
    for f in sorted((out / "sensors").glob("*.npz")):
        d, m = np.load(f), load_metadata(root / f.stem)
        ours += list(d["obj_feats"][:, i_box])
        ref += [a["num_lidar_pts"] for a in m["annotations"]]
        lid.append(len(d["lidar_pts"]))
        rad.append(len(d["radar_pts"]))
        n = sum(len([l for l in (out / "labels" / f"{f.stem}_{c}.txt").read_text().splitlines() if l.strip()]) for c in CAMERAS)
        nlab += n
        no_label += n == 0
        for k in ("lidar_map", "radar_map", "bev_lidar_front", "bev_lidar_full360", "obj_feats"):
            assert np.isfinite(d[k].astype(np.float32)).all(), (f.stem, k)
        P = d["lidar_pts"][:, :3].astype(np.float64)
        Tm, K = d["T_cam_from_vehicle"], d["K_canvas"]
        for ci, cam in enumerate(CAMERAS):
            for a in m["annotations"]:
                cb = a.get("camera_boxes", {}).get(cam)
                if not cb or cb["visible_fraction"] < 0.8:
                    continue
                c, _, R = box_in_vehicle(m, a)
                mk = points_in_box(P, c, R, a["size"])
                if not mk.any():
                    continue
                X = P[mk] @ Tm[ci][:3, :3].T + Tm[ci][:3, 3]
                X = X[X[:, 2] > 0.5]
                u = K[ci][0, 0] * X[:, 0] / X[:, 2] + K[ci][0, 2]
                v = K[ci][1, 1] * X[:, 1] / X[:, 2] + K[ci][1, 2]
                x1, y1, x2, y2 = box_to_canvas(cb["bbox_xyxy_clipped"])
                tot += len(u)
                hit += int(((u >= x1 - 4) & (u <= x2 + 4) & (v >= y1 - 4) & (v <= y2 + 4)).sum())
    ours, ref = np.array(ours), np.array(ref)
    rel = np.abs(ours - ref) / np.maximum(ref, 1)
    print(f"samples={len(lid)} objects={len(ref)} corr={np.corrcoef(ours, ref)[0, 1]:.3f} "
          f"within10%/+-2={np.mean((rel <= .1) | (abs(ours - ref) <= 2)):.2f}")
    print(f"projection inside 2D box: {hit / tot:.4f} ({tot} pts)")
    print(f"yolo labels={nlab} samples_without_labels={no_label}")
    print(f"lidar pts/sample min={min(lid)}; radar pts/sample min={min(rad)} zero_radar={sum(r == 0 for r in rad)}")
    return hit / tot > 0.95 and no_label == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--samples-root", default=str(DEFAULT_SAMPLES_ROOT))
    ap.add_argument("--per-split", type=int, default=2)
    a = ap.parse_args()
    out, root = Path(a.out), Path(a.samples_root)
    ok = aggregate(out, root)

    stats = json.loads((out / "norm_stats.json").read_text())
    manifest = load_split(out)
    for split in ("train", "val", "test"):
        toks = [t for t, s in sorted(manifest.items()) if s == split][:a.per_split]
        for t in toks:
            x = build_inputs(out, t, stats)
            assert x["image"].shape == (5, 3, 640, 640) and x["lidar"].shape == (5, 3, 640, 640)
            assert x["radar"].shape == (5, 4, 640, 640)
            assert all(np.isfinite(v).all() for v in x.values() if v.dtype.kind == "f")
            print(f"build_inputs OK [{split}] {t}")
            ok &= check_sample(out, t, load_metadata(root / t))
    print("\nOVERALL:", "PASS" if ok else "CHECK FAILURES")
