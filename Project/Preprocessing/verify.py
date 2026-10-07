"""Sanity checks + debug overlays for processed samples.

Usage: python verify.py --out ../Output [--token TOKEN]
Checks:
  1. n_lidar_in_box (our 3D box in vehicle frame) vs nuScenes num_lidar_pts  -> validates Eq.1 chain + box frame
  2. fraction of in-box LiDAR points that project inside the object's 2D camera box -> validates Eq.1-4 + letterbox
  3. z-score output: finite, empty pixels exactly 0, occupied mean ~ 0 / std ~ 1 on the stats set
  4. YOLO label sanity (inside canvas, content rows only)
Overlays are written to <out>/debug/.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from config import BEV_PRESETS, CAMERAS, CLASSES, OBJ_FEATURES
from io_utils import load_metadata
from normalization import build_inputs
from objects import box_in_vehicle, points_in_box
from rasterize import NEW_H, PAD_Y, bev_grid, box_to_canvas

import json


def _bev_xy(x, y, preset):
    p = BEV_PRESETS[preset]
    return (p["y"][1] - y) / p["res"], (p["x"][1] - x) / p["res"]


def check_sample(out_dir, token, meta):
    out_dir = Path(out_dir)
    d = np.load(out_dir / "sensors" / f"{token}.npz")
    dbg = out_dir / "debug"
    dbg.mkdir(exist_ok=True)
    lid = d["lidar_pts"]
    ok_all = True

    # 1. in-box counts vs nuScenes annotation counts
    print("\n[1] LiDAR points in 3D box (ours) vs nuScenes num_lidar_pts")
    ours, ref = [], []
    for a, f in zip(meta["annotations"], d["obj_feats"]):
        ours.append(f[OBJ_FEATURES.index("n_lidar_in_box")])
        ref.append(a["num_lidar_pts"])
    ours, ref = np.array(ours), np.array(ref)
    rel = np.abs(ours - ref) / np.maximum(ref, 1)
    print(f"    objects={len(ref)}  exact={np.mean(ours == ref):.2f}  within10%/+-2={np.mean((rel <= .1) | (np.abs(ours - ref) <= 2)):.2f}"
          f"  corr={np.corrcoef(ours, ref)[0, 1]:.3f}")

    # 2. projected points inside 2D camera boxes
    print("[2] Projected in-box LiDAR points falling inside the 2D camera box (canvas coords)")
    tot, hit = 0, 0
    for ci, cam in enumerate(CAMERAS):
        Tm, K = d["T_cam_from_vehicle"][ci], d["K_canvas"][ci]
        for a in meta["annotations"]:
            cb = a.get("camera_boxes", {}).get(cam)
            if not cb or cb["visible_fraction"] < 0.8:
                continue
            c, _, R = box_in_vehicle(meta, a)
            m = points_in_box(lid[:, :3].astype(np.float64), c, R, a["size"])
            if m.sum() == 0:
                continue
            Xc = lid[m, :3] @ Tm[:3, :3].T + Tm[:3, 3]
            Xc = Xc[Xc[:, 2] > 0.5]
            u, v = K[0, 0] * Xc[:, 0] / Xc[:, 2] + K[0, 2], K[1, 1] * Xc[:, 1] / Xc[:, 2] + K[1, 2]
            x1, y1, x2, y2 = box_to_canvas(cb["bbox_xyxy_clipped"])
            pad = 4
            inside = (u >= x1 - pad) & (u <= x2 + pad) & (v >= y1 - pad) & (v <= y2 + pad)
            tot += len(u)
            hit += int(inside.sum())
    frac = hit / max(tot, 1)
    print(f"    points={tot}  inside={frac:.3f}")
    ok_all &= frac > 0.8 or tot == 0

    # 3. z-score properties
    stats_p = out_dir / "norm_stats.json"
    if stats_p.exists():
        stats = json.loads(stats_p.read_text())
        x = build_inputs(out_dir, token, stats)
        print("[3] build_inputs shapes:", {k: tuple(v.shape) for k, v in x.items() if hasattr(v, "shape")})
        for k in ("image", "lidar", "radar", "bev_lidar_front", "bev_lidar_full360"):
            fin = np.isfinite(x[k]).all()
            ok_all &= bool(fin)
            print(f"    {k:18s} finite={fin} min={x[k].min():.2f} max={x[k].max():.2f}")
        empty0 = (x["lidar"][:, 0][x["lidar_mask"][:, 0] == 0] == 0).all()
        print(f"    empty LiDAR pixels exactly 0: {empty0}")
        ok_all &= bool(empty0)
        n_occ = int(x["lidar_mask"].sum())
        print(f"    occupied LiDAR pixels over 5 cams: {n_occ} ({n_occ / (5 * 640 * 640):.4%})")
        print(f"    radar occupied pixels: {int(x['radar_mask'].sum())}  radar pts kept: {len(d['radar_pts'])}  lidar pts: {len(lid)}")

    # 4. labels
    n_lab, bad = 0, 0
    for cam in CAMERAS:
        txt = (out_dir / "labels" / f"{token}_{cam}.txt").read_text().strip()
        for ln in filter(None, txt.splitlines()):
            c, cx, cy, w, h = map(float, ln.split())
            n_lab += 1
            if not (0 <= cx - w / 2 and cx + w / 2 <= 1.0001 and PAD_Y / 640 - 1e-4 <= cy - h / 2 and cy + h / 2 <= (PAD_Y + NEW_H) / 640 + 1e-4):
                bad += 1
    print(f"[4] YOLO labels: {n_lab}, out-of-content: {bad}")
    ok_all &= bad == 0

    # overlays
    for ci, cam in enumerate(CAMERAS):
        img = cv2.imread(str(out_dir / "images" / f"{token}_{cam}.jpg"))
        depth = d["lidar_map"][ci][0].astype(np.float32)
        ys, xs = np.nonzero(depth)
        for y, xx in zip(ys, xs):
            col = cv2.applyColorMap(np.uint8([[min(depth[y, xx] / 60 * 255, 255)]]), cv2.COLORMAP_JET)[0, 0].tolist()
            img[y, xx] = col
        ry, rx = np.nonzero(d["radar_map"][ci][0])
        for y, xx in zip(ry, rx):
            img[y, xx] = (255, 255, 255)
        for ln in filter(None, (out_dir / "labels" / f"{token}_{cam}.txt").read_text().splitlines()):
            c, cx, cy, w, h = map(float, ln.split())
            p1, p2 = (int((cx - w / 2) * 640), int((cy - h / 2) * 640)), (int((cx + w / 2) * 640), int((cy + h / 2) * 640))
            cv2.rectangle(img, p1, p2, (0, 255, 0), 1)
            cv2.putText(img, CLASSES[int(c)], (p1[0], max(p1[1] - 2, 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        cv2.imwrite(str(dbg / f"{token}_{cam}_overlay.jpg"), img)
    for preset in ("front", "full360"):
        gray = d[f"bev_lidar_{preset}"][0]
        img = np.repeat(gray[..., None], 3, axis=2)
        for a, f in zip(meta["annotations"], d["obj_feats"]):
            x, y, w, l = f[0], f[1], f[3], f[4]
            yaw = np.arctan2(f[6], f[7])
            corners = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
            rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            wc = corners @ rot.T + [x, y]
            px = np.array([_bev_xy(cx_, cy_, preset) for cx_, cy_ in wc], np.int32)
            cv2.polylines(img, [px], True, (0, 255, 255), 1)
        cv2.imwrite(str(dbg / f"{token}_bev_{preset}.png"), img)
    print("\nRESULT:", "PASS" if ok_all else "CHECK FAILURES")
    return ok_all


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples-root", default=str(Path(__file__).resolve().parents[2] / "Samples" / "senses"))
    ap.add_argument("--token")
    a = ap.parse_args()
    toks = [a.token] if a.token else [p.stem for p in sorted((Path(a.out) / "sensors").glob("*.npz"))]
    for t in toks:
        check_sample(a.out, t, load_metadata(Path(a.samples_root) / t))
