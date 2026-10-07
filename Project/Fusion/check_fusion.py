"""Checks for the 3.3 fusion stage (no training): python check_fusion.py [--n-align 24] [--batch 2]"""
import argparse

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from ultralytics import YOLO

import fusion_paths as fp
from config import CAMERAS
from fusion_data import FusionViews, collate, labelled_views
from fusion_model import TAPS, FusionYOLO
from geometry import bev_points, lift_to_camera, project


def check_dataset():
    print("[1] labelled views (kept) per split")
    ok = True
    for split in ("train", "val", "test"):
        v = labelled_views(split)
        print(f"    {split:5s}: {len(v)} views from {len({t for t, _ in v})} samples (of {len({t for t, _ in v}) * 5} possible)")
        ok &= len(v) > 0
    return ok


def check_alignment(ds, n):
    """Project full-resolution BEV cells to the canvas; compare with the independent image-plane LiDAR map."""
    print("[2] BEV cells projected to the camera vs image-plane LiDAR map (independent rasterization)")
    rng = np.random.default_rng(0)
    hits, tot, lift_cov = 0, 0, []
    for i in rng.choice(len(ds), min(n, len(ds)), replace=False):
        it = ds[i]
        tok, cam = it["token"], it["camera"]
        pts, valid, _ = bev_points(it["bev_gray"][None], ds.preset, it["z0"][None], 1)
        u, v, _, ok = project(pts, it["T"][None], it["K"][None])
        sel = (ok & valid)[0]
        u, v = u[0][sel].round().long().numpy(), v[0][sel].round().long().numpy()
        depth = np.load(fp.OUT / "sensors" / f"{tok}.npz")["lidar_map"][CAMERAS.index(cam)][0] > 0
        near = cv2.dilate(depth.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        inside = (u >= 0) & (u < 640) & (v >= 0) & (v < 640)
        hits += int(near[v[inside], u[inside]].sum())
        tot += int(inside.sum())
        feat = torch.ones(1, 1, pts.shape[1])
        lift = lift_to_camera(feat, pts, valid, it["T"][None], it["K"][None], 8)[0, 0] > 0
        lift_cov.append(float(lift.float().mean()))
    frac = hits / max(tot, 1)
    print(f"    BEV cells landing on/near a LiDAR-map pixel: {frac:.3f} ({tot} cells); P3 grid cells reached: {np.mean(lift_cov):.3f}")
    return frac > 0.85


def check_transfer(model, batch):
    """Camera-only path must reproduce the original pretrained YOLO11s neck outputs; also report weight transfer."""
    print("[3] YOLO11s weight transfer")
    n, tot = len(model.transferred), model.total_params
    print(f"    transferred tensors: {n}/{tot} (rest = Detect class head, nc 80 -> 6)")
    ref = YOLO(str(fp.YOLO_WEIGHTS)).model.eval()
    got = {}
    hooks = [ref.model[i].register_forward_hook(lambda m, a, o: got.__setitem__(m.i, o)) for i in (16, 19, 22)]
    model.eval()
    with torch.no_grad():
        ref(batch["image"])
        ours = {}
        hs = [model.layers[i].register_forward_hook(lambda m, a, o: ours.__setitem__(m.i, o)) for i in (16, 19, 22)]
        model(batch["image"], batch["radar"], batch["bev"], batch["bev_gray"], batch["z0"], batch["T"], batch["K"], fuse=False)
    for h in hooks + hs:
        h.remove()
    diff = max(float((got[i] - ours[i]).abs().max()) for i in (16, 19, 22))
    print(f"    camera-only neck outputs vs original YOLO11s: max abs diff = {diff:.2e}")
    return diff < 1e-4 and n > 0.9 * tot


def check_forward(model, batch):
    print("[4] fused forward pass")
    model.eval()
    args = (batch["image"], batch["radar"], batch["bev"], batch["bev_gray"], batch["z0"], batch["T"], batch["K"])
    with torch.no_grad():
        preds, aux = model(*args)
        shapes = [tuple(f.shape) for f in aux["fused"]]
        print(f"    fused P3/P4/P5: {shapes}; attention: {[tuple(a.shape) for a in aux['attention']]}")
        finite = all(torch.isfinite(f).all() for f in aux["fused"])
        out = preds[0] if isinstance(preds, (tuple, list)) else preds
        print(f"    detect output type: {type(preds).__name__}; first tensor {tuple(out.shape) if torch.is_tensor(out) else type(out).__name__}")
        model.set_weights(1.0, 0.0, 0.0)
        cam_only = model(*args)[1]["fused"][0]
        model.set_weights(1 / 3, 1 / 3, 1 / 3)
        equal = model(*args)[1]["fused"][0]
    changed = float((cam_only - equal).abs().mean()) > 0
    print(f"    weights (1,0,0) vs (1/3,1/3,1/3) change the fused feature: {changed}; all finite: {bool(finite)}")
    ok = shapes == [(len(batch["z0"]), c, 640 // s, 640 // s) for c, s in zip(model.channels, (8, 16, 32))]
    return ok and finite and changed


def count(module):
    return sum(p.numel() for p in module.parameters()) / 1e6


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-align", type=int, default=24)
    ap.add_argument("--batch", type=int, default=2)
    a = ap.parse_args()
    ok = check_dataset()
    ds = FusionViews("train")
    ok &= check_alignment(ds, a.n_align)
    batch = next(iter(DataLoader(ds, batch_size=a.batch, collate_fn=collate)))
    print(f"    batch: image {tuple(batch['image'].shape)} radar {tuple(batch['radar'].shape)} bev {tuple(batch['bev'].shape)} "
          f"labels {tuple(batch['bboxes'].shape)}")
    model = FusionYOLO()
    ok &= check_transfer(model, batch)
    ok &= check_forward(model, batch)
    print(f"params (M): total {count(model):.2f} | yolo {count(model.det):.2f} | lidar {count(model.lidar):.2f} | "
          f"radar {count(model.radar):.2f} | fusion {count(model.levels):.2f}")
    print("\nOVERALL:", "PASS" if ok else "CHECK FAILURES")
