"""Section 3.1 Z-score normalization: z = (x - mu) / sigma, statistics from the TRAIN split only.

Raw (physical-unit) maps are stored on disk. Statistics are computed over occupied pixels/cells only so the
large empty background does not dominate; empty pixels stay exactly 0 after normalization (= the mean).
"""
import json
from pathlib import Path

import cv2
import numpy as np

from config import CAMERAS, OBJ_FEATURES, OBJ_MASK_FEATURES
from io_utils import load_split
from rasterize import NEW_H, PAD_Y

# key -> index of the channel whose value > 0 marks an occupied pixel/cell
MAP_KEYS = {
    "lidar_map": 0, "radar_map": 0,
    "bev_lidar_front": 0, "bev_lidar_full360": 0,
}
EPS = 1e-6


class RunningStats:
    def __init__(self, n_ch):
        self.n, self.s, self.ss = 0, np.zeros(n_ch), np.zeros(n_ch)

    def update(self, x):  # x: (N, C)
        x = np.asarray(x, np.float64)
        self.n += len(x)
        self.s += x.sum(0)
        self.ss += (x ** 2).sum(0)

    def result(self):
        if self.n == 0:
            return np.zeros_like(self.s), np.ones_like(self.s)
        mean = self.s / self.n
        var = np.maximum(self.ss / self.n - mean ** 2, 0.0)
        return mean, np.maximum(np.sqrt(var), EPS)


def occupied_values(arr, mask_ch):
    """arr (..., C, H, W) -> (N_occupied, C)."""
    a = np.moveaxis(arr.astype(np.float32), -3, -1).reshape(-1, arr.shape[-3])
    return a[a[:, mask_ch] > 0]


def image_content_values(img):
    """uint8 (H, W, 3) letterboxed -> content pixels only (padding excluded), (N, 3)."""
    return img[PAD_Y:PAD_Y + NEW_H].reshape(-1, 3)


def compute_stats(out_dir, split="train", image_stride=4):
    out_dir = Path(out_dir)
    acc, img_acc, obj_acc = {}, RunningStats(3), RunningStats(len(OBJ_FEATURES))
    n_samples = 0
    assign = load_split(out_dir)
    for f in sorted((out_dir / "sensors").glob("*.npz")):
        if assign[f.stem] != split:
            continue
        d = np.load(f)
        n_samples += 1
        tok = f.stem
        for key, mc in MAP_KEYS.items():
            v = occupied_values(d[key], mc)
            acc.setdefault(key, RunningStats(v.shape[1])).update(v)
        for cam in CAMERAS:
            img = cv2.cvtColor(cv2.imread(str(out_dir / "images" / f"{tok}_{cam}.jpg")), cv2.COLOR_BGR2RGB)
            img_acc.update(image_content_values(img)[::image_stride].astype(np.float64) / 255.0)
        if len(d["obj_feats"]):
            obj_acc.update(d["obj_feats"])
    stats = {"split": split, "n_samples": n_samples,
             "image": dict(zip(("mean", "std"), (x.tolist() for x in img_acc.result())))}
    for key, a in acc.items():
        stats[key] = dict(zip(("mean", "std"), (x.tolist() for x in a.result())))
    om, os_ = obj_acc.result()
    om, os_ = om.copy(), os_.copy()
    for i, name in enumerate(OBJ_FEATURES):
        if name in OBJ_MASK_FEATURES:
            om[i], os_[i] = 0.0, 1.0
    stats["obj"] = {"mean": om.tolist(), "std": os_.tolist(), "features": OBJ_FEATURES}
    (out_dir / "norm_stats.json").write_text(json.dumps(stats, indent=1))
    return stats


def zscore_map(arr, stat, mask_ch):
    """arr (..., C, H, W) raw -> z-scored float32; empty pixels -> 0."""
    mean = np.asarray(stat["mean"], np.float32)[:, None, None]
    std = np.asarray(stat["std"], np.float32)[:, None, None]
    a = arr.astype(np.float32)
    occ = a[..., mask_ch:mask_ch + 1, :, :] > 0
    return np.where(occ, (a - mean) / std, 0.0).astype(np.float32)


def zscore_image(img_u8, stat):
    """uint8 (H, W, 3) -> float32 (3, H, W); padding rows set to 0."""
    x = img_u8.astype(np.float32) / 255.0
    x = (x - np.asarray(stat["mean"], np.float32)) / np.asarray(stat["std"], np.float32)
    x[:PAD_Y] = 0.0
    x[PAD_Y + NEW_H:] = 0.0
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def zscore_obj(feats, stat):
    return ((feats - np.asarray(stat["mean"], np.float32)) / np.asarray(stat["std"], np.float32)).astype(np.float32)


def build_inputs(out_dir, token, stats, cam=None):
    """Load one processed sample and return z-scored, pixel-aligned tensors for fusion/YOLO/LSTM stages.

    Returns dict with image (3,640,640), lidar (3,..), radar (4,..), occupancy masks, BEV maps, object features.
    `cam` is a camera name; None returns all five stacked on a leading axis.
    """
    out_dir = Path(out_dir)
    d = np.load(out_dir / "sensors" / f"{token}.npz")
    cams = CAMERAS if cam is None else [cam]
    idx = [CAMERAS.index(c) for c in cams]
    imgs = []
    for c in cams:
        im = cv2.cvtColor(cv2.imread(str(out_dir / "images" / f"{token}_{c}.jpg")), cv2.COLOR_BGR2RGB)
        imgs.append(zscore_image(im, stats["image"]))
    res = {
        "image": np.stack(imgs),
        "lidar": zscore_map(d["lidar_map"][idx], stats["lidar_map"], 0),
        "radar": zscore_map(d["radar_map"][idx], stats["radar_map"], 0),
        "lidar_mask": (d["lidar_map"][idx][:, :1] > 0).astype(np.float32),
        "radar_mask": (d["radar_map"][idx][:, :1] > 0).astype(np.float32),
        "obj_feats": zscore_obj(d["obj_feats"], stats["obj"]),
        "obj_class": d["obj_class"], "obj_instance": d["obj_instance"],
    }
    for k in ("bev_lidar_front", "bev_lidar_full360"):
        res[k] = zscore_map(d[k], stats[k], MAP_KEYS[k])
    if cam is not None:
        for k in ("image", "lidar", "radar", "lidar_mask", "radar_mask"):
            res[k] = res[k][0]
    return res
