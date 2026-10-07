"""Fusion dataset: one item = one camera view that has at least one labelled object, with its LiDAR BEV and radar map.

Views without any object label (about the sample-collection gaps) are dropped, together with their sensor data.
"""
import json

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import fusion_paths as fp
from calibration import sensor_to_vehicle
from config import CAMERAS, LIDAR
from io_utils import load_metadata, load_split
from normalization import zscore_image, zscore_map


def labelled_views(split, out_dir=fp.OUT):
    """[(sample_token, camera)] in the split whose label file has at least one object."""
    manifest = load_split(out_dir)
    views = []
    for tok in sorted(t for t, s in manifest.items() if s == split):
        for cam in CAMERAS:
            if (out_dir / "labels" / f"{tok}_{cam}.txt").read_text().strip():
                views.append((tok, cam))
    return views


class FusionViews(Dataset):
    def __init__(self, split, out_dir=fp.OUT, samples_root=fp.SAMPLES_ROOT, preset="full360"):
        self.out, self.root, self.preset = out_dir, samples_root, preset
        self.stats = json.loads((out_dir / "norm_stats.json").read_text())
        self.views = labelled_views(split, out_dir)
        self._z0 = {}

    def __len__(self):
        return len(self.views)

    def lidar_z0(self, tok):
        if tok not in self._z0:
            self._z0[tok] = float(sensor_to_vehicle(load_metadata(self.root / tok), LIDAR).T[2])
        return self._z0[tok]

    def __getitem__(self, i):
        tok, cam = self.views[i]
        ci = CAMERAS.index(cam)
        d = np.load(self.out / "sensors" / f"{tok}.npz")
        img = cv2.cvtColor(cv2.imread(str(self.out / "images" / f"{tok}_{cam}.jpg")), cv2.COLOR_BGR2RGB)

        radar_raw = d["radar_map"][ci]
        radar = np.concatenate([zscore_map(radar_raw, self.stats["radar_map"], 0), (radar_raw[:1] > 0).astype(np.float32)])

        key = f"bev_lidar_{self.preset}"
        bev_raw = d[key]
        bev = np.concatenate([zscore_map(bev_raw, self.stats[key], 0), (bev_raw > 0).astype(np.float32)])

        lab = np.loadtxt(self.out / "labels" / f"{tok}_{cam}.txt", ndmin=2, dtype=np.float32).reshape(-1, 5)
        return {
            "image": torch.from_numpy(zscore_image(img, self.stats["image"])),
            "radar": torch.from_numpy(radar),
            "bev": torch.from_numpy(bev),
            "bev_gray": torch.from_numpy(bev_raw.astype(np.float32)),
            "z0": torch.tensor(self.lidar_z0(tok), dtype=torch.float32),
            "T": torch.from_numpy(d["T_cam_from_vehicle"][ci].astype(np.float32)),
            "K": torch.from_numpy(d["K_canvas"][ci].astype(np.float32)),
            "labels": torch.from_numpy(lab),
            "token": tok, "camera": cam,
        }


INPUT_KEYS = ("image", "radar", "bev", "bev_gray", "z0", "T", "K")


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def model_inputs(batch):
    return tuple(batch[k] for k in INPUT_KEYS)


def collate(items):
    """Stack tensors; labels become Ultralytics-style batch_idx / cls / bboxes."""
    out = {k: torch.stack([it[k] for it in items]) for k in ("image", "radar", "bev", "bev_gray", "z0", "T", "K")}
    lab = torch.cat([it["labels"] for it in items])
    idx = torch.cat([torch.full((len(it["labels"]),), j, dtype=torch.float32) for j, it in enumerate(items)])
    out.update(batch_idx=idx, cls=lab[:, :1], bboxes=lab[:, 1:],
               token=[it["token"] for it in items], camera=[it["camera"] for it in items])
    return out
