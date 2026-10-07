"""Section 3.5 data: per-object observation sequences (sliding windows over a track) built from export_detections.py output.

One step = fused P4 RoI feature (256) + detector class scores (6) + confidence (1) + box cx, cy, w, h on the canvas (4) + log1p(dt)
(1; time since the previous observation of the same object, so gaps in a track are visible instead of breaking it) = 268.
A window ends at an observation and holds up to N previous observations of the same object (causal; shorter at the start of a track,
right-padded, `length` gives the valid steps). The label is the object's class. At most one observation per (object, sample): the
camera view with the largest detected box.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from cls_paths import OUT  # noqa: F401  (also sets sys.path)
from config import CANVAS

STEP_DIM = 268


def load_observations(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def step_features(o):
    """Raw (unnormalised) per-observation vector without dt: (n, 267)."""
    b = o["box"]
    cxcywh = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1) / CANVAS
    return np.concatenate([o["feat"], o["scores"], o["conf"][:, None], cxcywh], 1).astype(np.float32)


def best_view_per_sample(o):
    """Indices keeping, per (instance, sample), the observation with the largest detected box."""
    area = (o["box"][:, 2] - o["box"][:, 0]) * (o["box"][:, 3] - o["box"][:, 1])
    best = {}
    for i, key in enumerate(zip(o["instance"], o["sample"])):
        if key not in best or area[i] > area[best[key]]:
            best[key] = i
    return np.asarray(sorted(best.values()), int)


def feature_stats(o, idx):
    """Mean / std (train split only) of the 267 raw features; dt is handled separately."""
    x = step_features(o)[idx]
    return x.mean(0), x.std(0) + 1e-6


def build_tracks(o, split, stats):
    """-> list of dicts {x (L, 268) normalised, label, det (L,) detector class, instance} sorted by time, for one split."""
    mean, std = stats
    idx = best_view_per_sample(o)
    idx = idx[o["split"][idx] == split]
    x_all = (step_features(o) - mean) / std
    tracks = []
    for inst in sorted(set(o["instance"][idx])):
        ii = idx[o["instance"][idx] == inst]
        ii = ii[np.argsort(o["timestamp"][ii])]
        dt = np.zeros(len(ii), np.float32)
        dt[1:] = np.diff(o["timestamp"][ii]) / 1e6
        x = np.concatenate([x_all[ii], np.log1p(dt)[:, None]], 1).astype(np.float32)
        tracks.append({"x": x, "label": int(o["gt_cls"][ii[0]]), "det": o["det_cls"][ii].astype(int), "instance": inst})
    return tracks


class Windows(Dataset):
    """Every observation of every track is the end of one window of up to `seq_len` steps."""

    def __init__(self, tracks, seq_len):
        self.seq_len, self.items = seq_len, []
        for ti, t in enumerate(tracks):
            for e in range(len(t["x"])):
                self.items.append((ti, e))
        self.tracks = tracks

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ti, e = self.items[i]
        t = self.tracks[ti]
        s = max(0, e - self.seq_len + 1)
        x = np.zeros((self.seq_len, t["x"].shape[1]), np.float32)
        x[:e - s + 1] = t["x"][s:e + 1]
        return torch.from_numpy(x), e - s + 1, t["label"], ti, e

    def last_of_track(self):
        """Dataset indices of each track's final window (the prediction that has seen the whole history)."""
        last = {}
        for i, (ti, e) in enumerate(self.items):
            last[ti] = i
        return [last[ti] for ti in sorted(last)]


def collate(items):
    x, n, y, ti, e = zip(*items)
    return torch.stack(x), torch.tensor(n), torch.tensor(y), torch.tensor(ti), torch.tensor(e)


def synthetic_observations(out_dir=OUT, dim=256, seed=0):
    """Dry-run stand-in with the REAL track structure (scenes, split, timestamps) and class-dependent random features.

    Only exercises the LSTM / GOA / metrics code when the untrained trial detector yields too few matches. Never use for results.
    """
    import json
    from config import CLASS_ID
    from io_utils import load_split
    rng = np.random.default_rng(seed)
    split = load_split(out_dir)
    tracks = json.loads((out_dir / "tracks.json").read_text())
    protos = rng.normal(size=(len(CLASS_ID), dim + 6))
    rec = {k: [] for k in ("split", "instance", "sample", "scene", "camera", "gt_cls", "det_cls", "conf", "box", "scores", "feat", "timestamp")}
    for inst, lst in tracks.items():
        for e in lst:
            c = CLASS_ID[e["class"]]
            v = protos[c] + rng.normal(scale=2.0, size=dim + 6)
            det = c if rng.random() < 0.8 else int(rng.integers(len(CLASS_ID)))
            x1, y1 = rng.uniform(0, 500, 2)
            for k, val in (("split", split[e["sample"]]), ("instance", inst), ("sample", e["sample"]), ("scene", e["scene"]),
                           ("camera", "CAM_FRONT"), ("gt_cls", c), ("det_cls", det), ("conf", float(rng.uniform(0.3, 0.9))),
                           ("box", np.array([x1, y1, x1 + rng.uniform(20, 120), y1 + rng.uniform(20, 120)], np.float32)),
                           ("scores", v[dim:].astype(np.float32)), ("feat", v[:dim].astype(np.float32)), ("timestamp", e["timestamp"])):
                rec[k].append(val)
    out = {k: np.asarray(v) for k, v in rec.items()}
    out["timestamp"] = out["timestamp"].astype(np.int64)
    return out
