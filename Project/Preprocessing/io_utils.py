"""Readers for nuScenes sample folders (metadata.json, camera JPG, LiDAR .pcd.bin, radar .pcd)."""
import json
from pathlib import Path

import cv2
import numpy as np

from config import RADAR_FILTER

_PCD_TYPES = {("F", 4): "<f4", ("F", 8): "<f8", ("I", 1): "i1", ("I", 2): "<i2", ("I", 4): "<i4",
              ("U", 1): "u1", ("U", 2): "<u2", ("U", 4): "<u4"}


def load_metadata(sample_dir):
    with open(Path(sample_dir) / "metadata.json", encoding="utf8") as f:
        return json.load(f)


def load_split(out_dir):
    """{sample_token: 'train'|'val'|'test'} -- the scene-grouped split (whole scenes in one split, no track leakage)."""
    return json.loads((Path(out_dir) / "splits.json").read_text())["scene_grouped"]


def load_image_rgb(sample_dir, cam):
    img = cv2.imread(str(Path(sample_dir) / "camera" / f"{cam}.jpg"), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"{cam}.jpg missing in {sample_dir}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_lidar(sample_dir):
    """Returns (N, 5) float32: x, y, z, intensity, ring in the LiDAR sensor frame."""
    pts = np.fromfile(Path(sample_dir) / "lidar" / "LIDAR_TOP.pcd.bin", dtype=np.float32)
    return pts.reshape(-1, 5)


def load_radar(sample_dir, name, apply_filter=True):
    """Returns a structured array of radar detections in the radar sensor frame."""
    raw = (Path(sample_dir) / "radar" / f"{name}.pcd").read_bytes()
    marker = b"DATA binary\n"
    cut = raw.index(marker) + len(marker)
    header = {}
    for line in raw[:cut].decode("ascii").splitlines():
        parts = line.split()
        if parts and not line.startswith("#"):
            header[parts[0]] = parts[1:]
    n = int(header["POINTS"][0])
    dtype = np.dtype([(f, _PCD_TYPES[(t, int(s))]) for f, t, s in
                      zip(header["FIELDS"], header["TYPE"], header["SIZE"])])
    # Files carry trailing padding after POINTS * itemsize bytes; read only the declared payload.
    pts = np.frombuffer(raw[cut:cut + n * dtype.itemsize], dtype=dtype)
    if apply_filter:
        keep = (np.isin(pts["invalid_state"], list(RADAR_FILTER["invalid_states"]))
                & np.isin(pts["dyn_prop"], list(RADAR_FILTER["dynprop_states"]))
                & np.isin(pts["ambig_state"], list(RADAR_FILTER["ambig_states"])))
        pts = pts[keep]
    return pts
