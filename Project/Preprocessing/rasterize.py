"""Image-plane and BEV rasterization onto the common 640x640 canvas (fusion-ready, pixel-aligned)."""
import cv2
import numpy as np

from calibration import intrinsics, project_to_image, reference_vehicle_to_camera
from config import BEV_PRESETS, CANVAS, MIN_DEPTH, PAD_VALUE, SPLAT_LIDAR, SPLAT_RADAR, SRC_H, SRC_W

SCALE = CANVAS / SRC_W
NEW_H = int(round(SRC_H * SCALE))
PAD_Y = (CANVAS - NEW_H) // 2  # content occupies rows [PAD_Y, PAD_Y + NEW_H)


def letterbox_image(img_rgb):
    resized = cv2.resize(img_rgb, (CANVAS, NEW_H), interpolation=cv2.INTER_AREA)
    out = np.full((CANVAS, CANVAS, 3), PAD_VALUE, np.uint8)
    out[PAD_Y:PAD_Y + NEW_H] = resized
    return out


def to_canvas(u, v):
    return u * SCALE, v * SCALE + PAD_Y


def box_to_canvas(xyxy):
    x1, y1, x2, y2 = xyxy
    return [x1 * SCALE, y1 * SCALE + PAD_Y, x2 * SCALE, y2 * SCALE + PAD_Y]


def project_cloud(meta, cam, pts_ref_vehicle):
    """Eq. (1) + (3)-(4): reference-vehicle points -> canvas (u, v), depth, validity mask."""
    Xc = reference_vehicle_to_camera(meta, cam).apply(pts_ref_vehicle)
    u, v, d = project_to_image(Xc, intrinsics(meta, cam))
    ok = (d > MIN_DEPTH) & (u >= 0) & (u < SRC_W) & (v >= 0) & (v < SRC_H)
    uc, vc = to_canvas(u, v)
    return uc, vc, d, ok


def _disc(radius):
    r = int(radius)
    return [(dx, dy) for dy in range(-r, r + 1) for dx in range(-r, r + 1) if dx * dx + dy * dy <= radius * radius]


def splat(uc, vc, depth, values, radius):
    """Draw points into a (C+1, H, W) float32 map; channel 0 is depth, then `values` columns. Nearest wins."""
    C = values.shape[1]
    out = np.zeros((C + 1, CANVAS, CANVAS), np.float32)
    if len(uc) == 0:
        return out
    offs = np.array(_disc(radius))
    x = np.round(uc).astype(np.int64)[:, None] + offs[None, :, 0]
    y = np.round(vc).astype(np.int64)[:, None] + offs[None, :, 1]
    pid = np.broadcast_to(np.arange(len(uc))[:, None], x.shape)
    x, y, pid = x.ravel(), y.ravel(), pid.ravel()
    keep = (x >= 0) & (x < CANVAS) & (y >= PAD_Y) & (y < PAD_Y + NEW_H)
    x, y, pid = x[keep], y[keep], pid[keep]
    order = np.argsort(-depth[pid], kind="stable")  # far first, near last -> near overwrites
    x, y, pid = x[order], y[order], pid[order]
    out[0, y, x] = depth[pid]
    for c in range(C):
        out[c + 1, y, x] = values[pid, c]
    return out


def lidar_image_maps(meta, cam, lidar_v, intensity):
    """(3, 640, 640): depth [m], intensity, height z in vehicle frame [m]; 0 depth = empty."""
    uc, vc, d, ok = project_cloud(meta, cam, lidar_v)
    vals = np.stack([intensity[ok], lidar_v[ok, 2]], 1)
    return splat(uc[ok], vc[ok], d[ok], vals, SPLAT_LIDAR)


def radar_image_maps(meta, cam, radar_xyz, rcs, vxy):
    """(4, 640, 640): depth [m], rcs, vx, vy (vehicle frame, ego-compensated); 0 depth = empty."""
    uc, vc, d, ok = project_cloud(meta, cam, radar_xyz)
    vals = np.concatenate([rcs[ok, None], vxy[ok]], 1)
    return splat(uc[ok], vc[ok], d[ok], vals, SPLAT_RADAR)


def bev_grid(preset):
    p = BEV_PRESETS[preset]
    n_rows = int(round((p["x"][1] - p["x"][0]) / p["res"]))
    n_cols = int(round((p["y"][1] - p["y"][0]) / p["res"]))
    assert n_rows == CANVAS and n_cols == CANVAS, (preset, n_rows, n_cols)
    return p


def _bev_index(xyz, p):
    row = np.floor((p["x"][1] - xyz[:, 0]) / p["res"]).astype(np.int64)  # row 0 = farthest ahead
    col = np.floor((p["y"][1] - xyz[:, 1]) / p["res"]).astype(np.int64)  # col 0 = leftmost
    ok = ((row >= 0) & (row < CANVAS) & (col >= 0) & (col < CANVAS)
          & (xyz[:, 2] >= p["z"][0]) & (xyz[:, 2] <= p["z"][1]))
    return row, col, ok


def lidar_bev(lidar_rel, preset):
    """Eq. (8): (1, 640, 640) uint8 gray image g_ij = h_ij = max z per cell mapped to (0, 255).

    lidar_rel z is relative to the LiDAR origin. Empty cells are 0; occupied cells are >= 1.
    """
    p = bev_grid(preset)
    row, col, ok = _bev_index(lidar_rel, p)
    h = np.full((CANVAS, CANVAS), -np.inf, np.float32)
    np.maximum.at(h, (row[ok], col[ok]), lidar_rel[ok, 2])
    gray = np.round((h - p["z"][0]) / (p["z"][1] - p["z"][0]) * 255.0)
    return np.where(np.isfinite(h), np.clip(gray, 1, 255), 0).astype(np.uint8)[None]