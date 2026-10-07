"""Spatial alignment (manuscript Eq. 7 + Eq. 3-4): BEV cells -> 3D points -> camera feature grid."""
import torch
import torch.nn.functional as F

import fusion_paths  # noqa: F401
from config import BEV_PRESETS, CANVAS, MIN_DEPTH
from rasterize import NEW_H, PAD_Y


def bev_cell_centres(preset, stride=1, device=None):
    """Vehicle-frame (x, y) of BEV cell centres after `stride`-fold downsampling, each (n, n); row 0 = farthest ahead."""
    p = BEV_PRESETS[preset]
    n = CANVAS // stride
    off = (torch.arange(n, device=device, dtype=torch.float32) + 0.5) * p["res"] * stride
    x = p["x"][1] - off
    y = p["y"][1] - off
    return x[:, None].expand(n, n), y[None, :].expand(n, n)


def gray_to_height(gray, preset, z0):
    """Invert Eq. 8: gray (B,1,H,W) in 0..255 -> height in the vehicle frame [m]. z0 (B,) = LiDAR origin height."""
    zmin, zmax = BEV_PRESETS[preset]["z"]
    return zmin + gray / 255.0 * (zmax - zmin) + z0.view(-1, 1, 1, 1)


def bev_points(gray, preset, z0, stride):
    """Occupied BEV cells (max-pooled by `stride`) as vehicle-frame 3D points. Returns pts (B,N,3), valid (B,N), pooled (B,1,n,n)."""
    g = F.max_pool2d(gray, stride) if stride > 1 else gray
    X, Y = bev_cell_centres(preset, stride, gray.device)
    Z = gray_to_height(g, preset, z0)
    B = gray.shape[0]
    pts = torch.stack([X.expand(B, -1, -1), Y.expand(B, -1, -1), Z[:, 0]], -1).reshape(B, -1, 3)
    return pts, (g > 0).reshape(B, -1), g


def project(pts, T, K):
    """Eq. 1 + 3-4: vehicle points (B,N,3) -> canvas pixels. T (B,4,4) camera<-vehicle, K (B,3,3) canvas intrinsics.

    Returns u, v, depth, ok; ok keeps points in front of the camera that land on the real image rows of the canvas.
    """
    Xc = pts @ T[:, :3, :3].transpose(1, 2) + T[:, :3, 3].unsqueeze(1)
    z = Xc[..., 2]
    zs = z.clamp(min=1e-6)
    u = K[:, 0, 0:1] * Xc[..., 0] / zs + K[:, 0, 2:3]
    v = K[:, 1, 1:2] * Xc[..., 1] / zs + K[:, 1, 2:3]
    ok = (z > MIN_DEPTH) & (u >= 0) & (u < CANVAS) & (v >= PAD_Y) & (v < PAD_Y + NEW_H)
    return u, v, z, ok


def lift_to_camera(feat, pts, valid, T, K, stride):
    """Scatter BEV features (B,C,N) onto the camera feature grid of the given stride (max where several cells land together).

    Returns (B, C, 640/stride, 640/stride); camera pixels that receive no BEV cell stay 0.
    """
    B, C, _ = feat.shape
    n = CANVAS // stride
    u, v, _, ok = project(pts, T, K)
    ok = ok & valid
    iu = (u / stride).floor().long().clamp(0, n - 1)
    iv = (v / stride).floor().long().clamp(0, n - 1)
    idx = torch.where(ok, iv * n + iu, torch.full_like(iu, n * n))
    out = feat.new_zeros(B, C, n * n + 1)
    out.scatter_reduce_(2, idx.unsqueeze(1).expand(-1, C, -1), feat, reduce="amax", include_self=False)
    return out[..., :n * n].reshape(B, C, n, n)
