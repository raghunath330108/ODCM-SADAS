"""Section 3.3 fusion model: pretrained YOLO11s camera branch + LiDAR BEV branch + radar branch, fused at P3/P4/P5.

Eq. 11  F_fused  = wc*Fc + wl*Fl + wr*Fr            (all three aligned on the camera feature grid)
Eq. 13  Fm'      = Conv(Fm) + LSTM(Fm)              (per modality, C/3 channels each)
Eq. 14  F_LRV    = Cat(Fc', Fl', Fr')
Eq. 15  F_fusion = sigmoid(W1 relu(W0 MaxPool(F_img)) + W1 relu(W0 MaxPool(F_LRV)))
The fused feature fed to the YOLO neck is F_fused * F_fusion.

Stages: encode (modality features + attention, independent of w) -> fuse (Eq. 11, depends on w) -> decode (neck + head).
ASSA only changes w, so it can cache encode() and rerun fuse() + decode().
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.modules import Conv
from ultralytics.nn.tasks import DetectionModel

import fusion_paths as fp
from geometry import bev_points, lift_to_camera

TAPS = (4, 6, 10)  # YOLO11s backbone outputs P3, P4, P5
STRIDES = (8, 16, 32)


class LiDARBEVBranch(nn.Module):
    """Eq. 8 BEV height map -> BEV features -> (Eq. 7 + 3-4) lifted onto each camera feature grid."""

    def __init__(self, channels, preset="full360", mid=64):
        super().__init__()
        self.preset, self.stride = preset, 4
        self.encoder = nn.Sequential(Conv(2, 16, 3, 2), Conv(16, 32, 3, 2), Conv(32, mid, 3, 1))
        self.refine = nn.ModuleList(nn.Sequential(Conv(mid, c, 3, 1), Conv(c, c, 3, 1)) for c in channels)

    def forward(self, bev, bev_gray, z0, T, K):
        feat = self.encoder(bev)
        out = []
        with torch.autocast(device_type=bev.device.type, enabled=False):  # keep metric geometry in fp32
            pts, valid, _ = bev_points(bev_gray.float(), self.preset, z0.float(), self.stride)
            flat = feat.float().flatten(2)
            lifted = [lift_to_camera(flat, pts, valid, T.float(), K.float(), s) for s in STRIDES]
        for lf, refine in zip(lifted, self.refine):
            out.append(refine(F.max_pool2d(lf, 3, 1, 1)))  # 3x3 max splat so sparse cells cover their footprint
        return out


class RadarBranch(nn.Module):
    """Image-plane radar map (Eq. 5-6 projection, already pixel-aligned) -> features at P3/P4/P5."""

    def __init__(self, channels, in_ch=5):
        super().__init__()
        self.stem = nn.Sequential(Conv(in_ch, 16, 3, 2), Conv(16, 32, 3, 2), Conv(32, 64, 3, 2))
        self.d4 = Conv(64, 128, 3, 2)
        self.d5 = Conv(128, 256, 3, 2)
        self.out = nn.ModuleList([Conv(64, channels[0], 1), Conv(128, channels[1], 1), Conv(256, channels[2], 1)])

    def forward(self, radar):
        p3 = self.stem(radar)
        p4 = self.d4(p3)
        p5 = self.d5(p4)
        return [o(p) for o, p in zip(self.out, (p3, p4, p5))]


class SpatialLSTM(nn.Module):
    """LSTM that scans each feature-map row (a sequence of C-dim pixel features); no time axis is involved."""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.lstm = nn.LSTM(c_in, c_out, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.permute(0, 2, 3, 1).reshape(B * H, W, C)
        out, _ = self.lstm(seq)
        return out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()


class FusionLevel(nn.Module):
    def __init__(self, c, reduction=16):
        super().__init__()
        base = c // 3
        parts = [base, base, c - 2 * base]  # C/3 per modality; concatenation gives exactly C
        self.conv = nn.ModuleList(Conv(c, p, 3, 1) for p in parts)
        self.lstm = nn.ModuleList(SpatialLSTM(c, p) for p in parts)
        mid = max(c // reduction, 8)
        self.w0 = nn.Conv2d(c, mid, 1)
        self.w1 = nn.Conv2d(mid, c, 1)

    def attention(self, feats):
        """Eq. 13-15 (does not depend on the fusion weights): channel attention (B, C, 1, 1)."""
        flrv = torch.cat([cv(f) + ls(f) for f, cv, ls in zip(feats, self.conv, self.lstm)], 1)
        mp = lambda t: F.adaptive_max_pool2d(t, 1)
        return torch.sigmoid(self.w1(F.relu(self.w0(mp(feats[0])))) + self.w1(F.relu(self.w0(mp(flrv)))))

    @staticmethod
    def combine(feats, w, att):
        """Eq. 11 gated by Eq. 15."""
        return sum(wi * f for wi, f in zip(w, feats)) * att


class FusionYOLO(nn.Module):
    def __init__(self, weights=fp.YOLO_WEIGHTS, nc=6, preset="full360"):
        super().__init__()
        ref = YOLO(str(weights)).model
        det = DetectionModel("yolo11s.yaml", nc=nc, verbose=False)
        src, dst = ref.float().state_dict(), det.state_dict()
        self.transferred = [k for k in dst if k in src and src[k].shape == dst[k].shape]
        det.load_state_dict({k: src[k] for k in self.transferred}, strict=False)
        self.total_params = len(dst)
        det.args = get_cfg()  # loss gains (box / cls / dfl) for training
        self.det = det
        self.channels = tuple(self.layers[i].cv2.conv.out_channels for i in TAPS)
        self.lidar = LiDARBEVBranch(self.channels, preset)
        self.radar = RadarBranch(self.channels)
        self.levels = nn.ModuleList(FusionLevel(c) for c in self.channels)
        self.register_buffer("w", torch.full((3,), 1.0 / 3.0))  # (wc, wl, wr); equal until ASSA tunes it

    @property
    def layers(self):
        return self.det.model

    def set_weights(self, wc, wl, wr):
        w = torch.tensor([wc, wl, wr], dtype=torch.float32)
        assert (w >= 0).all() and (w <= 1).all() and abs(float(w.sum()) - 1.0) < 1e-4, "Eq. 9-10: weights in [0,1], sum 1"
        self.w.copy_(w)

    def encode(self, image, radar, bev, bev_gray, z0, T, K):
        """Per-modality features at P3/P4/P5 and the Eq. 15 attention; independent of the fusion weights."""
        cam, x = [], image
        for m in self.layers[:TAPS[-1] + 1]:
            x = m(x)
            if m.i in TAPS:
                cam.append(x)
        fl = self.lidar(bev, bev_gray, z0, T, K)
        fr = self.radar(radar)
        att = [lv.attention((c, l, r)) for lv, c, l, r in zip(self.levels, cam, fl, fr)]
        return {"cam": cam, "lidar": fl, "radar": fr, "att": att}

    def fuse(self, enc, w=None):
        w = self.w if w is None else w
        return [lv.combine((c, l, r), w, a) for lv, c, l, r, a in zip(self.levels, enc["cam"], enc["lidar"], enc["radar"], enc["att"])]

    def decode(self, feats):
        """YOLO11s neck + Detect head from the three backbone-level maps."""
        y = dict(zip(TAPS, feats))
        x = feats[-1]
        for m in self.layers[TAPS[-1] + 1:-1]:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y[m.i] = x
        return self.layers[-1]([y[16], y[19], y[22]])

    def forward(self, image, radar, bev, bev_gray, z0, T, K, fuse=True):
        enc = self.encode(image, radar, bev, bev_gray, z0, T, K)
        feats = self.fuse(enc) if fuse else enc["cam"]
        return self.decode(feats), {"fused": feats, "attention": enc["att"]}
