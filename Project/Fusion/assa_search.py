"""ASSA search of the global fusion weights (wc, wl, wr) on the validation split with the trained detector fixed.

Weight-independent features (modality features + Eq. 15 attention) are computed once and cached; every candidate only
reruns Eq. 11 + neck + head + NMS. Fitness (Eq. 12): 0.25*Accuracy + 0.25*F1 + 0.25*mAP@0.5 - 0.25*Time_norm,
Time_norm = latency / latency(equal weights). ASSA minimises -Fitness.

    python assa_search.py [--ckpt PATH] [--iters 30] [--n-max 20] [--n-min 5]
    python assa_search.py --trial        (4 views, 3 iterations; untrained weights are fine for checking the mechanics)
"""
import argparse
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import fusion_paths as fp
from assa import assa
from fusion_data import FusionViews, collate, model_inputs, to_device
from fusion_eval import DetectionStats, batch_gts, detections
from fusion_model import FusionYOLO

ALPHA = BETA = GAMMA = LAMBDA = 0.25


def to_cpu(enc):
    return {k: [t.half().cpu() for t in v] for k, v in enc.items()}


def to_dev(enc, device):
    return {k: [t.to(device).float() for t in v] for k, v in enc.items()}


class Evaluator:
    def __init__(self, model, loader, device, amp, timing_batches=2, repeats=3):
        self.model, self.device, self.amp = model.eval(), device, amp
        self.cache, self.gts = [], []
        enc_times = []
        with torch.no_grad():
            for batch in loader:
                b = to_device(batch, device)
                t0 = time.perf_counter()
                with torch.autocast(device_type=device.type, enabled=amp):
                    enc = model.encode(*model_inputs(b))
                self.sync()
                enc_times.append((time.perf_counter() - t0, len(b["z0"])))
                self.cache.append(to_cpu(enc))
                self.gts.append(batch_gts(batch))
        self.n_views = sum(n for _, n in enc_times)
        self.encode_per_view = sum(t for t, _ in enc_times) / self.n_views
        self.timing_batches, self.repeats = timing_batches, repeats
        self.memo = {}
        self.t_ref = None
        self.t_ref = self.latency(model.w.cpu().numpy())  # equal-weight latency (model.w starts at 1/3 each)

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def run(self, w, batches, nms=True):
        wt = torch.as_tensor(w, dtype=torch.float32, device=self.device)
        for idx in batches:
            enc = to_dev(self.cache[idx], self.device)
            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                feats = self.model.fuse(enc, wt)
                y = self.model.decode(feats)[0]
            yield idx, (detections(y) if nms else y)

    @torch.no_grad()
    def latency(self, w):
        """Seconds per view: encode (weight independent, measured once) + fuse + neck + head (min over repeats, fixed subset).
        NMS is excluded: its cost depends on how many boxes the candidate produces, which is noise for this comparison."""
        batches = range(min(self.timing_batches, len(self.cache)))
        n = sum(len(self.gts[i]) for i in batches)
        best = float("inf")
        for _ in range(self.repeats):
            self.sync()
            t0 = time.perf_counter()
            for _ in self.run(w, batches, nms=False):
                pass
            self.sync()
            best = min(best, time.perf_counter() - t0)
        return self.encode_per_view + best / n

    @torch.no_grad()
    def metrics(self, w):
        stats = DetectionStats()
        for idx, preds in self.run(w, range(len(self.cache))):
            stats.update(preds, self.gts[idx])
        m = stats.compute()
        m["time_norm"] = self.latency(w) / self.t_ref
        m["fitness"] = ALPHA * m["accuracy"] + BETA * m["f1"] + GAMMA * m["map50"] - LAMBDA * m["time_norm"]
        return m

    def neg_fitness(self, w):
        key = tuple(np.round(w, 6))
        if key not in self.memo:
            self.memo[key] = self.metrics(w)
        return -self.memo[key]["fitness"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--n-max", type=int, default=20)
    ap.add_argument("--n-min", type=int, default=5)
    ap.add_argument("--lam", type=float, default=1.5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-views", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--trial", action="store_true")
    a = ap.parse_args()
    if a.trial:
        a.iters, a.n_max, a.n_min, a.max_views, a.batch = 3, 4, 2, 4, 2
    out = fp.OUT / ("fusion_trial" if a.trial else "fusion") if a.out is None else fp.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else "cpu" if a.device == "auto" else a.device)
    amp = device.type == "cuda" and not a.no_amp

    model = FusionYOLO()
    ckpt = fp.Path(a.ckpt) if a.ckpt else out / "best.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
        print(f"loaded {ckpt}")
    elif a.trial:
        print("trial: no checkpoint found -> untrained model (metrics meaningless, mechanics only)")
    else:
        raise SystemExit(f"checkpoint not found: {ckpt} (train first with train_fusion.py)")
    model.set_weights(1 / 3, 1 / 3, 1 / 3)
    model.to(device)

    ds = FusionViews(a.split)
    if a.max_views:
        ds = Subset(ds, range(min(a.max_views, len(ds))))
    loader = DataLoader(ds, batch_size=a.batch, shuffle=False, collate_fn=collate)
    t0 = time.time()
    ev = Evaluator(model, loader, device, amp)
    print(f"cached {ev.n_views} {a.split} views in {time.time() - t0:.0f}s on {device}; "
          f"encode {ev.encode_per_view * 1000:.1f} ms/view; equal-weight latency {ev.t_ref * 1000:.1f} ms/view")

    equal = ev.metrics(np.full(3, 1 / 3))
    print("equal weights:", {k: round(v, 4) for k, v in equal.items()})
    t0 = time.time()
    w, f, hist = assa(ev.neg_fitness, n_max=a.n_max, n_min=a.n_min, iters=a.iters, lam=a.lam, seed=a.seed)
    best = ev.metrics(w)
    print(f"ASSA done in {time.time() - t0:.0f}s, {len(ev.memo)} distinct candidates evaluated")
    print(f"best weights (wc, wl, wr) = {np.round(w, 4).tolist()}  sum={w.sum():.4f}")
    print("best metrics:", {k: round(v, 4) for k, v in best.items()})

    res = {"weights": {"wc": float(w[0]), "wl": float(w[1]), "wr": float(w[2])}, "neg_fitness": f, "metrics": best,
           "equal_weight_metrics": equal, "split": a.split, "views": ev.n_views, "device": str(device),
           "settings": {k: v for k, v in vars(a).items()}, "history": hist,
           "candidates": [{"w": list(k), **v} for k, v in ev.memo.items()]}
    (out / "assa_result.json").write_text(json.dumps(res, indent=2))
    print(f"saved {out / 'assa_result.json'}")


if __name__ == "__main__":
    main()
