"""Section 3.4 -> 3.5 bridge: run the trained fusion detector and export one observation record per detected labelled object.

For every labelled object (GT box -> instance token, via the same rule that wrote the label files) the detection with the
highest confidence among those with IoU >= 0.5 is kept. The record stores what the LSTM later receives as one time step:
RoIAlign-pooled fused P4 feature (256-d), detector box / confidence / class scores. Objects the detector misses (no match)
produce no observation, exactly as they would not appear in a detector-driven track.

Also reports the detector metrics per split (Table 8: mAP@0.5, mAP@0.5:0.95, mean matched IoU, inference time per view).

    python export_detections.py [--ckpt PATH] [--assa PATH] [--out DIR] [--conf 0.001]
    python export_detections.py --dry-run         (trial checkpoint, 2 scenes per split; mechanics only)
"""
import argparse
import json
import statistics
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.ops import roi_align
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.nms import non_max_suppression

import cls_paths as cp
from config import CLASS_ID
from fusion_data import FusionViews, collate, model_inputs, to_device
from fusion_eval import DetectionStats, batch_gts
from fusion_model import FusionYOLO
from io_utils import load_metadata
from objects import labelled_annotations

STRIDE_P4 = 16
FEAT_DIM = 256


def sample_scenes(out_dir):
    tracks = json.loads((out_dir / "tracks.json").read_text())
    return {e["sample"]: e["scene"] for lst in tracks.values() for e in lst}


def timestamps(out_dir, tokens):
    return {t: int(np.load(out_dir / "sensors" / f"{t}.npz")["timestamp"]) for t in tokens}


def load_weights(path):
    if path is not None and path.exists():
        w = json.loads(path.read_text())["weights"]
        return [w["wc"], w["wl"], w["wr"]], str(path)
    return [1 / 3] * 3, "equal (no assa_result.json found)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--assa", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou-match", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-scenes", type=int, default=None, help="use only the first N scenes of each split")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.dry_run:
        a.max_scenes, a.batch = a.max_scenes or 2, 4
    fus = cp.OUT / ("fusion_trial" if a.dry_run else "fusion")
    out = cp.Path(a.out) if a.out else cp.OUT / ("classification_trial" if a.dry_run else "classification")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else "cpu" if a.device == "auto" else a.device)
    amp = device.type == "cuda" and not a.no_amp

    model = FusionYOLO()
    ckpt = cp.Path(a.ckpt) if a.ckpt else fus / "best.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
        print(f"loaded {ckpt}")
    elif a.dry_run:
        print("dry run: no checkpoint -> untrained model (mechanics only)")
    else:
        raise SystemExit(f"checkpoint not found: {ckpt} (train first with Fusion/train_fusion.py)")
    weights, weights_src = load_weights(cp.Path(a.assa) if a.assa else fus / "assa_result.json")
    model.set_weights(*weights)
    model.to(device).eval()
    print(f"fusion weights (wc, wl, wr) = {np.round(weights, 4).tolist()}  from {weights_src}")

    scenes = sample_scenes(cp.OUT)
    rec = {k: [] for k in ("split", "instance", "sample", "scene", "camera", "gt_cls", "det_cls", "conf", "box", "scores", "feat")}
    report = {"checkpoint": str(ckpt), "weights": weights, "weights_source": weights_src, "conf": a.conf, "splits": {}}
    for split in ("train", "val", "test"):
        ds = FusionViews(split)
        if a.max_scenes:
            keep = sorted({scenes[t] for t, _ in ds.views})[:a.max_scenes]
            ds.views = [(t, c) for t, c in ds.views if scenes[t] in keep]
        loader = DataLoader(ds, batch_size=a.batch, shuffle=False, collate_fn=collate)
        stats, ious, times, n_obs, n_gt = DetectionStats(), [], [], 0, 0
        for bi, batch in enumerate(loader):
            b = to_device(batch, device)
            t0 = time.perf_counter()
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=amp):
                feats = model.fuse(model.encode(*model_inputs(b)))
                y = model.decode(feats)[0]
            preds, idxs = non_max_suppression(y.float(), conf_thres=a.conf, iou_thres=0.7, max_det=300, return_idxs=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) / len(b["z0"]))
            gts = batch_gts(batch)
            stats.update(preds, gts)
            p4 = feats[1].float()
            for j, (tok, cam) in enumerate(zip(batch["token"], batch["camera"])):
                anns = list(labelled_annotations(load_metadata(ds.root / tok), cam))
                g, p = gts[j], preds[j].detach().cpu()
                assert len(anns) == len(g), f"label rows != annotations for {tok} {cam}"
                n_gt += len(g)
                if not len(p):
                    continue
                iou = box_iou(g[:, 1:], p[:, :4])
                for k, (ann, box) in enumerate(anns):
                    assert np.allclose(box, g[k, 1:].numpy(), atol=1.0), f"annotation/label box mismatch {tok} {cam}"
                    ok = iou[k] >= a.iou_match
                    if not ok.any():
                        continue
                    d = int(torch.where(ok, p[:, 4], torch.full_like(p[:, 4], -1)).argmax())
                    ious.append(float(iou[k, d]))
                    roi = torch.cat([torch.zeros(1, 1, device=p4.device), preds[j][d:d + 1, :4]], 1)
                    roi[0, 0] = j
                    f = roi_align(p4, roi, output_size=3, spatial_scale=1.0 / STRIDE_P4, aligned=True).mean((2, 3))[0]
                    rec["split"].append(split), rec["instance"].append(ann["instance_token"]), rec["sample"].append(tok)
                    rec["scene"].append(scenes[tok]), rec["camera"].append(cam), rec["gt_cls"].append(CLASS_ID[ann["class_name"]])
                    rec["det_cls"].append(int(p[d, 5])), rec["conf"].append(float(p[d, 4])), rec["box"].append(p[d, :4].numpy())
                    rec["scores"].append(y[j, 4:, idxs[j][d]].float().cpu().numpy()), rec["feat"].append(f.cpu().numpy())
                    n_obs += 1
        m = stats.compute()
        steady = times[1:] if len(times) > 1 else times
        m.update(views=len(ds), objects=n_gt, observations=n_obs, mean_matched_iou=float(np.mean(ious)) if ious else 0.0,
                 ms_per_view=1000 * statistics.median(steady) if steady else 0.0)
        report["splits"][split] = m
        print(f"{split}: {len(ds)} views, {n_gt} objects -> {n_obs} observations | mAP50={m['map50']:.3f} mAP={m['map']:.3f} "
              f"IoU={m['mean_matched_iou']:.3f} {m['ms_per_view']:.0f} ms/view")

    ts = timestamps(cp.OUT, set(rec["sample"]))
    arrays = {k: np.asarray(v) for k, v in rec.items() if k not in ("box", "scores", "feat")}
    arrays["timestamp"] = np.asarray([ts[t] for t in rec["sample"]], np.int64)
    arrays["box"] = np.asarray(rec["box"], np.float32).reshape(-1, 4)
    arrays["scores"] = np.asarray(rec["scores"], np.float32).reshape(-1, len(CLASS_ID))
    arrays["feat"] = np.asarray(rec["feat"], np.float32).reshape(-1, FEAT_DIM)
    np.savez_compressed(out / "observations.npz", **arrays)
    (out / "detector_report.json").write_text(json.dumps(report, indent=2))
    print(f"saved {len(arrays['conf'])} observations -> {out / 'observations.npz'}")


if __name__ == "__main__":
    main()
