"""Sections 3.5 + 3.6: GOA search of the LSTM hyperparameters (Table 3), final LSTM training, test evaluation.

Pipeline: observations.npz (export_detections.py) -> per-object windows -> GOA (fitness = validation precision, Eq. 36; each
candidate trained for --goa-epochs) -> final training with the best hyperparameters for --epochs (best-validation checkpoint)
-> metrics on val / test (window level and track level) plus the detector-only baseline on the same observations.

    python goa_lstm.py                      (real run: --pop 10 --iters 15 --goa-epochs 50 --epochs 300; manuscript Table 4 uses 3000)
    python goa_lstm.py --dry-run            (tiny settings on the trial export; synthetic features if the trial detector matched too little)
"""
import argparse
import csv
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import cls_paths as cp
from cls_metrics import classification_metrics, confusion_matrix
from config import CLASSES as CLASS_NAMES
from goa import goa
from lstm_model import SequenceClassifier
from sequences import Windows, build_tracks, collate, feature_stats, load_observations, synthetic_observations

NC = len(CLASS_NAMES)
FIT_KEY = {"precision": "precision_macro", "accuracy": "accuracy", "f1": "f1_macro"}


def loader(tracks, seq_len, batch, shuffle, seed=0):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(Windows(tracks, seq_len), batch_size=batch, shuffle=shuffle, collate_fn=collate, generator=g)


@torch.no_grad()
def predict(model, dl, device):
    model.eval()
    logits, ys, tis, es = [], [], [], []
    for x, n, y, ti, e in dl:
        logits.append(model(x.to(device), n).cpu()), ys.append(y), tis.append(ti), es.append(e)
    return torch.cat(logits), torch.cat(ys), torch.cat(tis), torch.cat(es)


def train_lstm(params, train_tracks, val_tracks, epochs, device, seed, fit_key, log=None):
    """Adam + cross-entropy; returns (best val state dict, best epoch, history rows). 'Best' = highest validation fitness metric."""
    torch.manual_seed(seed)
    in_dim = train_tracks[0]["x"].shape[1]
    model = SequenceClassifier(in_dim, params["hidden"], params["layers"], params["dropout"], NC).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"])
    ce = nn.CrossEntropyLoss()
    tr = loader(train_tracks, params["seq_len"], params["batch"], True, seed)
    va = loader(val_tracks, params["seq_len"], 64, False)
    best, best_state, best_ep, hist = -1.0, None, 0, []
    for ep in range(1, epochs + 1):
        model.train()
        tot, hit, cnt = 0.0, 0, 0
        for x, n, y, _, _ in tr:
            x, y = x.to(device), y.to(device)
            out = model(x, n)
            loss = ce(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot, hit, cnt = tot + loss.item() * len(y), hit + (out.argmax(1) == y).sum().item(), cnt + len(y)
        logits, ys, _, _ = predict(model, va, device)
        vloss = ce(logits, ys).item()
        m = classification_metrics(confusion_matrix(ys.numpy(), logits.argmax(1).numpy(), NC))
        row = {"epoch": ep, "train_loss": tot / cnt, "train_acc": hit / cnt, "val_loss": vloss, "val_acc": m["accuracy"], "val_fitness": m[fit_key]}
        hist.append(row)
        if row["val_fitness"] > best:
            best, best_ep = row["val_fitness"], ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if log and (ep == 1 or ep % max(1, epochs // 10) == 0 or ep == epochs):
            log(f"  epoch {ep}/{epochs} train loss {row['train_loss']:.4f} acc {row['train_acc']:.3f} | val loss {vloss:.4f} acc {row['val_acc']:.3f} fit {row['val_fitness']:.3f}")
    return best_state, best_ep, hist


def evaluate(model, tracks, seq_len, device):
    """Window level (every observation is a prediction), track level (final window), and the detector-only baseline on the same items."""
    ds = Windows(tracks, seq_len)
    logits, ys, tis, es = predict(model, DataLoader(ds, batch_size=64, collate_fn=collate), device)
    pred, y = logits.argmax(1).numpy(), ys.numpy()
    det_win = np.asarray([tracks[ti]["det"][e] for ti, e in zip(tis.tolist(), es.tolist())])
    last = ds.last_of_track()
    det_vote = np.asarray([np.bincount(tracks[ti]["det"], minlength=NC).argmax() for ti in tis[last].tolist()])
    out = {}
    for name, yt, yp in (("lstm_window", y, pred), ("detector_window", y, det_win), ("lstm_track", y[last], pred[last]), ("detector_track_vote", y[last], det_vote)):
        cm = confusion_matrix(yt, yp, NC)
        out[name] = {**classification_metrics(cm), "confusion_matrix": cm.tolist()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--pop", type=int, default=10)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--goa-epochs", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--fitness", choices=list(FIT_KEY), default="precision")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.dry_run:
        a.pop, a.iters, a.goa_epochs, a.epochs = 3, 2, 2, 3
    out = cp.Path(a.out) if a.out else cp.OUT / ("classification_trial" if a.dry_run else "classification")
    obs_path = cp.Path(a.obs) if a.obs else out / "observations.npz"
    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else "cpu" if a.device == "auto" else a.device)
    fit_key = FIT_KEY[a.fitness]

    synthetic = False
    obs = load_observations(obs_path) if obs_path.exists() else None
    enough = obs is not None and (obs["split"] == "train").sum() >= 50 and len(set(obs["gt_cls"][obs["split"] == "train"])) >= 2
    if not enough:
        if not a.dry_run:
            raise SystemExit(f"no usable observations at {obs_path} (run export_detections.py first)")
        print("DRY RUN: the trial detector produced too few observations -> SYNTHETIC features on the real track structure (mechanics only)")
        obs, synthetic = synthetic_observations(), True
    from sequences import best_view_per_sample
    idx = best_view_per_sample(obs)
    stats = feature_stats(obs, idx[obs["split"][idx] == "train"])
    tracks = {s: build_tracks(obs, s, stats) for s in ("train", "val", "test")}
    for s, t in tracks.items():
        print(f"{s}: {len(t)} tracks, {sum(len(x['x']) for x in t)} observations, classes {np.bincount([x['label'] for x in t], minlength=NC).tolist()}")
    if not tracks["train"] or not tracks["val"]:
        raise SystemExit("train or val split has no tracks")

    def fitness(p):
        t0 = time.time()
        state, ep, hist = train_lstm(p, tracks["train"], tracks["val"], a.goa_epochs, device, a.seed, fit_key)
        f = hist[ep - 1]["val_fitness"]
        print(f"  candidate {p} -> val {a.fitness} {f:.4f} (best epoch {ep}, {time.time() - t0:.1f}s)")
        return f

    print(f"GOA: population {a.pop}, {a.iters} iterations, {a.goa_epochs} epochs per candidate, fitness = val {a.fitness}")
    t0 = time.time()
    best_p, best_f, hist, cands = goa(fitness, pop=a.pop, iters=a.iters, seed=a.seed)
    tuning_s = time.time() - t0
    print(f"GOA done in {tuning_s:.0f}s, {len(cands)} distinct candidates; best {best_p} fitness {best_f:.4f}")

    print(f"final training: {a.epochs} epochs with {best_p}")
    t0 = time.time()
    state, best_ep, log_rows = train_lstm(best_p, tracks["train"], tracks["val"], a.epochs, device, a.seed, fit_key, log=print)
    train_s = time.time() - t0
    model = SequenceClassifier(tracks["train"][0]["x"].shape[1], best_p["hidden"], best_p["layers"], best_p["dropout"], NC).to(device)
    model.load_state_dict(state)
    t0 = time.time()
    results = {s: evaluate(model, tracks[s], best_p["seq_len"], device) for s in ("val", "test") if tracks[s]}
    infer = (time.time() - t0) / max(1, sum(len(Windows(tracks[s], 1)) for s in results))

    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state, "params": best_p, "feature_mean": stats[0], "feature_std": stats[1]}, out / "lstm_best.pt")
    with open(out / "lstm_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log_rows[0]))
        w.writeheader(), w.writerows(log_rows)
    (out / "goa_result.json").write_text(json.dumps({
        "best_params": best_p, "best_fitness": best_f, "fitness_metric": a.fitness, "tuning_time_s": tuning_s,
        "history": hist, "candidates": cands, "settings": {"pop": a.pop, "iters": a.iters, "goa_epochs": a.goa_epochs, "seed": a.seed},
        "synthetic": synthetic}, indent=2))
    (out / "lstm_metrics.json").write_text(json.dumps({
        "params": best_p, "epochs": a.epochs, "best_epoch": best_ep, "train_time_s": train_s, "lstm_infer_s_per_window": infer,
        "class_names": list(CLASS_NAMES), "results": results, "synthetic": synthetic}, indent=2))
    for s, r in results.items():
        for k in ("lstm_window", "detector_window", "lstm_track", "detector_track_vote"):
            m = r[k]
            print(f"{s:5s} {k:20s} n={m['n']:4d} acc={m['accuracy']:.3f} P={m['precision_macro']:.3f} R={m['recall_macro']:.3f} F1={m['f1_macro']:.3f} MCC={m['mcc']:.3f}")
    print(f"saved to {out}" + ("  [SYNTHETIC - mechanics only]" if synthetic else ""))


if __name__ == "__main__":
    main()
