"""Final metrics collection: merges the detector (3.4), ASSA (3.3), GOA (3.6) and LSTM (3.5) results into one report.

Writes into the classification output directory: final_report.json / final_report.md, per-class CSVs, confusion-matrix CSVs
and PNGs (manuscript Fig. 5), training curves (Fig. 8/9).

    python collect_metrics.py [--cls DIR] [--fusion DIR]        python collect_metrics.py --dry-run
"""
import argparse
import csv
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import cls_paths as cp  # noqa: E402
from config import CLASSES  # noqa: E402

LEVELS = ("lstm_window", "detector_window", "lstm_track", "detector_track_vote")
SCALAR = ("accuracy", "precision_macro", "recall_macro", "f1_macro", "mcc")


def read_json(p):
    return json.loads(p.read_text()) if p.exists() else None


def plot_confusion(cm, title, path):
    cm = np.asarray(cm)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right"), ax.set_yticks(range(len(CLASSES)), CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("predicted"), ax.set_ylabel("true"), ax.set_title(title)
    fig.tight_layout(), fig.savefig(path, dpi=150), plt.close(fig)


def plot_curves(rows, path):
    ep = [r["epoch"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(ep, [r["train_acc"] for r in rows], label="train"), ax[0].plot(ep, [r["val_acc"] for r in rows], label="validation")
    ax[0].set(xlabel="epoch", ylabel="accuracy", title="LSTM accuracy"), ax[0].legend()
    ax[1].plot(ep, [r["train_loss"] for r in rows], label="train"), ax[1].plot(ep, [r["val_loss"] for r in rows], label="validation")
    ax[1].set(xlabel="epoch", ylabel="loss", title="LSTM loss"), ax[1].legend()
    fig.tight_layout(), fig.savefig(path, dpi=150), plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", default=None)
    ap.add_argument("--fusion", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    tag = "_trial" if a.dry_run else ""
    cdir = cp.Path(a.cls) if a.cls else cp.OUT / f"classification{tag}"
    fdir = cp.Path(a.fusion) if a.fusion else cp.OUT / f"fusion{tag}"
    det, assa, goa_r, lstm = (read_json(cdir / "detector_report.json"), read_json(fdir / "assa_result.json"),
                              read_json(cdir / "goa_result.json"), read_json(cdir / "lstm_metrics.json"))
    if lstm is None or goa_r is None:
        raise SystemExit(f"missing goa_lstm.py output in {cdir}")
    synthetic = bool(lstm.get("synthetic"))
    report = {"synthetic_features": synthetic, "detector": det, "assa": None if assa is None else {
        k: assa[k] for k in ("weights", "metrics", "equal_weight_metrics", "views", "split")},
        "goa": {k: goa_r[k] for k in ("best_params", "best_fitness", "fitness_metric", "tuning_time_s", "settings")},
        "lstm": {k: lstm[k] for k in ("params", "epochs", "best_epoch", "train_time_s", "lstm_infer_s_per_window")}, "classification": {}}

    md = ["# Final results" + ("  (DRY RUN - SYNTHETIC FEATURES, NOT RESULTS)" if synthetic or a.dry_run else ""), ""]
    if det:
        md += ["## Detector (3.4, fused YOLO11s)", "", "| split | views | objects | mAP@0.5 | mAP@0.5:0.95 | mean IoU | P | R | F1 | ms/view |", "|---|---|---|---|---|---|---|---|---|---|"]
        for s, m in det["splits"].items():
            md.append(f"| {s} | {m['views']} | {m['objects']} | {m['map50']:.3f} | {m['map']:.3f} | {m['mean_matched_iou']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['ms_per_view']:.1f} |")
        md += ["", f"Fusion weights (wc, wl, wr): {np.round(det['weights'], 4).tolist()} ({det['weights_source']})", ""]
    if assa:
        e, b = assa["equal_weight_metrics"], assa["metrics"]
        md += ["## ASSA weight tuning (val)", "", "| | Accuracy | F1 | mAP@0.5 | Time_norm | Fitness |", "|---|---|---|---|---|---|",
               f"| equal weights | {e['accuracy']:.3f} | {e['f1']:.3f} | {e['map50']:.3f} | {e['time_norm']:.3f} | {e['fitness']:.3f} |",
               f"| ASSA | {b['accuracy']:.3f} | {b['f1']:.3f} | {b['map50']:.3f} | {b['time_norm']:.3f} | {b['fitness']:.3f} |", ""]
    p = goa_r["best_params"]
    md += ["## GOA-tuned LSTM (3.5 / 3.6)", "", f"Best hyperparameters: lr={p['lr']:.5f}, dropout={p['dropout']:.3f}, hidden={p['hidden']}, batch={p['batch']}, "
           f"layers={p['layers']}, sequence length={p['seq_len']}; GOA fitness (val {goa_r['fitness_metric']}) = {goa_r['best_fitness']:.4f}; "
           f"tuning time {goa_r['tuning_time_s']:.0f} s; final training {lstm['epochs']} epochs ({lstm['train_time_s']:.0f} s), best epoch {lstm['best_epoch']}.", ""]
    for s, res in lstm["results"].items():
        md += [f"### {s} split", "", "| level | n | Accuracy | Precision | Recall | F1 | MCC |", "|---|---|---|---|---|---|---|"]
        for lv in LEVELS:
            m = res[lv]
            md.append(f"| {lv} | {m['n']} | " + " | ".join(f"{m[k]:.3f}" for k in SCALAR) + " |")
            plot_confusion(m["confusion_matrix"], f"{s}: {lv}", cdir / f"confusion_{s}_{lv}.png")
            np.savetxt(cdir / f"confusion_{s}_{lv}.csv", np.asarray(m["confusion_matrix"]), fmt="%d", delimiter=",", header=",".join(CLASSES), comments="")
        md += ["", "Per class (LSTM, window level):", "", "| class | support | Accuracy (one-vs-rest) | Precision | Recall | F1 |", "|---|---|---|---|---|---|"]
        pc = res["lstm_window"]["per_class"]
        with open(cdir / f"per_class_{s}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["class", "support", "accuracy_ovr", "precision", "recall", "f1"])
            for i, c in enumerate(CLASSES):
                row = [pc["support"][i], pc["accuracy_ovr"][i], pc["precision"][i], pc["recall"][i], pc["f1"][i]]
                w.writerow([c] + [f"{v:.4f}" if isinstance(v, float) else v for v in row])
                md.append(f"| {c} | {row[0]} | {row[1]:.3f} | {row[2]:.3f} | {row[3]:.3f} | {row[4]:.3f} |")
        md.append("")
        report["classification"][s] = {lv: {k: res[lv][k] for k in (*SCALAR, "n", "per_class", "confusion_matrix")} for lv in LEVELS}
    rows = list(csv.DictReader(open(cdir / "lstm_log.csv")))
    plot_curves([{k: float(v) for k, v in r.items()} for r in rows], cdir / "training_curves.png")
    md += ["Computation: detector " + (f"{det['splits']['test']['ms_per_view']:.1f} ms/view (test), " if det else "") +
           f"LSTM {lstm['lstm_infer_s_per_window'] * 1000:.3f} ms/window.", ""]
    (cdir / "final_report.json").write_text(json.dumps(report, indent=2))
    (cdir / "final_report.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"saved report, CSVs and figures to {cdir}")


if __name__ == "__main__":
    main()
