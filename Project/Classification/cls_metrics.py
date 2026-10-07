"""Classification metrics from a confusion matrix (rows = true class, columns = predicted class)."""
import numpy as np


def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), np.int64)
    np.add.at(cm, (np.asarray(y_true, int), np.asarray(y_pred, int)), 1)
    return cm


def classification_metrics(cm):
    """Overall accuracy, per-class P/R/F1 (one-vs-rest accuracy too, as in manuscript Table 7), macro averages and MCC.

    Macro averages skip classes that have neither samples nor predictions.
    """
    cm = np.asarray(cm, np.float64)
    total, tp = cm.sum(), np.diag(cm)
    support, predicted = cm.sum(1), cm.sum(0)
    fp, fn = predicted - tp, support - tp
    tn = total - tp - fp - fn
    div = lambda a, b: np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=b > 0)
    precision, recall = div(tp, predicted), div(tp, support)
    f1 = div(2 * precision * recall, precision + recall)
    acc_ovr = div(tp + tn, np.full_like(tp, total))
    used = (support > 0) | (predicted > 0)
    pk, tk = (cm * np.eye(len(cm))).sum(), total
    cov_pt, cov_pp, cov_tt = pk * tk - (support * predicted).sum(), tk ** 2 - (predicted ** 2).sum(), tk ** 2 - (support ** 2).sum()
    mcc = cov_pt / np.sqrt(cov_pp * cov_tt) if cov_pp > 0 and cov_tt > 0 else 0.0
    return {
        "n": int(total), "accuracy": float(tp.sum() / total) if total else 0.0,
        "accuracy_ovr_macro": float(acc_ovr[used].mean()) if used.any() else 0.0,
        "precision_macro": float(precision[used].mean()) if used.any() else 0.0,
        "recall_macro": float(recall[used].mean()) if used.any() else 0.0,
        "f1_macro": float(f1[used].mean()) if used.any() else 0.0,
        "mcc": float(mcc),
        "per_class": {"precision": precision.tolist(), "recall": recall.tolist(), "f1": f1.tolist(),
                      "accuracy_ovr": acc_ovr.tolist(), "support": support.astype(int).tolist()},
    }
