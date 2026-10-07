"""Detection metrics for the fitness (Eq. 12) and for model selection.

mAP@0.5 / mAP@0.5:0.95 follow the Ultralytics definition. Accuracy and F1 use one operating point
(confidence >= conf_op, IoU >= 0.5), micro-averaged over all objects:
    Accuracy = TP / (TP + FP + FN)        F1 = 2TP / (2TP + FP + FN)
"""
import numpy as np
import torch
from ultralytics.utils.metrics import ap_per_class, box_iou
from ultralytics.utils.nms import non_max_suppression

import fusion_paths  # noqa: F401
from config import CANVAS
from fusion_data import model_inputs, to_device

IOU_THRS = np.linspace(0.5, 0.95, 10)


def batch_gts(batch):
    """Per-image ground truth (m, 5) = [cls, x1, y1, x2, y2] in canvas pixels."""
    out = []
    for j in range(len(batch["z0"])):
        sel = batch["batch_idx"] == j
        c, b = batch["cls"][sel], batch["bboxes"][sel] * CANVAS
        xy = torch.cat([b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2], 1)
        out.append(torch.cat([c, xy], 1).cpu())
    return out


def match(pred_cls, gt_cls, iou):
    """Greedy one-to-one matching per IoU threshold -> (Np, 10) bool (Ultralytics rule)."""
    correct = np.zeros((len(pred_cls), len(IOU_THRS)), bool)
    iou = iou * (gt_cls[:, None] == pred_cls[None])
    for i, t in enumerate(IOU_THRS):
        m = np.array(np.nonzero(iou >= t)).T
        if m.shape[0]:
            if m.shape[0] > 1:
                m = np.concatenate([m, iou[m[:, 0], m[:, 1]][:, None]], 1)
                m = m[m[:, 2].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1].astype(int), i] = True
    return correct


class DetectionStats:
    def __init__(self, conf_op=0.25):
        self.conf_op = conf_op
        self.tp, self.conf, self.pcls, self.tcls = [], [], [], []

    def update(self, preds, gts):
        for p, g in zip(preds, gts):
            p, g = p.detach().cpu(), g.cpu()
            self.tcls.append(g[:, 0].numpy())
            if len(p) == 0:
                continue
            if len(g):
                correct = match(p[:, 5].numpy(), g[:, 0].numpy(), box_iou(g[:, 1:], p[:, :4]).numpy())
            else:
                correct = np.zeros((len(p), len(IOU_THRS)), bool)
            self.tp.append(correct)
            self.conf.append(p[:, 4].numpy())
            self.pcls.append(p[:, 5].numpy())

    def compute(self):
        tcls = np.concatenate(self.tcls) if self.tcls else np.zeros(0)
        res = dict(map50=0.0, map=0.0, accuracy=0.0, f1=0.0, precision=0.0, recall=0.0, n_gt=int(len(tcls)))
        if not self.tp:
            return res
        tp, conf, pcls = np.concatenate(self.tp), np.concatenate(self.conf), np.concatenate(self.pcls)
        ap = ap_per_class(tp, conf, pcls, tcls)[5]
        if len(ap):
            res["map50"], res["map"] = float(ap[:, 0].mean()), float(ap.mean())
        sel = conf >= self.conf_op
        TP = float(tp[sel, 0].sum())
        FP, FN = float(sel.sum()) - TP, len(tcls) - TP
        res.update(accuracy=TP / max(TP + FP + FN, 1), f1=2 * TP / max(2 * TP + FP + FN, 1),
                   precision=TP / max(TP + FP, 1), recall=TP / max(TP + FN, 1))
        return res


def detections(y, conf=0.001, iou=0.7):
    """Eval-mode Detect output (B, 4+nc, N) -> list of (n, 6) [x1, y1, x2, y2, conf, cls]."""
    return non_max_suppression(y.float(), conf_thres=conf, iou_thres=iou, max_det=300)


@torch.no_grad()
def evaluate_model(model, loader, device, amp=False, max_batches=None, conf_op=0.25):
    model.eval()
    stats = DetectionStats(conf_op)
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = to_device(batch, device)
        with torch.autocast(device_type=device.type, enabled=amp):
            y = model(*model_inputs(batch))[0][0]
        stats.update(detections(y), batch_gts(batch))
    return stats.compute()
