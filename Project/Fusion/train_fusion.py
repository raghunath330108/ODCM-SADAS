"""Train the fused detector ONCE with equal fusion weights (1/3 each); ASSA tunes the weights afterwards.

Real run (GPU machine):  python train_fusion.py
Trial run (any machine): python train_fusion.py --trial        (1 epoch, 2 tiny batches; nothing meaningful is learned)
"""
import argparse
import csv
import math
import time

import torch
from torch.utils.data import DataLoader, Subset

import fusion_paths as fp
from fusion_data import FusionViews, collate, model_inputs, to_device
from fusion_eval import evaluate_model
from fusion_model import FusionYOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr-yolo", type=float, default=1e-4)
    ap.add_argument("--lr-new", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--warmup-iters", type=int, default=200)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--trial", action="store_true")
    a = ap.parse_args()
    if a.trial:
        a.epochs, a.batch, a.workers, a.max_batches = 1, 2, 0, 2
    out = fp.OUT / ("fusion_trial" if a.trial else "fusion") if a.out is None else fp.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else "cpu" if a.device == "auto" else a.device)
    amp = device.type == "cuda" and not a.no_amp
    torch.manual_seed(0)
    train_ds, val_ds = FusionViews("train"), FusionViews("val")
    if a.trial:
        train_ds, val_ds = Subset(train_ds, range(a.batch * 2)), Subset(val_ds, range(a.batch * 2))
    mk = lambda ds, shuffle: DataLoader(ds, batch_size=a.batch, shuffle=shuffle, drop_last=shuffle, num_workers=a.workers,
                                        collate_fn=collate, persistent_workers=a.workers > 0)
    train_loader, val_loader = mk(train_ds, True), mk(val_ds, False)
    print(f"device={device} amp={amp} train views={len(train_ds)} val views={len(val_ds)} batch={a.batch}")

    model = FusionYOLO().to(device)
    criterion = model.det.init_criterion()
    yolo_ids = {id(p) for p in model.det.parameters()}
    groups = [{"params": [p for p in model.parameters() if id(p) in yolo_ids], "lr": a.lr_yolo},
              {"params": [p for p in model.parameters() if id(p) not in yolo_ids], "lr": a.lr_new}]
    opt = torch.optim.AdamW(groups, weight_decay=a.wd)
    n_it = len(train_loader) if a.max_batches is None else min(len(train_loader), a.max_batches)
    total = a.epochs * n_it
    warm = min(a.warmup_iters, total // 2) if total > 1 else 0
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))
    scaler = torch.amp.GradScaler(enabled=amp)

    log = open(out / "train_log.csv", "w", newline="")
    wr = csv.writer(log)
    wr.writerow(["epoch", "box", "cls", "dfl", "lr_new", "map50", "map", "accuracy", "f1", "seconds"])
    best = -1.0
    for epoch in range(1, a.epochs + 1):
        model.train()
        t0, run = time.time(), torch.zeros(3)
        for i, batch in enumerate(train_loader):
            if a.max_batches is not None and i >= a.max_batches:
                break
            batch = to_device(batch, device)
            with torch.autocast(device_type=device.type, enabled=amp):
                preds, _ = model(*model_inputs(batch))
                loss, items = criterion(preds, batch)
                loss = loss.sum()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += torch.stack([torch.as_tensor(v, dtype=torch.float32).cpu() for v in items.values()]) if isinstance(items, dict) else items.cpu()
        run /= max(i + 1 if a.max_batches is None else min(i + 1, a.max_batches), 1)
        m = {}
        if epoch % a.eval_every == 0 or epoch == a.epochs:
            m = evaluate_model(model, val_loader, device, amp, a.max_batches)
        wr.writerow([epoch, *[f"{v:.4f}" for v in run], f"{opt.param_groups[1]['lr']:.2e}", m.get("map50", ""), m.get("map", ""),
                     m.get("accuracy", ""), m.get("f1", ""), f"{time.time() - t0:.0f}"])
        log.flush()
        print(f"epoch {epoch}/{a.epochs} loss box/cls/dfl {run[0]:.3f}/{run[1]:.3f}/{run[2]:.3f} "
              + (f"val mAP50 {m['map50']:.4f} mAP {m['map']:.4f} acc {m['accuracy']:.4f} f1 {m['f1']:.4f} " if m else "")
              + f"({time.time() - t0:.0f}s)", flush=True)
        ck = {"model": model.state_dict(), "epoch": epoch, "metrics": m, "args": vars(a)}
        torch.save(ck, out / "last.pt")
        if m and m["map50"] >= best:
            best = m["map50"]
            torch.save(ck, out / "best.pt")
    log.close()
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
