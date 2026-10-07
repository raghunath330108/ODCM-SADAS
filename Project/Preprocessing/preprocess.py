"""Preprocessing driver: sections 3.1 (z-score) and 3.2 (calibration) for ODCM-SADAS.

Usage (run from this folder):
  python preprocess.py process --samples-root ../../Samples/senses --out ../Output [--limit N]
  python preprocess.py stats   --out ../Output          # train-split mean/std -> norm_stats.json

Output layout:
  images/{token}_{cam}.jpg   640x640 letterboxed RGB (uint8) for YOLO11s
  labels/{token}_{cam}.txt   YOLO labels on the same canvas
  sensors/{token}.npz        raw physical-unit LiDAR/RADAR maps (image plane + BEV), points, objects, calibration
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from calibration import intrinsics, reference_vehicle_to_camera, sensor_to_reference_vehicle, sensor_to_vehicle
from config import CAMERAS, DEFAULT_OUT, DEFAULT_SAMPLES_ROOT, LIDAR, RADARS
from io_utils import load_image_rgb, load_lidar, load_metadata, load_radar
from objects import object_table, yolo_labels
from rasterize import SCALE, PAD_Y, letterbox_image, lidar_bev, lidar_image_maps, radar_image_maps


def lidar_points_vehicle(meta, sample_dir):
    raw = load_lidar(sample_dir)
    xyz = sensor_to_reference_vehicle(meta, LIDAR).apply(raw[:, :3].astype(np.float64))
    return xyz, raw[:, 3].astype(np.float32)


def radar_points_vehicle(meta, sample_dir):
    """All radars merged into the reference vehicle frame (Eq. 5-6 generalised, ego-motion compensated)."""
    xyz, rcs, vxy, dyn, sid = [], [], [], [], []
    for i, name in enumerate(RADARS):
        p = load_radar(sample_dir, name)
        if len(p) == 0:
            continue
        pose = sensor_to_reference_vehicle(meta, name)
        loc = np.stack([p["x"], p["y"], p["z"]], 1).astype(np.float64)
        vel = np.stack([p["vx_comp"], p["vy_comp"], np.zeros(len(p))], 1).astype(np.float64)
        xyz.append(pose.apply(loc))
        vxy.append(pose.rotate(vel)[:, :2])
        rcs.append(p["rcs"].astype(np.float32))
        dyn.append(p["dyn_prop"].astype(np.float32))
        sid.append(np.full(len(p), i, np.float32))
    if not xyz:
        return np.zeros((0, 3)), np.zeros(0, np.float32), np.zeros((0, 2)), np.zeros(0), np.zeros(0)
    return (np.concatenate(xyz), np.concatenate(rcs), np.concatenate(vxy).astype(np.float32),
            np.concatenate(dyn), np.concatenate(sid))


def canvas_intrinsics(K):
    Kc = K.copy()
    Kc[0] *= SCALE
    Kc[1] *= SCALE
    Kc[1, 2] += PAD_Y
    return Kc


def process_sample(sample_dir, out_dir):
    sample_dir, out_dir = Path(sample_dir), Path(out_dir)
    meta = load_metadata(sample_dir)
    tok = meta["sample_token"]
    for sub in ("images", "labels", "sensors"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    lidar_v, lidar_int = lidar_points_vehicle(meta, sample_dir)
    radar_v, radar_rcs, radar_vxy, radar_dyn, radar_sid = radar_points_vehicle(meta, sample_dir)

    lidar_map, radar_map, T_cam, K_canvas = [], [], [], []
    for cam in CAMERAS:
        img = letterbox_image(load_image_rgb(sample_dir, cam))
        cv2.imwrite(str(out_dir / "images" / f"{tok}_{cam}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        (out_dir / "labels" / f"{tok}_{cam}.txt").write_text("\n".join(yolo_labels(meta, cam)))
        lidar_map.append(lidar_image_maps(meta, cam, lidar_v, lidar_int))
        radar_map.append(radar_image_maps(meta, cam, radar_v, radar_rcs, radar_vxy))
        T_cam.append(reference_vehicle_to_camera(meta, cam).matrix())
        K_canvas.append(canvas_intrinsics(intrinsics(meta, cam)))

    z0 = sensor_to_vehicle(meta, LIDAR).T[2]  # BEV heights are relative to the LiDAR origin
    lidar_rel = lidar_v - np.array([0, 0, z0])
    bev = {f"bev_lidar_{preset}": lidar_bev(lidar_rel, preset) for preset in ("front", "full360")}

    feats, cls, inst, atok = object_table(meta, lidar_v, lidar_int, radar_v, radar_rcs, radar_vxy)
    np.savez_compressed(
        out_dir / "sensors" / f"{tok}.npz",
        split=np.array(meta["split"]), scene=np.array(meta["scene_token"]),
        timestamp=np.array(meta["timestamp"], np.int64), cameras=np.array(CAMERAS),
        lidar_map=np.stack(lidar_map).astype(np.float16), radar_map=np.stack(radar_map).astype(np.float16),
        lidar_pts=np.concatenate([lidar_v, lidar_int[:, None]], 1).astype(np.float32),
        radar_pts=np.concatenate([radar_v, radar_rcs[:, None], radar_vxy, radar_dyn[:, None],
                                  radar_sid[:, None]], 1).astype(np.float32),
        T_cam_from_vehicle=np.stack(T_cam).astype(np.float64), K_canvas=np.stack(K_canvas).astype(np.float64),
        obj_feats=feats, obj_class=cls, obj_instance=inst, obj_ann_token=atok, **bev)
    return tok


def cmd_process(args):
    root = Path(args.samples_root)
    dirs = sorted(p for p in root.iterdir() if (p / "metadata.json").exists())
    if args.limit:
        dirs = dirs[:args.limit]
    done = []
    for i, d in enumerate(dirs, 1):
        done.append(process_sample(d, args.out))
        print(f"[{i}/{len(dirs)}] {done[-1]}", flush=True)
    return done


def cmd_stats(args):
    from normalization import compute_stats
    s = compute_stats(args.out, split=args.split)
    print(f"norm_stats.json written from {s['n_samples']} '{args.split}' samples")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("process")
    p.add_argument("--samples-root", default=str(DEFAULT_SAMPLES_ROOT))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(fn=cmd_process)
    p = sub.add_parser("stats")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--split", default="train")
    p.set_defaults(fn=cmd_stats)
    a = ap.parse_args()
    a.fn(a)
