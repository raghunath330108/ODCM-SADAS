"""Per-object processing: 3D boxes in the vehicle frame, in-box sensor statistics, YOLO labels."""
import numpy as np
from pyquaternion import Quaternion

from calibration import vehicle_to_global
from config import CANVAS, CLASS_ID, LIDAR, OBJ_FEATURES
from rasterize import PAD_Y, NEW_H, box_to_canvas

RADAR_BOX_MARGIN = 1.0  # metres; radar positions are coarse, so its in-box test uses an enlarged footprint
MIN_BOX_PX = 2.0


def box_in_vehicle(meta, ann):
    """Annotation (global) -> centre, yaw, rotation matrix in the vehicle frame at the LiDAR timestamp."""
    ego = vehicle_to_global(meta, LIDAR)
    R_box = ego.R.T @ Quaternion(ann["rotation"]).rotation_matrix
    c = ego.inv().apply(np.asarray(ann["translation"], dtype=np.float64)[None])[0]
    yaw = float(np.arctan2(R_box[1, 0], R_box[0, 0]))
    return c, yaw, R_box


def points_in_box(pts, c, R, size, margin_xy=0.0):
    w, l, h = size  # nuScenes: [width (y), length (x), height (z)]
    local = (pts - c) @ R
    return (np.abs(local[:, 0]) <= l / 2 + margin_xy) & (np.abs(local[:, 1]) <= w / 2 + margin_xy) \
        & (np.abs(local[:, 2]) <= h / 2 + 0.25)


def object_table(meta, lidar_v, lidar_int, radar_v, radar_rcs, radar_vxy):
    """Returns feats (N, len(OBJ_FEATURES)) float32, class ids, visibility, instance tokens, ann tokens."""
    rows, cls, inst, toks = [], [], [], []
    for a in meta["annotations"]:
        c, yaw, R = box_in_vehicle(meta, a)
        w, l, h = a["size"]
        rng = float(np.hypot(c[0], c[1]))
        bearing = float(np.arctan2(c[1], c[0]))
        li = points_in_box(lidar_v, c, R, a["size"]) if len(lidar_v) else np.zeros(0, bool)
        ri = points_in_box(radar_v, c, R, a["size"], RADAR_BOX_MARGIN) if len(radar_v) else np.zeros(0, bool)
        nl, nr = int(li.sum()), int(ri.sum())
        f = {
            "x": c[0], "y": c[1], "z": c[2], "w": w, "l": l, "h": h,
            "yaw_sin": np.sin(yaw), "yaw_cos": np.cos(yaw), "range": rng,
            "bearing_sin": np.sin(bearing), "bearing_cos": np.cos(bearing),
            "visibility": a["visibility"],
            "num_lidar_pts_ann": a["num_lidar_pts"], "num_radar_pts_ann": a["num_radar_pts"],
            "n_lidar_in_box": nl, "n_radar_in_box": nr,
            "lidar_int_mean": float(lidar_int[li].mean()) if nl else 0.0,
            "radar_rcs_mean": float(radar_rcs[ri].mean()) if nr else 0.0,
            "radar_vx_mean": float(radar_vxy[ri, 0].mean()) if nr else 0.0,
            "radar_vy_mean": float(radar_vxy[ri, 1].mean()) if nr else 0.0,
            "has_lidar": float(nl > 0), "has_radar": float(nr > 0),
        }
        rows.append([f[k] for k in OBJ_FEATURES])
        cls.append(CLASS_ID[a["class_name"]])
        inst.append(a["instance_token"])
        toks.append(a["token"])
    feats = np.asarray(rows, np.float32).reshape(-1, len(OBJ_FEATURES))
    return feats, np.asarray(cls, np.int64), np.asarray(inst), np.asarray(toks)


def labelled_annotations(meta, cam):
    """Yield (annotation, (x1, y1, x2, y2) on the 640x640 canvas) for every object kept as a label in this camera."""
    for a in meta["annotations"]:
        cb = a.get("camera_boxes", {}).get(cam)
        if not cb:
            continue
        x1, y1, x2, y2 = box_to_canvas(cb["bbox_xyxy_clipped"])
        y1, y2 = max(y1, PAD_Y), min(y2, PAD_Y + NEW_H)
        x1, x2 = max(x1, 0.0), min(x2, float(CANVAS))
        if x2 - x1 < MIN_BOX_PX or y2 - y1 < MIN_BOX_PX:
            continue
        yield a, (x1, y1, x2, y2)


def yolo_labels(meta, cam):
    """Normalised YOLO lines (cls cx cy w h) on the 640x640 letterboxed canvas for one camera."""
    return [f"{CLASS_ID[a['class_name']]} {(x1 + x2) / 2 / CANVAS:.6f} {(y1 + y2) / 2 / CANVAS:.6f} "
            f"{(x2 - x1) / CANVAS:.6f} {(y2 - y1) / CANVAS:.6f}" for a, (x1, y1, x2, y2) in labelled_annotations(meta, cam)]
