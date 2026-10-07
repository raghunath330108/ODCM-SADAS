"""Shared constants for ODCM-SADAS preprocessing (manuscript sections 3.1 and 3.2)."""
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLES_ROOT = CODE_ROOT / "Samples" / "scenes"
DEFAULT_METADATA_DIR = DEFAULT_SAMPLES_ROOT
DEFAULT_MANIFEST = CODE_ROOT / "collection_manifest.json"
DEFAULT_OUT = CODE_ROOT / "Project" / "Output"

# Six classes (manuscript Table 1). Order defines the YOLO class ids.
CLASSES = ["bicycle", "bus", "car", "motorcycle", "person", "truck"]
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}

CAMERAS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
RADARS = ["RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT", "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT"]
LIDAR = "LIDAR_TOP"

# Common canvas shared by every modality (YOLO11 stride 32 -> 640 = 20 x 32).
SRC_W, SRC_H = 1600, 900
CANVAS = 640
PAD_VALUE = 114  # Ultralytics letterbox gray
MIN_DEPTH = 0.5  # metres; points closer than this to the camera plane are dropped

# Splat radius (pixels) used when drawing sparse LiDAR/RADAR points into image-plane maps.
SPLAT_LIDAR = 2
SPLAT_RADAR = 4

# nuScenes default RadarPointCloud state filters.
RADAR_FILTER = {"invalid_states": {0}, "dynprop_states": set(range(8)), "ambig_states": {3}}

# BEV grids. Both give 640x640 natively (no resampling).
#   front  : manuscript window, 0.0625 m cells
#   full360: +-40 m around the vehicle, 0.125 m cells (covers back cameras)
# z limits are heights relative to the LiDAR sensor (manuscript Z in [-2, 0.5]).
BEV_PRESETS = {
    "front": {"x": (0.0, 40.0), "y": (-20.0, 20.0), "z": (-2.0, 0.5), "res": 0.0625},
    "full360": {"x": (-40.0, 40.0), "y": (-40.0, 40.0), "z": (-2.0, 0.5), "res": 0.125},
}

# Object-level feature vector (per annotation) kept for the LSTM stage.
OBJ_FEATURES = [
    "x", "y", "z", "w", "l", "h", "yaw_sin", "yaw_cos", "range", "bearing_sin", "bearing_cos",
    "visibility", "num_lidar_pts_ann", "num_radar_pts_ann",
    "n_lidar_in_box", "n_radar_in_box", "lidar_int_mean", "radar_rcs_mean", "radar_vx_mean",
    "radar_vy_mean", "has_lidar", "has_radar",
]
OBJ_MASK_FEATURES = {"has_lidar", "has_radar", "visibility"}  # not z-scored
