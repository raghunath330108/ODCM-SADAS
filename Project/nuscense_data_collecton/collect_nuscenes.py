"""Collect a camera-visible, class-balanced nuScenes subset."""

import argparse
import json
import logging
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation


LOGGER = logging.getLogger("nuscenes_collection")
CAMERA_CHANNELS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)
LIDAR_CHANNEL = "LIDAR_TOP"
RADAR_CHANNELS = (
    "RADAR_FRONT",
    "RADAR_FRONT_LEFT",
    "RADAR_FRONT_RIGHT",
    "RADAR_BACK_LEFT",
    "RADAR_BACK_RIGHT",
)
REQUIRED_CHANNELS = CAMERA_CHANNELS + (LIDAR_CHANNEL,) + RADAR_CHANNELS
CLASS_TO_CATEGORY = {
    "car": "vehicle.car",
    "truck": "vehicle.truck",
    "person": "human.pedestrian.adult",
    "bus": "vehicle.bus.rigid",
    "bicycle": "vehicle.bicycle",
    "motorcycle": "vehicle.motorcycle",
}
CATEGORY_TO_CLASS = {category: name for name, category in CLASS_TO_CATEGORY.items()}
CLASS_COLORS = {
    "car": (40, 220, 40),
    "truck": (255, 160, 30),
    "person": (30, 210, 255),
    "bus": (255, 80, 200),
    "bicycle": (255, 235, 40),
    "motorcycle": (180, 100, 255),
}
BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
SPLIT_NAMES = ("train", "val", "test")
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
IMAGE_SIZE = (1600, 900)
SAMPLE_QUOTA = 250
RANDOM_SEED = 42


def parse_args(argv=None):
    workspace = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Select, split, and collect camera-visible nuScenes samples."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="nuScenes release directory containing the selected version folder",
    )
    parser.add_argument(
        "--version",
        default="v1.0-trainval",
        help="nuScenes metadata version directory (default: v1.0-trainval)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=workspace / "Samples",
        help="New output directory (default: ./Samples)",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED, help="Deterministic selection seed"
    )
    parser.add_argument(
        "--quota-per-class",
        type=int,
        default=SAMPLE_QUOTA,
        help=f"Target eligible frame memberships per class (default: {SAMPLE_QUOTA})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report eligibility and split counts without copying data",
    )
    return parser.parse_args(argv)


def load_json(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, allow_nan=False)
        file.write("\n")
    temporary_path.replace(path)


def rotation_matrix(quaternion_wxyz):
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("nuScenes rotations must be finite wxyz quaternions")
    return Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()


def allowed_image_box(category_name, bbox_xyxy, width, height):
    x1, y1, x2, y2 = bbox_xyxy
    if not np.isfinite(bbox_xyxy).all() or x2 <= x1 or y2 <= y1:
        return None
    area = (x2 - x1) * (y2 - y1)
    visible_width = max(0.0, min(float(width), x2) - max(0.0, x1))
    visible_height = max(0.0, min(float(height), y2) - max(0.0, y1))
    visible_fraction = (visible_width * visible_height) / area
    tolerance = 0.10 if category_name in {
        "vehicle.truck",
        "vehicle.bus.rigid",
    } else 0.0
    if visible_fraction + 1e-9 < 1.0 - tolerance:
        return None
    clipped = [
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    ]
    return {
        "bbox_xyxy": [float(value) for value in bbox_xyxy],
        "bbox_xyxy_clipped": clipped,
        "visible_fraction": float(visible_fraction),
    }


def project_annotation(annotation, camera_transform, width, height):
    """Project a global nuScenes 3D box into one camera image."""
    size = np.asarray(annotation["size"], dtype=np.float64)
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
        return None

    # nuScenes boxes are ordered width, length, height in the box's local frame.
    length, width_box, height_box = size[1], size[0], size[2]
    x_corners = length / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y_corners = width_box / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = height_box / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    local_corners = np.vstack((x_corners, y_corners, z_corners))
    global_corners = (
        rotation_matrix(annotation["rotation"]) @ local_corners
        + np.asarray(annotation["translation"], dtype=np.float64)[:, None]
    )

    rotation_global_camera, translation_global_camera, intrinsic = camera_transform
    camera_corners = rotation_global_camera.T @ (
        global_corners - translation_global_camera[:, None]
    )
    near_plane = 1e-5
    projected_points = [
        camera_corners[:, index]
        for index in range(camera_corners.shape[1])
        if camera_corners[2, index] > near_plane
    ]
    for first, second in BOX_EDGES:
        z_first = camera_corners[2, first]
        z_second = camera_corners[2, second]
        if (z_first - near_plane) * (z_second - near_plane) < 0:
            fraction = (near_plane - z_first) / (z_second - z_first)
            projected_points.append(
                camera_corners[:, first]
                + fraction * (camera_corners[:, second] - camera_corners[:, first])
            )
    if not projected_points:
        return None

    points = np.stack(projected_points, axis=1)
    image_points = intrinsic @ points
    image_points = image_points[:2] / image_points[2:3]
    if not np.isfinite(image_points).all():
        return None
    x1, y1 = image_points.min(axis=1)
    x2, y2 = image_points.max(axis=1)
    return allowed_image_box(
        annotation["_category_name"], (x1, y1, x2, y2), width, height
    )


def sample_split_targets(total):
    if total < 0:
        raise ValueError("sample count cannot be negative")
    train = int(total * SPLIT_RATIOS["train"] + 0.5)
    validation = int(total * SPLIT_RATIOS["val"] + 0.5)
    return {"train": train, "val": validation, "test": total - train - validation}


def solve_split_assignments(
    candidate_labels, class_targets, seed, candidate_visibility=None
):
    """Assign whole frames to splits while meeting per-class quotas exactly."""
    sample_tokens = sorted(candidate_labels)
    if not sample_tokens:
        raise ValueError("No eligible complete, camera-visible samples were found")

    variable_count = len(sample_tokens) * len(SPLIT_NAMES)
    class_split_pairs = [
        (class_name, split_name)
        for class_name in CLASS_TO_CATEGORY
        for split_name in SPLIT_NAMES
    ]
    class_split_rows = {
        pair: index for index, pair in enumerate(class_split_pairs)
    }
    class_constraint_count = len(class_split_rows)
    row_indices = []
    column_indices = []
    values = []
    for sample_index, token in enumerate(sample_tokens):
        sample_row = class_constraint_count + sample_index
        for split_index, split_name in enumerate(SPLIT_NAMES):
            variable = sample_index * len(SPLIT_NAMES) + split_index
            row_indices.append(sample_row)
            column_indices.append(variable)
            values.append(1.0)
            for class_name in candidate_labels[token]:
                row_indices.append(class_split_rows[(class_name, split_name)])
                column_indices.append(variable)
                values.append(1.0)

    constraints_matrix = coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(class_constraint_count + len(sample_tokens), variable_count),
    ).tocsc()
    targets_by_class = {
        class_name: sample_split_targets(target)
        for class_name, target in class_targets.items()
    }
    lower = np.full(class_constraint_count + len(sample_tokens), -np.inf)
    upper = np.full(class_constraint_count + len(sample_tokens), np.inf)
    upper[class_constraint_count:] = 1.0
    for (class_name, split_name), row in class_split_rows.items():
        required = targets_by_class.get(class_name, {}).get(split_name, 0)
        lower[row] = required

    rng = np.random.default_rng(seed)
    low_visibility_penalty = sum(class_targets.values()) + 1
    sample_costs = []
    for token in sample_tokens:
        low_visibility_classes = 0
        if candidate_visibility is not None:
            low_visibility_classes = sum(
                candidate_visibility[token][class_name] == 2
                for class_name in candidate_labels[token]
            )
        sample_costs.append(
            low_visibility_classes * low_visibility_penalty
            + 1.0
            + rng.random() * 1e-5
        )
    objective = np.repeat(sample_costs, len(SPLIT_NAMES))
    result = milp(
        c=objective,
        integrality=np.ones(variable_count, dtype=np.int8),
        bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
        constraints=LinearConstraint(constraints_matrix, lower, upper),
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        raise RuntimeError(
            "Could not satisfy all exact per-class split counts with whole frames: "
            f"{result.message}"
        )

    assignments = {}
    for sample_index, token in enumerate(sample_tokens):
        chosen = np.flatnonzero(
            result.x[sample_index * len(SPLIT_NAMES):(sample_index + 1) * len(SPLIT_NAMES)]
            > 0.5
        )
        if len(chosen) > 1:
            raise RuntimeError(f"Solver assigned sample {token} to multiple splits")
        if len(chosen) == 1:
            assignments[token] = SPLIT_NAMES[int(chosen[0])]

    actual = {
        class_name: {
            split_name: sum(
                split == split_name and class_name in candidate_labels[token]
                for token, split in assignments.items()
            )
            for split_name in SPLIT_NAMES
        }
        for class_name in CLASS_TO_CATEGORY
    }
    if any(
        actual[class_name][split_name] < targets_by_class[class_name][split_name]
        for class_name in CLASS_TO_CATEGORY
        for split_name in SPLIT_NAMES
    ):
        raise RuntimeError(
            f"Split solver returned class counts below their minimum targets: {actual}; "
            f"minimums {targets_by_class}"
        )
    return assignments, targets_by_class


def load_source_tables(dataset_root, version_dir):
    sensors = load_json(version_dir / "sensor.json")
    channel_by_sensor = {row["token"]: row["channel"] for row in sensors}
    calibrations = load_json(version_dir / "calibrated_sensor.json")
    calibration_by_token = {row["token"]: row for row in calibrations}
    channel_by_calibration = {
        row["token"]: channel_by_sensor[row["sensor_token"]] for row in calibrations
    }

    sample_data_by_sample = defaultdict(dict)
    for row in load_json(version_dir / "sample_data.json"):
        if not row["is_key_frame"]:
            continue
        channel = channel_by_calibration[row["calibrated_sensor_token"]]
        if channel in REQUIRED_CHANNELS:
            sample_data_by_sample[row["sample_token"]][channel] = row

    sample_rows = load_json(version_dir / "sample.json")
    samples = {row["token"]: row for row in sample_rows}
    scene_by_token = {
        row["token"]: row["name"] for row in load_json(version_dir / "scene.json")
    }

    annotation_categories = {
        row["token"]: row["name"] for row in load_json(version_dir / "category.json")
    }
    category_by_instance = {
        row["token"]: annotation_categories[row["category_token"]]
        for row in load_json(version_dir / "instance.json")
    }
    annotations_by_sample = defaultdict(list)
    for row in load_json(version_dir / "sample_annotation.json"):
        category_name = category_by_instance[row["instance_token"]]
        if category_name not in CATEGORY_TO_CLASS:
            continue
        row["_category_name"] = category_name
        row["_visibility_level"] = int(row["visibility_token"])
        if row["_visibility_level"] >= 2:
            annotations_by_sample[row["sample_token"]].append(row)

    pose_tokens = {
        row["ego_pose_token"]
        for channels in sample_data_by_sample.values()
        for channel, row in channels.items()
        if channel in REQUIRED_CHANNELS
    }
    pose_by_token = {
        row["token"]: row
        for row in load_json(version_dir / "ego_pose.json")
        if row["token"] in pose_tokens
    }
    return (
        samples,
        scene_by_token,
        sample_data_by_sample,
        annotations_by_sample,
        calibration_by_token,
        pose_by_token,
    )


def camera_transform(sample_data, calibration_by_token, pose_by_token):
    calibration = calibration_by_token[sample_data["calibrated_sensor_token"]]
    pose = pose_by_token[sample_data["ego_pose_token"]]
    intrinsic = np.asarray(calibration["camera_intrinsic"], dtype=np.float64)
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError("Camera calibration is missing a finite 3x3 intrinsic matrix")
    sensor_rotation = rotation_matrix(calibration["rotation"])
    ego_rotation = rotation_matrix(pose["rotation"])
    rotation_global_camera = ego_rotation @ sensor_rotation
    translation_global_camera = (
        ego_rotation @ np.asarray(calibration["translation"], dtype=np.float64)
        + np.asarray(pose["translation"], dtype=np.float64)
    )
    return rotation_global_camera, translation_global_camera, intrinsic


def build_candidates(
    dataset_root,
    samples,
    sample_data_by_sample,
    annotations_by_sample,
    calibration_by_token,
    pose_by_token,
):
    complete_samples = {}
    missing_by_channel = Counter()
    for sample_token, sample in samples.items():
        channels = sample_data_by_sample.get(sample_token, {})
        for channel in REQUIRED_CHANNELS:
            row = channels.get(channel)
            if row is None or not (dataset_root / row["filename"]).is_file():
                missing_by_channel[channel] += 1
        if all(
            channel in channels
            and (dataset_root / channels[channel]["filename"]).is_file()
            for channel in REQUIRED_CHANNELS
        ):
            complete_samples[sample_token] = channels

    candidates = {}
    high_visibility_candidates = {name: set() for name in CLASS_TO_CATEGORY}
    candidate_visibility = {}
    for sample_token, channels in complete_samples.items():
        transforms = {}
        for channel in CAMERA_CHANNELS:
            row = channels[channel]
            if (row["width"], row["height"]) != IMAGE_SIZE:
                continue
            transforms[channel] = camera_transform(
                row, calibration_by_token, pose_by_token
            )
        eligible_annotations = []
        for annotation in annotations_by_sample.get(sample_token, ()):
            category_name = annotation["_category_name"]
            class_name = CATEGORY_TO_CLASS[category_name]
            boxes = {}
            for channel, transform in transforms.items():
                box = project_annotation(
                    annotation, transform, IMAGE_SIZE[0], IMAGE_SIZE[1]
                )
                if box is not None:
                    boxes[channel] = box
            if not boxes:
                continue
            level = annotation["_visibility_level"]
            if level >= 3:
                high_visibility_candidates[class_name].add(sample_token)
            eligible_annotations.append(
                {
                    "token": annotation["token"],
                    "instance_token": annotation["instance_token"],
                    "category_name": category_name,
                    "class_name": class_name,
                    "visibility": level,
                    "translation": annotation["translation"],
                    "size": annotation["size"],
                    "rotation": annotation["rotation"],
                    "num_lidar_pts": int(annotation.get("num_lidar_pts", 0)),
                    "num_radar_pts": int(annotation.get("num_radar_pts", 0)),
                    "camera_boxes": boxes,
                }
            )
        if eligible_annotations:
            candidates[sample_token] = {
                "channels": channels,
                "annotations": eligible_annotations,
            }
    candidate_labels = {}
    class_candidates = {name: set() for name in CLASS_TO_CATEGORY}
    for sample_token, candidate in candidates.items():
        labels = set(candidate_visibility.get(sample_token, {}))
        for annotation in candidate["annotations"]:
            class_name = annotation["class_name"]
            labels.add(class_name)
            candidate_visibility.setdefault(sample_token, {})[class_name] = max(
                candidate_visibility.get(sample_token, {}).get(class_name, 0),
                annotation["visibility"],
            )
        for class_name in labels:
            class_candidates[class_name].add(sample_token)
        if labels:
            candidate_labels[sample_token] = labels

    return (
        candidates,
        candidate_labels,
        class_candidates,
        candidate_visibility,
        high_visibility_candidates,
        complete_samples,
        missing_by_channel,
    )


def validate_selected_images(dataset_root, selected_tokens, candidates):
    invalid = {}
    for token in sorted(selected_tokens):
        for channel in CAMERA_CHANNELS:
            row = candidates[token]["channels"][channel]
            image_path = dataset_root / row["filename"]
            try:
                with Image.open(image_path) as image:
                    if image.size != IMAGE_SIZE:
                        invalid[token] = (
                            f"{channel} is {image.width}x{image.height}, "
                            f"expected {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}"
                        )
                        break
                    image.verify()
            except OSError as error:
                invalid[token] = f"{channel} cannot be decoded: {error}"
                break
    return invalid


def draw_box_overlays(camera_dir, annotations):
    overlay_dir = camera_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for channel in CAMERA_CHANNELS:
        source = camera_dir / f"{channel}.jpg"
        with Image.open(source) as image:
            image = image.convert("RGB")
            draw = ImageDraw.Draw(image)
            for annotation in annotations:
                box = annotation["camera_boxes"].get(channel)
                if box is None:
                    continue
                x1, y1, x2, y2 = box["bbox_xyxy_clipped"]
                color = CLASS_COLORS[annotation["class_name"]]
                draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
                draw.text(
                    (x1, max(0, y1 - 16)),
                    f"{annotation['class_name']} v{annotation['visibility']}",
                    fill=color,
                    stroke_width=1,
                    stroke_fill=(0, 0, 0),
                )
            image.save(overlay_dir / f"{channel}.jpg", quality=95)


def calibration_record(sample_data, calibration_by_token, pose_by_token):
    calibrated_sensor = calibration_by_token[sample_data["calibrated_sensor_token"]]
    ego_pose = pose_by_token[sample_data["ego_pose_token"]]
    return {
        "sample_data_token": sample_data["token"],
        "timestamp": sample_data["timestamp"],
        "source_filename": sample_data["filename"],
        "calibrated_sensor_token": calibrated_sensor["token"],
        "calibrated_sensor": {
            "translation": calibrated_sensor["translation"],
            "rotation": calibrated_sensor["rotation"],
            "camera_intrinsic": calibrated_sensor.get("camera_intrinsic", []),
        },
        "ego_pose_token": ego_pose["token"],
        "ego_pose": {
            "translation": ego_pose["translation"],
            "rotation": ego_pose["rotation"],
        },
    }


def collect_sample(
    dataset_root,
    staging_dir,
    sample_token,
    split_name,
    sample,
    scene_name,
    candidate,
    calibration_by_token,
    pose_by_token,
):
    sample_dir = staging_dir / "scenes" / sample_token
    camera_dir = sample_dir / "camera"
    lidar_dir = sample_dir / "lidar"
    radar_dir = sample_dir / "radar"
    camera_dir.mkdir(parents=True)
    lidar_dir.mkdir()
    radar_dir.mkdir()

    channel_records = candidate["channels"]
    for channel in CAMERA_CHANNELS:
        source = dataset_root / channel_records[channel]["filename"]
        shutil.copy2(source, camera_dir / f"{channel}.jpg")

    lidar_source = dataset_root / channel_records[LIDAR_CHANNEL]["filename"]
    shutil.copy2(lidar_source, lidar_dir / "LIDAR_TOP.pcd.bin")

    for channel in RADAR_CHANNELS:
        source = dataset_root / channel_records[channel]["filename"]
        shutil.copy2(source, radar_dir / f"{channel}.pcd")

    annotations = candidate["annotations"]
    camera_box_counts = Counter(
        channel
        for annotation in annotations
        for channel in annotation["camera_boxes"]
    )
    primary_camera = max(
        CAMERA_CHANNELS,
        key=lambda channel: (
            camera_box_counts[channel],
            -CAMERA_CHANNELS.index(channel),
        ),
    )
    draw_box_overlays(camera_dir, annotations)
    sensor_calibrations = {
        channel: calibration_record(
            channel_records[channel], calibration_by_token, pose_by_token
        )
        for channel in REQUIRED_CHANNELS
    }
    sensor_calibrations["CAM_FRONT"]["calibrated_sensor"].update(
        calibration_by_token[
            channel_records["CAM_FRONT"]["calibrated_sensor_token"]
        ]
    )
    for channel in REQUIRED_CHANNELS:
        record = sensor_calibrations[channel]["calibrated_sensor"]
        if "token" in record:
            record.pop("token")
        if "sensor_token" in record:
            record.pop("sensor_token")

    primary_calibration = sensor_calibrations[primary_camera]["calibrated_sensor"]
    lidar_calibration = sensor_calibrations[LIDAR_CHANNEL]["calibrated_sensor"]
    radar_calibration = sensor_calibrations["RADAR_FRONT"]["calibrated_sensor"]
    metadata = {
        "sample_token": sample_token,
        "scene_token": sample["scene_token"],
        "scene_name": scene_name,
        "timestamp": sample["timestamp"],
        "split": split_name,
        "primary_camera": primary_camera,
        "image_size": {"width": IMAGE_SIZE[0], "height": IMAGE_SIZE[1]},
        "visibility_policy": (
            "Visibility 3/4 is preferred; visibility 2 is used only as needed "
            "to meet exact joint per-class and split counts."
        ),
        "calibration": {
            "camera": {
                "translation": primary_calibration["translation"],
                "rotation": primary_calibration["rotation"],
                "camera_intrinsic": primary_calibration["camera_intrinsic"],
            },
            "lidar": {
                "translation": lidar_calibration["translation"],
                "rotation": lidar_calibration["rotation"],
            },
            "radar": {
                "translation": radar_calibration["translation"],
                "rotation": radar_calibration["rotation"],
            },
        },
        "sensor_calibrations": sensor_calibrations,
        "annotations": annotations,
        "labels": sorted({annotation["class_name"] for annotation in annotations}),
        "point_count_source": "nuScenes sample_annotation num_lidar_pts and num_radar_pts",
    }
    write_json(sample_dir / "metadata.json", metadata)


def class_split_counts(candidate_labels, assignments):
    return {
        class_name: {
            split_name: sum(
                assigned_split == split_name and class_name in candidate_labels[token]
                for token, assigned_split in assignments.items()
            )
            for split_name in SPLIT_NAMES
        }
        for class_name in CLASS_TO_CATEGORY
    }


def run(args):
    dataset_root = args.dataset_root.expanduser().resolve()
    version_dir = (dataset_root / args.version).resolve()
    output_dir = args.output_dir.expanduser().resolve()
    staging_dir = output_dir.with_name(f"{output_dir.name}.staging")
    if not dataset_root.is_dir() or not version_dir.is_dir():
        raise SystemExit(f"nuScenes root/version directory not found: {version_dir}")
    if output_dir.exists() or staging_dir.exists():
        raise SystemExit(
            f"Refusing to overwrite existing output or staging directory: "
            f"{output_dir} / {staging_dir}"
        )
    if args.quota_per_class <= 0:
        raise ValueError("--quota-per-class must be positive")

    (
        samples,
        scene_by_token,
        sample_data_by_sample,
        annotations_by_sample,
        calibration_by_token,
        pose_by_token,
    ) = load_source_tables(dataset_root, version_dir)
    (
        candidates,
        candidate_labels,
        class_candidates,
        candidate_visibility,
        high_visibility_candidates,
        complete_samples,
        missing_by_channel,
    ) = build_candidates(
        dataset_root,
        samples,
        sample_data_by_sample,
        annotations_by_sample,
        calibration_by_token,
        pose_by_token,
    )
    class_targets = {
        class_name: min(args.quota_per_class, len(class_candidates[class_name]))
        for class_name in CLASS_TO_CATEGORY
    }
    for class_name, count in class_targets.items():
        if count < args.quota_per_class:
            LOGGER.warning(
                "%s has only %d eligible frames after visibility fallback; "
                "the output will report this shortfall",
                class_name,
                count,
            )

    assignments, split_targets = solve_split_assignments(
        candidate_labels, class_targets, args.seed, candidate_visibility
    )
    invalid_images = validate_selected_images(
        dataset_root, assignments, candidates
    )
    if invalid_images:
        descriptions = "; ".join(
            f"{token}: {reason}" for token, reason in sorted(invalid_images.items())
        )
        raise RuntimeError(
            "Selected camera images failed the 1600x900/decode check. "
            "No files were copied. Re-run after resolving these source files: "
            f"{descriptions}"
        )

    split_counts = class_split_counts(candidate_labels, assignments)
    LOGGER.info(
        "Eligible complete frames: %d of %d; selected unique frames: %d",
        len(complete_samples),
        len(samples),
        len(assignments),
    )
    for class_name in CLASS_TO_CATEGORY:
        vis_2_selected = sum(
            assigned_split in SPLIT_NAMES
            and candidate_visibility[token].get(class_name) == 2
            for token, assigned_split in assignments.items()
        )
        LOGGER.info(
            "%s: candidates=%d (visibility 3/4=%d, visibility 2-only=%d), "
            "selected train/val/test=%d/%d/%d",
            class_name,
            len(class_candidates[class_name]),
            len(high_visibility_candidates[class_name]),
            len(class_candidates[class_name] - high_visibility_candidates[class_name]),
            split_counts[class_name]["train"],
            split_counts[class_name]["val"],
            split_counts[class_name]["test"],
        )
        if vis_2_selected:
            LOGGER.info("%s includes %d visibility-2-only selected frames", class_name, vis_2_selected)
    if args.dry_run:
        return {
            "selected_frames": len(assignments),
            "class_split_counts": split_counts,
            "split_targets": split_targets,
        }

    staging_dir.mkdir(parents=True)
    for index, (sample_token, split_name) in enumerate(sorted(assignments.items()), 1):
        collect_sample(
            dataset_root,
            staging_dir,
            sample_token,
            split_name,
            samples[sample_token],
            scene_by_token[samples[sample_token]["scene_token"]],
            candidates[sample_token],
            calibration_by_token,
            pose_by_token,
        )
        if index % 100 == 0 or index == len(assignments):
            LOGGER.info("Collected %d/%d unique frames", index, len(assignments))

    split_tokens = {
        split_name: sorted(
            token for token, assigned_split in assignments.items()
            if assigned_split == split_name
        )
        for split_name in SPLIT_NAMES
    }
    manifest = {
        "source": {
            "dataset_root": str(dataset_root),
            "version": args.version,
            "sample_count": len(samples),
        },
        "selection": {
            "seed": args.seed,
            "sample_quota_per_class": args.quota_per_class,
            "quota_semantics": (
                "Minimum per-class memberships in each sample-level split; actual counts may exceed "
                "targets when co-occurring labels make exact counts infeasible."
            ),
            "sample_level_split": True,
            "scene_grouping": False,
            "image_size": {"width": IMAGE_SIZE[0], "height": IMAGE_SIZE[1]},
            "primary_camera_policy": (
                "Choose the non-CAM_BACK camera with the most qualifying target boxes; "
                "ties prefer the front-facing camera order."
            ),
            "visibility_policy": (
                "Visibility 3/4 is preferred; visibility 2 is used only as needed "
                "to meet exact joint per-class and split counts."
            ),
            "truck_bus_max_box_area_outside_image": 0.10,
            "other_classes_max_box_area_outside_image": 0.0,
            "visibility_2_fallback_reason": (
                "Used only where necessary to make exact class and split counts "
                "jointly feasible when frames contain multiple target classes."
            ),
            "point_count_filter": None,
            "point_count_source": (
                "nuScenes sample_annotation num_lidar_pts and num_radar_pts; "
                "counts are recorded but do not filter samples"
            ),
            "box_projection": "3D annotation corners projected using each camera's calibrated sensor and ego pose",
        },
        "file_completeness": {
            "required_channels": list(REQUIRED_CHANNELS),
            "excluded_channel": "CAM_BACK",
            "complete_frames": len(complete_samples),
            "incomplete_frames_skipped": len(samples) - len(complete_samples),
            "missing_keyframe_files_by_channel": dict(sorted(missing_by_channel.items())),
        },
        "class_counts": {
            class_name: {
                "eligible_candidates": len(class_candidates[class_name]),
                "visibility_3_or_4_candidates": len(
                    high_visibility_candidates[class_name]
                ),
                "visibility_2_candidates": len(
                    class_candidates[class_name] - high_visibility_candidates[class_name]
                ),
                "target": class_targets[class_name],
                "shortfall": args.quota_per_class - class_targets[class_name],
                "selected_visibility_3_or_4": sum(
                    candidate_visibility[token][class_name] >= 3
                    for token in assignments
                    if class_name in candidate_labels[token]
                ),
                "selected_visibility_2": sum(
                    candidate_visibility[token][class_name] == 2
                    for token in assignments
                    if class_name in candidate_labels[token]
                ),
                "splits": split_counts[class_name],
            }
            for class_name in CLASS_TO_CATEGORY
        },
        "splits": split_tokens,
        "unique_selected_frames": len(assignments),
        "multiple_classes_per_frame_allowed": True,
        "storage": {
            "raw_sensor_files_copied": True,
            "duplicate_npy_point_arrays_written": False,
            "point_cloud_conversion": (
                "Deferred to preprocessing; only original LiDAR and radar files are stored."
            ),
            "box_overlays": "camera/overlays/<CAMERA_CHANNEL>.jpg",
        },
    }
    write_json(staging_dir / "metadata" / "collection_manifest.json", manifest)
    staging_dir.replace(output_dir)
    LOGGER.info("Collection complete: %s", output_dir)
    return manifest


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run(args)


if __name__ == "__main__":
    main()
