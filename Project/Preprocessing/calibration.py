"""Section 3.2 Sensor Calibration using nuScenes full 6-DoF extrinsics.

Frames follow Fig. 2: sensor frame (Fl / Fr / Fc) -> vehicle frame Fv (ego) -> camera frame Fc
-> image plane -> pixels (u, v). A Pose maps child -> parent: X_parent = R @ X_child + T,
which is Eq. (1)/(5) with R = Rsv, T = Tsv. Quaternions are nuScenes (w, x, y, z).
"""
from dataclasses import dataclass

import numpy as np
from pyquaternion import Quaternion


@dataclass(frozen=True)
class Pose:
    R: np.ndarray  # (3, 3)
    T: np.ndarray  # (3,)

    def __matmul__(self, other):  # (self @ other) maps other's child frame into self's parent frame
        return Pose(self.R @ other.R, self.R @ other.T + self.T)

    def inv(self):
        return Pose(self.R.T, -self.R.T @ self.T)

    def apply(self, pts):
        return pts @ self.R.T + self.T

    def rotate(self, vecs):
        return vecs @ self.R.T

    def matrix(self):
        M = np.eye(4)
        M[:3, :3], M[:3, 3] = self.R, self.T
        return M


def pose_from_dict(d):
    return Pose(Quaternion(d["rotation"]).rotation_matrix, np.asarray(d["translation"], dtype=np.float64))


def sensor_to_vehicle(meta, name):
    """Rsv, Tsv: sensor -> ego (vehicle) frame."""
    return pose_from_dict(meta["sensor_calibrations"][name]["calibrated_sensor"])


def vehicle_to_global(meta, name):
    """Ego pose at the sensor's own timestamp."""
    return pose_from_dict(meta["sensor_calibrations"][name]["ego_pose"])


def intrinsics(meta, cam):
    return np.asarray(meta["sensor_calibrations"][cam]["calibrated_sensor"]["camera_intrinsic"], dtype=np.float64)


def sensor_to_reference_vehicle(meta, name, ref=None):
    """Sensor -> vehicle frame at the reference (LiDAR) timestamp, compensating ego motion via global frame."""
    from config import LIDAR
    ref = ref or LIDAR
    return vehicle_to_global(meta, ref).inv() @ vehicle_to_global(meta, name) @ sensor_to_vehicle(meta, name)


def reference_vehicle_to_camera(meta, cam, ref=None):
    """Eq. (1) with ego-motion compensation: reference vehicle frame -> camera frame at the camera timestamp."""
    from config import LIDAR
    ref = ref or LIDAR
    return (sensor_to_vehicle(meta, cam).inv() @ vehicle_to_global(meta, cam).inv()
            @ vehicle_to_global(meta, ref))


def global_to_camera(meta, cam):
    return sensor_to_vehicle(meta, cam).inv() @ vehicle_to_global(meta, cam).inv()


def project_to_image(Xc, K):
    """Eq. (3)-(4): camera coords (N, 3) -> pixel (u, v) in the original image and depth Zc."""
    z = Xc[:, 2]
    safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = K[0, 0] * Xc[:, 0] / safe + K[0, 2]
    v = K[1, 1] * Xc[:, 1] / safe + K[1, 2]
    return u, v, z
