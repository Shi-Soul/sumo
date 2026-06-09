"""NumPy quaternion and pose utilities for G1 WBC tracking."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def normalize_quat(quat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    out = quat / np.maximum(norm, eps)
    if out.ndim == 1 and out[0] < 0:
        out = -out
    elif out.ndim > 1:
        out = np.where(out[..., :1] < 0, -out, out)
    return out


def quat_conjugate(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    out = quat.copy()
    out[..., 1:] *= -1.0
    return out


def quat_mul(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lhs = np.asarray(lhs, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    lw, lx, ly, lz = np.moveaxis(lhs, -1, 0)
    rw, rx, ry, rz = np.moveaxis(rhs, -1, 0)
    return np.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        axis=-1,
    )


def quat_rotate(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    quat = normalize_quat(quat)
    vec = np.asarray(vec, dtype=np.float64)
    zeros = np.zeros((*vec.shape[:-1], 1), dtype=np.float64)
    vec_quat = np.concatenate([zeros, vec], axis=-1)
    return quat_mul(quat_mul(quat, vec_quat), quat_conjugate(quat))[..., 1:]


def quat_rotate_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    return quat_rotate(quat_conjugate(normalize_quat(quat)), vec)


def matrix_from_quat(quat: np.ndarray) -> np.ndarray:
    quat = normalize_quat(quat)
    xyzw = np.concatenate([quat[..., 1:], quat[..., :1]], axis=-1)
    return Rotation.from_quat(xyzw.reshape(-1, 4)).as_matrix().reshape(*quat.shape[:-1], 3, 3)


def rot6d_from_quat(quat: np.ndarray) -> np.ndarray:
    return matrix_from_quat(quat)[..., :2].reshape(*quat.shape[:-1], 6)


def subtract_frame_transforms(
    anchor_pos_w: np.ndarray,
    anchor_quat_w: np.ndarray,
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return body pose expressed in the anchor frame."""
    anchor_quat_inv = quat_conjugate(normalize_quat(anchor_quat_w))
    pos_b = quat_rotate(anchor_quat_inv, np.asarray(body_pos_w) - np.asarray(anchor_pos_w))
    quat_b = quat_mul(anchor_quat_inv, body_quat_w)
    return pos_b, normalize_quat(quat_b)


def quat_geodesic_error(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lhs = normalize_quat(lhs)
    rhs = normalize_quat(rhs)
    dots = np.abs(np.sum(lhs * rhs, axis=-1))
    dots = np.clip(dots, -1.0, 1.0)
    return 2.0 * np.arccos(dots)


def slerp_wxyz(times: np.ndarray, quats_wxyz: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    """Spherical interpolation for MuJoCo wxyz quaternions."""
    times = np.asarray(times, dtype=np.float64)
    query_times = np.asarray(query_times, dtype=np.float64)
    quats_wxyz = normalize_quat(quats_wxyz)
    xyzw = np.concatenate([quats_wxyz[:, 1:], quats_wxyz[:, :1]], axis=-1)
    slerp = Slerp(times, Rotation.from_quat(xyzw))
    out_xyzw = slerp(query_times).as_quat()
    return normalize_quat(np.concatenate([out_xyzw[:, 3:4], out_xyzw[:, :3]], axis=-1))


def finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) <= 1:
        return np.zeros_like(values)
    return np.gradient(values, dt, axis=0, edge_order=1)


def angular_velocity_from_quat(quats_wxyz: np.ndarray, dt: float) -> np.ndarray:
    quats_wxyz = normalize_quat(quats_wxyz)
    if len(quats_wxyz) <= 1:
        return np.zeros((len(quats_wxyz), 3), dtype=np.float64)
    rot = Rotation.from_quat(np.concatenate([quats_wxyz[:, 1:], quats_wxyz[:, :1]], axis=-1))
    rotvec = rot.as_rotvec()
    return finite_difference(rotvec, dt)
