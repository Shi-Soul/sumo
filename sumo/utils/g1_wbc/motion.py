"""Standalone motion loading and preprocessing for G1 WBC tracking."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal

import numpy as np

from sumo.utils.g1_wbc.constants import (
    CONTACT_BODY_NAMES,
    DEFAULT_MOTION_FILE,
    ISAACLAB_TO_MUJOCO_BODY_REINDEX,
    ISAACLAB_TO_MUJOCO_JOINT_REINDEX,
    MUJOCO_BODY_NAMES,
)
from sumo.utils.g1_wbc.math import normalize_quat

MotionType = Literal["isaaclab", "mujoco"]


@dataclass(frozen=True)
class G1WBCMotion:
    """Motion data in MuJoCo joint/body order."""

    path: Path
    motion_type: MotionType
    fps: float
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    body_lin_vel_w: np.ndarray
    body_ang_vel_w: np.ndarray
    contact_mask: np.ndarray

    @property
    def dt(self) -> float:
        return 1.0 / self.fps

    @property
    def num_frames(self) -> int:
        return int(self.joint_pos.shape[0])

    @property
    def duration(self) -> float:
        return max(0.0, (self.num_frames - 1) * self.dt)

    def frame_index(self, time_s: float) -> int:
        return int(np.clip(round(time_s * self.fps), 0, self.num_frames - 1))

    def root_qpos(self, frame: int) -> np.ndarray:
        pelvis_idx = MUJOCO_BODY_NAMES.index("pelvis")
        return np.concatenate([self.body_pos_w[frame, pelvis_idx], self.body_quat_w[frame, pelvis_idx]])

    @cached_property
    def _trajectory_controls(self) -> np.ndarray:
        root = np.stack([self.root_qpos(i) for i in range(self.num_frames)], axis=0)
        controls = np.concatenate([root, self.joint_pos], axis=-1)
        controls.setflags(write=False)
        return controls

    def trajectory_controls(self) -> np.ndarray:
        return self._trajectory_controls


def load_motion(
    motion_file: str | Path = DEFAULT_MOTION_FILE,
    motion_type: MotionType = "mujoco",
    contact_height_threshold: float = 0.08,
    contact_speed_threshold: float = 0.35,
) -> G1WBCMotion:
    """Load a tracking_bfm-compatible G1 npz without importing tracking_bfm."""
    path = Path(motion_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Motion file does not exist: {path}")

    data = np.load(path)
    required = (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "fps",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Motion file {path} is missing required arrays: {missing}")

    if motion_type not in ("isaaclab", "mujoco"):
        raise ValueError(f"Unsupported motion_type: {motion_type}")

    joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
    joint_vel = np.asarray(data["joint_vel"], dtype=np.float64)
    body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float64)
    body_quat_w = normalize_quat(np.asarray(data["body_quat_w"], dtype=np.float64))
    body_lin_vel_w = np.asarray(data["body_lin_vel_w"], dtype=np.float64)
    body_ang_vel_w = np.asarray(data["body_ang_vel_w"], dtype=np.float64)

    if motion_type == "isaaclab":
        joint_pos = joint_pos[:, ISAACLAB_TO_MUJOCO_JOINT_REINDEX]
        joint_vel = joint_vel[:, ISAACLAB_TO_MUJOCO_JOINT_REINDEX]
        body_pos_w = body_pos_w[:, ISAACLAB_TO_MUJOCO_BODY_REINDEX]
        body_quat_w = body_quat_w[:, ISAACLAB_TO_MUJOCO_BODY_REINDEX]
        body_lin_vel_w = body_lin_vel_w[:, ISAACLAB_TO_MUJOCO_BODY_REINDEX]
        body_ang_vel_w = body_ang_vel_w[:, ISAACLAB_TO_MUJOCO_BODY_REINDEX]

    fps_arr = np.asarray(data["fps"]).reshape(-1)
    fps = float(fps_arr[0])
    contact_mask = estimate_contact_mask(
        body_pos_w,
        body_lin_vel_w,
        height_threshold=contact_height_threshold,
        speed_threshold=contact_speed_threshold,
    )
    return G1WBCMotion(
        path=path,
        motion_type=motion_type,
        fps=fps,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
        contact_mask=contact_mask,
    )


def estimate_contact_mask(
    body_pos_w: np.ndarray,
    body_lin_vel_w: np.ndarray,
    *,
    height_threshold: float,
    speed_threshold: float,
) -> np.ndarray:
    """Estimate left/right foot contact from foot height and speed."""
    foot_indices = [MUJOCO_BODY_NAMES.index(name) for name in CONTACT_BODY_NAMES]
    foot_height = body_pos_w[:, foot_indices, 2]
    min_height = np.percentile(foot_height, 5, axis=0, keepdims=True)
    rel_height = foot_height - min_height
    speed = np.linalg.norm(body_lin_vel_w[:, foot_indices], axis=-1)
    raw = (rel_height <= height_threshold) & (speed <= speed_threshold)
    return smooth_contact_mask(raw)


def smooth_contact_mask(mask: np.ndarray, min_run: int = 2) -> np.ndarray:
    """Remove one-frame contact flicker while preserving hard transitions."""
    mask = np.asarray(mask, dtype=bool).copy()
    if mask.ndim != 2:
        raise ValueError(f"Expected contact mask with shape (T, feet), got {mask.shape}")
    for foot in range(mask.shape[1]):
        values = mask[:, foot]
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and values[end] == values[start]:
                end += 1
            if end - start < min_run:
                left = values[start - 1] if start > 0 else values[end] if end < len(values) else values[start]
                right = values[end] if end < len(values) else left
                if left == right:
                    values[start:end] = left
            start = end
        mask[:, foot] = values
    return mask
