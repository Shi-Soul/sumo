"""Tracking policy observation adapter and ONNX runtime."""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from sumo.utils.g1_wbc.constants import (
    ACTION_DIM,
    ACTION_SCALE_BY_PATTERN,
    ANCHOR_BODY_NAME,
    DEFAULT_JOINT_POSITIONS,
    LIMB_EE_BODY_NAMES,
    MUJOCO_JOINT_NAMES,
    OBS_DIM,
    POLICY_DECIMATION,
    resolve_policy_path,
)
from sumo.utils.g1_wbc.math import (
    normalize_quat,
    quat_rotate_inverse,
    rot6d_from_quat,
    subtract_frame_transforms,
)


@dataclass
class ReferenceFrame:
    """Reference data needed by the wbteleop actor at one policy step."""

    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    anchor_ang_vel_w: np.ndarray

    @property
    def command(self) -> np.ndarray:
        return np.concatenate([self.joint_pos, self.joint_vel])


class History:
    """Chronological fixed-length history with first-frame backfill."""

    def __init__(self, length: int) -> None:
        self.length = int(length)
        self._items: deque[np.ndarray] = deque(maxlen=self.length)

    def reset(self) -> None:
        self._items.clear()

    def append(self, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=np.float32).copy()
        if not self._items:
            for _ in range(self.length):
                self._items.append(value.copy())
        else:
            self._items.append(value)

    def flat(self) -> np.ndarray:
        if not self._items:
            raise RuntimeError("History is empty")
        return np.concatenate(list(self._items), axis=-1)


class TrackingPolicyRuntime:
    """Build 886-dim observations and apply the WBC ONNX policy."""

    def __init__(
        self,
        model: mujoco.MjModel,
        policy: str | Path = "bcrl",
        *,
        decimation: int = POLICY_DECIMATION,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise ImportError("onnxruntime is required for G1 WBC policy inference") from exc

        self.model = model
        self.policy_path = resolve_policy_path(policy)
        self.decimation = int(decimation)
        self.default_joint_pos = np.asarray(DEFAULT_JOINT_POSITIONS, dtype=np.float64)
        self.action_scale = action_scale_for_joints(MUJOCO_JOINT_NAMES)
        self.last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._step_count = 0
        self._held_ctrl = self.default_joint_pos.copy()

        self.body_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in MUJOCO_BODY_NAMES_SAFE}
        self.limb_body_ids = [self.body_ids[name] for name in LIMB_EE_BODY_NAMES]
        self.anchor_body_id = self.body_ids[ANCHOR_BODY_NAME]

        self.ref_limb_history = History(5)
        self.robot_limb_history = History(5)
        self.gravity_history = History(5)
        self.base_ang_vel_history = History(5)
        self.joint_pos_history = History(5)
        self.joint_vel_history = History(5)
        self.action_history = History(5)

        self.session = ort.InferenceSession(str(self.policy_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        output_shape = self.session.get_outputs()[0].shape
        if int(input_shape[-1]) != OBS_DIM or int(output_shape[-1]) != ACTION_DIM:
            raise ValueError(
                f"Unexpected policy IO for {self.policy_path}: input={input_shape}, output={output_shape}; "
                f"expected (*,{OBS_DIM}) -> (*,{ACTION_DIM})"
            )

    def reset(self) -> None:
        self.last_action[:] = 0.0
        self._step_count = 0
        self._held_ctrl = self.default_joint_pos.copy()
        for hist in (
            self.ref_limb_history,
            self.robot_limb_history,
            self.gravity_history,
            self.base_ang_vel_history,
            self.joint_pos_history,
            self.joint_vel_history,
            self.action_history,
        ):
            hist.reset()

    def step(self, data: mujoco.MjData, reference: ReferenceFrame) -> np.ndarray:
        """Return actuator position targets for the current sim step."""
        if self._step_count % self.decimation == 0:
            obs = self.build_observation(data, reference)
            [action] = self.session.run([self.output_name], {self.input_name: obs[None, :]})
            self.last_action = np.asarray(action[0], dtype=np.float32)
            self._held_ctrl = self.last_action.astype(np.float64) * self.action_scale + self.default_joint_pos
        self._step_count += 1
        return self._held_ctrl.copy()

    def build_observation(self, data: mujoco.MjData, reference: ReferenceFrame) -> np.ndarray:
        """Build the wbteleop actor observation in tracking_bfm term order."""
        ref_limb = limb_pose_in_anchor_frame(
            reference.body_pos_w,
            reference.body_quat_w,
            LIMB_EE_BODY_NAMES,
            ANCHOR_BODY_NAME,
        )
        robot_body_pos, robot_body_quat = robot_body_pose(self.model, data)
        robot_limb = limb_pose_in_anchor_frame(robot_body_pos, robot_body_quat, LIMB_EE_BODY_NAMES, ANCHOR_BODY_NAME)

        root_quat = normalize_quat(np.asarray(data.qpos[3:7]))
        projected_gravity = quat_rotate_inverse(root_quat, np.array([0.0, 0.0, -1.0]))
        base_ang_vel = quat_rotate_inverse(root_quat, np.asarray(data.qvel[3:6]))
        joint_pos_rel = np.asarray(data.qpos[7:36]) - self.default_joint_pos
        joint_vel_rel = np.asarray(data.qvel[6:35])

        self.ref_limb_history.append(ref_limb)
        self.robot_limb_history.append(robot_limb)
        self.gravity_history.append(projected_gravity)
        self.base_ang_vel_history.append(base_ang_vel)
        self.joint_pos_history.append(joint_pos_rel)
        self.joint_vel_history.append(joint_vel_rel)
        self.action_history.append(self.last_action)

        obs = np.concatenate(
            [
                reference.command.astype(np.float32),
                self.ref_limb_history.flat(),
                reference.anchor_ang_vel_w.astype(np.float32),
                self.robot_limb_history.flat(),
                self.gravity_history.flat(),
                self.base_ang_vel_history.flat(),
                self.joint_pos_history.flat(),
                self.joint_vel_history.flat(),
                self.action_history.flat(),
            ]
        ).astype(np.float32)
        if obs.shape != (OBS_DIM,):
            raise ValueError(f"Expected observation shape {(OBS_DIM,)}, got {obs.shape}")
        return obs


def action_scale_for_joints(joint_names: tuple[str, ...]) -> np.ndarray:
    scales = np.ones(len(joint_names), dtype=np.float64)
    for i, name in enumerate(joint_names):
        for pattern, value in ACTION_SCALE_BY_PATTERN.items():
            if re.fullmatch(pattern, name):
                scales[i] = value
                break
    return scales


def robot_body_pose(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    body_pos = np.zeros((len(MUJOCO_BODY_NAMES_SAFE), 3), dtype=np.float64)
    body_quat = np.zeros((len(MUJOCO_BODY_NAMES_SAFE), 4), dtype=np.float64)
    for i, name in enumerate(MUJOCO_BODY_NAMES_SAFE):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        body_pos[i] = data.xpos[body_id]
        body_quat[i] = data.xquat[body_id]
    return body_pos, normalize_quat(body_quat)


def limb_pose_in_anchor_frame(
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    body_names: tuple[str, ...],
    anchor_body_name: str,
) -> np.ndarray:
    anchor_idx = MUJOCO_BODY_NAMES_SAFE.index(anchor_body_name)
    body_indices = [MUJOCO_BODY_NAMES_SAFE.index(name) for name in body_names]
    anchor_pos = np.repeat(body_pos_w[anchor_idx : anchor_idx + 1], len(body_indices), axis=0)
    anchor_quat = np.repeat(body_quat_w[anchor_idx : anchor_idx + 1], len(body_indices), axis=0)
    pos_b, quat_b = subtract_frame_transforms(anchor_pos, anchor_quat, body_pos_w[body_indices], body_quat_w[body_indices])
    return np.concatenate([pos_b, rot6d_from_quat(quat_b)], axis=-1).reshape(-1).astype(np.float32)


# Keep this alias local to make accidental body-order changes visible in tests.
from sumo.utils.g1_wbc.constants import MUJOCO_BODY_NAMES as MUJOCO_BODY_NAMES_SAFE  # noqa: E402
