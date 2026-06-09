"""G1 whole-body-control tracking tasks for SUMO MPC."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
from judo.tasks.base import Task, TaskConfig

from sumo.utils.g1_wbc.constants import (
    CONTACT_BODY_NAMES,
    DEFAULT_MOTION_FILE,
    DEFAULT_POLICY_VARIANT,
    EE_REWARD_BODY_NAMES,
    G1_XML_PATH,
    JOINT_POS_SLICE,
    MUJOCO_BODY_NAMES,
    TASK_CONTROL_DIM,
)
from sumo.utils.g1_wbc.math import quat_geodesic_error, subtract_frame_transforms
from sumo.utils.g1_wbc.motion import load_motion
from sumo.utils.g1_wbc.reference import build_reference_frames, motion_controls_at_times, normalize_controls


def _env_path(name: str, default: Path) -> str:
    return os.environ.get(name, str(default))


@dataclass
class G1WBCConfig(TaskConfig):
    """Configuration shared by the G1 WBC tasks."""

    motion_file: str = field(default_factory=lambda: _env_path("SUMO_G1_WBC_MOTION_FILE", DEFAULT_MOTION_FILE))
    motion_type: Literal["isaaclab", "mujoco"] = field(
        default_factory=lambda: os.environ.get("SUMO_G1_WBC_MOTION_TYPE", "mujoco")  # type: ignore[return-value]
    )
    policy: str = field(default_factory=lambda: os.environ.get("SUMO_G1_WBC_POLICY", DEFAULT_POLICY_VARIANT))
    fall_threshold: float = 0.55
    success_local_ee_rmse: float = 0.12
    root_pos_weight: float = 2.0
    root_ori_weight: float = 1.0
    joint_pos_weight: float = 1.0
    joint_vel_weight: float = 0.05
    ee_pos_weight: float = 6.0
    ee_ori_weight: float = 0.5
    contact_mismatch_weight: float = 8.0
    contact_no_ref_weight: float = 20.0
    contact_force_weight: float = 0.002
    contact_switch_weight: float = 0.5
    smooth_joint_weight: float = 0.02
    smooth_root_weight: float = 0.1
    fall_penalty: float = 2500.0


class G1WBCBase(Task[G1WBCConfig]):
    """Base task whose controls are whole-body reference trajectories."""

    config_t = G1WBCConfig
    reward_mode: Literal["ee", "joint"] = "ee"

    def __init__(self, model_path: str = str(G1_XML_PATH)) -> None:
        super().__init__(model_path)
        self.name = f"g1_wbc_{self.reward_mode}"
        self.motion = load_motion(self.config.motion_file, self.config.motion_type)
        self._body_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in MUJOCO_BODY_NAMES]
        self._ee_reward_indices = [MUJOCO_BODY_NAMES.index(name) for name in EE_REWARD_BODY_NAMES]
        self._contact_body_indices = [MUJOCO_BODY_NAMES.index(name) for name in CONTACT_BODY_NAMES]
        self._ctrlrange = self._build_ctrlrange()
        self._last_metrics: dict[str, float] = {}

    @property
    def nu(self) -> int:
        return TASK_CONTROL_DIM

    @property
    def actuator_ctrlrange(self) -> np.ndarray:
        return self._ctrlrange

    @property
    def reset_pose(self) -> np.ndarray:
        return np.concatenate([self.motion.root_qpos(0), self.motion.joint_pos[0]])

    def reset(self) -> None:
        self.data.qpos[:] = self.reset_pose
        self.data.qvel[:] = np.zeros_like(self.data.qvel)
        if self.motion.num_frames > 0:
            self.data.qvel[6:35] = self.motion.joint_vel[0]
            pelvis_idx = MUJOCO_BODY_NAMES.index("pelvis")
            self.data.qvel[:3] = self.motion.body_lin_vel_w[0, pelvis_idx]
            self.data.qvel[3:6] = self.motion.body_ang_vel_w[0, pelvis_idx]
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)

    def optimizer_warm_start(self) -> np.ndarray:
        frame = self.motion.frame_index(float(self.data.time))
        return np.concatenate([self.motion.root_qpos(frame), self.motion.joint_pos[frame]])

    def task_to_sim_ctrl(self, controls: np.ndarray) -> np.ndarray:
        return normalize_controls(controls)

    def reference_controls_for_times(self, times: np.ndarray) -> np.ndarray:
        return motion_controls_at_times(self.motion, times)

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict | None = None,
    ) -> np.ndarray:
        controls = normalize_controls(controls)
        qpos = states[:, 1:, : self.model.nq] if states.shape[1] == controls.shape[1] + 1 else states[:, :, : self.model.nq]
        horizon = min(qpos.shape[1], controls.shape[1])
        qpos = qpos[:, :horizon]
        controls = controls[:, :horizon]
        sensors = sensors[:, :horizon] if sensors.size else np.zeros((controls.shape[0], horizon, 4))
        target_controls = self._target_controls_for_horizon(horizon)

        rewards = np.zeros(controls.shape[0], dtype=np.float64)
        local_ee_errors = []
        for batch in range(controls.shape[0]):
            if self.reward_mode == "ee":
                pose_reward, local_err = self._ee_reward(qpos[batch], target_controls)
            else:
                pose_reward, local_err = self._joint_reward(qpos[batch], states[batch], target_controls)
            contact_reward = self._contact_reward(sensors[batch], horizon)
            smooth_reward = self._smoothness_reward(controls[batch])
            fall_penalty = self._fall_penalty(qpos[batch])
            rewards[batch] = pose_reward + contact_reward + smooth_reward + fall_penalty
            local_ee_errors.append(local_err)
        self._last_metrics = {
            "local_ee_rmse_best": float(np.min(local_ee_errors)) if local_ee_errors else float("nan"),
            "local_ee_rmse_mean": float(np.mean(local_ee_errors)) if local_ee_errors else float("nan"),
            "reward_best": float(np.max(rewards)) if rewards.size else float("nan"),
        }
        return rewards

    def success(self, model: mujoco.MjModel, data: mujoco.MjData, metadata: dict | None = None) -> bool:
        return bool(self._last_metrics.get("local_ee_rmse_best", np.inf) <= self.config.success_local_ee_rmse)

    def get_sim_metadata(self) -> dict:
        return {"g1_wbc_metrics": dict(self._last_metrics)}

    def _build_ctrlrange(self) -> np.ndarray:
        trajectory = self.motion.trajectory_controls()
        lower = np.full(TASK_CONTROL_DIM, -np.inf, dtype=np.float64)
        upper = np.full(TASK_CONTROL_DIM, np.inf, dtype=np.float64)
        lower[:3] = np.nanmin(trajectory[:, :3], axis=0) - np.array([0.6, 0.6, 0.25])
        upper[:3] = np.nanmax(trajectory[:, :3], axis=0) + np.array([0.6, 0.6, 0.35])
        lower[3:7] = -1.0
        upper[3:7] = 1.0
        for i, joint_id in enumerate(range(1, self.model.njnt)):
            if self.model.jnt_limited[joint_id]:
                lower[7 + i], upper[7 + i] = self.model.jnt_range[joint_id]
            else:
                lower[7 + i], upper[7 + i] = -np.pi, np.pi
        return np.stack([lower, upper], axis=-1)

    def _reference_contact_mask(self, horizon: int) -> np.ndarray:
        times = float(self.data.time) + self.model.opt.timestep * (np.arange(horizon) + 1)
        frames = [self.motion.frame_index(float(t)) for t in times]
        return self.motion.contact_mask[frames].astype(np.float64)

    def _target_controls_for_horizon(self, horizon: int) -> np.ndarray:
        times = float(self.data.time) + self.model.opt.timestep * (np.arange(horizon) + 1)
        return self.reference_controls_for_times(times)

    def _ee_reward(self, qpos_seq: np.ndarray, controls: np.ndarray) -> tuple[float, float]:
        executed_pos, executed_quat = self._fk_sequence(qpos_seq)
        reference_frames = build_reference_frames(self.model, controls, self.model.opt.timestep)
        ref_pos = np.stack([frame.body_pos_w for frame in reference_frames], axis=0)
        ref_quat = np.stack([frame.body_quat_w for frame in reference_frames], axis=0)

        ee = self._ee_reward_indices
        global_pos_err = np.linalg.norm(executed_pos[:, ee] - ref_pos[:, ee], axis=-1)
        global_ori_err = quat_geodesic_error(executed_quat[:, ee], ref_quat[:, ee])
        local_pos_err = self._local_body_position_error(executed_pos, executed_quat, ref_pos, ref_quat, ee)
        local_rmse = float(np.sqrt(np.mean(local_pos_err**2)))
        reward = (
            -self.config.ee_pos_weight * float(np.mean(global_pos_err))
            -self.config.ee_pos_weight * float(np.mean(local_pos_err))
            -self.config.ee_ori_weight * float(np.mean(global_ori_err))
        )
        return reward, local_rmse

    def _joint_reward(self, qpos_seq: np.ndarray, states: np.ndarray, controls: np.ndarray) -> tuple[float, float]:
        qvel_seq = states[1 : len(qpos_seq) + 1, self.model.nq :] if states.shape[0] == len(qpos_seq) + 1 else states[: len(qpos_seq), self.model.nq :]
        reference_qvel = np.zeros_like(qvel_seq)
        reference_qvel[:, 6:35] = np.gradient(controls[:, JOINT_POS_SLICE], self.model.opt.timestep, axis=0, edge_order=1)
        root_pos_err = np.linalg.norm(qpos_seq[:, :3] - controls[:, :3], axis=-1)
        root_ori_err = quat_geodesic_error(qpos_seq[:, 3:7], controls[:, 3:7])
        joint_pos_err = np.linalg.norm(qpos_seq[:, 7:36] - controls[:, JOINT_POS_SLICE], axis=-1)
        joint_vel_err = np.linalg.norm(qvel_seq[:, 6:35] - reference_qvel[:, 6:35], axis=-1)
        reward = (
            -self.config.root_pos_weight * float(np.mean(root_pos_err))
            -self.config.root_ori_weight * float(np.mean(root_ori_err))
            -self.config.joint_pos_weight * float(np.mean(joint_pos_err))
            -self.config.joint_vel_weight * float(np.mean(joint_vel_err))
        )
        local_err = self._joint_local_ee_rmse(qpos_seq, controls)
        return reward, local_err

    def _joint_local_ee_rmse(self, qpos_seq: np.ndarray, controls: np.ndarray) -> float:
        executed_pos, executed_quat = self._fk_sequence(qpos_seq)
        reference_frames = build_reference_frames(self.model, controls, self.model.opt.timestep)
        ref_pos = np.stack([frame.body_pos_w for frame in reference_frames], axis=0)
        ref_quat = np.stack([frame.body_quat_w for frame in reference_frames], axis=0)
        err = self._local_body_position_error(executed_pos, executed_quat, ref_pos, ref_quat, self._ee_reward_indices)
        return float(np.sqrt(np.mean(err**2)))

    def _local_body_position_error(
        self,
        executed_pos: np.ndarray,
        executed_quat: np.ndarray,
        ref_pos: np.ndarray,
        ref_quat: np.ndarray,
        body_indices: list[int],
    ) -> np.ndarray:
        anchor_idx = MUJOCO_BODY_NAMES.index("pelvis")
        exec_anchor_pos = np.repeat(executed_pos[:, anchor_idx : anchor_idx + 1], len(body_indices), axis=1)
        exec_anchor_quat = np.repeat(executed_quat[:, anchor_idx : anchor_idx + 1], len(body_indices), axis=1)
        ref_anchor_pos = np.repeat(ref_pos[:, anchor_idx : anchor_idx + 1], len(body_indices), axis=1)
        ref_anchor_quat = np.repeat(ref_quat[:, anchor_idx : anchor_idx + 1], len(body_indices), axis=1)
        exec_local, _ = subtract_frame_transforms(
            exec_anchor_pos,
            exec_anchor_quat,
            executed_pos[:, body_indices],
            executed_quat[:, body_indices],
        )
        ref_local, _ = subtract_frame_transforms(
            ref_anchor_pos,
            ref_anchor_quat,
            ref_pos[:, body_indices],
            ref_quat[:, body_indices],
        )
        return np.linalg.norm(exec_local - ref_local, axis=-1)

    def _contact_reward(self, sensor_seq: np.ndarray, horizon: int) -> float:
        if horizon <= 0:
            return 0.0
        ref_mask = self._reference_contact_mask(horizon)
        exec_mask = sensor_seq[:, :2]
        forces = sensor_seq[:, 2:4]
        mismatch = np.abs(exec_mask - ref_mask)
        no_ref_contact = exec_mask * (1.0 - ref_mask)
        switch_count = np.abs(np.diff(exec_mask, axis=0)).sum() if horizon > 1 else 0.0
        force_penalty = forces * (1.0 - ref_mask) + np.maximum(forces - 350.0, 0.0) * ref_mask
        return -(
            self.config.contact_mismatch_weight * float(np.mean(mismatch))
            + self.config.contact_no_ref_weight * float(np.mean(no_ref_contact))
            + self.config.contact_force_weight * float(np.mean(force_penalty))
            + self.config.contact_switch_weight * float(switch_count / max(horizon - 1, 1))
        )

    def _smoothness_reward(self, controls: np.ndarray) -> float:
        if len(controls) <= 2:
            return 0.0
        joint_acc = np.diff(controls[:, JOINT_POS_SLICE], n=2, axis=0)
        root_acc = np.diff(controls[:, :3], n=2, axis=0)
        return -(
            self.config.smooth_joint_weight * float(np.mean(np.linalg.norm(joint_acc, axis=-1)))
            + self.config.smooth_root_weight * float(np.mean(np.linalg.norm(root_acc, axis=-1)))
        )

    def _fall_penalty(self, qpos_seq: np.ndarray) -> float:
        if np.any(qpos_seq[:, 2] <= self.config.fall_threshold):
            return -self.config.fall_penalty
        return 0.0

    def _fk_sequence(self, qpos_seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        data = mujoco.MjData(self.model)
        body_pos = np.zeros((len(qpos_seq), len(MUJOCO_BODY_NAMES), 3), dtype=np.float64)
        body_quat = np.zeros((len(qpos_seq), len(MUJOCO_BODY_NAMES), 4), dtype=np.float64)
        for t, qpos in enumerate(qpos_seq):
            data.qpos[:] = qpos
            data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, data)
            body_pos[t] = data.xpos[self._body_ids]
            body_quat[t] = data.xquat[self._body_ids]
        return body_pos, body_quat


class G1WBCEE(G1WBCBase):
    reward_mode: Literal["ee", "joint"] = "ee"


class G1WBCJoint(G1WBCBase):
    reward_mode: Literal["ee", "joint"] = "joint"
