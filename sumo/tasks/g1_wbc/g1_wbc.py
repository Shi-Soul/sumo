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
    ACTION_SENSOR_DIM,
    ACTION_SENSOR_START,
    CONTACT_SENSOR_DIM,
    DEFAULT_MOTION_FILE,
    DEFAULT_POLICY_VARIANT,
    G1_XML_PATH,
    JOINT_POS_SLICE,
    MUJOCO_BODY_NAMES,
    POLICY_DT,
    TASK_CONTROL_DIM,
    UPPER_EE_BODY_NAMES,
    UPPER_EE_SENSOR_DIM,
    UPPER_EE_SENSOR_START,
)
from sumo.utils.g1_wbc.math import quat_geodesic_error
from sumo.utils.g1_wbc.model import configure_wbc_model
from sumo.utils.g1_wbc.motion import load_motion
from sumo.utils.g1_wbc.reference import (
    motion_controls_at_times,
    motion_policy_qvel_at_times,
    motion_qvel_at_times,
    normalize_controls,
)


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
    root_pos_weight: float = 4.0
    root_ori_weight: float = 4.0
    joint_pos_weight: float = 2.0
    joint_vel_weight: float = 0.05
    ee_pos_weight: float = 30.0
    ee_ori_weight: float = 0.5
    contact_mismatch_weight: float = 200.0
    contact_no_ref_weight: float = 400.0
    contact_force_weight: float = 0.0005
    contact_switch_weight: float = 40.0
    smooth_joint_weight: float = 80.0
    smooth_root_weight: float = 160.0
    reference_root_weight: float = 0.2
    reference_ori_weight: float = 0.05
    reference_joint_weight: float = 0.2
    stability_height_weight: float = 30.0
    stability_ori_weight: float = 4.0
    fall_penalty: float = 2500.0
    accept_local_improvement: float = 5e-4
    accept_joint_improvement: float = 5e-4
    accept_root_improvement: float = 5e-4
    accept_local_tolerance: float = 2e-3
    accept_joint_tolerance: float = 1e-3
    accept_root_tolerance: float = 2e-3
    accept_contact_tolerance: float = 0.0
    accept_contact_force_tolerance: float = 10.0
    accept_contact_improvement: float = 2e-3
    accept_contact_switch_improvement: float = 5e-4
    accept_contact_force_improvement: float = 2.0
    accept_smooth_tolerance: float = 3e-3
    accept_smooth_improvement: float = 2e-5
    accept_upper_ee_tolerance: float = 5e-4
    accept_upper_ee_improvement: float = 5e-4
    accept_root_ori_tolerance: float = 2e-3
    accept_root_ori_improvement: float = 5e-4


class G1WBCBase(Task[G1WBCConfig]):
    """Base task whose controls are whole-body reference trajectories."""

    config_t = G1WBCConfig
    reward_mode: Literal["ee", "joint"] = "ee"

    def __init__(self, model_path: str = str(G1_XML_PATH)) -> None:
        super().__init__(model_path)
        configure_wbc_model(self.model)
        configure_wbc_model(self.sim_model)
        self.name = f"g1_wbc_{self.reward_mode}"
        self.motion = load_motion(self.config.motion_file, self.config.motion_type)
        self._upper_ee_body_indices = [MUJOCO_BODY_NAMES.index(name) for name in UPPER_EE_BODY_NAMES]
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
            self.data.qvel[:] = motion_qvel_at_times(self.motion, np.asarray([0.0]))[0]
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)

    def optimizer_warm_start(self) -> np.ndarray:
        frame = self.motion.frame_index(float(self.data.time))
        return np.concatenate([self.motion.root_qpos(frame), self.motion.joint_pos[frame]])

    def task_to_sim_ctrl(self, controls: np.ndarray) -> np.ndarray:
        return normalize_controls(controls)

    def reference_controls_for_times(self, times: np.ndarray) -> np.ndarray:
        return motion_controls_at_times(self.motion, times)

    def reference_policy_qvel_for_times(self, times: np.ndarray) -> np.ndarray:
        return motion_policy_qvel_at_times(self.motion, times)

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
        if horizon <= 0:
            return np.zeros(controls.shape[0], dtype=np.float64)
        sensors = sensors[:, :horizon] if sensors.size else np.zeros((controls.shape[0], horizon, CONTACT_SENSOR_DIM))
        target_controls = self._target_controls_for_horizon(horizon)
        command_reference = self._command_reference_for_horizon(horizon)

        metrics = self._tracking_metrics(qpos, sensors, controls, target_controls, command_reference)
        upper_weight = self.config.ee_pos_weight if self.reward_mode == "ee" else 0.7 * self.config.ee_pos_weight
        joint_weight = self.config.joint_pos_weight if self.reward_mode == "joint" else 0.5 * self.config.joint_pos_weight
        rewards = -(
            self.config.contact_mismatch_weight * metrics["contact_mismatch"]
            + self.config.contact_no_ref_weight * metrics["contact_no_ref"]
            + self.config.contact_switch_weight * metrics["contact_switch"]
            + self.config.contact_force_weight * metrics["contact_force"]
            + self.config.smooth_joint_weight * metrics["smooth_joint"]
            + self.config.smooth_root_weight * metrics["smooth_root"]
            + upper_weight * metrics["upper_ee"]
            + self.config.root_pos_weight * metrics["root"]
            + self.config.root_ori_weight * metrics["root_ori"]
            + joint_weight * metrics["joint"]
            + self.config.reference_root_weight * metrics["reference_root"]
            + self.config.reference_ori_weight * metrics["reference_ori"]
            + self.config.reference_joint_weight * metrics["reference_joint"]
        )
        rewards = rewards - self.config.fall_penalty * metrics["fallen"].astype(np.float64)
        self._last_metrics = {
            "local_ee_rmse_best": float(np.min(metrics["upper_ee"])) if metrics["upper_ee"].size else float("nan"),
            "local_ee_rmse_mean": float(np.mean(metrics["upper_ee"])) if metrics["upper_ee"].size else float("nan"),
            "upper_ee_rmse_best": float(np.min(metrics["upper_ee"])) if metrics["upper_ee"].size else float("nan"),
            "contact_total_best": float(np.min(metrics["contact"])) if metrics["contact"].size else float("nan"),
            "smooth_best": float(np.min(metrics["smooth"])) if metrics["smooth"].size else float("nan"),
            "reward_best": float(np.max(rewards)) if rewards.size else float("nan"),
        }
        return rewards

    def success(self, model: mujoco.MjModel, data: mujoco.MjData, metadata: dict | None = None) -> bool:
        return bool(self._last_metrics.get("local_ee_rmse_best", np.inf) <= self.config.success_local_ee_rmse)

    def get_sim_metadata(self) -> dict:
        return {"g1_wbc_metrics": dict(self._last_metrics)}

    def select_mpc_candidate(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        rewards: np.ndarray,
    ) -> int:
        """Select candidates lexicographically by contact, smoothness, upper EE, and root tracking."""
        metrics = self._candidate_tracking_metrics(states, sensors, controls)
        if not metrics:
            return int(np.argmax(rewards))

        ref_contact = metrics["contact"][0]
        ref_smooth = metrics["smooth"][0]
        ref_upper_ee = metrics["upper_ee"][0]
        ref_root = metrics["root"][0]
        ref_root_ori = metrics["root_ori"][0]
        ref_joint = metrics["joint"][0]
        contact_component_ok = np.ones_like(metrics["contact"], dtype=bool)
        contact_component_improved = np.zeros_like(metrics["contact"], dtype=bool)
        for key in ("contact_mismatch", "contact_no_ref", "contact_switch"):
            if key in metrics:
                contact_component_ok &= metrics[key] <= metrics[key][0] + self.config.accept_contact_tolerance
                improvement = (
                    self.config.accept_contact_switch_improvement
                    if key == "contact_switch"
                    else self.config.accept_contact_improvement
                )
                improved = metrics[key] <= metrics[key][0] - improvement
                contact_component_improved |= improved
        if "contact_force" in metrics:
            contact_component_ok &= (
                metrics["contact_force"] <= metrics["contact_force"][0] + self.config.accept_contact_force_tolerance
            )
            contact_component_improved |= metrics["contact_force"] <= (
                metrics["contact_force"][0] - self.config.accept_contact_force_improvement
            )
        for key in ("contact_force_max", "contact_force_mean"):
            if key in metrics:
                contact_component_ok &= metrics[key] <= metrics[key][0] + self.config.accept_contact_force_tolerance
                improved = metrics[key] <= metrics[key][0] - self.config.accept_contact_force_improvement
                contact_component_improved |= improved
        contact_equal = (metrics["contact"] <= ref_contact + self.config.accept_contact_tolerance) & contact_component_ok
        smooth_component_ok = metrics["smooth"] <= ref_smooth + self.config.accept_smooth_tolerance
        smooth_equal = smooth_component_ok
        contact_improved = (metrics["contact"] <= ref_contact - self.config.accept_contact_improvement) | contact_component_improved
        smooth_improved = metrics["smooth"] <= ref_smooth - self.config.accept_smooth_improvement
        if "action_smooth" in metrics:
            smooth_improved |= metrics["action_smooth"] <= metrics["action_smooth"][0] - self.config.accept_smooth_improvement
        lower_priority_improved = (
            (metrics["upper_ee"] <= ref_upper_ee - self.config.accept_upper_ee_improvement)
            | (metrics["root"] <= ref_root - self.config.accept_root_improvement)
            | (metrics["root_ori"] <= ref_root_ori - self.config.accept_root_ori_improvement)
        )

        acceptable = (
            ~metrics["fallen"]
            & contact_equal
            & smooth_equal
            & (metrics["upper_ee"] <= ref_upper_ee + self.config.accept_upper_ee_tolerance)
            & (metrics["root"] <= ref_root + self.config.accept_root_tolerance)
            & (metrics["root_ori"] <= ref_root_ori + self.config.accept_root_ori_tolerance)
            & (metrics["joint"] <= ref_joint + self.config.accept_joint_tolerance)
        )
        contact_priority_ok = contact_improved & contact_component_ok
        improved = (
            contact_priority_ok
            | (contact_equal & smooth_improved)
            | (contact_equal & smooth_equal & lower_priority_improved)
        )
        candidate_mask = acceptable & improved
        candidate_mask[0] = False
        if not np.any(candidate_mask):
            return 0

        score = -(
            20000.0 * metrics.get("contact_no_ref", np.zeros_like(metrics["contact"]))
            + 10000.0 * metrics.get("contact_mismatch", metrics["contact"])
            + 5000.0 * metrics.get("contact_switch", np.zeros_like(metrics["contact"]))
            + 1000.0 * metrics.get("contact_force", np.zeros_like(metrics["contact"]))
            + 100.0 * metrics.get("contact_force_max", np.zeros_like(metrics["contact"]))
            + 100.0 * metrics.get("contact_force_mean", np.zeros_like(metrics["contact"]))
            + 1000.0 * metrics["smooth"]
            + 1000.0 * metrics.get("action_smooth", np.zeros_like(metrics["contact"]))
            + 100.0 * metrics["upper_ee"]
            + 50.0 * metrics["root"]
            + 50.0 * metrics["root_ori"]
            + 10.0 * metrics["joint"]
        )
        score = np.where(candidate_mask, score, -np.inf)
        return int(np.argmax(score))

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

    def _candidate_tracking_metrics(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
    ) -> dict[str, np.ndarray]:
        controls = normalize_controls(controls)
        qpos = states[:, 1:, : self.model.nq] if states.shape[1] == controls.shape[1] + 1 else states[:, :, : self.model.nq]
        horizon = min(qpos.shape[1], controls.shape[1])
        if horizon <= 0 or qpos.shape[0] == 0:
            return {}
        qpos = qpos[:, :horizon]
        controls = controls[:, :horizon]
        sensors = sensors[:, :horizon] if sensors.size else np.zeros((controls.shape[0], horizon, CONTACT_SENSOR_DIM))
        target = self._target_controls_for_horizon(horizon)
        command_reference = self._command_reference_for_horizon(horizon)
        return self._tracking_metrics(qpos, sensors, controls, target, command_reference)

    def _tracking_metrics(
        self,
        qpos: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        target: np.ndarray,
        command_reference: np.ndarray,
    ) -> dict[str, np.ndarray]:
        root = np.sqrt(np.mean(np.sum((qpos[:, :, :3] - target[None, :, :3]) ** 2, axis=-1), axis=1))
        root_ori = np.mean(quat_geodesic_error(qpos[:, :, 3:7], target[None, :, 3:7]), axis=1)
        joint = np.sqrt(np.mean((qpos[:, :, 7:36] - target[None, :, JOINT_POS_SLICE]) ** 2, axis=(1, 2)))
        smooth_joint, smooth_root = self._smoothness_cost(qpos)
        reference_root = np.sqrt(
            np.mean(np.sum((controls[:, :, :3] - command_reference[None, :, :3]) ** 2, axis=-1), axis=1)
        )
        reference_ori = np.mean(quat_geodesic_error(controls[:, :, 3:7], command_reference[None, :, 3:7]), axis=1)
        reference_joint = np.sqrt(
            np.mean((controls[:, :, JOINT_POS_SLICE] - command_reference[None, :, JOINT_POS_SLICE]) ** 2, axis=(1, 2))
        )

        ref_mask = self._reference_contact_mask(controls.shape[1])
        exec_mask = sensors[:, :, :2] if sensors.shape[-1] >= 2 else np.zeros((*sensors.shape[:2], 2))
        forces = sensors[:, :, 2:4] if sensors.shape[-1] >= CONTACT_SENSOR_DIM else np.zeros((*sensors.shape[:2], 2))
        contact_mismatch = np.mean(np.abs(exec_mask - ref_mask[None]), axis=(1, 2))
        contact_no_ref = np.mean(exec_mask * (1.0 - ref_mask[None]), axis=(1, 2))
        contact_switch = (
            np.mean(np.abs(np.diff(exec_mask, axis=1)), axis=(1, 2)) if controls.shape[1] > 1 else np.zeros(qpos.shape[0])
        )
        contact_force = np.mean(
            forces * (1.0 - ref_mask[None]) + np.maximum(forces - 350.0, 0.0) * ref_mask[None],
            axis=(1, 2),
        )
        force_active = exec_mask > 0.5
        force_count = np.maximum(np.sum(force_active, axis=(1, 2)), 1.0)
        contact_force_mean = np.sum(forces * force_active, axis=(1, 2)) / force_count
        contact_force_max = np.max(forces, axis=(1, 2)) if forces.size else np.zeros(qpos.shape[0], dtype=np.float64)
        contact = contact_mismatch + 2.0 * contact_no_ref + 0.5 * contact_switch

        upper_ee = self._upper_ee_global_rmse(sensors, controls.shape[1])
        fallen = np.any(qpos[:, :, 2] <= self.config.fall_threshold, axis=1)
        smooth = smooth_joint + smooth_root
        action_smooth = self._sensor_action_smoothness_cost(sensors)
        if action_smooth is None:
            action_smooth = self._action_smoothness_cost(controls)
        return {
            "root": root,
            "root_ori": root_ori,
            "joint": joint,
            "upper_ee": upper_ee,
            "contact": contact,
            "contact_mismatch": contact_mismatch,
            "contact_no_ref": contact_no_ref,
            "contact_switch": contact_switch,
            "contact_force": contact_force,
            "contact_force_mean": contact_force_mean,
            "contact_force_max": contact_force_max,
            "smooth": smooth,
            "smooth_joint": smooth_joint,
            "smooth_root": smooth_root,
            "action_smooth": action_smooth,
            "reference_root": reference_root,
            "reference_ori": reference_ori,
            "reference_joint": reference_joint,
            "fallen": fallen,
        }

    def _smoothness_cost(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if qpos.shape[1] <= 2:
            zeros = np.zeros(qpos.shape[0], dtype=np.float64)
            return zeros, zeros
        joint_acc = np.diff(qpos[:, :, 7:36], n=2, axis=1)
        root_acc = np.diff(qpos[:, :, :3], n=2, axis=1)
        joint_cost = np.mean(np.linalg.norm(joint_acc, axis=-1), axis=1)
        root_cost = np.mean(np.linalg.norm(root_acc, axis=-1), axis=1)
        return joint_cost, root_cost

    def _action_smoothness_cost(self, controls: np.ndarray) -> np.ndarray:
        if controls.shape[1] <= 1:
            return np.zeros(controls.shape[0], dtype=np.float64)
        root_delta = np.diff(controls[:, :, :3], axis=1)
        joint_delta = np.diff(controls[:, :, JOINT_POS_SLICE], axis=1)
        return np.mean(np.linalg.norm(root_delta, axis=-1) + np.linalg.norm(joint_delta, axis=-1), axis=1)

    def _sensor_action_smoothness_cost(self, sensors: np.ndarray) -> np.ndarray | None:
        if sensors.shape[-1] < ACTION_SENSOR_START + ACTION_SENSOR_DIM:
            return None
        if sensors.shape[1] <= 1:
            return np.zeros(sensors.shape[0], dtype=np.float64)
        actions = sensors[:, :, ACTION_SENSOR_START : ACTION_SENSOR_START + ACTION_SENSOR_DIM]
        return np.mean(np.linalg.norm(np.diff(actions, axis=1), axis=-1), axis=1)

    def _upper_ee_global_rmse(self, sensors: np.ndarray, horizon: int) -> np.ndarray:
        if sensors.shape[-1] < UPPER_EE_SENSOR_START + UPPER_EE_SENSOR_DIM:
            return np.zeros(sensors.shape[0], dtype=np.float64)
        ee_pos = sensors[:, :horizon, UPPER_EE_SENSOR_START : UPPER_EE_SENSOR_START + UPPER_EE_SENSOR_DIM]
        ee_pos = ee_pos.reshape(sensors.shape[0], horizon, len(UPPER_EE_BODY_NAMES), 3)
        ref_pos = self._reference_upper_ee_positions(horizon)
        return np.sqrt(np.mean(np.sum((ee_pos - ref_pos[None]) ** 2, axis=-1), axis=(1, 2)))

    def _reference_upper_ee_positions(self, horizon: int) -> np.ndarray:
        times = float(self.data.time) + self.model.opt.timestep * (np.arange(horizon) + 1)
        return self._motion_body_positions_at_times(times, self._upper_ee_body_indices)

    def _motion_body_positions_at_times(self, times: np.ndarray, body_indices: list[int]) -> np.ndarray:
        frame = np.clip(np.asarray(times, dtype=np.float64) * self.motion.fps, 0.0, self.motion.num_frames - 1)
        lo = np.floor(frame).astype(np.int64)
        hi = np.minimum(lo + 1, self.motion.num_frames - 1)
        alpha = (frame - lo)[:, None, None]
        return (1.0 - alpha) * self.motion.body_pos_w[lo][:, body_indices] + alpha * self.motion.body_pos_w[hi][:, body_indices]

    def _reference_contact_mask(self, horizon: int) -> np.ndarray:
        times = float(self.data.time) + self.model.opt.timestep * (np.arange(horizon) + 1)
        frames = [self.motion.frame_index(float(t)) for t in times]
        return self.motion.contact_mask[frames].astype(np.float64)

    def _target_controls_for_horizon(self, horizon: int) -> np.ndarray:
        times = float(self.data.time) + self.model.opt.timestep * (np.arange(horizon) + 1)
        return self.reference_controls_for_times(times)

    def _command_reference_for_horizon(self, horizon: int) -> np.ndarray:
        times = float(self.data.time) + POLICY_DT + self.model.opt.timestep * np.arange(horizon)
        return self.reference_controls_for_times(times)


class G1WBCEE(G1WBCBase):
    reward_mode: Literal["ee", "joint"] = "ee"


class G1WBCJoint(G1WBCBase):
    reward_mode: Literal["ee", "joint"] = "joint"
