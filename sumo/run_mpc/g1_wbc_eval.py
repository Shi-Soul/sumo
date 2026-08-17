"""Evaluation CLI for G1 WBC no-MPC and MPC tracking methods."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import mujoco
import numpy as np
import tyro

from sumo.tasks.g1_wbc import G1WBCEE, G1WBCJoint
from sumo.utils.g1_wbc.constants import (
    DEFAULT_MOTION_FILE,
    DEFAULT_POLICY_VARIANT,
    G1_XML_PATH,
    MUJOCO_BODY_NAMES,
    POLICY_DT,
    UPPER_EE_BODY_NAMES,
)
from sumo.utils.g1_wbc.math import quat_geodesic_error, subtract_frame_transforms
from sumo.utils.g1_wbc.model import load_wbc_model
from sumo.utils.g1_wbc.motion import load_motion
from sumo.utils.g1_wbc.reference import motion_controls_at_times, motion_policy_qvel_at_times, motion_qvel_at_times
from sumo.utils.g1_wbc.rollout import build_foot_geom_side, contact_sensor_values

Method = Literal["no_mpc", "ee_mpc", "joint_mpc"]

PRIORITY_METRIC_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "contact",
        (
            "contact_no_ref_violation_rate",
            "contact_mismatch_rate",
            "contact_switch_rate",
            "contact_force_max",
            "contact_force_mean",
        ),
    ),
    ("smooth", ("motion_smoothness", "action_smoothness")),
    ("upper_global", ("upper_global_rmse",)),
    ("global", ("root_pos_rmse", "root_ori_mean")),
    ("other_tracking", ("joint_rmse", "local_ee_rmse")),
)

COMPARISON_METRIC_TOLERANCES: dict[str, float] = {
    "contact_force_max": 2.0,
    "contact_force_mean": 0.25,
}


@dataclass
class G1WBCEvalConfig:
    """Config for G1 WBC evaluation."""

    motion_file: str = str(DEFAULT_MOTION_FILE)
    motion_path: str = ""
    motion_type: Literal["isaaclab", "mujoco"] = "mujoco"
    policy: str = DEFAULT_POLICY_VARIANT
    methods: tuple[Method, ...] = ("no_mpc", "ee_mpc", "joint_mpc")
    episode_length_s: float = 0.0
    max_motions: int = 1
    output_dir: str = "run_mpc/results/g1_wbc_eval"
    visualize: bool = False
    mpc_optimizer: str = "cem"
    mpc_num_rollouts: int = 0
    mpc_max_opt_iters: int = 0
    success_local_ee_rmse: float = 0.12
    comparison_tolerance: float = 1e-6


def main(config: G1WBCEvalConfig | None = None) -> list[dict]:
    config = tyro.cli(G1WBCEvalConfig) if config is None else config
    motions = _resolve_motion_files(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    episodes_to_save = []

    for motion_file in motions:
        for method in config.methods:
            start = time.perf_counter()
            if method == "no_mpc":
                episode = run_no_mpc_episode(config, motion_file)
            else:
                episode = run_mpc_episode(config, motion_file, method)
            elapsed = time.perf_counter() - start
            metrics = compute_metrics(episode, success_threshold=config.success_local_ee_rmse)
            metrics.update(
                {
                    "method": method,
                    "motion_file": str(motion_file),
                    "policy": config.policy,
                    "runtime_s": elapsed,
                }
            )
            episode["metrics"] = metrics
            results.append(metrics)
            episodes_to_save.append((motion_file, method, episode))

    annotate_priority_comparisons(results, tolerance=config.comparison_tolerance)
    for motion_file, method, episode in episodes_to_save:
        save_episode(output_dir, motion_file, method, episode)

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    csv_path = output_dir / "metrics.csv"
    write_metrics_csv(csv_path, results)
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved metrics CSV to {csv_path}")
    return results


def annotate_priority_comparisons(rows: list[dict], *, tolerance: float = 1e-6) -> None:
    """Annotate MPC rows with strict no-MPC comparison following the WBC priority order."""
    baselines = {
        (row.get("motion_file"), row.get("policy")): row
        for row in rows
        if row.get("method") == "no_mpc" and row.get("motion_file") is not None
    }
    for row in rows:
        method = row.get("method")
        if method == "no_mpc":
            row.update(
                {
                    "priority_compare_status": "baseline",
                    "priority_better_than_no_mpc": False,
                    "priority_first_regression_group": "",
                    "priority_first_improvement_group": "",
                    "priority_num_regressions": 0,
                }
            )
            continue
        baseline = baselines.get((row.get("motion_file"), row.get("policy")))
        if baseline is None:
            row.update(
                {
                    "priority_compare_status": "missing_no_mpc",
                    "priority_better_than_no_mpc": False,
                    "priority_first_regression_group": "",
                    "priority_first_improvement_group": "",
                    "priority_num_regressions": 0,
                }
            )
            continue
        row.update(priority_compare_to_baseline(row, baseline, tolerance=tolerance))


def priority_compare_to_baseline(row: dict, baseline: dict, *, tolerance: float = 1e-6) -> dict:
    """Return lower-is-better comparison annotations against no-MPC in priority order."""
    deltas: dict[str, float] = {}
    first_regression_group = ""
    first_improvement_group = ""
    num_regressions = 0
    group_outcomes: list[tuple[str, bool, bool, int]] = []
    for group, metric_names in PRIORITY_METRIC_GROUPS:
        group_regressed = False
        group_improved = False
        group_regressions = 0
        for metric_name in metric_names:
            if metric_name not in row or metric_name not in baseline:
                continue
            value = row.get(metric_name)
            base_value = baseline.get(metric_name)
            if not _is_finite_number(value) or not _is_finite_number(base_value):
                continue
            delta = float(value) - float(base_value)
            deltas[f"delta_vs_no_mpc_{metric_name}"] = delta
            metric_tolerance = max(tolerance, COMPARISON_METRIC_TOLERANCES.get(metric_name, 0.0))
            if delta > metric_tolerance:
                group_regressed = True
                group_regressions += 1
            elif delta < -metric_tolerance:
                group_improved = True
        group_outcomes.append((group, group_regressed, group_improved, group_regressions))

    for group, _group_regressed, group_improved, _group_regressions in group_outcomes:
        if group_improved:
            first_improvement_group = group
            break

    decisive_improvement_group = ""
    for group, group_regressed, group_improved, group_regressions in group_outcomes:
        if group_regressed:
            first_regression_group = group
            num_regressions = group_regressions
            break
        if group_improved:
            decisive_improvement_group = group
            break

    better = bool(decisive_improvement_group) and num_regressions == 0
    if better:
        status = f"better_{decisive_improvement_group}"
    elif first_regression_group:
        status = f"regressed_{first_regression_group}"
    else:
        status = "tied"
    return {
        **deltas,
        "priority_compare_status": status,
        "priority_better_than_no_mpc": better,
        "priority_first_regression_group": first_regression_group,
        "priority_first_improvement_group": first_improvement_group,
        "priority_num_regressions": int(num_regressions),
    }


def _is_finite_number(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _resolve_motion_files(config: G1WBCEvalConfig) -> list[Path]:
    if config.motion_path:
        root = Path(config.motion_path).expanduser().resolve()
        files = sorted(root.rglob("*.npz"))
    else:
        files = [Path(config.motion_file).expanduser().resolve()]
    if config.max_motions > 0:
        files = files[: config.max_motions]
    if not files:
        raise ValueError("No motion files found for evaluation")
    return files


def apply_mpc_compute_budget(config: G1WBCEvalConfig, optimizer_config, controller_config) -> None:
    """Apply optional MPC compute-budget overrides without changing task semantics."""
    if config.mpc_num_rollouts > 0:
        optimizer_config.num_rollouts = int(config.mpc_num_rollouts)
    if config.mpc_max_opt_iters > 0:
        controller_config.max_opt_iters = int(config.mpc_max_opt_iters)


def episode_length_for_motion(config: G1WBCEvalConfig, motion_file: Path) -> float:
    """Return requested episode length; nonpositive means run the full motion."""
    if config.episode_length_s > 0.0:
        return float(config.episode_length_s)
    motion = load_motion(motion_file, config.motion_type)
    return max(motion.duration, motion.dt)


def run_no_mpc_episode(config: G1WBCEvalConfig, motion_file: Path) -> dict:
    motion = load_motion(motion_file, config.motion_type)
    model = load_wbc_model()
    data = mujoco.MjData(model)
    os.environ["SUMO_G1_WBC_POLICY"] = config.policy
    from sumo.app.dora.g1_wbc_simulation import SimBackendG1WBC

    backend = SimBackendG1WBC(model)
    data.qpos[:] = np.concatenate([motion.root_qpos(0), motion.joint_pos[0]])
    data.qvel[:] = 0.0
    data.qvel[:] = motion_qvel_at_times(motion, np.asarray([0.0]))[0]
    mujoco.mj_forward(model, data)

    episode_length_s = episode_length_for_motion(config, motion_file)
    max_steps = max(1, int(np.ceil(episode_length_s / model.opt.timestep)))
    qpos_traj = []
    qvel_traj = []
    ctrl_traj = []
    time_traj = []
    reference_controls = []
    contact_time_traj = []
    contact_traj = []
    reference_contact = []
    foot_geom_side = build_foot_geom_side(model)
    upper_ee_body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in UPPER_EE_BODY_NAMES]
    upper_ee_traj = []
    upper_ee_time_traj = []
    sim_dt = float(model.opt.timestep)
    command_times = POLICY_DT + sim_dt * np.arange(max_steps, dtype=np.float64)
    recorded_times = sim_dt * (np.arange(max_steps, dtype=np.float64) + 1.0)
    command_controls = motion_controls_at_times(motion, command_times)
    command_qvels = motion_policy_qvel_at_times(motion, command_times)
    recorded_reference_controls = motion_controls_at_times(motion, recorded_times)
    recorded_reference_contact = motion.contact_mask[
        np.clip(np.round(recorded_times * motion.fps).astype(np.int64), 0, motion.num_frames - 1)
    ]
    for step in range(max_steps):
        ref_control = command_controls[step]
        ref_qvel = command_qvels[step]
        backend.sim(model, data, ref_control, ref_qvel)
        time_traj.append(float(data.time))
        qpos_traj.append(np.array(data.qpos))
        qvel_traj.append(np.array(data.qvel))
        ctrl_traj.append(np.array(data.ctrl))
        reference_controls.append(recorded_reference_controls[step])
        contact_time_traj.append(float(data.time))
        contact_traj.append(contact_sensor_values(model, data, foot_geom_side))
        reference_contact.append(recorded_reference_contact[step])
        upper_ee_time_traj.append(float(data.time))
        upper_ee_traj.append(np.asarray(data.xpos[upper_ee_body_ids], dtype=np.float64).copy())
    upper_ee_time_traj = np.asarray(upper_ee_time_traj)
    upper_ee_reference = _reference_body_positions(
        motion,
        upper_ee_time_traj,
        [MUJOCO_BODY_NAMES.index(name) for name in UPPER_EE_BODY_NAMES],
    )
    return {
        "time_traj": np.asarray(time_traj),
        "qpos_traj": np.asarray(qpos_traj),
        "qvel_traj": np.asarray(qvel_traj),
        "ctrl_traj": np.asarray(ctrl_traj),
        "reference_controls": np.asarray(reference_controls),
        "contact_time_traj": np.asarray(contact_time_traj),
        "contact_traj": np.asarray(contact_traj),
        "reference_contact": np.asarray(reference_contact),
        "upper_ee_time_traj": upper_ee_time_traj,
        "upper_ee_traj": np.asarray(upper_ee_traj),
        "reference_upper_ee_traj": upper_ee_reference,
        "method": "no_mpc",
    }


def run_mpc_episode(config: G1WBCEvalConfig, motion_file: Path, method: Method) -> dict:
    from judo.optimizers import get_registered_optimizers

    from sumo.controller import ControllerConfig, G1WBCController
    from sumo.run_mpc.run_mpc import run_single_episode
    from sumo.utils.g1_wbc.rollout import G1WBCRolloutBackend

    os.environ["SUMO_G1_WBC_MOTION_FILE"] = str(motion_file)
    os.environ["SUMO_G1_WBC_MOTION_TYPE"] = config.motion_type
    os.environ["SUMO_G1_WBC_POLICY"] = config.policy
    task = G1WBCEE() if method == "ee_mpc" else G1WBCJoint()
    optimizer_cls, optimizer_config_cls = get_registered_optimizers()[config.mpc_optimizer]
    optimizer_config = optimizer_config_cls()
    optimizer_config.set_override("g1_wbc_ee" if method == "ee_mpc" else "g1_wbc_joint")
    controller_config = ControllerConfig()
    controller_config.set_override("g1_wbc_ee" if method == "ee_mpc" else "g1_wbc_joint")
    apply_mpc_compute_budget(config, optimizer_config, controller_config)
    optimizer = optimizer_cls(optimizer_config, task.nu)
    controller = G1WBCController(
        controller_config,
        task,
        optimizer,
        rollout_backend="mujoco_g1_wbc",
        rollout_backend_registry={"mujoco_g1_wbc": G1WBCRolloutBackend},
    )

    class _Sim:
        supports_reference_qvel = True

        def __init__(self, task):
            from sumo.app.dora.g1_wbc_simulation import SimBackendG1WBC

            self.task = task
            self.backend = SimBackendG1WBC(task.sim_model)
            self.foot_geom_side = build_foot_geom_side(task.sim_model)
            self.upper_ee_body_ids = [
                mujoco.mj_name2id(task.sim_model, mujoco.mjtObj.mjOBJ_BODY, name) for name in UPPER_EE_BODY_NAMES
            ]
            self.contact_time_traj = []
            self.contact_traj = []
            self.upper_ee_time_traj = []
            self.upper_ee_traj = []
            self.refined_time_traj = []
            self.refined_controls = []

        def step(self, command, reference_qvel=None):
            self.refined_time_traj.append(float(self.task.data.time) + POLICY_DT)
            self.refined_controls.append(np.asarray(command, dtype=np.float64).copy())
            self.task.pre_sim_step()
            self.backend.sim(self.task.sim_model, self.task.data, self.task.task_to_sim_ctrl(command), reference_qvel)
            self.task.post_sim_step()
            self.contact_time_traj.append(float(self.task.data.time))
            self.contact_traj.append(contact_sensor_values(self.task.sim_model, self.task.data, self.foot_geom_side))
            self.upper_ee_time_traj.append(float(self.task.data.time))
            self.upper_ee_traj.append(np.asarray(self.task.data.xpos[self.upper_ee_body_ids], dtype=np.float64).copy())

        def reset(self):
            self.backend.reset()
            self.contact_time_traj.clear()
            self.contact_traj.clear()
            self.upper_ee_time_traj.clear()
            self.upper_ee_traj.clear()
            self.refined_time_traj.clear()
            self.refined_controls.clear()

        def get_sim_metadata(self):
            return self.backend.get_sim_metadata()

    sim_dt = float(task.sim_model.opt.timestep)
    episode_length_s = max(0.0, episode_length_for_motion(config, motion_file) - 0.5 * sim_dt)
    episode_config = type(
        "EpisodeConfig",
        (),
        {
            "episode_length_s": episode_length_s,
            "viz_dt": sim_dt,
            "num_episodes": 1,
            "record_all_data": False,
            "record_qvel": True,
            "record_xpos": False,
            "record_xquat": False,
            "record_ctrl": True,
            "record_sensordata": False,
            "record_mocap": False,
            "record_traces": False,
            "record_rollouts": False,
            "record_rollout_controls": False,
            "record_rollout_sensors": False,
        },
    )()
    sim = _Sim(task)
    episode = run_single_episode(episode_config, task, controller, sim, viser_model=None, episode_idx=0)
    episode["method"] = method
    episode["mpc_num_rollouts"] = int(optimizer_config.num_rollouts)
    episode["mpc_max_opt_iters"] = int(controller_config.max_opt_iters)
    if "qpos_traj" in episode and episode["qpos_traj"].size:
        times = episode["time_traj"] + task.sim_model.opt.timestep
        episode["reference_controls"] = task.reference_controls_for_times(times)
    episode["refined_time_traj"] = np.asarray(sim.refined_time_traj)
    episode["refined_controls"] = np.asarray(sim.refined_controls)
    if episode["refined_time_traj"].size:
        episode["refined_reference_controls"] = task.reference_controls_for_times(episode["refined_time_traj"])
    episode["contact_time_traj"] = np.asarray(sim.contact_time_traj)
    episode["contact_traj"] = np.asarray(sim.contact_traj)
    episode["upper_ee_time_traj"] = np.asarray(sim.upper_ee_time_traj)
    episode["upper_ee_traj"] = np.asarray(sim.upper_ee_traj)
    if episode["contact_time_traj"].size:
        episode["reference_contact"] = np.asarray(
            [task.motion.contact_mask[task.motion.frame_index(float(t))] for t in episode["contact_time_traj"]]
        )
    if episode["upper_ee_time_traj"].size:
        episode["reference_upper_ee_traj"] = _reference_body_positions(
            task.motion,
            episode["upper_ee_time_traj"],
            [MUJOCO_BODY_NAMES.index(name) for name in UPPER_EE_BODY_NAMES],
        )
    return episode


def compute_metrics(episode: dict, *, success_threshold: float) -> dict:
    qpos = np.asarray(episode.get("qpos_traj", np.empty((0, 36))))
    refs = np.asarray(episode.get("reference_controls", np.empty((0, 36))))
    if qpos.size == 0 or refs.size == 0:
        return {"success": False}
    n = min(len(qpos), len(refs))
    qpos = qpos[:n]
    refs = refs[:n]
    root_pos_rmse = float(np.sqrt(np.mean(np.sum((qpos[:, :3] - refs[:, :3]) ** 2, axis=-1))))
    root_ori_mean = float(np.mean(quat_geodesic_error(qpos[:, 3:7], refs[:, 3:7])))
    joint_rmse = float(np.sqrt(np.mean((qpos[:, 7:36] - refs[:, 7:36]) ** 2)))
    local_ee_rmse = float(_local_ee_rmse(qpos, refs))
    ctrl = np.asarray(episode.get("ctrl_traj", np.empty((0, 29))))
    action_smoothness = float(np.mean(np.linalg.norm(np.diff(ctrl, axis=0), axis=-1))) if len(ctrl) > 1 else 0.0
    motion_smoothness = _motion_smoothness(qpos)
    fallen = bool(np.any(qpos[:, 2] < 0.55))
    metrics = {
        "success": bool(local_ee_rmse <= success_threshold and not fallen),
        "mpc_num_rollouts": int(episode.get("mpc_num_rollouts", 0)),
        "mpc_max_opt_iters": int(episode.get("mpc_max_opt_iters", 0)),
        "root_pos_rmse": root_pos_rmse,
        "root_ori_mean": root_ori_mean,
        "joint_rmse": joint_rmse,
        "local_ee_rmse": local_ee_rmse,
        "motion_smoothness": motion_smoothness,
        "action_smoothness": action_smoothness,
        "fallen": fallen,
        "num_frames": int(n),
    }
    metrics.update(_contact_metrics(episode))
    metrics.update(_upper_ee_metrics(episode))
    if "refined_controls" in episode:
        refined = np.asarray(episode["refined_controls"])
        refined_refs = np.asarray(episode.get("refined_reference_controls", refs))
        m = min(len(refined), len(refined_refs))
        if m:
            metrics["refined_root_delta_rmse"] = float(
                np.sqrt(np.mean(np.sum((refined[:m, :3] - refined_refs[:m, :3]) ** 2, axis=-1)))
            )
            metrics["refined_joint_delta_rmse"] = float(
                np.sqrt(np.mean((refined[:m, 7:36] - refined_refs[:m, 7:36]) ** 2))
            )
    return metrics


def _motion_smoothness(qpos: np.ndarray) -> float:
    if qpos.shape[0] <= 2:
        return 0.0
    root_acc = np.diff(qpos[:, :3], n=2, axis=0)
    joint_acc = np.diff(qpos[:, 7:36], n=2, axis=0)
    return float(np.mean(np.linalg.norm(root_acc, axis=-1)) + np.mean(np.linalg.norm(joint_acc, axis=-1)))


def _upper_ee_metrics(episode: dict) -> dict:
    upper_ee = np.asarray(episode.get("upper_ee_traj", np.empty((0, len(UPPER_EE_BODY_NAMES), 3))))
    ref = np.asarray(episode.get("reference_upper_ee_traj", np.empty((0, len(UPPER_EE_BODY_NAMES), 3))))
    if upper_ee.size == 0 or ref.size == 0:
        return {}
    n = min(len(upper_ee), len(ref))
    err = upper_ee[:n] - ref[:n]
    return {"upper_global_rmse": float(np.sqrt(np.mean(np.sum(err * err, axis=-1))))}


def _reference_body_positions(motion, times: np.ndarray, body_indices: list[int]) -> np.ndarray:
    frame = np.clip(np.asarray(times, dtype=np.float64) * motion.fps, 0.0, motion.num_frames - 1)
    lo = np.floor(frame).astype(np.int64)
    hi = np.minimum(lo + 1, motion.num_frames - 1)
    alpha = (frame - lo)[:, None, None]
    return (1.0 - alpha) * motion.body_pos_w[lo][:, body_indices] + alpha * motion.body_pos_w[hi][:, body_indices]


def _contact_metrics(episode: dict) -> dict:
    contact = np.asarray(episode.get("contact_traj", np.empty((0, 4))))
    ref_contact = np.asarray(episode.get("reference_contact", np.empty((0, 2))))
    if contact.size == 0 or ref_contact.size == 0:
        return {}
    n = min(len(contact), len(ref_contact))
    exec_mask = contact[:n, :2]
    forces = contact[:n, 2:4]
    ref_mask = ref_contact[:n].astype(np.float64)
    switch_rate = float(np.mean(np.abs(np.diff(exec_mask, axis=0)))) if n > 1 else 0.0
    force_when_contact = forces[exec_mask > 0.5]
    return {
        "contact_mismatch_rate": float(np.mean(np.abs(exec_mask - ref_mask))),
        "contact_no_ref_violation_rate": float(np.mean(exec_mask * (1.0 - ref_mask))),
        "contact_switch_rate": switch_rate,
        "contact_force_mean": float(np.mean(force_when_contact)) if force_when_contact.size else 0.0,
        "contact_force_max": float(np.max(forces)) if forces.size else 0.0,
    }


def _local_ee_rmse(qpos: np.ndarray, refs: np.ndarray) -> float:
    model = mujoco.MjModel.from_xml_path(str(G1_XML_PATH))
    data = mujoco.MjData(model)
    names = ("left_wrist_yaw_link", "right_wrist_yaw_link", "left_ankle_roll_link", "right_ankle_roll_link")
    anchor_idx = MUJOCO_BODY_NAMES.index("pelvis")
    body_indices = [MUJOCO_BODY_NAMES.index(name) for name in names]
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in MUJOCO_BODY_NAMES]
    errors = []
    for q, ref in zip(qpos, refs, strict=False):
        data.qpos[:] = q
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        exec_pos = data.xpos[body_ids].copy()
        exec_quat = data.xquat[body_ids].copy()
        data.qpos[:] = np.concatenate([ref[:3], ref[3:7], ref[7:36]])
        mujoco.mj_forward(model, data)
        ref_pos = data.xpos[body_ids].copy()
        ref_quat = data.xquat[body_ids].copy()
        exec_anchor_pos = np.repeat(exec_pos[anchor_idx : anchor_idx + 1], len(body_indices), axis=0)
        exec_anchor_quat = np.repeat(exec_quat[anchor_idx : anchor_idx + 1], len(body_indices), axis=0)
        ref_anchor_pos = np.repeat(ref_pos[anchor_idx : anchor_idx + 1], len(body_indices), axis=0)
        ref_anchor_quat = np.repeat(ref_quat[anchor_idx : anchor_idx + 1], len(body_indices), axis=0)
        exec_local, _ = subtract_frame_transforms(exec_anchor_pos, exec_anchor_quat, exec_pos[body_indices], exec_quat[body_indices])
        ref_local, _ = subtract_frame_transforms(ref_anchor_pos, ref_anchor_quat, ref_pos[body_indices], ref_quat[body_indices])
        errors.append(np.linalg.norm(exec_local - ref_local, axis=-1))
    return float(np.sqrt(np.mean(np.asarray(errors) ** 2)))


def save_episode(output_dir: Path, motion_file: Path, method: str, episode: dict) -> None:
    stem = _episode_file_stem(motion_file)
    path = output_dir / f"{stem}_{method}.h5"
    with h5py.File(path, "w") as f:
        f.attrs["motion_file"] = str(motion_file)
        f.attrs["method"] = method
        group = f.create_group(method)
        for key, value in episode.items():
            if isinstance(value, np.ndarray):
                group.create_dataset(key, data=value)
        metrics = episode.get("metrics", {})
        for key, value in metrics.items():
            group.attrs[key] = value


def _episode_file_stem(motion_file: Path) -> str:
    stem = motion_file.stem
    if motion_file.name == "motion.npz" and motion_file.parent.name:
        stem = motion_file.parent.name
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)


def write_metrics_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    lines = [",".join(keys)]
    for row in rows:
        lines.append(",".join(str(row.get(key, "")) for key in keys))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
