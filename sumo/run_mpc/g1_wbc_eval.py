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
from sumo.utils.g1_wbc.constants import DEFAULT_MOTION_FILE, DEFAULT_POLICY_VARIANT, G1_XML_PATH, MUJOCO_BODY_NAMES
from sumo.utils.g1_wbc.math import quat_geodesic_error, subtract_frame_transforms
from sumo.utils.g1_wbc.motion import load_motion
from sumo.utils.g1_wbc.policy import TrackingPolicyRuntime
from sumo.utils.g1_wbc.reference import build_reference_frames, motion_controls_at_times
from sumo.utils.g1_wbc.rollout import build_foot_geom_side, contact_sensor_values

Method = Literal["no_mpc", "ee_mpc", "joint_mpc"]


@dataclass
class G1WBCEvalConfig:
    """Config for G1 WBC evaluation."""

    motion_file: str = str(DEFAULT_MOTION_FILE)
    motion_path: str = ""
    motion_type: Literal["isaaclab", "mujoco"] = "mujoco"
    policy: str = DEFAULT_POLICY_VARIANT
    methods: tuple[Method, ...] = ("no_mpc", "ee_mpc", "joint_mpc")
    episode_length_s: float = 1.0
    max_motions: int = 1
    output_dir: str = "run_mpc/results/g1_wbc_eval"
    visualize: bool = False
    mpc_optimizer: str = "cem"
    mpc_num_rollouts: int = 0
    mpc_max_opt_iters: int = 0
    success_local_ee_rmse: float = 0.12


def main(config: G1WBCEvalConfig | None = None) -> list[dict]:
    config = tyro.cli(G1WBCEvalConfig) if config is None else config
    motions = _resolve_motion_files(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

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
                    "mpc_num_rollouts": int(config.mpc_num_rollouts),
                    "mpc_max_opt_iters": int(config.mpc_max_opt_iters),
                    "runtime_s": elapsed,
                }
            )
            episode["metrics"] = metrics
            results.append(metrics)
            save_episode(output_dir, motion_file, method, episode)

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    csv_path = output_dir / "metrics.csv"
    write_metrics_csv(csv_path, results)
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved metrics CSV to {csv_path}")
    return results


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


def run_no_mpc_episode(config: G1WBCEvalConfig, motion_file: Path) -> dict:
    motion = load_motion(motion_file, config.motion_type)
    model = mujoco.MjModel.from_xml_path(str(G1_XML_PATH))
    data = mujoco.MjData(model)
    runtime = TrackingPolicyRuntime(model, config.policy)
    data.qpos[:] = np.concatenate([motion.root_qpos(0), motion.joint_pos[0]])
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    max_steps = min(int(config.episode_length_s / model.opt.timestep), motion.num_frames * 4)
    qpos_traj = []
    qvel_traj = []
    ctrl_traj = []
    time_traj = []
    reference_controls = []
    contact_time_traj = []
    contact_traj = []
    reference_contact = []
    foot_geom_side = build_foot_geom_side(model)
    for step in range(max_steps):
        time_s = step * model.opt.timestep
        ref_controls = motion_controls_at_times(motion, time_s + model.opt.timestep * np.arange(2))
        ref = build_reference_frames(model, ref_controls, model.opt.timestep)[0]
        data.ctrl[:] = runtime.step(data, ref)
        mujoco.mj_step(model, data)
        time_traj.append(float(data.time))
        qpos_traj.append(np.array(data.qpos))
        qvel_traj.append(np.array(data.qvel))
        ctrl_traj.append(np.array(data.ctrl))
        reference_controls.append(ref_controls[0])
        contact_time_traj.append(float(data.time))
        contact_traj.append(contact_sensor_values(model, data, foot_geom_side))
        reference_contact.append(motion.contact_mask[motion.frame_index(float(data.time))])
    return {
        "time_traj": np.asarray(time_traj),
        "qpos_traj": np.asarray(qpos_traj),
        "qvel_traj": np.asarray(qvel_traj),
        "ctrl_traj": np.asarray(ctrl_traj),
        "reference_controls": np.asarray(reference_controls),
        "contact_time_traj": np.asarray(contact_time_traj),
        "contact_traj": np.asarray(contact_traj),
        "reference_contact": np.asarray(reference_contact),
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
        def __init__(self, task):
            from sumo.app.dora.g1_wbc_simulation import SimBackendG1WBC

            self.task = task
            self.backend = SimBackendG1WBC(task.sim_model)
            self.foot_geom_side = build_foot_geom_side(task.sim_model)
            self.contact_time_traj = []
            self.contact_traj = []

        def step(self, command):
            self.task.pre_sim_step()
            self.backend.sim(self.task.sim_model, self.task.data, self.task.task_to_sim_ctrl(command))
            self.task.post_sim_step()
            self.contact_time_traj.append(float(self.task.data.time))
            self.contact_traj.append(contact_sensor_values(self.task.sim_model, self.task.data, self.foot_geom_side))

    episode_config = type(
        "EpisodeConfig",
        (),
        {
            "episode_length_s": config.episode_length_s,
            "viz_dt": 0.02,
            "num_episodes": 1,
            "record_all_data": True,
            "record_qvel": True,
            "record_xpos": True,
            "record_xquat": True,
            "record_ctrl": True,
            "record_sensordata": True,
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
    if "qpos_traj" in episode and episode["qpos_traj"].size:
        times = episode["time_traj"]
        episode["reference_controls"] = task.reference_controls_for_times(times)
    episode["contact_time_traj"] = np.asarray(sim.contact_time_traj)
    episode["contact_traj"] = np.asarray(sim.contact_traj)
    if episode["contact_time_traj"].size:
        episode["reference_contact"] = np.asarray(
            [task.motion.contact_mask[task.motion.frame_index(float(t))] for t in episode["contact_time_traj"]]
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
    fallen = bool(np.any(qpos[:, 2] < 0.55))
    metrics = {
        "success": bool(local_ee_rmse <= success_threshold and not fallen),
        "root_pos_rmse": root_pos_rmse,
        "root_ori_mean": root_ori_mean,
        "joint_rmse": joint_rmse,
        "local_ee_rmse": local_ee_rmse,
        "action_smoothness": action_smoothness,
        "fallen": fallen,
        "num_frames": int(n),
    }
    metrics.update(_contact_metrics(episode))
    if "refined_controls" in episode:
        refined = np.asarray(episode["refined_controls"])
        m = min(len(refined), len(refs))
        if m:
            metrics["refined_root_delta_rmse"] = float(np.sqrt(np.mean(np.sum((refined[:m, :3] - refs[:m, :3]) ** 2, axis=-1))))
            metrics["refined_joint_delta_rmse"] = float(np.sqrt(np.mean((refined[:m, 7:36] - refs[:m, 7:36]) ** 2)))
    return metrics


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
    stem = motion_file.stem.replace("/", "_")
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
