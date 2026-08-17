"""Compare SUMO native no-MPC G1 WBC against tracking_bfm."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import tyro

from sumo.utils.extensions import require_g1_extensions
from sumo.utils.g1_wbc.constants import (
    DEFAULT_MOTION_FILE,
    DEFAULT_POLICY_VARIANT,
    MUJOCO_JOINT_NAMES,
    POLICY_DECIMATION,
    POLICY_DT,
    SIM_DT,
    WXY_POLICY_VARIANTS,
    resolve_policy_path,
)
from sumo.utils.g1_wbc.model import load_wbc_model
from sumo.utils.g1_wbc.motion import load_motion
from sumo.utils.g1_wbc.reference import motion_controls_at_times, motion_policy_qvel_at_times, motion_qvel_at_times

_STATE_LAST_ACTION_OFFSET = 2 + 36
_STATE_HELD_CTRL_OFFSET = _STATE_LAST_ACTION_OFFSET + 29


@dataclass
class G1WBCCompareTrackingBFMConfig:
    """Config for direct no-MPC comparison with tracking_bfm."""

    motion_file: str = str(DEFAULT_MOTION_FILE)
    motion_type: str = "mujoco"
    policy: str = DEFAULT_POLICY_VARIANT
    steps: int = 500
    tracking_bfm_root: str = str(Path(__file__).resolve().parents[3] / "tracking_bfm")
    tracking_python: str = ""
    output_dir: str = "run_mpc/results/g1_wbc_compare_tracking_bfm"
    obs_atol: float = 5e-4
    direct_action_atol: float = 2e-4
    direct_ctrl_atol: float = 2e-4
    initial_state_atol: float = 1e-6
    model_options_atol: float = 1e-9
    first_action_atol: float = 2e-4
    action_atol: float = 5e-2
    ctrl_atol: float = 5e-2
    qpos_atol: float = 2e-2
    qvel_atol: float = 5e-1
    require_closed_loop: bool = False
    extend_short_motion: bool = True


def main(config: G1WBCCompareTrackingBFMConfig | None = None) -> dict:
    config = tyro.cli(G1WBCCompareTrackingBFMConfig) if config is None else config
    source_motion_file = Path(config.motion_file).expanduser().resolve()
    policy_path = resolve_policy_path(config.policy)
    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    motion_file = _motion_file_for_steps(source_motion_file, output_dir, config)
    config.motion_file = str(motion_file)

    with tempfile.TemporaryDirectory(prefix="g1_wbc_compare_") as tmp:
        tracking_path = Path(tmp) / "tracking_bfm.npz"
        _run_tracking_bfm_reference(config, motion_file, policy_path, tracking_path)
        tracking = dict(np.load(tracking_path))

    sumo = _run_sumo_native(config, motion_file, policy_path)
    sumo_forced = _run_sumo_native_on_tracking_states(config, motion_file, policy_path, tracking)
    np.savez(output_dir / "tracking_bfm.npz", **tracking)
    np.savez(output_dir / "sumo_native.npz", **sumo)
    np.savez(output_dir / "sumo_forced_states.npz", **sumo_forced)

    report = _compare_outputs(tracking, sumo, sumo_forced, config)
    report["source_motion_file"] = str(source_motion_file)
    report["effective_motion_file"] = str(motion_file)
    (output_dir / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)
    return report


def _motion_file_for_steps(
    motion_file: Path,
    output_dir: Path,
    config: G1WBCCompareTrackingBFMConfig,
) -> Path:
    if not config.extend_short_motion:
        return motion_file
    data = np.load(motion_file)
    if "joint_pos" not in data:
        return motion_file
    num_frames = int(np.asarray(data["joint_pos"]).shape[0])
    required_frames = int(config.steps) + 2
    if num_frames >= required_frames:
        return motion_file

    output_path = output_dir / f"{motion_file.stem}_hold_last_{required_frames}frames.npz"
    arrays = {}
    pad_count = required_frames - num_frames
    for key in data.files:
        value = np.asarray(data[key])
        if value.ndim > 0 and value.shape[0] == num_frames:
            pad = np.repeat(value[-1:], pad_count, axis=0)
            arrays[key] = np.concatenate([value, pad], axis=0)
        else:
            arrays[key] = value
    np.savez(output_path, **arrays)
    return output_path


def _run_tracking_bfm_reference(
    config: G1WBCCompareTrackingBFMConfig,
    motion_file: Path,
    policy_path: Path,
    output_path: Path,
) -> None:
    tracking_root = Path(config.tracking_bfm_root).expanduser().resolve()
    tracking_python = Path(config.tracking_python).expanduser() if config.tracking_python else tracking_root / ".venv/bin/python"
    if not tracking_python.is_file():
        raise FileNotFoundError(f"tracking_bfm Python not found: {tracking_python}")

    runner = r'''
import argparse
import numpy as np
import mujoco
import onnxruntime as ort
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.tracking.wbteleop.env_cfg import unitree_g1_flat_tracking_bfm_wbteleop_env_cfg

SUMO_JOINT_NAMES = %r


def configure_env(motion_file, motion_type):
    cfg = unitree_g1_flat_tracking_bfm_wbteleop_env_cfg(play=True)
    cfg.scene.num_envs = 1
    cfg.terminations = {}
    cfg.events = {}
    motion = cfg.commands["motion"]
    motion.motion_file = motion_file
    motion.motion_type = motion_type
    motion.sampling_mode = "start"
    motion.debug_vis = False
    motion.joint_position_range = (0.0, 0.0)
    actor = cfg.observations["actor"]
    actor.enable_corruption = False
    terms = actor.terms
    terms["ref_limb_ee_pose_b"].params["history_steps"] = 0
    terms["ref_limb_ee_pose_b"].params["future_steps"] = 1
    terms["ref_limb_ee_pose_b"].history_length = 5
    for name in ("robot_limb_ee_pose_b", "projected_gravity", "base_ang_vel", "joint_pos", "joint_vel", "actions"):
        terms[name].history_length = 5
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-file", required=True)
    parser.add_argument("--motion-type", required=True)
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    sess = ort.InferenceSession(args.policy_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    env = ManagerBasedRlEnv(configure_env(args.motion_file, args.motion_type), device="cpu")
    try:
        obs, _ = env.reset()
        model = env.sim.mj_model
        initial_qpos = env.sim.data.qpos.numpy()[0].copy()
        initial_qvel = env.sim.data.qvel.numpy()[0].copy()

        pre_qpos_traj = []
        pre_qvel_traj = []
        obs_traj = []
        action_traj = []
        qpos_traj = []
        qvel_traj = []
        ctrl_traj = []
        time_traj = []
        command_time_steps = []
        motion_term = env.command_manager.get_term("motion")
        for _ in range(args.steps):
            pre_qpos_traj.append(env.sim.data.qpos.numpy()[0].copy())
            pre_qvel_traj.append(env.sim.data.qvel.numpy()[0].copy())
            command_time_steps.append(int(motion_term.time_steps.detach().cpu().numpy()[0]))
            actor_obs = obs["actor"].detach().cpu().numpy().astype(np.float32)
            action = sess.run(None, {input_name: actor_obs})[0].astype(np.float32)
            obs_traj.append(actor_obs[0].copy())
            action_traj.append(action[0].copy())
            obs, _, _, _, _ = env.step(torch.as_tensor(action, dtype=torch.float32))
            qpos_traj.append(env.sim.data.qpos.numpy()[0].copy())
            qvel_traj.append(env.sim.data.qvel.numpy()[0].copy())
            ctrl = env.sim.data.ctrl.numpy()[0].copy()
            ctrl_by_joint = {}
            for actuator_id in range(model.nu):
                joint_id = int(model.actuator_trnid[actuator_id, 0])
                joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                joint_name = joint_name.split("/")[-1]
                ctrl_by_joint[joint_name] = ctrl[actuator_id]
            ctrl_traj.append(np.asarray([ctrl_by_joint[name] for name in SUMO_JOINT_NAMES], dtype=np.float64))
            time_traj.append(env.sim.data.time.numpy()[0].copy())

        np.savez(
            args.output,
            initial_qpos=initial_qpos,
            initial_qvel=initial_qvel,
            pre_qpos_traj=np.asarray(pre_qpos_traj),
            pre_qvel_traj=np.asarray(pre_qvel_traj),
            obs_traj=np.asarray(obs_traj),
            action_traj=np.asarray(action_traj),
            qpos_traj=np.asarray(qpos_traj),
            qvel_traj=np.asarray(qvel_traj),
            ctrl_traj=np.asarray(ctrl_traj),
            time_traj=np.asarray(time_traj),
            command_time_steps=np.asarray(command_time_steps, dtype=np.int64),
            model_options=np.asarray([
                model.opt.timestep,
                float(model.opt.integrator),
                float(model.opt.cone),
                model.opt.impratio,
                float(model.opt.iterations),
                float(model.opt.ls_iterations),
            ], dtype=np.float64),
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
''' % (tuple(MUJOCO_JOINT_NAMES),)
    subprocess.run(
        [
            str(tracking_python),
            "-c",
            runner,
            "--motion-file",
            str(motion_file),
            "--motion-type",
            config.motion_type,
            "--policy-path",
            str(policy_path),
            "--steps",
            str(config.steps),
            "--output",
            str(output_path),
        ],
        cwd=str(tracking_root),
        check=True,
    )


def _run_sumo_native(config: G1WBCCompareTrackingBFMConfig, motion_file: Path, policy_path: Path) -> dict[str, np.ndarray]:
    motion = load_motion(motion_file, config.motion_type)  # type: ignore[arg-type]
    model = load_wbc_model()
    data = mujoco.MjData(model)
    data.qpos[:] = np.concatenate([motion.root_qpos(0), motion.joint_pos[0]])
    data.qvel[:] = motion_qvel_at_times(motion, np.asarray([0.0]))[0]
    data.time = 0.0
    mujoco.mj_forward(model, data)

    g1_extensions = require_g1_extensions()
    policy_state = np.zeros(g1_extensions.g1_wbc_policy_state_dim(), dtype=np.float32)

    action_traj = []
    held_ctrl_traj = []
    qpos_traj = []
    qvel_traj = []
    ctrl_traj = []
    time_traj = []
    for env_step in range(config.steps):
        command_time_s = (env_step + 1) * POLICY_DT
        for substep in range(POLICY_DECIMATION):
            control = motion_controls_at_times(motion, np.asarray([command_time_s]))[0]
            reference_qvel = motion_policy_qvel_at_times(motion, np.asarray([command_time_s]))[0]
            x0 = np.concatenate([data.qpos, data.qvel])
            policy_state = g1_extensions.sim_g1_wbc(
                model,
                data,
                x0,
                control,
                policy_state,
                str(policy_path),
                reference_qvel,
            )
            if substep == 0:
                state = np.asarray(policy_state, dtype=np.float32)
                action_traj.append(state[_STATE_LAST_ACTION_OFFSET : _STATE_LAST_ACTION_OFFSET + 29].astype(np.float64))
                held_ctrl_traj.append(state[_STATE_HELD_CTRL_OFFSET : _STATE_HELD_CTRL_OFFSET + 29].astype(np.float64))
        qpos_traj.append(np.asarray(data.qpos, dtype=np.float64).copy())
        qvel_traj.append(np.asarray(data.qvel, dtype=np.float64).copy())
        ctrl_traj.append(np.asarray(data.ctrl, dtype=np.float64).copy())
        time_traj.append(np.asarray(data.time, dtype=np.float64).copy())

    return {
        "initial_qpos": np.concatenate([motion.root_qpos(0), motion.joint_pos[0]]),
        "initial_qvel": motion_qvel_at_times(motion, np.asarray([0.0]))[0],
        "action_traj": np.asarray(action_traj),
        "held_ctrl_traj": np.asarray(held_ctrl_traj),
        "qpos_traj": np.asarray(qpos_traj),
        "qvel_traj": np.asarray(qvel_traj),
        "ctrl_traj": np.asarray(ctrl_traj),
        "time_traj": np.asarray(time_traj),
        "model_options": np.asarray(
            [
                model.opt.timestep,
                float(model.opt.integrator),
                float(model.opt.cone),
                model.opt.impratio,
                float(model.opt.iterations),
                float(model.opt.ls_iterations),
            ],
            dtype=np.float64,
        ),
        "policy_dt": np.asarray(POLICY_DT, dtype=np.float64),
    }


def _run_sumo_native_on_tracking_states(
    config: G1WBCCompareTrackingBFMConfig,
    motion_file: Path,
    policy_path: Path,
    tracking: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    motion = load_motion(motion_file, config.motion_type)  # type: ignore[arg-type]
    model = load_wbc_model()
    data = mujoco.MjData(model)
    g1_extensions = require_g1_extensions()
    policy_state = np.zeros(g1_extensions.g1_wbc_policy_state_dim(), dtype=np.float32)

    pre_qpos = np.asarray(tracking["pre_qpos_traj"], dtype=np.float64)
    pre_qvel = np.asarray(tracking["pre_qvel_traj"], dtype=np.float64)
    command_time_steps = np.asarray(tracking["command_time_steps"], dtype=np.int64)
    command_times = command_time_steps.astype(np.float64) * motion.dt

    obs_traj = []
    action_traj = []
    held_ctrl_traj = []
    policy_state_traj = []
    for step_idx in range(config.steps):
        x0 = np.concatenate([pre_qpos[step_idx], pre_qvel[step_idx]])
        control = motion_controls_at_times(motion, np.asarray([command_times[step_idx]]))[0]
        reference_qvel = motion_policy_qvel_at_times(motion, np.asarray([command_times[step_idx]]))[0]
        debug = g1_extensions.debug_g1_wbc_policy_step(
            model,
            data,
            x0,
            control,
            policy_state,
            str(policy_path),
            reference_qvel,
        )
        policy_state = np.asarray(debug["policy_state"], dtype=np.float32)
        obs_traj.append(np.asarray(debug["observation"], dtype=np.float32).copy())
        action_traj.append(np.asarray(debug["action"], dtype=np.float32).copy())
        held_ctrl_traj.append(np.asarray(debug["held_ctrl"], dtype=np.float32).copy())
        policy_state_traj.append(policy_state.copy())

    return {
        "obs_traj": np.asarray(obs_traj),
        "action_traj": np.asarray(action_traj),
        "held_ctrl_traj": np.asarray(held_ctrl_traj),
        "policy_state_traj": np.asarray(policy_state_traj),
    }


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) if a.size and b.size else float("nan")


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    delta = np.asarray(a) - np.asarray(b)
    return float(np.sqrt(np.mean(delta * delta))) if delta.size else float("nan")


def _first_exceed_step(a: np.ndarray, b: np.ndarray, atol: float) -> int | None:
    delta = np.max(np.abs(np.asarray(a) - np.asarray(b)), axis=tuple(range(1, np.asarray(a).ndim)))
    exceeded = np.nonzero(delta > atol)[0]
    return int(exceeded[0]) if exceeded.size else None


def _obs_segment_diffs(tracking_obs: np.ndarray, sumo_obs: np.ndarray) -> dict[str, float]:
    segments = {
        "command": (0, 58),
        "ref_limb_ee_pose_b": (58, 238),
        "motion_ref_ang_vel": (238, 241),
        "robot_limb_ee_pose_b": (241, 421),
        "projected_gravity": (421, 436),
        "base_ang_vel": (436, 451),
        "joint_pos": (451, 596),
        "joint_vel": (596, 741),
        "actions": (741, 886),
    }
    return {name: _max_abs(tracking_obs[:, start:end], sumo_obs[:, start:end]) for name, (start, end) in segments.items()}


def _compare_outputs(
    tracking: dict[str, np.ndarray],
    sumo: dict[str, np.ndarray],
    sumo_forced: dict[str, np.ndarray],
    config: G1WBCCompareTrackingBFMConfig,
) -> dict:
    forced_obs_linf = _max_abs(tracking["obs_traj"], sumo_forced["obs_traj"])
    forced_action_linf = _max_abs(tracking["action_traj"], sumo_forced["action_traj"])
    forced_ctrl_linf = _max_abs(tracking["ctrl_traj"], sumo_forced["held_ctrl_traj"])
    action_linf = _max_abs(tracking["action_traj"], sumo["action_traj"])
    ctrl_linf = _max_abs(tracking["ctrl_traj"], sumo["held_ctrl_traj"])
    first_action_linf = _max_abs(tracking["action_traj"][:1], sumo["action_traj"][:1])
    first_ctrl_linf = _max_abs(tracking["ctrl_traj"][:1], sumo["held_ctrl_traj"][:1])
    qpos_linf = _max_abs(tracking["qpos_traj"], sumo["qpos_traj"])
    qvel_linf = _max_abs(tracking["qvel_traj"], sumo["qvel_traj"])
    initial_state_passed = (
        _max_abs(tracking["initial_qpos"], sumo["initial_qpos"]) <= config.initial_state_atol
        and _max_abs(tracking["initial_qvel"], sumo["initial_qvel"]) <= config.initial_state_atol
        and _max_abs(tracking["model_options"], sumo["model_options"]) <= config.model_options_atol
    )
    direct_policy_passed = bool(
        initial_state_passed
        and forced_obs_linf <= config.obs_atol
        and forced_action_linf <= config.direct_action_atol
        and forced_ctrl_linf <= config.direct_ctrl_atol
    )
    closed_loop_passed = bool(
        action_linf <= config.action_atol
        and first_action_linf <= config.first_action_atol
        and ctrl_linf <= config.ctrl_atol
        and qpos_linf <= config.qpos_atol
        and qvel_linf <= config.qvel_atol
    )
    report = {
        "passed": bool(direct_policy_passed and (closed_loop_passed or not config.require_closed_loop)),
        "direct_policy_passed": direct_policy_passed,
        "closed_loop_passed": closed_loop_passed,
        "steps": int(config.steps),
        "motion_file": str(Path(config.motion_file).expanduser().resolve()),
        "policy": str(config.policy),
        "thresholds": {
            "obs_atol": config.obs_atol,
            "direct_action_atol": config.direct_action_atol,
            "direct_ctrl_atol": config.direct_ctrl_atol,
            "initial_state_atol": config.initial_state_atol,
            "model_options_atol": config.model_options_atol,
            "action_atol": config.action_atol,
            "first_action_atol": config.first_action_atol,
            "ctrl_atol": config.ctrl_atol,
            "qpos_atol": config.qpos_atol,
            "qvel_atol": config.qvel_atol,
            "require_closed_loop": config.require_closed_loop,
        },
        "diff": {
            "initial_qpos_linf": _max_abs(tracking["initial_qpos"], sumo["initial_qpos"]),
            "initial_qvel_linf": _max_abs(tracking["initial_qvel"], sumo["initial_qvel"]),
            "forced_obs_linf": forced_obs_linf,
            "forced_obs_rmse": _rmse(tracking["obs_traj"], sumo_forced["obs_traj"]),
            "forced_action_linf": forced_action_linf,
            "forced_action_rmse": _rmse(tracking["action_traj"], sumo_forced["action_traj"]),
            "forced_ctrl_linf": forced_ctrl_linf,
            "forced_ctrl_rmse": _rmse(tracking["ctrl_traj"], sumo_forced["held_ctrl_traj"]),
            "action_linf": action_linf,
            "first_action_linf": first_action_linf,
            "ctrl_linf": ctrl_linf,
            "first_ctrl_linf": first_ctrl_linf,
            "qpos_linf": qpos_linf,
            "qpos_rmse": _rmse(tracking["qpos_traj"], sumo["qpos_traj"]),
            "qvel_linf": qvel_linf,
            "qvel_rmse": _rmse(tracking["qvel_traj"], sumo["qvel_traj"]),
            "time_linf": _max_abs(tracking["time_traj"], sumo["time_traj"]),
            "model_options_linf": _max_abs(tracking["model_options"], sumo["model_options"]),
        },
        "first_exceed_step": {
            "forced_obs": _first_exceed_step(tracking["obs_traj"], sumo_forced["obs_traj"], config.obs_atol),
            "forced_action": _first_exceed_step(
                tracking["action_traj"], sumo_forced["action_traj"], config.direct_action_atol
            ),
            "forced_ctrl": _first_exceed_step(tracking["ctrl_traj"], sumo_forced["held_ctrl_traj"], config.direct_ctrl_atol),
            "closed_loop_action": _first_exceed_step(tracking["action_traj"], sumo["action_traj"], config.action_atol),
            "closed_loop_qpos": _first_exceed_step(tracking["qpos_traj"], sumo["qpos_traj"], config.qpos_atol),
            "closed_loop_qvel": _first_exceed_step(tracking["qvel_traj"], sumo["qvel_traj"], config.qvel_atol),
        },
        "forced_obs_segment_linf": _obs_segment_diffs(tracking["obs_traj"], sumo_forced["obs_traj"]),
    }
    return report


if __name__ == "__main__":
    main()
