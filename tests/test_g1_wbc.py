from __future__ import annotations

import numpy as np
from judo.controller.controller import ControllerConfig
from judo.optimizers import get_registered_optimizers
from judo.tasks import get_registered_tasks

import sumo.controller  # noqa: F401
import sumo.tasks  # noqa: F401
from sumo.controller.g1_wbc_controller import G1WBCController, G1WBCControlSpline
from sumo.run_mpc.g1_wbc_eval import G1WBCEvalConfig, apply_mpc_compute_budget, episode_length_for_motion
from sumo.tasks.g1_wbc import G1WBCEE, G1WBCJoint
from sumo.utils.g1_wbc.constants import (
    ACTION_DIM,
    DEFAULT_MOTION_FILE,
    ISAACLAB_BODY_NAMES,
    ISAACLAB_JOINT_NAMES,
    ISAACLAB_TO_MUJOCO_BODY_REINDEX,
    ISAACLAB_TO_MUJOCO_JOINT_REINDEX,
    MUJOCO_BODY_NAMES,
    MUJOCO_JOINT_NAMES,
    OBS_DIM,
    TASK_CONTROL_DIM,
)
from sumo.utils.g1_wbc.math import normalize_quat, slerp_wxyz
from sumo.utils.g1_wbc.motion import load_motion
from sumo.utils.g1_wbc.policy import TrackingPolicyRuntime, action_scale_for_joints
from sumo.utils.g1_wbc.reference import build_reference_frames, interpolate_controls
from sumo.utils.g1_wbc.rollout import G1WBCRolloutBackend


def test_g1_wbc_tasks_registered() -> None:
    tasks = get_registered_tasks()
    assert tasks["g1_wbc_ee"].rollout_backend == "mujoco_g1_wbc"
    assert tasks["g1_wbc_joint"].simulation_backend == "mujoco_g1_wbc"


def test_g1_wbc_motion_loader_smoke() -> None:
    motion = load_motion(DEFAULT_MOTION_FILE, "mujoco")
    assert motion.joint_pos.shape[1] == ACTION_DIM
    assert motion.body_pos_w.shape[1] == len(MUJOCO_BODY_NAMES)
    assert motion.contact_mask.shape == (motion.num_frames, 2)
    assert np.isfinite(motion.trajectory_controls()).all()


def test_g1_wbc_reindex_tables_match_names() -> None:
    assert len(ISAACLAB_TO_MUJOCO_JOINT_REINDEX) == len(MUJOCO_JOINT_NAMES)
    assert len(ISAACLAB_TO_MUJOCO_BODY_REINDEX) == len(MUJOCO_BODY_NAMES)
    for mujoco_idx, isaac_idx in enumerate(ISAACLAB_TO_MUJOCO_JOINT_REINDEX):
        assert ISAACLAB_JOINT_NAMES[isaac_idx] == MUJOCO_JOINT_NAMES[mujoco_idx]
    for mujoco_idx, isaac_idx in enumerate(ISAACLAB_TO_MUJOCO_BODY_REINDEX):
        assert ISAACLAB_BODY_NAMES[isaac_idx] == MUJOCO_BODY_NAMES[mujoco_idx]


def test_g1_wbc_task_contracts() -> None:
    for task_cls in (G1WBCEE, G1WBCJoint):
        task = task_cls()
        assert task.nu == TASK_CONTROL_DIM
        assert task.actuator_ctrlrange.shape == (TASK_CONTROL_DIM, 2)
        assert task.task_to_sim_ctrl(task.optimizer_warm_start()).shape == (TASK_CONTROL_DIM,)


def test_g1_wbc_policy_observation_shape() -> None:
    task = G1WBCEE()
    task.reset()
    runtime = TrackingPolicyRuntime(task.model, "bcrl")
    controls = task.motion.trajectory_controls()[:1]
    ref = build_reference_frames(task.model, controls, task.model.opt.timestep)[0]
    obs = runtime.build_observation(task.data, ref)
    assert obs.shape == (OBS_DIM,)
    ctrl = runtime.step(task.data, ref)
    assert ctrl.shape == (ACTION_DIM,)
    assert np.isfinite(ctrl).all()


def test_g1_wbc_rollout_backend_short_smoke() -> None:
    task = G1WBCEE()
    task.reset()
    backend = G1WBCRolloutBackend(task.model, num_threads=1, policy="bcrl")
    x0 = np.concatenate([task.data.qpos, task.data.qvel])
    controls = task.motion.trajectory_controls()[:3][None, :, :]
    states, sensors, policy_out = backend.rollout(x0, controls)
    assert policy_out is None
    assert states.shape == (1, 4, task.model.nq + task.model.nv)
    assert sensors.shape == (1, 3, 4)
    assert np.isfinite(states).all()
    assert np.isfinite(sensors).all()


def test_g1_wbc_action_scale_contains_joint_specific_values() -> None:
    scales = action_scale_for_joints(MUJOCO_JOINT_NAMES)
    assert scales.shape == (ACTION_DIM,)
    left_hip = MUJOCO_JOINT_NAMES.index("left_hip_pitch_joint")
    left_wrist_yaw = MUJOCO_JOINT_NAMES.index("left_wrist_yaw_joint")
    assert scales[left_hip] > scales[left_wrist_yaw]


def test_slerp_wxyz_normalizes_and_interpolates() -> None:
    quats = normalize_quat(np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]))
    out = slerp_wxyz(np.array([0.0, 1.0]), quats, np.array([0.5]))
    assert out.shape == (1, 4)
    np.testing.assert_allclose(np.linalg.norm(out, axis=-1), 1.0)


def test_g1_wbc_control_interpolation_uses_root_quat_slerp() -> None:
    controls = np.zeros((2, TASK_CONTROL_DIM), dtype=np.float64)
    controls[:, 3] = 1.0
    controls[1, 3:7] = normalize_quat(np.array([0.0, 0.0, 0.0, 1.0]))
    out = interpolate_controls(np.array([0.0, 1.0]), controls, np.array([0.5]))
    np.testing.assert_allclose(out[:, 3:7], slerp_wxyz(np.array([0.0, 1.0]), controls[:, 3:7], np.array([0.5])))
    np.testing.assert_allclose(np.linalg.norm(out[:, 3:7], axis=-1), 1.0)


def test_g1_wbc_controller_initializes_knots_from_reference_horizon() -> None:
    task = G1WBCEE()
    optimizer_cls, optimizer_config_cls = get_registered_optimizers()["cem"]
    optimizer_config = optimizer_config_cls()
    optimizer_config.set_override("g1_wbc_ee")
    optimizer = optimizer_cls(optimizer_config, task.nu)
    controller_config = ControllerConfig()
    controller_config.set_override("g1_wbc_ee")
    controller = G1WBCController(
        controller_config,
        task,
        optimizer,
        rollout_backend="mujoco_g1_wbc",
        rollout_backend_registry={"mujoco_g1_wbc": G1WBCRolloutBackend},
    )

    task.data.time = 0.05
    controller.time = 0.05
    controller._set_reference_nominal_knots()
    expected = task.reference_controls_for_times(controller.times)
    np.testing.assert_allclose(controller.nominal_knots, expected)
    assert isinstance(controller.spline, G1WBCControlSpline)


def test_g1_wbc_compute_budget_overrides_do_not_change_problem_definition() -> None:
    _, optimizer_config_cls = get_registered_optimizers()["cem"]
    optimizer_config = optimizer_config_cls()
    optimizer_config.set_override("g1_wbc_ee")
    controller_config = ControllerConfig()
    controller_config.set_override("g1_wbc_ee")

    original_num_nodes = optimizer_config.num_nodes
    original_horizon = controller_config.horizon
    config = G1WBCEvalConfig(
        mpc_num_rollouts=optimizer_config.num_rollouts + 3,
        mpc_max_opt_iters=controller_config.max_opt_iters + 2,
    )
    apply_mpc_compute_budget(config, optimizer_config, controller_config)

    assert optimizer_config.num_rollouts == config.mpc_num_rollouts
    assert controller_config.max_opt_iters == config.mpc_max_opt_iters
    assert optimizer_config.num_nodes == original_num_nodes
    assert controller_config.horizon == original_horizon


def test_g1_wbc_eval_default_episode_length_is_full_motion() -> None:
    motion = load_motion(DEFAULT_MOTION_FILE, "mujoco")
    config = G1WBCEvalConfig(motion_file=str(DEFAULT_MOTION_FILE), motion_type="mujoco", episode_length_s=0.0)
    assert episode_length_for_motion(config, DEFAULT_MOTION_FILE) == motion.duration
