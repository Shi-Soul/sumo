from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import mujoco
from judo.controller.controller import ControllerConfig
from judo.optimizers import get_registered_optimizers
from judo.tasks import get_registered_tasks

import sumo.controller  # noqa: F401
import sumo.tasks  # noqa: F401
from sumo.controller.g1_wbc_controller import G1WBCController, G1WBCControlSpline, G1WBCReferenceResidualSpline
from sumo.run_mpc.g1_wbc_eval import (
    G1WBCEvalConfig,
    annotate_priority_comparisons,
    apply_mpc_compute_budget,
    compute_metrics,
    episode_length_for_motion,
    run_no_mpc_episode,
)
from sumo.tasks.g1_wbc import G1WBCEE, G1WBCJoint
from sumo.utils.g1_wbc.constants import (
    ACTION_DIM,
    DEFAULT_MOTION_FILE,
    ISAACLAB_BODY_NAMES,
    ISAACLAB_JOINT_NAMES,
    ISAACLAB_TO_MUJOCO_BODY_REINDEX,
    ISAACLAB_TO_MUJOCO_JOINT_REINDEX,
    JOINT_POS_SLICE,
    MUJOCO_BODY_NAMES,
    MUJOCO_JOINT_NAMES,
    OBS_DIM,
    POLICY_DT,
    TASK_CONTROL_DIM,
    WBC_ROLLOUT_SENSOR_DIM,
    resolve_policy_path,
)
from sumo.tasks.g1.g1_base import XML_PATH as G1_BASE_XML_PATH
from sumo.utils.g1_wbc.math import normalize_quat, slerp_wxyz
from sumo.utils.g1_wbc.model import load_wbc_model
from sumo.utils.g1_wbc.motion import load_motion
from sumo.utils.g1_wbc.reference import (
    controls_to_policy_qvel,
    interpolate_controls,
    motion_policy_qvel_at_times,
    motion_qvel_at_times,
)
from sumo.utils.g1_wbc.rollout import G1WBCRolloutBackend
from sumo.utils.extensions import require_g1_extensions


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


def test_g1_wbc_model_uses_tracking_bfm_simulator_semantics() -> None:
    wbc_model = load_wbc_model()
    base_model = mujoco.MjModel.from_xml_path(G1_BASE_XML_PATH)
    assert np.all(wbc_model.actuator_ctrllimited == 0)
    assert np.all(base_model.actuator_ctrllimited == 1)
    assert wbc_model.opt.timestep == 0.005
    assert wbc_model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert wbc_model.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL
    assert wbc_model.opt.impratio == 1.0
    assert wbc_model.opt.iterations == 10
    assert wbc_model.opt.ls_iterations == 20


def test_g1_wbc_native_policy_single_step() -> None:
    task = G1WBCEE()
    task.reset()
    g1_extensions = require_g1_extensions()
    state = np.zeros(g1_extensions.g1_wbc_policy_state_dim(), dtype=np.float32)
    x0 = np.concatenate([task.data.qpos, task.data.qvel])
    command = task.reference_controls_for_times(np.asarray([POLICY_DT]))[0]
    next_state = g1_extensions.sim_g1_wbc(
        task.model,
        task.data,
        x0,
        command,
        state,
        str(resolve_policy_path("bcrl")),
        motion_policy_qvel_at_times(task.motion, np.asarray([POLICY_DT]))[0],
    )
    assert state.shape == next_state.shape
    assert next_state.shape[0] > OBS_DIM
    assert task.data.ctrl.shape == (ACTION_DIM,)
    assert np.isfinite(task.data.ctrl).all()


def test_g1_wbc_rollout_backend_short_smoke() -> None:
    task = G1WBCEE()
    task.reset()
    backend = G1WBCRolloutBackend(task.model, num_threads=1, policy="bcrl")
    x0 = np.concatenate([task.data.qpos, task.data.qvel])
    controls = task.motion.trajectory_controls()[:3][None, :, :]
    states, sensors, policy_out = backend.rollout(x0, controls)
    assert policy_out is None
    assert states.shape == (1, 4, task.model.nq + task.model.nv)
    assert sensors.shape == (1, 3, WBC_ROLLOUT_SENSOR_DIM)
    assert np.isfinite(states).all()
    assert np.isfinite(sensors).all()


def test_g1_wbc_native_policy_state_dim_is_opaque_history() -> None:
    g1_extensions = require_g1_extensions()
    assert g1_extensions.g1_wbc_policy_state_dim() > OBS_DIM


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


def test_g1_wbc_motion_qvel_converts_root_ang_vel_to_body_frame() -> None:
    controls = np.zeros((1, TASK_CONTROL_DIM), dtype=np.float64)
    controls[0, 3:7] = normalize_quat(np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]))
    body_lin_vel = np.zeros((1, len(MUJOCO_BODY_NAMES), 3), dtype=np.float64)
    body_ang_vel = np.zeros_like(body_lin_vel)
    body_ang_vel[0, MUJOCO_BODY_NAMES.index("pelvis")] = np.array([1.0, 0.0, 0.0])
    motion = SimpleNamespace(
        num_frames=1,
        dt=0.02,
        body_lin_vel_w=body_lin_vel,
        body_ang_vel_w=body_ang_vel,
        joint_vel=np.zeros((1, ACTION_DIM), dtype=np.float64),
        trajectory_controls=lambda: controls,
    )

    qvel = motion_qvel_at_times(motion, np.asarray([0.0]))[0]

    np.testing.assert_allclose(qvel[3:6], np.array([0.0, -1.0, 0.0]), atol=1e-8)


def test_g1_wbc_refined_policy_qvel_uses_torso_angular_velocity() -> None:
    task = G1WBCEE()
    controls = np.tile(task.motion.trajectory_controls()[0], (3, 1))
    waist_yaw_idx = MUJOCO_JOINT_NAMES.index("waist_yaw_joint")
    controls[:, 7 + waist_yaw_idx] += np.array([0.0, 0.05, 0.10])

    qvel = controls_to_policy_qvel(task.model, controls, task.model.opt.timestep)

    np.testing.assert_allclose(qvel[:, :3], 0.0)
    assert np.max(np.abs(qvel[:, 3:6])) > 1.0
    np.testing.assert_allclose(qvel[:, 6 + waist_yaw_idx], 10.0)


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
    np.testing.assert_allclose(controller.nominal_residual_knots, 0.0)
    np.testing.assert_allclose(controller.spline(controller.times), expected)
    assert isinstance(controller.spline, G1WBCReferenceResidualSpline)


def test_g1_wbc_controller_action_reference_qvel_uses_refined_spline() -> None:
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

    dt = task.model.opt.timestep
    times = np.asarray([0.0, dt, 2.0 * dt])
    controls = np.tile(task.motion.trajectory_controls()[0], (3, 1))
    waist_yaw_idx = MUJOCO_JOINT_NAMES.index("waist_yaw_joint")
    controls[:, 7 + waist_yaw_idx] += np.array([0.0, 0.05, 0.10])
    reference_controls = controls.copy()
    reference_controls[:, 7 + waist_yaw_idx] -= 0.05
    reference_motion_qvel = np.zeros((1, task.model.nv), dtype=np.float64)
    reference_motion_qvel[0, 6 + waist_yaw_idx] = 3.0

    def fake_reference_controls_for_times(query_times):
        query_times = np.asarray(query_times, dtype=np.float64)
        return interpolate_controls(times, reference_controls, query_times)

    def fake_reference_policy_qvel_for_times(query_times):
        return np.repeat(reference_motion_qvel, len(np.atleast_1d(query_times)), axis=0)

    task.reference_controls_for_times = fake_reference_controls_for_times
    task.reference_policy_qvel_for_times = fake_reference_policy_qvel_for_times
    controller.times = times
    controller.update_spline(times, controls)

    expected = (
        reference_motion_qvel[0]
        + controls_to_policy_qvel(task.model, controls, dt)[1]
        - controls_to_policy_qvel(task.model, reference_controls, dt)[1]
    )
    np.testing.assert_allclose(controller.action_reference_qvel(float(dt)), expected)

    shifted_times = times + 0.05
    shifted_reference = reference_controls.copy()

    def fake_shifted_reference_controls_for_times(query_times):
        query_times = np.asarray(query_times, dtype=np.float64)
        return interpolate_controls(shifted_times, shifted_reference, query_times)

    task.reference_controls_for_times = fake_shifted_reference_controls_for_times
    controller.times = shifted_times
    controller.update_spline(shifted_times, controls)
    expected_start = (
        reference_motion_qvel[0]
        + controls_to_policy_qvel(task.model, controls[:2], dt)[0]
        - controls_to_policy_qvel(task.model, shifted_reference[:2], dt)[0]
    )
    np.testing.assert_allclose(controller.action_reference_qvel(float(shifted_times[0])), expected_start)


def test_g1_wbc_controller_keeps_reference_candidate_on_spline_path() -> None:
    task = G1WBCEE()
    original_reference_controls_for_times = task.reference_controls_for_times

    def fake_reference_controls_for_times(times):
        times = np.asarray(times, dtype=np.float64)
        controls = np.zeros((len(times), TASK_CONTROL_DIM), dtype=np.float64)
        controls[:, 0] = times**2
        controls[:, 3] = 1.0
        controls[:, JOINT_POS_SLICE] = 0.01 * times[:, None]
        return controls

    task.reference_controls_for_times = fake_reference_controls_for_times
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

    task.data.time = 0.1
    controller.time = 0.1
    controller._pre_optimization()
    rollout_controls = controller._sample_controls()

    np.testing.assert_allclose(controller.candidate_residual_knots[0], 0.0)
    np.testing.assert_allclose(controller.candidate_knots[0], controller._reference_knots)
    rollout_times = controller.policy_command_time(controller.time + controller.rollout_times)
    expected_rollout = task.reference_controls_for_times(rollout_times)
    np.testing.assert_allclose(rollout_controls[0], expected_rollout)
    reference_qvels = controller.rollout_backend.reference_qvels
    assert reference_qvels.shape == (controller.optimizer_cfg.num_rollouts, len(rollout_times), task.model.nv)
    np.testing.assert_allclose(reference_qvels[0], task.reference_policy_qvel_for_times(rollout_times), atol=1e-10)
    assert np.isfinite(reference_qvels).all()

    task.reference_controls_for_times = original_reference_controls_for_times


def test_g1_wbc_controller_does_not_warm_start_residual_by_default() -> None:
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

    controller.time = 0.1
    controller._pre_optimization()
    old_times = controller._new_times.copy()
    residuals = np.zeros_like(controller._reference_knots)
    residuals[:, 0] = np.linspace(0.002, 0.01, residuals.shape[0])
    controller.times = old_times
    controller._update_residual_spline(old_times, residuals)

    controller.time = 0.12
    controller._pre_optimization()
    np.testing.assert_allclose(controller.nominal_residual_knots, 0.0)

    controller._sample_controls()
    np.testing.assert_allclose(controller.candidate_residual_knots[0], 0.0)
    np.testing.assert_allclose(controller.candidate_knots[0], controller._reference_knots)


def test_g1_wbc_controller_action_uses_selected_spline_not_reference_bypass() -> None:
    task = G1WBCEE()

    def fake_reference_controls_for_times(times):
        times = np.asarray(times, dtype=np.float64)
        controls = np.zeros((len(times), TASK_CONTROL_DIM), dtype=np.float64)
        controls[:, 0] = times**2
        controls[:, 3] = 1.0
        return controls

    task.reference_controls_for_times = fake_reference_controls_for_times
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

    times = np.asarray([0.0, 1.0])
    controller.times = times
    controller.update_spline(times, fake_reference_controls_for_times(times))

    action = controller.action(0.5)
    np.testing.assert_allclose(action[0], 0.5)

    qvel = controller.action_reference_qvel(0.5)
    assert qvel.shape == (task.model.nv,)
    assert np.isfinite(qvel).all()


def test_g1_wbc_controller_uses_best_sampled_candidate_after_optimization() -> None:
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

    controller._pre_optimization()
    residuals = np.zeros((controller.optimizer_cfg.num_rollouts, controller.optimizer_cfg.num_nodes, task.nu))
    residuals[1, :, 0] = 0.01
    controller.candidate_residual_knots = residuals
    controller.candidate_knots = controller._compose_reference_residual(controller._reference_knots, residuals)
    controller.rewards = np.zeros(controller.optimizer_cfg.num_rollouts)
    controller.rewards[1] = 1.0
    controller._post_optimization()

    np.testing.assert_allclose(controller.nominal_residual_knots, residuals[1])
    np.testing.assert_allclose(controller.nominal_knots, controller.candidate_knots[1])


def test_g1_wbc_candidate_gate_rejects_pose_worse_high_reward() -> None:
    task = G1WBCJoint()
    task.reset()
    horizon = 4
    target = task._target_controls_for_horizon(horizon)
    controls = np.tile(target, (2, 1, 1))
    states = np.zeros((2, horizon + 1, task.model.nq + task.model.nv), dtype=np.float64)
    states[:, 0, : task.model.nq] = target[0]
    states[0, 1:, : task.model.nq] = target
    worse = target.copy()
    worse[:, 0] += 0.05
    states[1, 1:, : task.model.nq] = worse
    sensors = np.zeros((2, horizon, 4), dtype=np.float64)
    rewards = np.asarray([0.0, 100.0], dtype=np.float64)

    assert task.select_mpc_candidate(states, sensors, controls, rewards) == 0


def test_g1_wbc_candidate_gate_accepts_upper_improvement_without_contact_or_smooth_regression() -> None:
    task = G1WBCJoint()
    task.reset()
    states = np.zeros((3, 2, task.model.nq + task.model.nv), dtype=np.float64)
    sensors = np.zeros((3, 1, 4), dtype=np.float64)
    controls = np.zeros((3, 1, TASK_CONTROL_DIM), dtype=np.float64)
    rewards = np.asarray([0.0, -1.0, 10.0], dtype=np.float64)
    metrics = {
        "fallen": np.asarray([False, False, False]),
        "contact": np.asarray([0.0, 0.0, 0.01]),
        "contact_mismatch": np.asarray([0.0, 0.0, 0.01]),
        "contact_no_ref": np.asarray([0.0, 0.0, 0.0]),
        "contact_switch": np.asarray([0.0, 0.0, 0.0]),
        "contact_force": np.asarray([0.0, 0.0, 0.0]),
        "smooth": np.asarray([0.001, 0.001, 0.001]),
        "action_smooth": np.asarray([0.01, 0.012, 0.01]),
        "upper_ee": np.asarray([0.03, 0.03 - 2.0 * task.config.accept_upper_ee_improvement, 0.0]),
        "root": np.asarray([0.02, 0.02, 0.0]),
        "root_ori": np.asarray([0.01, 0.01, 0.0]),
        "joint": np.asarray([0.05, 0.05, 0.0]),
    }
    task._candidate_tracking_metrics = lambda *_args: metrics  # type: ignore[method-assign]

    assert task.select_mpc_candidate(states, sensors, controls, rewards) == 1


def test_g1_wbc_candidate_gate_rejects_smooth_regression_for_upper_improvement() -> None:
    task = G1WBCJoint()
    task.reset()
    states = np.zeros((2, 2, task.model.nq + task.model.nv), dtype=np.float64)
    sensors = np.zeros((2, 1, 4), dtype=np.float64)
    controls = np.zeros((2, 1, TASK_CONTROL_DIM), dtype=np.float64)
    rewards = np.asarray([0.0, 10.0], dtype=np.float64)
    metrics = {
        "fallen": np.asarray([False, False]),
        "contact": np.asarray([0.0, 0.0]),
        "contact_mismatch": np.asarray([0.0, 0.0]),
        "contact_no_ref": np.asarray([0.0, 0.0]),
        "contact_switch": np.asarray([0.0, 0.0]),
        "contact_force": np.asarray([0.0, 0.0]),
        "smooth": np.asarray([0.001, 0.001 + 10.0 * task.config.accept_smooth_tolerance]),
        "action_smooth": np.asarray([0.01, 0.01]),
        "upper_ee": np.asarray([0.03, 0.03 - 2.0 * task.config.accept_upper_ee_improvement]),
        "root": np.asarray([0.02, 0.02]),
        "root_ori": np.asarray([0.01, 0.01]),
        "joint": np.asarray([0.05, 0.05]),
    }
    task._candidate_tracking_metrics = lambda *_args: metrics  # type: ignore[method-assign]

    assert task.select_mpc_candidate(states, sensors, controls, rewards) == 0


def test_g1_wbc_candidate_gate_rejects_contact_component_regression() -> None:
    task = G1WBCJoint()
    task.reset()
    states = np.zeros((2, 2, task.model.nq + task.model.nv), dtype=np.float64)
    sensors = np.zeros((2, 1, 4), dtype=np.float64)
    controls = np.zeros((2, 1, TASK_CONTROL_DIM), dtype=np.float64)
    rewards = np.asarray([0.0, 10.0], dtype=np.float64)
    metrics = {
        "fallen": np.asarray([False, False]),
        "contact": np.asarray([0.01, 0.0]),
        "contact_mismatch": np.asarray([0.0, 0.0]),
        "contact_no_ref": np.asarray([0.0, 0.01]),
        "contact_switch": np.asarray([0.0, 0.0]),
        "contact_force": np.asarray([0.0, 0.0]),
        "smooth": np.asarray([0.001, 0.001]),
        "action_smooth": np.asarray([0.01, 0.01]),
        "upper_ee": np.asarray([0.03, 0.0]),
        "root": np.asarray([0.02, 0.0]),
        "root_ori": np.asarray([0.01, 0.0]),
        "joint": np.asarray([0.05, 0.0]),
    }
    task._candidate_tracking_metrics = lambda *_args: metrics  # type: ignore[method-assign]

    assert task.select_mpc_candidate(states, sensors, controls, rewards) == 0


def test_g1_wbc_candidate_gate_does_not_hard_reject_action_smooth_increase() -> None:
    task = G1WBCJoint()
    task.reset()
    states = np.zeros((2, 2, task.model.nq + task.model.nv), dtype=np.float64)
    sensors = np.zeros((2, 1, 4), dtype=np.float64)
    controls = np.zeros((2, 1, TASK_CONTROL_DIM), dtype=np.float64)
    rewards = np.asarray([0.0, 10.0], dtype=np.float64)
    metrics = {
        "fallen": np.asarray([False, False]),
        "contact": np.asarray([0.0, 0.0]),
        "contact_mismatch": np.asarray([0.0, 0.0]),
        "contact_no_ref": np.asarray([0.0, 0.0]),
        "contact_switch": np.asarray([0.0, 0.0]),
        "contact_force": np.asarray([0.0, 0.0]),
        "contact_force_max": np.asarray([0.0, 0.0]),
        "contact_force_mean": np.asarray([0.0, 0.0]),
        "smooth": np.asarray([0.001, 0.001]),
        "action_smooth": np.asarray([0.01, 0.01 + 0.5 * task.config.accept_smooth_tolerance]),
        "upper_ee": np.asarray([0.03, 0.03 - 2.0 * task.config.accept_upper_ee_improvement]),
        "root": np.asarray([0.02, 0.02]),
        "root_ori": np.asarray([0.01, 0.01]),
        "joint": np.asarray([0.05, 0.05]),
    }
    task._candidate_tracking_metrics = lambda *_args: metrics  # type: ignore[method-assign]

    assert task.select_mpc_candidate(states, sensors, controls, rewards) == 1


def test_g1_wbc_reward_regularizes_at_policy_command_times() -> None:
    task = G1WBCEE()
    task.reset()
    task.data.time = 0.12
    horizon = 4
    reference_calls = []

    def fake_reference_controls_for_times(times):
        times = np.asarray(times, dtype=np.float64)
        reference_calls.append(times.copy())
        return np.zeros((len(times), TASK_CONTROL_DIM), dtype=np.float64)

    task.reference_controls_for_times = fake_reference_controls_for_times

    task._command_reference_for_horizon(horizon)
    expected_command_times = task.data.time + POLICY_DT + task.model.opt.timestep * np.arange(horizon)
    np.testing.assert_allclose(reference_calls[-1], expected_command_times)

    task._target_controls_for_horizon(horizon)
    expected_target_times = task.data.time + task.model.opt.timestep * (np.arange(horizon) + 1)
    np.testing.assert_allclose(reference_calls[-1], expected_target_times)
    assert not np.allclose(expected_command_times, expected_target_times)


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


def test_g1_wbc_default_cem_budgets_match_tuned_eval_defaults() -> None:
    _, optimizer_config_cls = get_registered_optimizers()["cem"]
    ee_config = optimizer_config_cls()
    ee_config.set_override("g1_wbc_ee")
    joint_config = optimizer_config_cls()
    joint_config.set_override("g1_wbc_joint")

    assert ee_config.num_rollouts == 8
    assert ee_config.num_nodes == 9
    assert ee_config.num_elites == 3
    np.testing.assert_allclose(ee_config.sigma_min, 0.0006)
    np.testing.assert_allclose(ee_config.sigma_max, 0.006)
    assert joint_config.num_rollouts == 14
    assert joint_config.num_nodes == 9
    assert joint_config.num_elites == 3
    np.testing.assert_allclose(joint_config.sigma_min, 0.0002)
    np.testing.assert_allclose(joint_config.sigma_max, 0.002)


def test_g1_wbc_eval_default_episode_length_is_full_motion() -> None:
    motion = load_motion(DEFAULT_MOTION_FILE, "mujoco")
    config = G1WBCEvalConfig(motion_file=str(DEFAULT_MOTION_FILE), motion_type="mujoco", episode_length_s=0.0)
    assert episode_length_for_motion(config, DEFAULT_MOTION_FILE) == motion.duration


def test_g1_wbc_no_mpc_respects_explicit_episode_length() -> None:
    config = G1WBCEvalConfig(
        motion_file=str(DEFAULT_MOTION_FILE),
        motion_type="mujoco",
        methods=("no_mpc",),
        episode_length_s=0.03,
    )
    episode = run_no_mpc_episode(config, DEFAULT_MOTION_FILE)
    assert episode["time_traj"].shape[0] == int(np.ceil(config.episode_length_s / 0.005))


def test_g1_wbc_metrics_use_refined_reference_controls_for_delta() -> None:
    task = G1WBCEE()
    task.reset()
    qpos = np.tile(np.asarray(task.data.qpos), (2, 1))
    refs = qpos.copy()
    refined_refs = refs.copy()
    refined = refs.copy()
    refined_refs[:, 0] += 0.1
    refined[:, 0] += 0.3
    refined_refs[:, 7:36] += 0.02
    refined[:, 7:36] += 0.12

    metrics = compute_metrics(
        {
            "qpos_traj": qpos,
            "reference_controls": refs,
            "refined_controls": refined,
            "refined_reference_controls": refined_refs,
        },
        success_threshold=0.12,
    )

    np.testing.assert_allclose(metrics["refined_root_delta_rmse"], 0.2)
    np.testing.assert_allclose(metrics["refined_joint_delta_rmse"], 0.1)


def test_g1_wbc_metrics_record_effective_mpc_budget() -> None:
    task = G1WBCEE()
    task.reset()
    qpos = np.tile(np.asarray(task.data.qpos), (2, 1))

    metrics = compute_metrics(
        {
            "qpos_traj": qpos,
            "reference_controls": qpos.copy(),
            "mpc_num_rollouts": 16,
            "mpc_max_opt_iters": 1,
        },
        success_threshold=0.12,
    )

    assert metrics["mpc_num_rollouts"] == 16
    assert metrics["mpc_max_opt_iters"] == 1


def test_g1_wbc_priority_comparison_rejects_contact_regression() -> None:
    rows = [
        {
            "method": "no_mpc",
            "motion_file": "motion.npz",
            "policy": "bcrl",
            "contact_no_ref_violation_rate": 0.0,
            "contact_mismatch_rate": 0.01,
            "contact_switch_rate": 0.02,
            "contact_force_max": 100.0,
            "contact_force_mean": 50.0,
            "motion_smoothness": 0.1,
            "action_smoothness": 0.2,
            "upper_global_rmse": 0.03,
            "root_pos_rmse": 0.02,
            "root_ori_mean": 0.01,
            "joint_rmse": 0.04,
            "local_ee_rmse": 0.05,
        },
        {
            "method": "ee_mpc",
            "motion_file": "motion.npz",
            "policy": "bcrl",
            "contact_no_ref_violation_rate": 0.001,
            "contact_mismatch_rate": 0.01,
            "contact_switch_rate": 0.02,
            "contact_force_max": 100.0,
            "contact_force_mean": 50.0,
            "motion_smoothness": 0.1,
            "action_smoothness": 0.2,
            "upper_global_rmse": 0.02,
            "root_pos_rmse": 0.02,
            "root_ori_mean": 0.01,
            "joint_rmse": 0.04,
            "local_ee_rmse": 0.05,
        },
    ]

    annotate_priority_comparisons(rows)

    assert rows[0]["priority_compare_status"] == "baseline"
    assert rows[1]["priority_compare_status"] == "regressed_contact"
    assert rows[1]["priority_better_than_no_mpc"] is False
    assert rows[1]["priority_first_improvement_group"] == "upper_global"


def test_g1_wbc_priority_comparison_accepts_first_clean_improvement() -> None:
    rows = [
        {
            "method": "no_mpc",
            "motion_file": "motion.npz",
            "policy": "bc",
            "contact_no_ref_violation_rate": 0.0,
            "contact_mismatch_rate": 0.01,
            "contact_switch_rate": 0.02,
            "contact_force_max": 100.0,
            "contact_force_mean": 50.0,
            "motion_smoothness": 0.1,
            "action_smoothness": 0.2,
            "upper_global_rmse": 0.03,
            "root_pos_rmse": 0.02,
            "root_ori_mean": 0.01,
            "joint_rmse": 0.04,
            "local_ee_rmse": 0.05,
        },
        {
            "method": "joint_mpc",
            "motion_file": "motion.npz",
            "policy": "bc",
            "contact_no_ref_violation_rate": 0.0,
            "contact_mismatch_rate": 0.01,
            "contact_switch_rate": 0.02,
            "contact_force_max": 100.0,
            "contact_force_mean": 50.0,
            "motion_smoothness": 0.1,
            "action_smoothness": 0.2,
            "upper_global_rmse": 0.02,
            "root_pos_rmse": 0.02,
            "root_ori_mean": 0.01,
            "joint_rmse": 0.04,
            "local_ee_rmse": 0.05,
        },
    ]

    annotate_priority_comparisons(rows)

    assert rows[1]["priority_compare_status"] == "better_upper_global"
    assert rows[1]["priority_better_than_no_mpc"] is True
    assert rows[1]["priority_first_regression_group"] == ""


def test_g1_wbc_priority_comparison_is_lexicographic() -> None:
    rows = [
        {
            "method": "no_mpc",
            "motion_file": "motion.npz",
            "policy": "bc",
            "contact_no_ref_violation_rate": 0.01,
            "contact_mismatch_rate": 0.02,
            "contact_switch_rate": 0.03,
            "contact_force_max": 100.0,
            "contact_force_mean": 50.0,
            "motion_smoothness": 0.1,
            "action_smoothness": 0.2,
            "upper_global_rmse": 0.03,
            "root_pos_rmse": 0.02,
            "root_ori_mean": 0.01,
            "joint_rmse": 0.04,
            "local_ee_rmse": 0.05,
        },
        {
            "method": "joint_mpc",
            "motion_file": "motion.npz",
            "policy": "bc",
            "contact_no_ref_violation_rate": 0.009,
            "contact_mismatch_rate": 0.019,
            "contact_switch_rate": 0.03,
            "contact_force_max": 100.0,
            "contact_force_mean": 50.0,
            "motion_smoothness": 0.1,
            "action_smoothness": 0.21,
            "upper_global_rmse": 0.04,
            "root_pos_rmse": 0.03,
            "root_ori_mean": 0.02,
            "joint_rmse": 0.05,
            "local_ee_rmse": 0.06,
        },
    ]

    annotate_priority_comparisons(rows)

    assert rows[1]["priority_compare_status"] == "better_contact"
    assert rows[1]["priority_better_than_no_mpc"] is True
    assert rows[1]["priority_first_regression_group"] == ""
    assert rows[1]["delta_vs_no_mpc_action_smoothness"] > 0.0


def test_g1_wbc_priority_comparison_tolerates_tiny_force_noise_only() -> None:
    rows = [
        {
            "method": "no_mpc",
            "motion_file": "motion.npz",
            "policy": "bc",
            "contact_no_ref_violation_rate": 0.01,
            "contact_mismatch_rate": 0.02,
            "contact_switch_rate": 0.03,
            "contact_force_max": 100.0,
            "contact_force_mean": 50.0,
            "motion_smoothness": 0.1,
            "action_smoothness": 0.2,
            "upper_global_rmse": 0.03,
            "root_pos_rmse": 0.02,
            "root_ori_mean": 0.01,
            "joint_rmse": 0.04,
            "local_ee_rmse": 0.05,
        },
        {
            "method": "joint_mpc",
            "motion_file": "motion.npz",
            "policy": "bc",
            "contact_no_ref_violation_rate": 0.009,
            "contact_mismatch_rate": 0.019,
            "contact_switch_rate": 0.03,
            "contact_force_max": 100.0,
            "contact_force_mean": 50.02,
            "motion_smoothness": 0.1,
            "action_smoothness": 0.2,
            "upper_global_rmse": 0.03,
            "root_pos_rmse": 0.02,
            "root_ori_mean": 0.01,
            "joint_rmse": 0.04,
            "local_ee_rmse": 0.05,
        },
        {
            "method": "ee_mpc",
            "motion_file": "motion.npz",
            "policy": "bc",
            "contact_no_ref_violation_rate": 0.0101,
            "contact_mismatch_rate": 0.019,
            "contact_switch_rate": 0.03,
            "contact_force_max": 100.0,
            "contact_force_mean": 50.02,
            "motion_smoothness": 0.1,
            "action_smoothness": 0.2,
            "upper_global_rmse": 0.03,
            "root_pos_rmse": 0.02,
            "root_ori_mean": 0.01,
            "joint_rmse": 0.04,
            "local_ee_rmse": 0.05,
        },
    ]

    annotate_priority_comparisons(rows)

    assert rows[1]["priority_compare_status"] == "better_contact"
    assert rows[1]["priority_better_than_no_mpc"] is True
    assert rows[1]["delta_vs_no_mpc_contact_force_mean"] > 0.0
    assert rows[2]["priority_compare_status"] == "regressed_contact"
    assert rows[2]["priority_better_than_no_mpc"] is False


def test_g1_wbc_priority_comparison_tolerates_small_force_and_smooth_noise() -> None:
    rows = [
        {
            "method": "no_mpc",
            "motion_file": "motion.npz",
            "policy": "bc",
            "contact_no_ref_violation_rate": 0.01,
            "contact_mismatch_rate": 0.02,
            "contact_switch_rate": 0.03,
            "contact_force_max": 600.0,
            "contact_force_mean": 170.0,
            "motion_smoothness": 0.001,
            "action_smoothness": 0.02,
            "upper_global_rmse": 0.04,
            "root_pos_rmse": 0.03,
            "root_ori_mean": 0.02,
            "joint_rmse": 0.05,
            "local_ee_rmse": 0.06,
        },
        {
            "method": "ee_mpc",
            "motion_file": "motion.npz",
            "policy": "bc",
            "contact_no_ref_violation_rate": 0.01,
            "contact_mismatch_rate": 0.02,
            "contact_switch_rate": 0.03,
            "contact_force_max": 601.5,
            "contact_force_mean": 170.2,
            "motion_smoothness": 0.0010005,
            "action_smoothness": 0.019997,
            "upper_global_rmse": 0.039,
            "root_pos_rmse": 0.03,
            "root_ori_mean": 0.02,
            "joint_rmse": 0.05,
            "local_ee_rmse": 0.06,
        },
    ]

    annotate_priority_comparisons(rows)

    assert rows[1]["priority_compare_status"] == "better_smooth"
    assert rows[1]["priority_better_than_no_mpc"] is True
