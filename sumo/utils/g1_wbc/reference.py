"""Reference trajectory conversion helpers for G1 WBC policy inputs."""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.interpolate import interp1d

from sumo.utils.g1_wbc.constants import (
    JOINT_POS_SLICE,
    MUJOCO_BODY_NAMES,
    ROOT_POS_SLICE,
    ROOT_QUAT_SLICE,
    TASK_CONTROL_DIM,
)
from sumo.utils.g1_wbc.math import (
    angular_velocity_from_quat,
    finite_difference,
    normalize_quat,
    quat_rotate_inverse,
    slerp_wxyz,
)
from sumo.utils.g1_wbc.policy import ReferenceFrame

_NON_QUAT_INDICES = np.r_[0:3, 7:TASK_CONTROL_DIM]


def normalize_controls(controls: np.ndarray) -> np.ndarray:
    controls = np.asarray(controls, dtype=np.float64).copy()
    if controls.shape[-1] != TASK_CONTROL_DIM:
        raise ValueError(f"Expected controls last dim {TASK_CONTROL_DIM}, got {controls.shape}")
    controls[..., ROOT_QUAT_SLICE] = normalize_quat(controls[..., ROOT_QUAT_SLICE])
    return controls


def controls_to_qpos(control: np.ndarray) -> np.ndarray:
    control = normalize_controls(np.asarray(control))
    return np.concatenate([control[ROOT_POS_SLICE], control[ROOT_QUAT_SLICE], control[JOINT_POS_SLICE]])


def qpos_to_control(qpos: np.ndarray) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float64)
    return np.concatenate([qpos[:3], normalize_quat(qpos[3:7]), qpos[7:36]])


def controls_to_qvel(controls: np.ndarray, dt: float) -> np.ndarray:
    controls = normalize_controls(controls)
    qvel = np.zeros((len(controls), 35), dtype=np.float64)
    qvel[:, :3] = finite_difference(controls[:, ROOT_POS_SLICE], dt)
    qvel[:, 3:6] = angular_velocity_from_quat(controls[:, ROOT_QUAT_SLICE], dt)
    qvel[:, 6:35] = finite_difference(controls[:, JOINT_POS_SLICE], dt)
    return qvel


def controls_to_policy_qvel(model: mujoco.MjModel, controls: np.ndarray, dt: float) -> np.ndarray:
    """Convert refined controls into wbteleop policy qvel semantics.

    The policy's reference angular velocity field follows tracking_bfm's
    torso-link angular velocity command, not the free-joint root quaternion
    derivative. Joint velocities and root linear velocity still come directly
    from the whole-body trajectory samples.
    """
    controls = normalize_controls(controls)
    qvel = controls_to_qvel(controls, dt)
    if len(controls) == 0:
        return qvel

    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    if torso_id < 0:
        raise ValueError("G1 WBC model is missing torso_link")
    data = mujoco.MjData(model)
    torso_quat = np.zeros((len(controls), 4), dtype=np.float64)
    for idx, ctrl in enumerate(controls):
        data.qpos[:] = controls_to_qpos(ctrl)
        data.qvel[:] = qvel[idx]
        mujoco.mj_forward(model, data)
        torso_quat[idx] = data.xquat[torso_id]
    qvel[:, 3:6] = angular_velocity_from_quat(torso_quat, dt)
    return qvel


def build_reference_frames(
    model: mujoco.MjModel,
    controls: np.ndarray,
    dt: float,
) -> list[ReferenceFrame]:
    """Convert whole-body controls into policy reference frames."""
    controls = normalize_controls(controls)
    qvel = controls_to_qvel(controls, dt)
    data = mujoco.MjData(model)
    frames: list[ReferenceFrame] = []
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in MUJOCO_BODY_NAMES]
    for t, ctrl in enumerate(controls):
        data.qpos[:] = controls_to_qpos(ctrl)
        data.qvel[:] = qvel[t]
        mujoco.mj_forward(model, data)
        body_pos = np.asarray(data.xpos[body_ids], dtype=np.float64).copy()
        body_quat = normalize_quat(np.asarray(data.xquat[body_ids], dtype=np.float64).copy())
        frames.append(
            ReferenceFrame(
                joint_pos=np.asarray(ctrl[JOINT_POS_SLICE], dtype=np.float64).copy(),
                joint_vel=qvel[t, 6:35].copy(),
                body_pos_w=body_pos,
                body_quat_w=body_quat,
                anchor_ang_vel_w=qvel[t, 3:6].copy(),
            )
        )
    return frames


def build_reference_frame_from_qvel(
    model: mujoco.MjModel,
    control: np.ndarray,
    qvel: np.ndarray,
) -> ReferenceFrame:
    """Convert a single whole-body control and explicit velocity into a policy reference."""
    control = normalize_controls(np.asarray(control, dtype=np.float64))
    qvel = np.asarray(qvel, dtype=np.float64)
    if qvel.shape != (model.nv,):
        raise ValueError(f"Expected qvel shape {(model.nv,)}, got {qvel.shape}")

    data = mujoco.MjData(model)
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in MUJOCO_BODY_NAMES]
    data.qpos[:] = controls_to_qpos(control)
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
    return ReferenceFrame(
        joint_pos=control[JOINT_POS_SLICE].copy(),
        joint_vel=qvel[6:35].copy(),
        body_pos_w=np.asarray(data.xpos[body_ids], dtype=np.float64).copy(),
        body_quat_w=normalize_quat(np.asarray(data.xquat[body_ids], dtype=np.float64).copy()),
        anchor_ang_vel_w=qvel[3:6].copy(),
    )


def interpolate_controls(
    times: np.ndarray,
    controls: np.ndarray,
    query_times: np.ndarray | float,
    spline_order: str = "linear",
) -> np.ndarray:
    """Interpolate whole-body controls, using SLERP for the root quaternion."""
    times = np.asarray(times, dtype=np.float64)
    controls = normalize_controls(np.asarray(controls, dtype=np.float64))
    query = np.asarray(query_times, dtype=np.float64)
    scalar_query = query.ndim == 0
    query_1d = query.reshape(-1)

    if controls.ndim < 2 or controls.shape[-2] != len(times) or controls.shape[-1] != TASK_CONTROL_DIM:
        raise ValueError(f"Expected controls shape (..., {len(times)}, {TASK_CONTROL_DIM}), got {controls.shape}")
    if len(times) == 0:
        raise ValueError("Cannot interpolate an empty control trajectory")
    if len(times) == 1:
        out = np.repeat(controls[..., :1, :], len(query_1d), axis=-2)
        return out[..., 0, :] if scalar_query else out

    order = spline_order
    if order == "cubic" and len(times) < 4:
        order = "linear"

    fill_value = (controls[..., 0, _NON_QUAT_INDICES], controls[..., -1, _NON_QUAT_INDICES])
    linear = interp1d(
        times,
        controls[..., _NON_QUAT_INDICES],
        kind=order,
        axis=-2,
        copy=False,
        fill_value=fill_value,  # type: ignore[arg-type]
        bounds_error=False,
    )(query_1d)

    leading_shape = controls.shape[:-2]
    out = np.empty((*leading_shape, len(query_1d), TASK_CONTROL_DIM), dtype=np.float64)
    out[..., _NON_QUAT_INDICES] = linear

    clipped_query = np.clip(query_1d, times[0], times[-1])
    flat_controls = controls.reshape((-1, controls.shape[-2], controls.shape[-1]))
    flat_quat = np.empty((flat_controls.shape[0], len(query_1d), 4), dtype=np.float64)
    for idx, ctrl in enumerate(flat_controls):
        flat_quat[idx] = slerp_wxyz(times, ctrl[:, ROOT_QUAT_SLICE], clipped_query)
    out[..., ROOT_QUAT_SLICE] = flat_quat.reshape((*leading_shape, len(query_1d), 4))

    out = normalize_controls(out)
    return out[..., 0, :] if scalar_query else out


def motion_controls_at_times(motion, times: np.ndarray) -> np.ndarray:
    """Sample motion controls with continuous interpolation and root-quat SLERP."""
    motion_times = np.arange(motion.num_frames, dtype=np.float64) * motion.dt
    return interpolate_controls(motion_times, motion.trajectory_controls(), np.asarray(times, dtype=np.float64), "linear")


def motion_qvel_at_times(motion, times: np.ndarray) -> np.ndarray:
    """Sample physical root/joint qvel from motion data in MuJoCo qvel order."""
    query = np.asarray(times, dtype=np.float64).reshape(-1)
    motion_times = np.arange(motion.num_frames, dtype=np.float64) * motion.dt
    pelvis_idx = MUJOCO_BODY_NAMES.index("pelvis")
    root_quat = motion_controls_at_times(motion, query)[:, ROOT_QUAT_SLICE]
    if motion.num_frames == 1:
        root_lin = np.repeat(motion.body_lin_vel_w[:1, pelvis_idx], len(query), axis=0)
        root_ang = np.repeat(motion.body_ang_vel_w[:1, pelvis_idx], len(query), axis=0)
        joint_vel = np.repeat(motion.joint_vel[:1], len(query), axis=0)
    else:
        root_lin = interp1d(
            motion_times,
            motion.body_lin_vel_w[:, pelvis_idx],
            axis=0,
            copy=False,
            bounds_error=False,
            fill_value=(motion.body_lin_vel_w[0, pelvis_idx], motion.body_lin_vel_w[-1, pelvis_idx]),
        )(query)
        root_ang = interp1d(
            motion_times,
            motion.body_ang_vel_w[:, pelvis_idx],
            axis=0,
            copy=False,
            bounds_error=False,
            fill_value=(motion.body_ang_vel_w[0, pelvis_idx], motion.body_ang_vel_w[-1, pelvis_idx]),
        )(query)
        joint_vel = interp1d(
            motion_times,
            motion.joint_vel,
            axis=0,
            copy=False,
            bounds_error=False,
            fill_value=(motion.joint_vel[0], motion.joint_vel[-1]),
        )(query)
    qvel = np.zeros((len(query), 35), dtype=np.float64)
    qvel[:, :3] = root_lin
    qvel[:, 3:6] = quat_rotate_inverse(root_quat, root_ang)
    qvel[:, 6:35] = joint_vel
    return qvel


def motion_policy_qvel_at_times(motion, times: np.ndarray) -> np.ndarray:
    """Sample policy reference qvel matching tracking_bfm wbteleop observations."""
    qvel = motion_qvel_at_times(motion, times)
    query = np.asarray(times, dtype=np.float64).reshape(-1)
    motion_times = np.arange(motion.num_frames, dtype=np.float64) * motion.dt
    anchor_idx = MUJOCO_BODY_NAMES.index("torso_link")
    qvel[:, 3:6] = interp1d(
        motion_times,
        motion.body_ang_vel_w[:, anchor_idx],
        axis=0,
        copy=False,
        bounds_error=False,
        fill_value=(motion.body_ang_vel_w[0, anchor_idx], motion.body_ang_vel_w[-1, anchor_idx]),
    )(query)
    return qvel
