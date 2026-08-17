"""Controller specialization for G1 WBC whole-body reference controls."""

from __future__ import annotations

import warnings

import numpy as np
from judo.app.structs import MujocoState
from judo.controller.controller import Controller as JudoController
from judo.controller.controller import ControllerConfig
from judo.tasks.spot.spot_constants import POLICY_OUTPUT_DIM
from judo.utils.hierarchical_mj_rollout_backend import HierarchicalMJRolloutBackend
from judo.utils.normalization import IdentityNormalizer, normalizer_registry
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation

from sumo.utils.g1_wbc.constants import JOINT_POS_SLICE, MUJOCO_JOINT_NAMES, POLICY_DT, ROOT_POS_SLICE, ROOT_QUAT_SLICE
from sumo.utils.g1_wbc.reference import controls_to_policy_qvel, interpolate_controls, normalize_controls


class G1WBCControlSpline:
    """Callable spline that keeps root quaternions on S3 with SLERP."""

    def __init__(self, times: np.ndarray, controls: np.ndarray, spline_order: str) -> None:
        self.times = np.asarray(times, dtype=np.float64)
        self.controls = normalize_controls(controls)
        self.spline_order = spline_order

    def __call__(self, query_times: np.ndarray | float) -> np.ndarray:
        return interpolate_controls(self.times, self.controls, query_times, self.spline_order)


class G1WBCResidualSpline:
    """Spline for additive residuals; residuals are not normalized controls."""

    def __init__(self, times: np.ndarray, residuals: np.ndarray, spline_order: str) -> None:
        self.times = np.asarray(times, dtype=np.float64)
        self.residuals = np.asarray(residuals, dtype=np.float64)
        self.spline_order = spline_order

    def __call__(self, query_times: np.ndarray | float) -> np.ndarray:
        query = np.asarray(query_times, dtype=np.float64)
        scalar_query = query.ndim == 0
        query_1d = query.reshape(-1)
        if len(self.times) == 0:
            raise ValueError("Cannot interpolate an empty residual trajectory")
        if len(self.times) == 1:
            out = np.repeat(self.residuals[..., :1, :], len(query_1d), axis=-2)
            return out[..., 0, :] if scalar_query else out

        order = self.spline_order
        if order == "cubic" and len(self.times) < 4:
            order = "linear"
        interpolated = interp1d(
            self.times,
            self.residuals,
            kind=order,
            axis=-2,
            copy=False,
            fill_value=(self.residuals[..., 0, :], self.residuals[..., -1, :]),
            bounds_error=False,
        )(query_1d)
        return interpolated[..., 0, :] if scalar_query else interpolated


class G1WBCReferenceResidualSpline:
    """Full command spline represented as dense reference plus residual spline."""

    def __init__(
        self,
        reference_controls_for_times,
        residual_spline: G1WBCResidualSpline,
        compose_reference_residual,
    ) -> None:
        self.reference_controls_for_times = reference_controls_for_times
        self.residual_spline = residual_spline
        self.compose_reference_residual = compose_reference_residual

    def __call__(self, query_times: np.ndarray | float) -> np.ndarray:
        query = np.asarray(query_times, dtype=np.float64)
        scalar_query = query.ndim == 0
        query_1d = query.reshape(-1)
        reference = self.reference_controls_for_times(query_1d)
        residual = self.residual_spline(query_1d)
        out = self.compose_reference_residual(reference, residual)
        return out[0] if scalar_query else out


class G1WBCController(JudoController):
    """Judo controller with reference-horizon initialization and quaternion SLERP."""

    _EE_RESIDUAL_JOINT_PATTERNS = ("shoulder", "elbow", "wrist")
    _ROOT_DELTA_LOWER = np.array([-0.06, -0.06, 0.0], dtype=np.float64)
    _ROOT_DELTA_UPPER = np.array([0.06, 0.06, 0.04], dtype=np.float64)
    _JOINT_DELTA_LIMIT = 0.2
    _ROOT_ORI_DELTA_LIMIT = 0.15
    _ROOT_ROT_RESIDUAL_SLICE = slice(3, 6)
    _RESIDUAL_WARM_START_DECAY = 0.0

    def update_states(self, state_msg: MujocoState) -> None:
        super().update_states(state_msg)
        policy_state = self.system_metadata.get("g1_wbc_policy_state")
        if policy_state is not None:
            self._last_policy_output = np.asarray(policy_state, dtype=np.float32).reshape(-1)

    def reset(self) -> None:
        super().reset()
        self._set_reference_nominal_knots()

    def policy_command_time(self, sim_time: float | np.ndarray) -> float | np.ndarray:
        """Match tracking_bfm's wbteleop command timing."""
        return np.asarray(sim_time) + POLICY_DT if not np.isscalar(sim_time) else float(sim_time) + POLICY_DT

    def update_spline(self, times: np.ndarray, controls: np.ndarray) -> None:
        """Update the controller spline using SLERP for the root quaternion."""
        self.spline = G1WBCControlSpline(times, controls, self.spline_order)

    def _update_residual_spline(self, times: np.ndarray, residual_knots: np.ndarray) -> None:
        self.residual_spline = G1WBCResidualSpline(times, residual_knots, self.spline_order)
        self.spline = G1WBCReferenceResidualSpline(
            self.task.reference_controls_for_times,
            self.residual_spline,
            self._compose_reference_residual,
        )

    def action_reference_qvel(self, time: float) -> np.ndarray:
        """Return policy-reference qvel for the current refined spline command."""
        dt = float(self.task.sim_model.opt.timestep)
        center = float(time)
        spline_start = float(self.times[0]) if hasattr(self, "times") and len(self.times) else center
        spline_end = float(self.times[-1]) if hasattr(self, "times") and len(self.times) else center
        if center - dt < spline_start:
            query_times = np.asarray([center, center + dt], dtype=np.float64)
            qvel_index = 0
        elif center + dt > spline_end:
            query_times = np.asarray([center - dt, center], dtype=np.float64)
            qvel_index = 1
        else:
            query_times = np.asarray([center - dt, center, center + dt], dtype=np.float64)
            qvel_index = 1
        controls = self.spline(query_times)
        reference_controls = self.task.reference_controls_for_times(query_times)
        control_qvel = controls_to_policy_qvel(self.task.sim_model, controls, dt)[qvel_index]
        reference_spline_qvel = controls_to_policy_qvel(self.task.sim_model, reference_controls, dt)[qvel_index]
        reference_qvel_for_times = getattr(self.task, "reference_policy_qvel_for_times", None)
        if callable(reference_qvel_for_times):
            return reference_qvel_for_times(np.asarray([center], dtype=np.float64))[0] + control_qvel - reference_spline_qvel
        return control_qvel

    def _pre_optimization(self) -> None:
        self._ensure_valid_spline_nodes()
        assert self.current_state.shape == (self.model.nq + self.model.nv,), "Current state must be of shape (nq + nv,)"
        assert self.optimizer_cfg.num_rollouts > 0, "Need at least one rollout!"

        self._new_times = self.policy_command_time(self.time) + self.spline_timesteps
        self._reference_knots = normalize_controls(self.task.reference_controls_for_times(self._new_times))
        self.nominal_residual_knots = self._warm_start_residual_knots(self._new_times, self._reference_knots.shape)
        self.nominal_knots = self._compose_reference_residual(self._reference_knots, self.nominal_residual_knots)
        self._nominal_knots_normalized = self.nominal_residual_knots.copy()

        if self.rollout_backend.num_threads != self.optimizer_cfg.num_rollouts:
            num_problems = getattr(self.rollout_backend, "num_problems", None)
            if num_problems is not None:
                self.rollout_backend.update(self.optimizer_cfg.num_rollouts, num_problems)  # type: ignore[call-arg]
            else:
                self.rollout_backend.update(self.optimizer_cfg.num_rollouts)
            if isinstance(self.rollout_backend, HierarchicalMJRolloutBackend):
                self._last_policy_output = np.zeros((self.optimizer_cfg.num_rollouts, POLICY_OUTPUT_DIM))
            else:
                self._last_policy_output = None

        normalizer_cls = normalizer_registry.get(self.action_normalizer_type)
        if normalizer_cls is None:
            warnings.warn(
                f"Invalid action normalizer type '{self.action_normalizer_type}'. "
                f"Available types: {list(normalizer_registry.keys())}. "
                "Falling back to 'none' normalizer.",
                stacklevel=2,
            )
            normalizer_cls = IdentityNormalizer
        if not isinstance(self.action_normalizer, normalizer_cls):
            self.action_normalizer = self._init_action_normalizer()

        self.optimizer.pre_optimization(self.times, self._new_times)

    def _sample_controls(self) -> np.ndarray:
        sampled_residuals = self.optimizer.sample_control_knots(self._nominal_knots_normalized)
        self.candidate_residual_knots = self._constrain_residual_knots(sampled_residuals)
        self.candidate_residual_knots[0] = 0.0
        self._candidate_knots_normalized = self.candidate_residual_knots.copy()
        self.candidate_knots = self._compose_reference_residual(self._reference_knots, self.candidate_residual_knots)
        residual_splines = G1WBCResidualSpline(self._new_times, self.candidate_residual_knots, self.spline_order)
        rollout_times = self.policy_command_time(self.time + self.rollout_times)
        rollout_reference = self.task.reference_controls_for_times(rollout_times)
        rollout_residuals = residual_splines(rollout_times)
        rollout_controls = self._compose_reference_residual(rollout_reference, rollout_residuals)
        setattr(
            self.rollout_backend,
            "reference_qvels",
            self._candidate_policy_qvels(rollout_controls, rollout_reference, rollout_times),
        )
        return rollout_controls

    def _post_optimization(self) -> None:
        """Use the best evaluated WBC trajectory instead of the CEM elite mean."""
        if getattr(self, "candidate_knots", None) is not None and getattr(self, "rewards", None) is not None:
            select_candidate = getattr(self.task, "select_mpc_candidate", None)
            has_rollout_data = (
                callable(select_candidate)
                and getattr(self, "states", None) is not None
                and getattr(self, "rollout_controls", None) is not None
                and self.states.shape[0] == len(self.rewards)
                and self.rollout_controls.shape[0] == len(self.rewards)
                and np.any(self.states)
            )
            if has_rollout_data:
                best_idx = int(select_candidate(self.states, self.sensors, self.rollout_controls, self.rewards))
            else:
                best_idx = int(np.argmax(self.rewards))
            self.nominal_residual_knots = np.asarray(self.candidate_residual_knots[best_idx], dtype=np.float64).copy()
            self.nominal_knots = normalize_controls(self.candidate_knots[best_idx])
            self._nominal_knots_normalized = self.nominal_residual_knots.copy()
        else:
            self.nominal_residual_knots = np.zeros_like(self._reference_knots)
            self.nominal_knots = self._reference_knots.copy()
            self._nominal_knots_normalized = self.nominal_residual_knots.copy()
        self.times = self._new_times
        self._update_residual_spline(self.times, self.nominal_residual_knots)
        self.update_traces()

    def _set_reference_nominal_knots(self) -> None:
        if not hasattr(self.task, "reference_controls_for_times"):
            raise TypeError("G1WBCController requires task.reference_controls_for_times(times)")
        self.times = self.policy_command_time(self.task.data.time) + self.spline_timesteps
        self.nominal_knots = normalize_controls(self.task.reference_controls_for_times(self.times))
        self.nominal_residual_knots = np.zeros_like(self.nominal_knots)
        self.candidate_knots = np.tile(self.nominal_knots, (self.optimizer_cfg.num_rollouts, 1, 1))
        self.candidate_residual_knots = np.zeros_like(self.candidate_knots)
        self._update_residual_spline(self.times, self.nominal_residual_knots)

    def _warm_start_residual_knots(self, times: np.ndarray, expected_shape: tuple[int, ...]) -> np.ndarray:
        if self._RESIDUAL_WARM_START_DECAY <= 0.0:
            return np.zeros(expected_shape, dtype=np.float64)
        residual_spline = getattr(self, "residual_spline", None)
        if residual_spline is None:
            return np.zeros(expected_shape, dtype=np.float64)
        try:
            residuals = np.asarray(residual_spline(times), dtype=np.float64)
        except Exception:
            return np.zeros(expected_shape, dtype=np.float64)
        if residuals.shape != expected_shape or not np.isfinite(residuals).all():
            return np.zeros(expected_shape, dtype=np.float64)
        return self._RESIDUAL_WARM_START_DECAY * self._constrain_residual_knots(residuals)

    def _ensure_valid_spline_nodes(self) -> None:
        if self.optimizer_cfg.num_nodes < 4 and self.spline_order == "cubic":
            warnings.warn("Cubic splines require at least 4 nodes. Setting num_nodes=4.", stacklevel=2)
            self.optimizer_cfg.num_nodes = 4

    def _constrain_residual_knots(self, residuals: np.ndarray) -> np.ndarray:
        """Keep residual refinements local to the motion reference used by the WBC policy."""
        residuals = np.asarray(residuals, dtype=np.float64).copy()
        residuals = self._mask_residual_knots(residuals)
        residuals[..., ROOT_POS_SLICE] = np.clip(
            residuals[..., ROOT_POS_SLICE],
            self._ROOT_DELTA_LOWER,
            self._ROOT_DELTA_UPPER,
        )

        residuals[..., JOINT_POS_SLICE] = np.clip(
            residuals[..., JOINT_POS_SLICE],
            -self._JOINT_DELTA_LIMIT,
            self._JOINT_DELTA_LIMIT,
        )
        rotvec = residuals[..., self._ROOT_ROT_RESIDUAL_SLICE]
        angles = np.linalg.norm(rotvec, axis=-1, keepdims=True)
        scale = np.minimum(1.0, self._ROOT_ORI_DELTA_LIMIT / np.maximum(angles, 1e-12))
        residuals[..., self._ROOT_ROT_RESIDUAL_SLICE] = rotvec * scale
        residuals[..., 6] = 0.0
        return residuals

    def _mask_residual_knots(self, residuals: np.ndarray) -> np.ndarray:
        """Keep EE tracking residuals off the root and legs so contact/root gait stays comparable."""
        if getattr(self.task, "reward_mode", None) != "ee":
            return residuals
        mask = np.zeros(residuals.shape[-1], dtype=np.float64)
        for idx, joint_name in enumerate(MUJOCO_JOINT_NAMES):
            if any(pattern in joint_name for pattern in self._EE_RESIDUAL_JOINT_PATTERNS):
                mask[JOINT_POS_SLICE.start + idx] = 1.0
        return residuals * mask

    def _compose_reference_residual(self, reference: np.ndarray, residuals: np.ndarray) -> np.ndarray:
        reference = normalize_controls(np.asarray(reference, dtype=np.float64))
        residuals = self._constrain_residual_knots(residuals)
        full = np.broadcast_to(reference, residuals.shape).copy()
        full[..., ROOT_POS_SLICE] = full[..., ROOT_POS_SLICE] + residuals[..., ROOT_POS_SLICE]
        full[..., JOINT_POS_SLICE] = full[..., JOINT_POS_SLICE] + residuals[..., JOINT_POS_SLICE]

        rotvec = residuals[..., self._ROOT_ROT_RESIDUAL_SLICE].reshape(-1, 3)
        if rotvec.size:
            ref_quat = np.broadcast_to(reference, residuals.shape)[..., ROOT_QUAT_SLICE].reshape(-1, 4)
            ref_rot = Rotation.from_quat(np.concatenate([ref_quat[:, 1:], ref_quat[:, :1]], axis=-1))
            delta_rot = Rotation.from_rotvec(rotvec)
            quat_xyzw = (ref_rot * delta_rot).as_quat()
            full[..., ROOT_QUAT_SLICE] = np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, :3]], axis=-1).reshape(
                full[..., ROOT_QUAT_SLICE].shape
            )
        return normalize_controls(full)

    def _candidate_policy_qvels(
        self,
        controls: np.ndarray,
        reference_controls: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray | None:
        reference_qvel_for_times = getattr(self.task, "reference_policy_qvel_for_times", None)
        if not callable(reference_qvel_for_times):
            return None
        controls = normalize_controls(controls)
        reference_controls = normalize_controls(reference_controls)
        dt = float(self.task.sim_model.opt.timestep)
        reference_motion_qvel = reference_qvel_for_times(np.asarray(times, dtype=np.float64))
        reference_spline_qvel = controls_to_policy_qvel(self.task.sim_model, reference_controls, dt)
        qvels = np.empty((controls.shape[0], controls.shape[1], self.task.sim_model.nv), dtype=np.float64)
        for batch in range(controls.shape[0]):
            control_qvel = controls_to_policy_qvel(self.task.sim_model, controls[batch], dt)
            qvels[batch] = reference_motion_qvel + control_qvel - reference_spline_qvel
        return qvels

__all__ = [
    "ControllerConfig",
    "G1WBCControlSpline",
    "G1WBCReferenceResidualSpline",
    "G1WBCResidualSpline",
    "G1WBCController",
]
