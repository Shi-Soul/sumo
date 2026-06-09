"""Controller specialization for G1 WBC whole-body reference controls."""

from __future__ import annotations

import warnings

import numpy as np
from judo.controller.controller import Controller as JudoController
from judo.controller.controller import ControllerConfig

from sumo.utils.g1_wbc.reference import interpolate_controls, normalize_controls


class G1WBCControlSpline:
    """Callable spline that keeps root quaternions on S3 with SLERP."""

    def __init__(self, times: np.ndarray, controls: np.ndarray, spline_order: str) -> None:
        self.times = np.asarray(times, dtype=np.float64)
        self.controls = normalize_controls(controls)
        self.spline_order = spline_order

    def __call__(self, query_times: np.ndarray | float) -> np.ndarray:
        return interpolate_controls(self.times, self.controls, query_times, self.spline_order)


class G1WBCController(JudoController):
    """Judo controller with reference-horizon initialization and quaternion SLERP."""

    def reset(self) -> None:
        super().reset()
        self._set_reference_nominal_knots()

    def update_spline(self, times: np.ndarray, controls: np.ndarray) -> None:
        """Update the controller spline using SLERP for the root quaternion."""
        self.spline = G1WBCControlSpline(times, controls, self.spline_order)

    def _pre_optimization(self) -> None:
        self._ensure_valid_spline_nodes()
        self._set_reference_nominal_knots()
        super()._pre_optimization()

    def _sample_controls(self) -> np.ndarray:
        self._candidate_knots_normalized = self.optimizer.sample_control_knots(self._nominal_knots_normalized)
        self._candidate_knots_normalized = np.clip(
            self._candidate_knots_normalized,
            self.action_normalizer.normalize(self.task.actuator_ctrlrange[:, 0]),
            self.action_normalizer.normalize(self.task.actuator_ctrlrange[:, 1]),
        )
        self.candidate_knots = normalize_controls(self.action_normalizer.denormalize(self._candidate_knots_normalized))
        self._candidate_knots_normalized = self.action_normalizer.normalize(self.candidate_knots)
        candidate_splines = G1WBCControlSpline(self._new_times, self.candidate_knots, self.spline_order)
        return candidate_splines(self.time + self.rollout_times)

    def _set_reference_nominal_knots(self) -> None:
        if not hasattr(self.task, "reference_controls_for_times"):
            raise TypeError("G1WBCController requires task.reference_controls_for_times(times)")
        self.times = self.task.data.time + self.spline_timesteps
        self.nominal_knots = normalize_controls(self.task.reference_controls_for_times(self.times))
        self.candidate_knots = np.tile(self.nominal_knots, (self.optimizer_cfg.num_rollouts, 1, 1))
        self.update_spline(self.times, self.nominal_knots)

    def _ensure_valid_spline_nodes(self) -> None:
        if self.optimizer_cfg.num_nodes < 4 and self.spline_order == "cubic":
            warnings.warn("Cubic splines require at least 4 nodes. Setting num_nodes=4.", stacklevel=2)
            self.optimizer_cfg.num_nodes = 4


__all__ = ["ControllerConfig", "G1WBCControlSpline", "G1WBCController"]
