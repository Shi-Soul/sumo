"""Single-step simulation backend for G1 WBC tracking policies."""

from __future__ import annotations

import os

import mujoco
import numpy as np
from judo.simulation.base import Simulation

from sumo.utils.g1_wbc.constants import DEFAULT_POLICY_VARIANT
from sumo.utils.g1_wbc.policy import TrackingPolicyRuntime
from sumo.utils.g1_wbc.reference import build_reference_frame_from_qvel, controls_to_qvel, normalize_controls


class SimBackendG1WBC:
    """Advance MuJoCo by one step using the WBC tracking policy."""

    def __init__(self, model: mujoco.MjModel) -> None:
        policy = os.environ.get("SUMO_G1_WBC_POLICY", DEFAULT_POLICY_VARIANT)
        self.runtime = TrackingPolicyRuntime(model, policy)
        self._last_time = 0.0
        self._last_control: np.ndarray | None = None

    def reset(self) -> None:
        self.runtime.reset()
        self._last_time = 0.0
        self._last_control = None

    def sim(self, model: mujoco.MjModel, data: mujoco.MjData, command: np.ndarray) -> None:
        control = normalize_controls(np.asarray(command, dtype=np.float64).reshape(1, -1))[0]
        if self._last_control is None or float(data.time) < self._last_time:
            qvel = np.zeros(model.nv, dtype=np.float64)
        else:
            controls = np.stack([self._last_control, control], axis=0)
            qvel = controls_to_qvel(controls, model.opt.timestep)[-1]
        ref = build_reference_frame_from_qvel(model, control, qvel)
        data.ctrl[:] = self.runtime.step(data, ref)
        mujoco.mj_step(model, data)
        self._last_time = float(data.time)
        self._last_control = control.copy()


class G1WBCSimulation(Simulation):
    """Simulation backend for registered g1_wbc_* tasks."""

    def __init__(self, **kwargs) -> None:
        self._sim_backend: SimBackendG1WBC | None = None
        super().__init__(**kwargs)

    def set_task(self, task_name: str) -> None:
        super().set_task(task_name)
        self._sim_backend = SimBackendG1WBC(self.task.sim_model)

    def step(self, command: np.ndarray) -> None:
        if self.paused:
            return
        assert self._sim_backend is not None
        self.task.pre_sim_step()
        self._sim_backend.sim(self.task.sim_model, self.task.data, self.task.task_to_sim_ctrl(command))
        self.task.post_sim_step()
