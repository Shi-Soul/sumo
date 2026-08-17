"""Single-step simulation backend for G1 WBC tracking policies."""

from __future__ import annotations

import os

import mujoco
import numpy as np
from judo.simulation.base import Simulation

from sumo.utils.extensions import require_g1_extensions
from sumo.utils.g1_wbc.constants import DEFAULT_POLICY_VARIANT
from sumo.utils.g1_wbc.constants import resolve_policy_path
from sumo.utils.g1_wbc.reference import normalize_controls


class SimBackendG1WBC:
    """Advance MuJoCo by one step using the WBC tracking policy."""

    def __init__(self, model: mujoco.MjModel) -> None:
        policy = os.environ.get("SUMO_G1_WBC_POLICY", DEFAULT_POLICY_VARIANT)
        self.policy_path = str(resolve_policy_path(policy))
        self.g1_extensions = require_g1_extensions()
        self.previous_policy_state = np.zeros(self.g1_extensions.g1_wbc_policy_state_dim(), dtype=np.float32)

    def reset(self) -> None:
        self.previous_policy_state[:] = 0.0

    def get_sim_metadata(self) -> dict:
        return {"g1_wbc_policy_state": self.previous_policy_state.copy()}

    def sim(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        command: np.ndarray,
        reference_qvel: np.ndarray | None = None,
    ) -> None:
        control = normalize_controls(np.asarray(command, dtype=np.float64).reshape(1, -1))[0]
        x0 = np.concatenate([data.qpos, data.qvel])
        qvel_arg = None if reference_qvel is None else np.asarray(reference_qvel, dtype=np.float64).reshape(-1)
        self.previous_policy_state = self.g1_extensions.sim_g1_wbc(
            model,
            data,
            x0,
            control,
            self.previous_policy_state,
            self.policy_path,
            qvel_arg,
        )


class G1WBCSimulation(Simulation):
    """Simulation backend for registered g1_wbc_* tasks."""

    supports_reference_qvel = True

    def __init__(self, **kwargs) -> None:
        self._sim_backend: SimBackendG1WBC | None = None
        super().__init__(**kwargs)

    def set_task(self, task_name: str) -> None:
        super().set_task(task_name)
        self._sim_backend = SimBackendG1WBC(self.task.sim_model)

    def reset(self) -> None:
        if self._sim_backend is not None:
            self._sim_backend.reset()

    def get_sim_metadata(self) -> dict:
        if self._sim_backend is None:
            return {}
        return self._sim_backend.get_sim_metadata()

    def step(self, command: np.ndarray, reference_qvel: np.ndarray | None = None) -> None:
        if self.paused:
            return
        assert self._sim_backend is not None
        self.task.pre_sim_step()
        self._sim_backend.sim(self.task.sim_model, self.task.data, self.task.task_to_sim_ctrl(command), reference_qvel)
        self.task.post_sim_step()
