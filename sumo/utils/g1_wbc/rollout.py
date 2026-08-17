"""G1 WBC rollout backend using SUMO's local C++ extensions."""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np
from judo.utils.mj_rollout_backend import make_model_data_pairs
from judo.utils.rollout_backend import RolloutBackend
from mujoco import MjModel

from sumo.utils.extensions import require_g1_extensions
from sumo.utils.g1_wbc.constants import CONTACT_GEOM_PREFIXES, DEFAULT_POLICY_VARIANT, resolve_policy_path
from sumo.utils.g1_wbc.reference import normalize_controls


class G1WBCRolloutBackend(RolloutBackend):
    """Rollout backend that executes the wbteleop ONNX policy in C++."""

    def __init__(
        self,
        model: MjModel,
        num_threads: int,
        cutoff_time: float | None = None,
        policy: str | Path | None = None,
    ) -> None:
        self.model = model
        self.num_threads = num_threads
        if cutoff_time is None:
            cutoff_time = float(os.environ.get("SUMO_G1_WBC_ROLLOUT_CUTOFF_TIME", "2.0"))
        self.cutoff_time = cutoff_time
        self.policy = policy or os.environ.get("SUMO_G1_WBC_POLICY", DEFAULT_POLICY_VARIANT)
        self.policy_path = str(resolve_policy_path(self.policy))
        g1_extensions = require_g1_extensions()
        self._policy_state_dim = g1_extensions.g1_wbc_policy_state_dim()
        self._rollout_obj = g1_extensions.G1WBCRollout(
            nthread=num_threads,
            cutoff_time=cutoff_time,
            policy_path=self.policy_path,
        )
        self._models, self._datas = make_model_data_pairs(model, num_threads)
        self._foot_geom_side = _build_foot_geom_side(model)
        self.reference_qvels = None

    def rollout(
        self,
        x0: np.ndarray,
        controls: np.ndarray,
        last_policy_output=None,
    ) -> tuple[np.ndarray, np.ndarray, None]:
        controls = normalize_controls(controls)
        batch_size, horizon, _ = controls.shape
        if batch_size != len(self._models):
            raise ValueError(f"Expected {len(self._models)} rollouts, got {batch_size}")

        x0_batched = np.tile(x0, (batch_size, 1))
        if last_policy_output is None:
            initial_policy_state = np.zeros(self._policy_state_dim, dtype=np.float32)
        else:
            initial_policy_state = np.asarray(last_policy_output, dtype=np.float32).reshape(-1)
        if initial_policy_state.shape != (self._policy_state_dim,):
            raise ValueError(
                f"Expected G1 WBC policy state shape {(self._policy_state_dim,)}, got {initial_policy_state.shape}"
            )
        reference_qvels = None if self.reference_qvels is None else np.asarray(self.reference_qvels, dtype=np.float64)
        if reference_qvels is not None and reference_qvels.shape != (batch_size, horizon, self.model.nv):
            raise ValueError(
                f"Expected reference_qvels shape {(batch_size, horizon, self.model.nv)}, got {reference_qvels.shape}"
            )
        out_states, out_sensors = self._rollout_obj.rollout(
            self._models,
            self._datas,
            x0_batched,
            controls,
            initial_policy_state,
            reference_qvels,
        )
        return np.array(out_states), np.array(out_sensors), None

    def update(self, num_threads: int) -> None:
        self.num_threads = num_threads
        self._rollout_obj.close()
        g1_extensions = require_g1_extensions()
        self._policy_state_dim = g1_extensions.g1_wbc_policy_state_dim()
        self._rollout_obj = g1_extensions.G1WBCRollout(
            nthread=num_threads,
            cutoff_time=self.cutoff_time,
            policy_path=self.policy_path,
        )
        self._models, self._datas = make_model_data_pairs(self.model, num_threads)
        self.reference_qvels = None


def build_foot_geom_side(model: mujoco.MjModel) -> dict[int, int]:
    mapping = {}
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name.startswith(CONTACT_GEOM_PREFIXES[0]):
            mapping[geom_id] = 0
        elif name.startswith(CONTACT_GEOM_PREFIXES[1]):
            mapping[geom_id] = 1
    return mapping


_build_foot_geom_side = build_foot_geom_side


def contact_sensor_values(model: mujoco.MjModel, data: mujoco.MjData, foot_geom_side: dict[int, int]) -> np.ndarray:
    mask = np.zeros(2, dtype=np.float64)
    forces = np.zeros(2, dtype=np.float64)
    force6 = np.zeros(6, dtype=np.float64)
    for contact_idx in range(data.ncon):
        contact = data.contact[contact_idx]
        sides = []
        if int(contact.geom1) in foot_geom_side:
            sides.append(foot_geom_side[int(contact.geom1)])
        if int(contact.geom2) in foot_geom_side:
            sides.append(foot_geom_side[int(contact.geom2)])
        if not sides:
            continue
        mujoco.mj_contactForce(model, data, contact_idx, force6)
        force_mag = float(np.linalg.norm(force6[:3]))
        for side in sides:
            mask[side] = 1.0
            forces[side] += force_mag
    return np.concatenate([mask, forces])
