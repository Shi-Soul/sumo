"""Pure Python MuJoCo rollout backend for G1 WBC tracking policies."""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np
from judo.utils.mj_rollout_backend import make_model_data_pairs
from judo.utils.rollout_backend import RolloutBackend
from mujoco import MjModel

from sumo.utils.g1_wbc.constants import CONTACT_GEOM_PREFIXES, DEFAULT_POLICY_VARIANT
from sumo.utils.g1_wbc.policy import TrackingPolicyRuntime
from sumo.utils.g1_wbc.reference import build_reference_frames, normalize_controls


class G1WBCRolloutBackend(RolloutBackend):
    """Rollout backend that executes the wbteleop ONNX policy on refined references."""

    def __init__(
        self,
        model: MjModel,
        num_threads: int,
        cutoff_time: float = 0.2,
        policy: str | Path | None = None,
    ) -> None:
        self.model = model
        self.num_threads = num_threads
        self.cutoff_time = cutoff_time
        self.policy = policy or os.environ.get("SUMO_G1_WBC_POLICY", DEFAULT_POLICY_VARIANT)
        self._models, self._datas = make_model_data_pairs(model, num_threads)
        self._runtimes = [TrackingPolicyRuntime(m, self.policy) for m in self._models]
        self._foot_geom_side = _build_foot_geom_side(model)

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

        nstate = self.model.nq + self.model.nv
        states = np.zeros((batch_size, horizon + 1, nstate), dtype=np.float64)
        sensors = np.zeros((batch_size, horizon, 4), dtype=np.float64)
        x0_batched = np.tile(x0, (batch_size, 1))

        for i, (model, data, runtime) in enumerate(zip(self._models, self._datas, self._runtimes, strict=True)):
            runtime.reset()
            mujoco.mj_setState(model, data, x0_batched[i], mujoco.mjtState.mjSTATE_QPOS | mujoco.mjtState.mjSTATE_QVEL)
            mujoco.mj_forward(model, data)
            states[i, 0, : model.nq] = data.qpos
            states[i, 0, model.nq :] = data.qvel
            ref_frames = build_reference_frames(model, controls[i], model.opt.timestep)
            for t, ref in enumerate(ref_frames):
                data.ctrl[:] = runtime.step(data, ref)
                mujoco.mj_step(model, data)
                states[i, t + 1, : model.nq] = data.qpos
                states[i, t + 1, model.nq :] = data.qvel
                sensors[i, t] = contact_sensor_values(model, data, self._foot_geom_side)
        return states, sensors, None

    def update(self, num_threads: int) -> None:
        self.num_threads = num_threads
        self._models, self._datas = make_model_data_pairs(self.model, num_threads)
        self._runtimes = [TrackingPolicyRuntime(m, self.policy) for m in self._models]


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
