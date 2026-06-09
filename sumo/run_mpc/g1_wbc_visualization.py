"""Viser overlays for G1 WBC reference/refined trajectory comparison."""

from __future__ import annotations

import mujoco
import numpy as np

from sumo.utils.g1_wbc.constants import EE_REWARD_BODY_NAMES
from sumo.utils.g1_wbc.reference import controls_to_qpos, normalize_controls


class G1WBCReferenceOverlay:
    """Small marker overlay for reference and refined WBC body poses."""

    def __init__(self, server, model: mujoco.MjModel) -> None:
        self._model = model
        self._data = mujoco.MjData(model)
        self._body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in EE_REWARD_BODY_NAMES]
        self._reference = [
            server.scene.add_icosphere(
                f"g1_wbc_reference/{name}",
                radius=0.035,
                color=(0.0, 0.65, 1.0),
                position=(0.0, 0.0, 0.0),
            )
            for name in EE_REWARD_BODY_NAMES
        ]
        self._refined = [
            server.scene.add_icosphere(
                f"g1_wbc_refined/{name}",
                radius=0.025,
                color=(1.0, 0.8, 0.0),
                position=(0.0, 0.0, 0.0),
            )
            for name in EE_REWARD_BODY_NAMES
        ]

    def set_controls(self, reference_control: np.ndarray, refined_control: np.ndarray | None = None) -> None:
        ref_pos = self._body_positions(reference_control)
        for handle, pos in zip(self._reference, ref_pos, strict=True):
            handle.position = tuple(pos)
        if refined_control is not None:
            refined_pos = self._body_positions(refined_control)
            for handle, pos in zip(self._refined, refined_pos, strict=True):
                handle.position = tuple(pos)

    def _body_positions(self, control: np.ndarray) -> np.ndarray:
        control = normalize_controls(np.asarray(control, dtype=np.float64))
        self._data.qpos[:] = controls_to_qpos(control)
        self._data.qvel[:] = 0.0
        mujoco.mj_forward(self._model, self._data)
        return np.asarray(self._data.xpos[self._body_ids], dtype=np.float64).copy()


__all__ = ["G1WBCReferenceOverlay"]
