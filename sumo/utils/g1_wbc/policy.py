"""Lightweight data contracts for G1 WBC tracking references.

Policy inference and observation construction live in the native
``g1_extensions`` backend to match SUMO's existing G1 task path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReferenceFrame:
    """Reference data needed to score a WBC tracking step."""

    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    anchor_ang_vel_w: np.ndarray

    @property
    def command(self) -> np.ndarray:
        return np.concatenate([self.joint_pos, self.joint_vel])
