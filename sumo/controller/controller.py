# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Controller with G1 rollout backend support."""

from __future__ import annotations

from judo.controller.controller import Controller, ControllerConfig, make_spline
from judo.controller.controller import make_controller as _judo_make_controller
from omegaconf import DictConfig

from sumo.controller.g1_wbc_controller import G1WBCController
from sumo.utils.g1_wbc.constants import WBC_TASK_NAMES
from sumo.utils.g1_wbc.rollout import G1WBCRolloutBackend
from sumo.utils.mujoco import G1RolloutBackend


def make_controller(
    init_task: str,
    init_optimizer: str,
    task_registration_cfg: DictConfig | None = None,
    optimizer_registration_cfg: DictConfig | None = None,
) -> Controller:
    """Make a controller with G1 backend support."""
    controller_cls = G1WBCController if init_task in WBC_TASK_NAMES else Controller
    return _judo_make_controller(
        init_task=init_task,
        init_optimizer=init_optimizer,
        task_registration_cfg=task_registration_cfg,
        optimizer_registration_cfg=optimizer_registration_cfg,
        controller_cls=controller_cls,
        rollout_backend_registry={"mujoco_g1": G1RolloutBackend, "mujoco_g1_wbc": G1WBCRolloutBackend},
    )


__all__ = ["Controller", "ControllerConfig", "G1WBCController", "make_controller", "make_spline"]
