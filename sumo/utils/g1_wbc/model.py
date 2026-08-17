"""MuJoCo model helpers for tracking_bfm-compatible G1 WBC simulation."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from sumo.utils.g1_wbc.constants import G1_XML_PATH, SIM_DT


def configure_wbc_model(model: mujoco.MjModel) -> mujoco.MjModel:
    """Apply tracking_bfm runtime simulator semantics to a compiled G1 model."""
    model.opt.timestep = SIM_DT
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    model.opt.jacobian = mujoco.mjtJacobian.mjJAC_AUTO
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.impratio = 1.0
    model.opt.iterations = 10
    model.opt.ls_iterations = 20
    model.opt.tolerance = 1e-8
    model.opt.ls_tolerance = 0.01
    model.opt.gravity[:] = (0.0, 0.0, -9.81)

    # tracking_bfm position actuators intentionally allow targets beyond joint
    # limits; force is bounded by actuator_forcerange instead.
    model.actuator_ctrllimited[:] = 0
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        stiffness = float(model.actuator_gainprm[actuator_id, 0])
        effort = float(np.max(np.abs(model.actuator_forcerange[actuator_id])))
        if joint_id >= 0 and stiffness > 0.0 and effort > 0.0:
            delta = effort / stiffness
            model.actuator_ctrlrange[actuator_id, 0] = model.jnt_range[joint_id, 0] - delta
            model.actuator_ctrlrange[actuator_id, 1] = model.jnt_range[joint_id, 1] + delta
    return model


def load_wbc_model(model_path: str | Path = G1_XML_PATH) -> mujoco.MjModel:
    """Load a G1 model and apply WBC-only tracking_bfm simulator settings."""
    return configure_wbc_model(mujoco.MjModel.from_xml_path(str(model_path)))


__all__ = ["configure_wbc_model", "load_wbc_model"]
