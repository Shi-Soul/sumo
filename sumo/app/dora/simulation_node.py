# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from judo.app.dora.simulation_node import SimulationNode as JudoSimulationNode

import sumo.tasks  # noqa: F401 -- register all sumo tasks
from sumo.app.dora.g1_simulation import G1Simulation
from sumo.app.dora.g1_wbc_simulation import G1WBCSimulation


class SimulationNode(JudoSimulationNode):
    """Simulation node with G1 backend support."""

    def __init__(self, init_task: str = "spot_box_push", **kwargs) -> None:
        kwargs.setdefault("backend_registry", {"mujoco_g1": G1Simulation, "mujoco_g1_wbc": G1WBCSimulation})
        super().__init__(init_task=init_task, **kwargs)


__all__ = ["G1Simulation", "G1WBCSimulation", "SimulationNode"]
