import os

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
POLICY_DIR = os.path.join(_PACKAGE_DIR, "policy")
os.environ.setdefault("G1_EXTENSIONS_POLICY_DIR", POLICY_DIR)

from g1_extensions._g1_extensions import (
    G1Rollout,
    G1WBCRollout,
    debug_g1_wbc_policy_step,
    g1_wbc_policy_state_dim,
    rollout,
    sim,
    sim_g1,
    sim_g1_wbc,
)
