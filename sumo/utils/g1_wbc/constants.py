"""Standalone constants for SUMO G1 WBC tracking.

These values mirror the tracking_bfm G1 wbteleop contract without importing
tracking_bfm at runtime.
"""

from __future__ import annotations

from pathlib import Path

from sumo import MODEL_PATH, PACKAGE_ROOT

REPO_ROOT = PACKAGE_ROOT.parents[1]

G1_XML_PATH = MODEL_PATH / "xml" / "g1" / "g1.xml"
DEFAULT_MOTION_FILE = REPO_ROOT / "wxy" / "smoke_motion" / "g1_stand_mujoco.npz"

WXY_POLICY_VARIANTS = {
    "bc": REPO_ROOT / "wxy" / "0608_ckpt_bc" / "deploy_model_8000.onnx",
    "bcrl": REPO_ROOT / "wxy" / "0608_ckpt_bcrl" / "deploy_model_16000.onnx",
}
DEFAULT_POLICY_VARIANT = "bcrl"

POLICY_DT = 0.02
SIM_DT = 0.005
POLICY_DECIMATION = 4
OBS_DIM = 886
ACTION_DIM = 29
TASK_CONTROL_DIM = 36  # root pos(3) + root quat(4) + 29 joints

ROOT_POS_SLICE = slice(0, 3)
ROOT_QUAT_SLICE = slice(3, 7)
JOINT_POS_SLICE = slice(7, 36)

ISAACLAB_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

MUJOCO_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

ISAACLAB_BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)

MUJOCO_BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
)

ISAACLAB_TO_MUJOCO_JOINT_REINDEX = tuple(ISAACLAB_JOINT_NAMES.index(name) for name in MUJOCO_JOINT_NAMES)
ISAACLAB_TO_MUJOCO_BODY_REINDEX = tuple(ISAACLAB_BODY_NAMES.index(name) for name in MUJOCO_BODY_NAMES)

DEFAULT_JOINT_POSITIONS = (
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,
    0.0,
    0.0,
    0.0,
    0.2,
    0.2,
    0.0,
    0.6,
    0.0,
    0.0,
    0.0,
    0.2,
    -0.2,
    0.0,
    0.6,
    0.0,
    0.0,
    0.0,
)

ACTION_SCALE_BY_PATTERN = {
    ".*_elbow_joint": 0.43857731392336724,
    ".*_shoulder_pitch_joint": 0.43857731392336724,
    ".*_shoulder_roll_joint": 0.43857731392336724,
    ".*_shoulder_yaw_joint": 0.43857731392336724,
    ".*_wrist_roll_joint": 0.43857731392336724,
    ".*_hip_pitch_joint": 0.5475464629911068,
    ".*_hip_yaw_joint": 0.5475464629911068,
    "waist_yaw_joint": 0.5475464629911068,
    ".*_hip_roll_joint": 0.35066146637882434,
    ".*_knee_joint": 0.35066146637882434,
    ".*_wrist_pitch_joint": 0.07450087032950714,
    ".*_wrist_yaw_joint": 0.07450087032950714,
    "waist_pitch_joint": 0.43857731392336724,
    "waist_roll_joint": 0.43857731392336724,
    ".*_ankle_pitch_joint": 0.43857731392336724,
    ".*_ankle_roll_joint": 0.43857731392336724,
}

LIMB_EE_BODY_NAMES = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)
UPPER_EE_BODY_NAMES = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)
EE_REWARD_BODY_NAMES = (
    "torso_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)
ANCHOR_BODY_NAME = "pelvis"
CONTACT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
CONTACT_GEOM_PREFIXES = ("left_foot", "right_foot")
CONTACT_SENSOR_DIM = 4
UPPER_EE_SENSOR_START = CONTACT_SENSOR_DIM
UPPER_EE_SENSOR_DIM = 3 * len(UPPER_EE_BODY_NAMES)
ACTION_SENSOR_START = UPPER_EE_SENSOR_START + UPPER_EE_SENSOR_DIM
ACTION_SENSOR_DIM = ACTION_DIM
WBC_ROLLOUT_SENSOR_DIM = CONTACT_SENSOR_DIM + UPPER_EE_SENSOR_DIM + ACTION_SENSOR_DIM

WBC_TASK_NAMES = ("g1_wbc_ee", "g1_wbc_joint")


def resolve_policy_path(policy: str | Path) -> Path:
    """Resolve a policy variant name or explicit ONNX path."""
    policy_path = WXY_POLICY_VARIANTS.get(str(policy), Path(policy))
    return Path(policy_path).expanduser().resolve()
