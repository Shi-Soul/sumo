"""Viser player for saved G1 WBC evaluation episodes."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import mujoco
import numpy as np
import tyro
import viser
from judo.visualizers.model import ViserMjModel

from sumo.utils.g1_wbc.constants import G1_XML_PATH
from sumo.utils.g1_wbc.reference import controls_to_qpos


@dataclass
class G1WBCPlayResultConfig:
    """Configuration for replaying a saved G1 WBC HDF5 episode."""

    result_file: str
    group: str = ""
    port: int = 8080
    speed: float = 1.0
    stride: int = 1
    start_frame: int = 0
    end_frame: int = 0
    loop: bool = True
    show_reference: bool = True
    show_refined: bool = True
    allow_stale_refined: bool = False
    show_collision: bool = False
    dry_run: bool = False


class ControlSkeletonOverlay:
    """Lightweight body skeleton for reference/refined controls."""

    def __init__(
        self,
        server: viser.ViserServer,
        model: mujoco.MjModel,
        *,
        name: str,
        color: tuple[float, float, float],
        radius: float = 0.018,
        visible: bool = True,
    ) -> None:
        self._model = model
        self._data = mujoco.MjData(model)
        self._body_ids = list(range(1, model.nbody))
        self._line_pairs = [
            (int(model.body_parentid[body_id]), int(body_id))
            for body_id in self._body_ids
            if int(model.body_parentid[body_id]) > 0
        ]
        self._points = []
        for body_id in self._body_ids:
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
            handle = server.scene.add_icosphere(
                f"{name}/bodies/{body_name}",
                radius=radius,
                color=color,
                position=(0.0, 0.0, 0.0),
                visible=visible,
            )
            self._points.append(handle)
        initial_segments = np.zeros((max(len(self._line_pairs), 1), 2, 3), dtype=np.float64)
        self._lines = server.scene.add_line_segments(
            f"{name}/links",
            initial_segments,
            color,
            line_width=2.0,
            visible=visible,
        )

    def set_visible(self, visible: bool) -> None:
        for handle in self._points:
            handle.visible = visible
        self._lines.visible = visible

    def set_control(self, control: np.ndarray) -> None:
        self._data.qpos[:] = controls_to_qpos(control)
        self._data.qvel[:] = 0.0
        mujoco.mj_forward(self._model, self._data)

        positions = np.asarray(self._data.xpos, dtype=np.float64)
        for handle, body_id in zip(self._points, self._body_ids, strict=True):
            handle.position = tuple(positions[body_id])

        if self._line_pairs:
            segments = np.asarray([[positions[parent], positions[child]] for parent, child in self._line_pairs])
            self._lines.points = segments


def main(config: G1WBCPlayResultConfig | None = None) -> None:
    config = tyro.cli(G1WBCPlayResultConfig) if config is None else config
    episode = load_episode(Path(config.result_file), config.group)
    if config.dry_run:
        print_episode_summary(episode)
        return
    play_episode(config, episode)


def load_episode(path: Path, group_name: str = "") -> dict:
    """Load one episode group from a G1 WBC eval HDF5 file."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Result file does not exist: {path}")

    with h5py.File(path, "r") as f:
        group_path = _resolve_group_path(f, group_name)
        group = f[group_path]
        arrays = {key: np.asarray(value) for key, value in group.items() if isinstance(value, h5py.Dataset)}
        attrs = {key: _python_scalar(value) for key, value in group.attrs.items()}
        file_attrs = {key: _python_scalar(value) for key, value in f.attrs.items()}

    if "qpos_traj" not in arrays:
        raise ValueError(f"Group '{group_path}' in {path} does not contain qpos_traj")
    if arrays["qpos_traj"].ndim != 2 or arrays["qpos_traj"].shape[1] != 36:
        raise ValueError(f"Expected qpos_traj shape (T, 36), got {arrays['qpos_traj'].shape}")

    return {
        "path": path,
        "group_path": group_path,
        "arrays": arrays,
        "attrs": attrs,
        "file_attrs": file_attrs,
    }


def _resolve_group_path(file: h5py.File, group_name: str) -> str:
    if group_name:
        cleaned = group_name.strip("/")
        if cleaned not in file:
            raise ValueError(f"Group '{group_name}' does not exist in {file.filename}")
        return cleaned

    candidates: list[str] = []

    def visitor(name: str, obj) -> None:
        if isinstance(obj, h5py.Group) and "qpos_traj" in obj:
            candidates.append(name)

    file.visititems(visitor)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(f"No group containing qpos_traj found in {file.filename}")
    raise ValueError(f"Multiple episode groups found; pass --group. Candidates: {candidates}")


def _python_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def print_episode_summary(episode: dict) -> None:
    arrays = episode["arrays"]
    attrs = episode["attrs"]
    print(f"file: {episode['path']}")
    print(f"group: {episode['group_path']}")
    print(f"frames: {len(arrays['qpos_traj'])}")
    if "time_traj" in arrays and len(arrays["time_traj"]):
        print(f"time: {arrays['time_traj'][0]:.3f}s -> {arrays['time_traj'][-1]:.3f}s")
    method = attrs.get("method")
    if method is not None:
        print(f"method: {method}")
    for key in ("local_ee_rmse", "joint_rmse", "root_pos_rmse", "root_ori_mean", "contact_mismatch_rate", "fallen"):
        if key in attrs:
            print(f"{key}: {attrs[key]}")
    for key, value in sorted(arrays.items()):
        print(f"{key}: shape={value.shape}, dtype={value.dtype}")


def play_episode(config: G1WBCPlayResultConfig, episode: dict) -> None:
    arrays = episode["arrays"]
    qpos_traj = arrays["qpos_traj"]
    qvel_traj = arrays.get("qvel_traj")
    time_traj = arrays.get("time_traj")
    reference_controls = arrays.get("reference_controls")
    refined_controls = arrays.get("refined_controls")
    refined_time_traj = arrays.get("refined_time_traj")
    stale_refined = refined_controls is not None and refined_time_traj is None
    if stale_refined and not config.allow_stale_refined:
        print(
            "Warning: this HDF5 has old refined_controls without refined_time_traj; "
            "hiding the yellow refined overlay because it is not the real per-step MPC command."
        )
        refined_controls = None

    start = int(np.clip(config.start_frame, 0, len(qpos_traj) - 1))
    end = len(qpos_traj) if config.end_frame <= 0 else int(np.clip(config.end_frame, start + 1, len(qpos_traj)))
    stride = max(1, int(config.stride))

    spec = mujoco.MjSpec.from_file(str(G1_XML_PATH))
    model = spec.compile()
    data = mujoco.MjData(model)
    server = viser.ViserServer(port=config.port)
    robot = ViserMjModel(
        server,
        spec,
        geom_exclude_substring="" if config.show_collision else "collision",
    )
    reference_overlay = ControlSkeletonOverlay(
        server,
        model,
        name="reference",
        color=(0.0, 0.65, 1.0),
        radius=0.016,
        visible=config.show_reference and reference_controls is not None,
    )
    refined_overlay = ControlSkeletonOverlay(
        server,
        model,
        name="refined",
        color=(1.0, 0.75, 0.0),
        radius=0.014,
        visible=config.show_refined and refined_controls is not None,
    )

    play = server.gui.add_checkbox("Play", initial_value=True)
    loop = server.gui.add_checkbox("Loop", initial_value=bool(config.loop))
    show_reference = server.gui.add_checkbox(
        "Reference",
        initial_value=config.show_reference and reference_controls is not None,
        disabled=reference_controls is None,
    )
    show_refined = server.gui.add_checkbox(
        "Refined",
        initial_value=config.show_refined and refined_controls is not None,
        disabled=refined_controls is None,
    )
    speed = server.gui.add_slider("Speed", min=0.05, max=4.0, step=0.05, initial_value=max(0.05, float(config.speed)))
    frame = server.gui.add_slider("Frame", min=start, max=end - 1, step=stride, initial_value=start)

    def render(frame_index: int) -> None:
        idx = int(np.clip(frame_index, start, end - 1))
        data.qpos[:] = qpos_traj[idx]
        if qvel_traj is not None and len(qvel_traj) > idx:
            data.qvel[:] = qvel_traj[idx]
        else:
            data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        robot.set_data(data)

        time_s = float(time_traj[idx]) if time_traj is not None and len(time_traj) > idx else float(idx)
        reference_overlay.set_visible(bool(show_reference.value) and reference_controls is not None)
        refined_overlay.set_visible(bool(show_refined.value) and refined_controls is not None)
        if show_reference.value and reference_controls is not None:
            ref = _sample_by_frame_or_time(reference_controls, idx, time_s, time_traj, len(qpos_traj))
            reference_overlay.set_control(ref)
        if show_refined.value and refined_controls is not None:
            refined = _sample_by_frame_or_time(refined_controls, idx, time_s, refined_time_traj, len(qpos_traj))
            refined_overlay.set_control(refined)

    @frame.on_update
    def _(_) -> None:
        render(int(frame.value))

    @show_reference.on_update
    def _(_) -> None:
        render(int(frame.value))

    @show_refined.on_update
    def _(_) -> None:
        render(int(frame.value))

    render(start)
    print(f"Viser running at: http://localhost:{config.port}")
    print(f"Loaded {episode['path']} group={episode['group_path']} frames={len(qpos_traj)}")

    while True:
        if not play.value:
            time.sleep(0.05)
            continue
        idx = int(frame.value)
        render(idx)
        next_idx = idx + stride
        if next_idx >= end:
            if loop.value:
                next_idx = start
            else:
                play.value = False
                next_idx = end - 1
        frame.value = next_idx
        time.sleep(_frame_delay(time_traj, idx, next_idx, speed=float(speed.value)))


def _sample_by_frame_or_time(
    values: np.ndarray,
    frame_index: int,
    time_s: float,
    value_times: np.ndarray | None,
    num_frames: int,
) -> np.ndarray:
    if len(values) == num_frames:
        return values[int(np.clip(frame_index, 0, len(values) - 1))]
    if value_times is not None and len(value_times) == len(values):
        idx = int(np.searchsorted(value_times, time_s, side="left"))
        if idx >= len(values):
            idx = len(values) - 1
        if idx > 0 and abs(float(value_times[idx - 1]) - time_s) < abs(float(value_times[idx]) - time_s):
            idx -= 1
        return values[idx]
    return values[int(np.clip(frame_index, 0, len(values) - 1))]


def _frame_delay(time_traj: np.ndarray | None, idx: int, next_idx: int, *, speed: float) -> float:
    if time_traj is None or len(time_traj) <= max(idx, next_idx):
        return 1.0 / (60.0 * max(speed, 1e-6))
    dt = abs(float(time_traj[next_idx]) - float(time_traj[idx]))
    if dt <= 0.0:
        dt = 1.0 / 60.0
    return min(0.25, dt / max(speed, 1e-6))


if __name__ == "__main__":
    main()
