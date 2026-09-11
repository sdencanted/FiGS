"""Interactive Viser editor for FiGS drone-course keyframes.

Run from an environment with FiGS and viser installed, for example::

    python -m figs.utilities.course_editor --input configs/courses/infinity.json
    python -m figs.utilities.course_editor --input configs/courses/infinity.json \
        --gsplat-config gsplats/workspace/outputs/my_scene/config.yml
    python -m figs.utilities.course_editor --output configs/courses/new_course.json

The first command edits the supplied file in place when ``Save JSON`` is
pressed.  Supplying ``--output`` writes to that path instead.  A new course is
created when ``--input`` is omitted.

Timing defaults to automatic estimation and optimization. Manual waypoint
times and a fixed total duration are available in the Timing folder. Solve
trajectory previews arrival times and sampled control checks without a scene.
Saved timing settings also apply to downstream trajectory generation.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import threading
import time
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

from figs.utilities.course_timing import TIMING_MODES, estimate_times, timing_settings


_ROWS = ("x", "y", "z", "yaw")
_MIN_DERIVATIVES = 4  # d0 through d3 are always exposed in the editor.


def _to_editor_value(row: int, value: float) -> float:
    """Convert canonical FLU values to the editor's rendered FRD frame."""
    return -value if row in (1, 2, 3) else value


def _from_editor_value(row: int, value: float) -> float:
    """Convert a rendered FRD value back to canonical JSON coordinates."""
    return -value if row in (1, 2, 3) else value


def _new_course() -> dict[str, Any]:
    """Return a minimal, solver-compatible course with two editable poses."""
    frame = {
        "t": 0.0,
        "fo": [[0.0, 0.0, 0.0, None] for _ in _ROWS],
    }
    return {
        "waypoints": {
            "Nco": 6,
            "timing": timing_settings(),
            "keyframes": {"fo0": frame, "fo1": {**frame, "t": 1.0, "fo": [row.copy() for row in frame["fo"]]}},
        },
        "forces": None,
    }


def _as_number_or_none(value: Any, context: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a number or null")
    if not math.isfinite(float(value)):
        raise ValueError(f"{context} must be finite")
    return float(value)


def _normalise_course(course: MutableMapping[str, Any]) -> tuple[dict[str, Any], int]:
    """Validate and normalise editable flat-output matrices in place."""
    try:
        waypoints = course["waypoints"]
        keyframes = waypoints["keyframes"]
    except (KeyError, TypeError) as exc:
        raise ValueError("course must contain waypoints.keyframes") from exc
    if not isinstance(waypoints, MutableMapping) or not isinstance(keyframes, MutableMapping) or not keyframes:
        raise ValueError("waypoints.keyframes must be a non-empty object")
    waypoints["timing"] = timing_settings(waypoints.get("timing"))

    derivative_count = _MIN_DERIVATIVES
    for name, keyframe in keyframes.items():
        if not isinstance(keyframe, MutableMapping):
            raise ValueError(f"keyframe {name!r} must be an object")
        keyframe["t"] = _as_number_or_none(keyframe.get("t", 0.0), f"keyframe {name}.t")
        if keyframe["t"] is None:
            raise ValueError(f"keyframe {name}.t must be a number")
        fo = keyframe.get("fo")
        if not isinstance(fo, list) or len(fo) != len(_ROWS):
            raise ValueError(f"keyframe {name}.fo must contain x, y, z, and yaw rows")
        for row_index, row in enumerate(fo):
            if not isinstance(row, list) or not row:
                raise ValueError(f"keyframe {name}.fo[{row_index}] must be a non-empty array")
            derivative_count = max(derivative_count, len(row))
            fo[row_index] = [_as_number_or_none(value, f"keyframe {name}.fo[{row_index}]") for value in row]

    for keyframe in keyframes.values():
        for row in keyframe["fo"]:
            row.extend([None] * (derivative_count - len(row)))
    return dict(course), derivative_count


def _yaw_to_wxyz(yaw: float) -> tuple[float, float, float, float]:
    """Return the FRD pose quaternion: rendered yaw plus a fixed 180° roll."""
    half_yaw = yaw / 2.0
    # q = q_yaw * q_roll(pi), so the body frame is forward-right-down.
    return (0.0, math.cos(half_yaw), math.sin(half_yaw), 0.0)


def _wxyz_to_yaw(wxyz: tuple[float, float, float, float] | np.ndarray) -> float:
    w, x, y, z = (float(value) for value in wxyz)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _unwrap_angle(angle: float, reference: float) -> float:
    """Return the 2π-equivalent angle closest to ``reference``."""
    return reference + (angle - reference + math.pi) % (2.0 * math.pi) - math.pi


class CourseEditor:
    """Own the course model and keep Viser handles synchronised with it."""

    def __init__(
        self,
        course: dict[str, Any],
        derivative_count: int,
        output_path: Path,
        host: str,
        port: int,
        server: Any | None = None,
        scene_scale: float = 1.0,
        gsplat: Any | None = None,
    ) -> None:
        import numpy as np
        import viser

        self.np = np
        self.course = course
        self.keyframes: MutableMapping[str, Any] = self.course["waypoints"]["keyframes"]
        self.derivative_count = derivative_count
        self.output_path = output_path
        self.server = server if server is not None else viser.ViserServer(host=host, port=port)
        self.scene_scale = scene_scale
        self.gsplat = gsplat
        self.timing = self.course["waypoints"].setdefault("timing", timing_settings())
        self._manual_arrivals = {name: frame["t"] for name, frame in self.keyframes.items()}
        self.optimized_times: list[float] | None = None
        self._revision = 0
        self.lock = threading.RLock()
        self._syncing_gui = False
        self.pose_handles: dict[str, Any] = {}
        self.simulated_tro: Any | None = None
        self.simulated_xro: Any | None = None
        self.simulated_rgb: Any | None = None
        self.simulation_hz: int | None = None
        self.show_simulated_trajectory = False
        self.speed_markers_handle: Any | None = None
        self.selection = next(iter(self.keyframes))

        self.server.scene.add_grid(
            "/course/grid", width=30.0 * scene_scale, height=30.0 * scene_scale,
            plane="xy", cell_size=scene_scale, cell_thickness=1.0,
        )
        self._build_gui(viser)
        self._redraw_scene()
        print(f"Course editor: http://{host}:{port}")
        print(f"Save target: {self.output_path}")

    def _build_gui(self, viser: Any) -> None:
        self.server.gui.add_markdown("# Drone course editor\nDrag the selected pose gizmo to edit x/y/z and yaw. The frame axes show every keypoint pose.")
        with self.server.gui.add_folder("Course"):
            self.selected_gui = self.server.gui.add_dropdown("Selected keypoint", tuple(self.keyframes), initial_value=self.selection)
            self.selected_gui.on_update(lambda _: self._select(self.selected_gui.value))
            self.add_button = self.server.gui.add_button("Add keypoint")
            self.add_button.on_click(lambda _: self._add_keypoint())
            self.delete_button = self.server.gui.add_button("Delete selected", color="red")
            self.delete_button.on_click(lambda _: self._delete_selected())
            self.save_button = self.server.gui.add_button("Save JSON", color="green")
            self.save_button.on_click(lambda _: self._save())
            self.solve_button = self.server.gui.add_button("Solve trajectory")
            self.solve_button.on_click(lambda _: self._solve_timing())
            self.simulate_button = self.server.gui.add_button(
                "Run FiGS simulation", disabled=self.gsplat is None,
                hint="Run the Section 4 FiGS simulator using the current course and Gaussian splat.",
            )
            self.simulate_button.on_click(lambda _: self._run_simulation())
            self.route_toggle_button = self.server.gui.add_button("Show simulated trajectory", disabled=True)
            self.route_toggle_button.on_click(lambda _: self._toggle_route_display())
            self.status = self.server.gui.add_markdown("")

        with self.server.gui.add_folder("Timing"):
            self.timing_mode_gui = self.server.gui.add_dropdown(
                "Timing mode", tuple(TIMING_MODES.values()), initial_value=TIMING_MODES[self.timing["mode"]],
            )
            self.timing_mode_gui.on_update(lambda _: self._set_timing())
            self.aggressiveness_gui = self.server.gui.add_number(
                "Aggressiveness", initial_value=self.timing["aggressiveness"], min=0.01, step=0.1,
                hint="1 is the default. Higher values favor faster flight; this is not a speed limit.",
            )
            self.aggressiveness_gui.on_update(lambda _: self._set_timing())
            self.total_duration_gui = self.server.gui.add_number(
                "Total duration (s)", initial_value=self.timing["total_duration"], min=0.01, step=0.5,
            )
            self.total_duration_gui.on_update(lambda _: self._set_timing())
            self.timing_description = self.server.gui.add_markdown("")
            self.arrival_times_gui = self.server.gui.add_markdown("")
            self.feasibility_gui = self.server.gui.add_markdown("")

        with self.server.gui.add_folder("Simulation display"):
            self.speed_marker_period_gui = self.server.gui.add_number(
                "Speed marker period (s)", initial_value=1.0, min=0.01, step=0.1,
                hint="Time between simulated-trajectory speed markers.",
            )
            self.speed_marker_period_gui.on_update(lambda _: self._refresh_simulation_render())
            self.server.gui.add_markdown("Speed markers: blue = slower, red = faster.")
            self.video_output_gui = self.server.gui.add_text(
                "Simulation video (.mp4)", initial_value=str(self.output_path.with_suffix(".mp4")),
            )
            self.export_video_button = self.server.gui.add_button("Export simulation video", disabled=True)
            self.export_video_button.on_click(lambda _: self._export_simulation_video())

        with self.server.gui.add_folder("Selected keypoint"):
            self.name_gui = self.server.gui.add_text("Keypoint name", initial_value=self.selection)
            self.name_gui.on_update(lambda _: self._rename_selected())
            with self.server.gui.add_folder("Order"):
                self.move_first_button = self.server.gui.add_button("Move to first")
                self.move_first_button.on_click(lambda _: self._move_selected(0, first=True))
                self.move_earlier_button = self.server.gui.add_button("Move earlier")
                self.move_earlier_button.on_click(lambda _: self._move_selected(-1))
                self.move_later_button = self.server.gui.add_button("Move later")
                self.move_later_button.on_click(lambda _: self._move_selected(1))
                self.move_last_button = self.server.gui.add_button("Move to last")
                self.move_last_button.on_click(lambda _: self._move_selected(-1, last=True))
            self.time_gui = self.server.gui.add_number("Time (s)", initial_value=0.0, step=0.01)
            self.time_gui.on_update(lambda _: self._set_time(float(self.time_gui.value)))
            self.value_guis: list[list[tuple[Any, Any]]] = []
            for row_index, row_name in enumerate(_ROWS):
                row_guis: list[tuple[Any, Any]] = []
                with self.server.gui.add_folder(row_name):
                    for derivative in range(self.derivative_count):
                        specified = self.server.gui.add_checkbox(f"d{derivative} specified", initial_value=True)
                        value = self.server.gui.add_number(f"d{derivative} value", initial_value=0.0, step=0.01)
                        specified.on_update(lambda _, r=row_index, d=derivative: self._set_specified(r, d))
                        value.on_update(lambda _, r=row_index, d=derivative: self._set_value(r, d))
                        row_guis.append((specified, value))
                self.value_guis.append(row_guis)
        self._refresh_gui()

    def _refresh_timing(self) -> None:
        mode = self.timing["mode"]
        self.time_gui.disabled = mode != "manual"
        self.time_gui.label = "Arrival time (s)" if mode == "manual" else "Initial estimate (s)"
        self.aggressiveness_gui.visible = mode == "automatic"
        self.total_duration_gui.visible = mode == "total_duration"
        descriptions = {
            "automatic": "Arrival times are chosen automatically. Estimates update with the course; aggressiveness balances smoothness and duration.",
            "manual": "Advanced: fix each arrival time. Start at 0 and increase times in keypoint order. The solver optimizes smoothness with these times fixed.",
            "total_duration": "Fix the total flight duration. The solver distributes this time between keypoints to minimize the smoothness cost.",
        }
        self.timing_description.content = descriptions[mode]
        try:
            estimates = estimate_times(self.keyframes, self.timing)
        except ValueError as exc:
            self.arrival_times_gui.content = f"**Timing needs attention:** {exc}"
            return
        if mode != "manual":
            for frame, estimate in zip(self.keyframes.values(), estimates):
                frame["t"] = estimate
        heading = "Fixed arrival (s)" if mode == "manual" else "Initial estimate (s)"
        rows = [f"| Keypoint | {heading} | Solved arrival (s) |", "|---|---:|---:|"]
        for index, (name, estimate) in enumerate(zip(self.keyframes, estimates)):
            solved = "—" if self.optimized_times is None else f"{self.optimized_times[index]:.3f}"
            safe_name = name.replace("|", "\\|").replace("\n", " ")
            rows.append(f"| {safe_name} | {estimate:.3f} | {solved} |")
        self.arrival_times_gui.content = "\n".join(rows)

    def _set_timing(self) -> None:
        if self._syncing_gui:
            return
        with self.lock:
            value = {
                "mode": next(mode for mode, label in TIMING_MODES.items() if label == self.timing_mode_gui.value),
                "aggressiveness": self.aggressiveness_gui.value,
                "total_duration": self.total_duration_gui.value,
            }
            try:
                settings = timing_settings(value)
            except ValueError as exc:
                self._refresh_gui()
                self.status.content = str(exc)
                return
            if self.timing["mode"] == "manual":
                self._manual_arrivals = {name: frame["t"] for name, frame in self.keyframes.items()}
            elif settings["mode"] == "manual":
                for name, frame in self.keyframes.items():
                    frame["t"] = self._manual_arrivals.get(name, frame["t"])
            self.timing = settings
            self.course["waypoints"]["timing"] = self.timing
            self._course_changed()

    def _course_changed(self) -> None:
        """Invalidate results so they cannot be mistaken for the edited course."""
        self._revision += 1
        self.optimized_times = None
        self.simulated_tro = self.simulated_xro = self.simulated_rgb = None
        self.simulation_hz = None
        self.show_simulated_trajectory = False
        self.route_toggle_button.disabled = self.export_video_button.disabled = True
        self.route_toggle_button.label = "Show simulated trajectory"
        self.feasibility_gui.content = ""
        self.status.content = "Course changed. Solve or simulate to update the results."
        self._refresh_gui()
        self._update_route()

    def _validated_course(self) -> dict[str, Any]:
        """Snapshot the course, validating timing before saving or planning."""
        course = copy.deepcopy(self.course)
        course, _ = _normalise_course(course)
        times = estimate_times(course["waypoints"]["keyframes"], course["waypoints"]["timing"])
        for frame, arrival in zip(course["waypoints"]["keyframes"].values(), times):
            frame["t"] = arrival
        return course

    def _reference_report(self, flat_outputs: Any, reference: Any, lower: Any, upper: Any) -> str:
        """Report sampled reference demands against the simulation pilot limits."""
        np = self.np
        if not np.all(np.isfinite(flat_outputs)) or not np.all(np.isfinite(reference)):
            raise ValueError("The planned trajectory contains non-finite states or controls; revise its constraints or timing")
        controls = reference[:, -4:]
        exceeded = np.any((controls < np.asarray(lower) - 1e-6) | (controls > np.asarray(upper) + 1e-6), axis=0)
        labels = ("thrust", "roll rate", "pitch rate", "yaw rate")
        violations = ", ".join(label for label, failed in zip(labels, exceeded) if failed)
        result = f"**Control limits exceeded:** {violations}. Reduce aggressiveness or allow more time." if violations else "Sampled reference controls are within Viper's limits for carl."
        speed = np.linalg.norm(flat_outputs[:, :3, 1], axis=1).max()
        acceleration = np.linalg.norm(flat_outputs[:, :3, 2], axis=1).max()
        yaw_rate = np.abs(flat_outputs[:, 3, 1]).max()
        return f"{result}\n\nPeak sampled speed: **{speed:.2f} m/s**; acceleration: **{acceleration:.2f} m/s²**; yaw rate: **{yaw_rate:.2f} rad/s**.\n\nThese are sampled control checks; obstacle clearance is not checked."

    def _solve_timing(self) -> None:
        self.solve_button.disabled = True
        self.status.content = "Solving trajectory…"
        try:
            with self.lock:
                course = self._validated_course()
                revision = self._revision
            from figs.tsplines.min_time_snap import MinTimeSnap
            from figs.utilities import transform_helper as th
            from figs.dynamics.external_forces import ExternalForces

            # Use the same config root and presets as the editor simulation,
            # without importing the splat renderer for a trajectory-only solve.
            config_root = Path(__file__).resolve().parents[4] / "configs"
            with (config_root / "pilots/Viper.json").open(encoding="utf-8") as file:
                policy = json.load(file)
            with (config_root / "frames/carl.json").open(encoding="utf-8") as file:
                frame = json.load(file)
            mts = MinTimeSnap(course["waypoints"], 100, policy["plan"]["kT"], policy["plan"]["use_l2_time"])
            times, outputs = mts.get_desired_trajectory()
            reference = th.TsFO_to_tXU(times, outputs, frame["mass"], frame["motor_thrust_coeff"], ExternalForces(course.get("forces")))
            bounds = policy["track"]["bounds"]
            report = self._reference_report(outputs, reference, bounds["lower"], bounds["upper"])
            with self.lock:
                if revision != self._revision:
                    self.status.content = "Course changed during the solve. Solve again to update the results."
                    return
                self.optimized_times = mts.Tkf.tolist()
                self.feasibility_gui.content = report
                self._refresh_gui()
                self.status.content = f"Trajectory solved. Duration: {mts.Tkf[-1]:.3f} s."
        except Exception as exc:
            self.status.content = f"Trajectory solve failed: {exc}"
        finally:
            self.solve_button.disabled = False

    def _current(self) -> dict[str, Any]:
        return self.keyframes[self.selection]

    def _select(self, name: str) -> None:
        with self.lock:
            if name == self.selection:
                return
            self.selection = name
            # Keep the side-pane selection in sync when a rendered pose is clicked.
            self.selected_gui.value = name
            self._refresh_gui()
            self._redraw_scene()

    def _refresh_gui(self) -> None:
        self._syncing_gui = True
        try:
            self.timing_mode_gui.value = TIMING_MODES[self.timing["mode"]]
            self.aggressiveness_gui.value = self.timing["aggressiveness"]
            self.total_duration_gui.value = self.timing["total_duration"]
            self._refresh_timing()
            frame = self._current()
            self.name_gui.value = self.selection
            self.time_gui.value = float(frame["t"])
            for row_index, row_guis in enumerate(self.value_guis):
                for derivative, (specified, value) in enumerate(row_guis):
                    saved = frame["fo"][row_index][derivative]
                    specified.value = saved is not None
                    value.disabled = saved is None
                    if saved is not None:
                        # Numeric controls always show the canonical JSON value.
                        value.value = float(saved)
            self.delete_button.disabled = len(self.keyframes) <= 1
            selected_index = list(self.keyframes).index(self.selection)
            last_index = len(self.keyframes) - 1
            self.move_first_button.disabled = selected_index == 0
            self.move_earlier_button.disabled = selected_index == 0
            self.move_later_button.disabled = selected_index == last_index
            self.move_last_button.disabled = selected_index == last_index
        finally:
            self._syncing_gui = False

    def _sync_keyframe_options(self) -> None:
        """Synchronise the selected-keypoint widgets after a key/name change."""
        self.selected_gui.options = tuple(self.keyframes)
        self.selected_gui.value = self.selection

    def _move_selected(self, offset: int, *, first: bool = False, last: bool = False) -> None:
        """Move the selected keyframe while retaining its data and name."""
        with self.lock:
            names = list(self.keyframes)
            current_index = names.index(self.selection)
            destination = 0 if first else len(names) - 1 if last else current_index + offset
            destination = max(0, min(destination, len(names) - 1))
            if destination == current_index:
                return
            name = names.pop(current_index)
            names.insert(destination, name)
            self.keyframes = {keyframe_name: self.keyframes[keyframe_name] for keyframe_name in names}
            self.course["waypoints"]["keyframes"] = self.keyframes
            self._sync_keyframe_options()
            self._course_changed()
            self._redraw_scene()

    def _rename_selected(self) -> None:
        if self._syncing_gui:
            return
        with self.lock:
            old_name = self.selection
            new_name = self.name_gui.value.strip()
            if new_name == old_name:
                return
            if not new_name or "/" in new_name or "\\" in new_name:
                self.status.content = "Keypoint name must be non-empty and cannot contain a slash."
            elif new_name in self.keyframes:
                self.status.content = f"A keypoint named `{new_name}` already exists."
            else:
                if old_name in self._manual_arrivals:
                    self._manual_arrivals[new_name] = self._manual_arrivals.pop(old_name)
                self.keyframes = {
                    (new_name if keyframe_name == old_name else keyframe_name): keyframe
                    for keyframe_name, keyframe in self.keyframes.items()
                }
                self.course["waypoints"]["keyframes"] = self.keyframes
                self.selection = new_name
                self._sync_keyframe_options()
                self._course_changed()
                self._redraw_scene()
                return
            # Restore the canonical selected name after an invalid edit.
            self._syncing_gui = True
            try:
                self.name_gui.value = old_name
            finally:
                self._syncing_gui = False

    def _set_time(self, value: float) -> None:
        if not self._syncing_gui and self.timing["mode"] == "manual":
            with self.lock:
                self._current()["t"] = value
                self._course_changed()

    def _set_specified(self, row: int, derivative: int) -> None:
        if self._syncing_gui:
            return
        with self.lock:
            specified, value = self.value_guis[row][derivative]
            self._current()["fo"][row][derivative] = float(value.value) if specified.value else None
            value.disabled = not specified.value
            self._course_changed()
            self._redraw_scene()

    def _set_value(self, row: int, derivative: int) -> None:
        if not self._syncing_gui and self.value_guis[row][derivative][0].value:
            with self.lock:
                self._current()["fo"][row][derivative] = float(self.value_guis[row][derivative][1].value)
                self._course_changed()
                if derivative == 0:
                    self._redraw_scene()

    def _pose(self, name: str) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        """Resolve a keypoint pose without changing unspecified JSON values.

        A missing d0 is a constraint omission, not a zero.  To keep the pose
        drawable, it inherits the previous keypoint's resolved d0 for that
        row; the first unspecified value uses zero only as a visual fallback.
        """
        resolved = [0.0, 0.0, 0.0, 0.0]
        for keyframe_name, frame in self.keyframes.items():
            for row in range(len(_ROWS)):
                value = frame["fo"][row][0]
                if value is not None:
                    resolved[row] = float(value)
            if keyframe_name == name:
                break
        return (
            tuple(_to_editor_value(i, resolved[i]) * self.scene_scale for i in range(3)),
            _yaw_to_wxyz(_to_editor_value(3, resolved[3])),
        )

    def _redraw_scene(self) -> None:
        """Replace pose frames, the selected gizmo, and the straight-line course preview."""
        for handle in self.pose_handles.values():
            handle.remove()
        self.pose_handles.clear()
        ordered = list(self.keyframes.items())
        positions: list[tuple[float, float, float]] = []
        with self.server.atomic():
            for index, (name, frame) in enumerate(ordered):
                position, wxyz = self._pose(name)
                positions.append(position)
                pose_handle = self.server.scene.add_frame(
                    f"/course/keypoints/{name}", position=position, wxyz=wxyz,
                    axes_length=0.45 * self.scene_scale, axes_radius=0.015 * self.scene_scale, show_axes=True,
                )
                pose_handle.on_click(lambda _, keyframe_name=name: self._select(keyframe_name))
                self.pose_handles[name] = pose_handle
            self._update_route()

            position, wxyz = self._pose(self.selection)
            gizmo = self.server.scene.add_transform_controls(
                "/course/selected_gizmo", position=position, wxyz=wxyz, scale=0.6 * self.scene_scale,
                active_axes=(True, True, True), rotation_limits=((-0.0, 0.0), (-0.0, 0.0), (-1000.0, 1000.0)),
            )

        @gizmo.on_update
        def _(_: Any) -> None:
            with self.lock:
                frame = self._current()
                frame["fo"][0][0], frame["fo"][1][0], frame["fo"][2][0] = (
                    _from_editor_value(row, float(value) / self.scene_scale)
                    for row, value in enumerate(gizmo.position)
                )
                rendered_yaw = _unwrap_angle(
                    _wxyz_to_yaw(gizmo.wxyz),
                    _to_editor_value(3, float(frame["fo"][3][0] or 0.0)),
                )
                frame["fo"][3][0] = _from_editor_value(3, rendered_yaw)
                self._course_changed()
                # Updating the visible frame does not require recreating the gizmo mid-drag.
                self.pose_handles[self.selection].position = gizmo.position
                self.pose_handles[self.selection].wxyz = gizmo.wxyz
                self._update_route()

    def _update_route(self) -> None:
        if self.show_simulated_trajectory and self.simulated_tro is not None and self.simulated_xro is not None:
            self._draw_simulated_route()
            return
        positions = [self._pose(name)[0] for name in self.keyframes]
        points = self.np.asarray([[positions[i], positions[i + 1]] for i in range(len(positions) - 1)]) if len(positions) >= 2 else self.np.empty((0, 2, 3))
        self.server.scene.add_line_segments("/course/route", points=points, colors=(80, 190, 255), line_width=3.0)
        if self.speed_markers_handle is not None:
            self.speed_markers_handle.remove()
            self.speed_markers_handle = None

    def _draw_simulated_route(self) -> None:
        """Render the rollout and color periodic samples by translational speed."""
        assert self.simulated_tro is not None and self.simulated_xro is not None
        positions = self.simulated_xro[:, 0:3].copy()
        positions[:, 1:3] *= -1.0
        positions *= self.scene_scale
        points = (
            self.np.stack((positions[:-1], positions[1:]), axis=1)
            if len(positions) >= 2 else self.np.empty((0, 2, 3))
        )
        self.server.scene.add_line_segments(
            "/course/route", points=points, colors=(255, 170, 50), line_width=3.0,
        )

        period = max(float(self.speed_marker_period_gui.value), 0.01)
        sample_times = self.np.arange(self.simulated_tro[0], self.simulated_tro[-1] + period, period)
        sample_indices = self.np.unique(self.np.clip(self.np.searchsorted(self.simulated_tro, sample_times), 0, len(positions) - 1))
        marker_positions = positions[sample_indices]
        speeds = self.np.linalg.norm(self.simulated_xro[sample_indices, 3:6], axis=1)
        speed_range = float(speeds.max() - speeds.min()) if len(speeds) else 0.0
        normalized = (speeds - speeds.min()) / speed_range if speed_range > 1e-9 else self.np.zeros_like(speeds)
        colors = self.np.column_stack((255.0 * normalized, 80.0 + 120.0 * (1.0 - normalized), 255.0 * (1.0 - normalized))).astype(self.np.uint8)
        if self.speed_markers_handle is not None:
            self.speed_markers_handle.remove()
        self.speed_markers_handle = self.server.scene.add_point_cloud(
            "/course/simulation_speed_markers", points=marker_positions, colors=colors,
            point_size=0.12 * self.scene_scale, point_shape="circle",
        )

    def _refresh_simulation_render(self) -> None:
        if self.show_simulated_trajectory and self.simulated_tro is not None:
            with self.lock:
                self._draw_simulated_route()

    def _toggle_route_display(self) -> None:
        with self.lock:
            if self.simulated_tro is None or self.simulated_xro is None:
                self.status.content = "Run a simulation before showing its trajectory."
                return
            self.show_simulated_trajectory = not self.show_simulated_trajectory
            self.route_toggle_button.label = (
                "Show keypoint route" if self.show_simulated_trajectory else "Show simulated trajectory"
            )
            self._update_route()
            self.status.content = (
                "Showing simulated trajectory and speed markers."
                if self.show_simulated_trajectory else "Showing editable keypoint route."
            )

    def _export_simulation_video(self) -> None:
        if self.simulated_rgb is None or self.simulation_hz is None:
            self.status.content = "Run a simulation before exporting a video."
            return
        output_path = Path(self.video_output_gui.value)
        if output_path.suffix.lower() != ".mp4":
            self.status.content = "Simulation video path must end with .mp4."
            return
        self.export_video_button.disabled = True
        self.status.content = f"Exporting simulation video to {output_path}…"
        try:
            from figs.visualize.generate_videos import images_to_mp4

            output_path.parent.mkdir(parents=True, exist_ok=True)
            images_to_mp4(self.simulated_rgb, str(output_path), self.simulation_hz)
            self.status.content = f"Saved simulation video to `{output_path}`"
            print(f"Saved simulation video to {output_path}")
        except Exception as exc:
            self.status.content = f"Video export failed: {exc}"
            print(f"Video export failed: {exc}")
        finally:
            self.export_video_button.disabled = False

    def _write_course(self, path: Path) -> None:
        course = self._validated_course()
        with path.open("w", encoding="utf-8") as file:
            json.dump(course, file, indent=2, allow_nan=False)
            file.write("\n")

    def _run_simulation(self) -> None:
        if self.gsplat is None:
            self.status.content = "Simulation requires --gsplat-config."
            return
        self.simulate_button.disabled = True
        self.status.content = "Running FiGS simulation…"
        try:
            with self.lock:
                simulation_course = self._validated_course()
                revision = self._revision

            from figs.control.vehicle_rate_mpc import VehicleRateMPC
            from figs.simulator import Simulator

            simulator = Simulator(self.gsplat, "eval_single", "carl")
            simulator.update_forces(simulation_course.get("forces"))
            controller = VehicleRateMPC("Viper", simulation_course, "carl")
            report = self._reference_report(controller.FOd, controller.tXUd, controller.lbu, controller.ubu)
            t0, tf = controller.tXUd[0, 0], controller.tXUd[-1, 0]
            tro, xro, _, _, rgb, _, _ = simulator.simulate(controller, t0, tf, controller.tXUd[0, 1:11])
            with self.lock:
                if revision != self._revision:
                    self.status.content = "Course changed during simulation. Run again to update the results."
                    return
                self.optimized_times = controller.Tkf.tolist()
                self.feasibility_gui.content = report
                self._refresh_gui()
                self.simulated_tro = tro
                self.simulated_xro = xro
                self.simulated_rgb = rgb
                self.simulation_hz = int(controller.hz)
                self.show_simulated_trajectory = True
                self.route_toggle_button.disabled = False
                self.route_toggle_button.label = "Show keypoint route"
                self.export_video_button.disabled = False
                self._draw_simulated_route()
                self.status.content = "Simulation complete. Showing simulated trajectory and speed markers."
        except Exception as exc:
            self.status.content = f"Simulation failed: {exc}"
            print(f"Simulation failed: {exc}")
        finally:
            self.simulate_button.disabled = False

    def _add_keypoint(self) -> None:
        with self.lock:
            source = self._current()
            index = len(self.keyframes)
            while f"fo{index}" in self.keyframes:
                index += 1
            new_name = f"fo{index}"
            new_fo = [row.copy() for row in source["fo"]]
            last_time = float(next(reversed(self.keyframes.values()))["t"])
            self.keyframes[new_name] = {"t": last_time + 1.0, "fo": new_fo}
            self.selection = new_name
            self.selected_gui.options = tuple(self.keyframes)
            self.selected_gui.value = new_name
            self._course_changed()
            self._redraw_scene()

    def _delete_selected(self) -> None:
        with self.lock:
            if len(self.keyframes) <= 1:
                return
            self._manual_arrivals.pop(self.selection, None)
            del self.keyframes[self.selection]
            self.selection = next(iter(self.keyframes))
            self.selected_gui.options = tuple(self.keyframes)
            self.selected_gui.value = self.selection
            self._course_changed()
            self._redraw_scene()

    def _save(self) -> None:
        with self.lock:
            try:
                self.output_path.parent.mkdir(parents=True, exist_ok=True)
                self._write_course(self.output_path)
                self.status.content = f"Saved `{self.output_path}`"
                print(f"Saved {self.output_path}")
            except (OSError, ValueError) as exc:
                self.status.content = f"Unable to save course: {exc}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Edit FiGS drone-course keyframes in a Viser browser UI.")
    parser.add_argument("--input", type=Path, help="Existing course JSON to edit.")
    parser.add_argument("--output", type=Path, help="Write path; defaults to --input. Required when creating a course.")
    parser.add_argument("--host", default="127.0.0.1", help="Viser server host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port (default: 8080).")
    parser.add_argument(
        "--gsplat-config",
        type=Path,
        help="Nerfstudio experiment config.yml to load and render as the editor background.",
    )
    args = parser.parse_args()
    if args.input is None and args.output is None:
        parser.error("--output is required when --input is omitted")
    return args


def _start_ns_viewer(config_path: Path, host: str, port: int) -> tuple[Any, Any]:
    """Start Nerfstudio's eval viewer and return its server plus owning state.

    This follows ``nerfstudio.scripts.viewer.run_viewer`` so Gaussian splats
    are rendered on camera movement by Nerfstudio's render state machine.
    """
    from nerfstudio.utils.eval_utils import eval_setup
    from nerfstudio.viewer.viewer import Viewer
    # ``base_config`` imports this module while declaring ``LoggingConfig``.
    # Import it only after eval_setup has completed that configuration import;
    # importing writer first creates a base_config <-> writer import cycle.
    from nerfstudio.utils import writer
    from figs.render.gsplat import GSplat

    config, pipeline, _, step = eval_setup(config_path, eval_num_rays_per_chunk=None, test_mode="test")
    config.viewer.websocket_host = host
    config.viewer.websocket_port = port
    viewer = Viewer(
        config.viewer,
        log_filename=config.get_base_dir() / config.viewer.relative_log_filename,
        datapath=pipeline.datamanager.get_datapath(),
        pipeline=pipeline,
    )
    # Match ns-viewer's runtime setup; the render state machine consults the
    # global writer buffer while servicing interactive camera updates.
    config.logging.local_writer.enable = False
    writer.setup_local_writer(
        config.logging,
        max_iter=config.max_num_iterations,
        banner_messages=viewer.viewer_info,
    )
    assert pipeline.datamanager.train_dataset is not None
    viewer.init_scene(
        train_dataset=pipeline.datamanager.train_dataset,
        train_state="completed",
        eval_dataset=pipeline.datamanager.eval_dataset,
    )
    # Keep Nerfstudio's renderer, but present the editor as the only UI and
    # avoid obscuring the course with its dataset-camera frustums.
    viewer.set_camera_visibility(False)
    # Do not call ``gui.reset()`` here: viser 1.0.30 raises while removing a
    # populated tab group. Hiding the root handles keeps Nerfstudio's render
    # state intact but removes its controls from the side panel.
    root_gui = viewer.viser_server.gui._container_handle_from_uuid["root"]
    for handle in tuple(root_gui._children.values()):
        if hasattr(handle, "visible"):
            handle.visible = False
    viewer.viser_server.gui.set_panel_label("Drone course editor")
    viewer.viser_server.gui.configure_theme(
        titlebar_content=None,
        control_layout="collapsible",
        dark_mode=True,
        show_logo=False,
        show_share_button=False,
    )
    viewer.update_scene(step=step)
    viewer.gsplat = GSplat.from_pipeline(config, pipeline)
    return viewer.viser_server, viewer


def main() -> None:
    args = _parse_args()
    if args.input is None or not args.input.exists():
        course = _new_course()
        if args.input is not None:
            print(f"Creating new course at {args.input}")
    else:
        try:
            with args.input.open(encoding="utf-8") as file:
                course = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Unable to read {args.input}: {exc}") from exc
    try:
        course, derivative_count = _normalise_course(course)
    except ValueError as exc:
        raise SystemExit(f"Invalid course JSON: {exc}") from exc
    output_path = args.output if args.output is not None else args.input
    assert output_path is not None
    if args.gsplat_config is None:
        CourseEditor(course, derivative_count, output_path, args.host, args.port)
    else:
        if not args.gsplat_config.is_file():
            raise SystemExit(f"Nerfstudio config does not exist: {args.gsplat_config}")
        try:
            server, viewer = _start_ns_viewer(args.gsplat_config, args.host, args.port)
        except Exception as exc:
            raise SystemExit(f"Unable to start Nerfstudio viewer: {exc}") from exc
        editor = CourseEditor(
            course, derivative_count, output_path, args.host, args.port,
            server=server, scene_scale=10.0, gsplat=viewer.gsplat,
        )
        # Keep both owners alive; the editor adds its controls to the viewer's server.
        editor.ns_viewer = viewer
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
