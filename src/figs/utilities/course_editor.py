"""Interactive Viser editor for FiGS drone-course keyframes.

Run from an environment with FiGS and viser installed, for example::

    python -m figs.utilities.course_editor --input configs/courses/infinity.json
    python -m figs.utilities.course_editor --input configs/courses/infinity.json \
        --gsplat-config gsplats/workspace/outputs/my_scene/config.yml
    python -m figs.utilities.course_editor --output configs/courses/new_course.json

The first command edits the supplied file in place when ``Save JSON`` is
pressed.  Supplying ``--output`` writes to that path instead.  A new course is
created when ``--input`` is omitted.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
import tempfile
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

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
        "fo": [[0.0, None, None, None], [0.0, None, None, None], [0.0, None, None, None], [0.0, None, None, None]],
    }
    return {
        "waypoints": {
            "Nco": 6,
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

    derivative_count = _MIN_DERIVATIVES
    for name, keyframe in keyframes.items():
        if not isinstance(keyframe, MutableMapping):
            raise ValueError(f"keyframe {name!r} must be an object")
        keyframe["t"] = _as_number_or_none(keyframe.get("t"), f"keyframe {name}.t")
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
            self.simulate_button = self.server.gui.add_button(
                "Run FiGS simulation", disabled=self.gsplat is None,
                hint="Run the Section 4 FiGS simulator using the current course and Gaussian splat.",
            )
            self.simulate_button.on_click(lambda _: self._run_simulation())
            self.route_toggle_button = self.server.gui.add_button("Show simulated trajectory", disabled=True)
            self.route_toggle_button.on_click(lambda _: self._toggle_route_display())
            self.status = self.server.gui.add_markdown("")

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
            frame = self._current()
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
        finally:
            self._syncing_gui = False

    def _set_time(self, value: float) -> None:
        if not self._syncing_gui:
            with self.lock:
                self._current()["t"] = value

    def _set_specified(self, row: int, derivative: int) -> None:
        if self._syncing_gui:
            return
        with self.lock:
            specified, value = self.value_guis[row][derivative]
            self._current()["fo"][row][derivative] = float(value.value) if specified.value else None
            value.disabled = not specified.value
            self._redraw_scene()

    def _set_value(self, row: int, derivative: int) -> None:
        if not self._syncing_gui and self.value_guis[row][derivative][0].value:
            with self.lock:
                self._current()["fo"][row][derivative] = float(self.value_guis[row][derivative][1].value)
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
                frame["fo"][3][0] = _from_editor_value(3, _wxyz_to_yaw(gizmo.wxyz))
                self._refresh_gui()
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
        if self.simulated_tro is not None:
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
        with path.open("w", encoding="utf-8") as file:
            json.dump(self.course, file, indent=2, allow_nan=False)
            file.write("\n")

    def _run_simulation(self) -> None:
        if self.gsplat is None:
            self.status.content = "Simulation requires --gsplat-config."
            return
        self.simulate_button.disabled = True
        self.status.content = "Running FiGS simulation…"
        temporary_path: Path | None = None
        try:
            # The controller receives the serialized editor state, rather than
            # a mutable reference to the live editor model.
            with self.lock:
                with tempfile.NamedTemporaryFile(prefix="figs-course-editor-", suffix=".json", delete=False, mode="w", encoding="utf-8") as file:
                    temporary_path = Path(file.name)
                self._write_course(temporary_path)
            with temporary_path.open(encoding="utf-8") as file:
                simulation_course = json.load(file)

            from figs.control.vehicle_rate_mpc import VehicleRateMPC
            from figs.simulator import Simulator

            simulator = Simulator(self.gsplat, "eval_single", "carl")
            controller = VehicleRateMPC("Viper", simulation_course, "carl")
            t0, tf = controller.tXUd[0, 0], controller.tXUd[-1, 0]
            tro, xro, _, _, rgb, _, _ = simulator.simulate(controller, t0, tf, controller.tXUd[0, 1:11])
            with self.lock:
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
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            self.simulate_button.disabled = False

    def _add_keypoint(self) -> None:
        with self.lock:
            source = self._current()
            new_name = f"fo{len(self.keyframes)}"
            new_fo = [row.copy() for row in source["fo"]]
            self.keyframes[new_name] = {"t": float(source["t"]) + 1.0, "fo": new_fo}
            self.selection = new_name
            self.selected_gui.options = tuple(self.keyframes)
            self.selected_gui.value = new_name
            self._refresh_gui()
            self._redraw_scene()

    def _delete_selected(self) -> None:
        with self.lock:
            if len(self.keyframes) <= 1:
                return
            del self.keyframes[self.selection]
            self.selection = next(iter(self.keyframes))
            self.selected_gui.options = tuple(self.keyframes)
            self.selected_gui.value = self.selection
            self._refresh_gui()
            self._redraw_scene()

    def _save(self) -> None:
        with self.lock:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_course(self.output_path)
            self.status.content = f"Saved `{self.output_path}`"
            print(f"Saved {self.output_path}")


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
