# Flying in Gaussian Splats (FiGS)
See https://github.com/StanfordMSL/FiGS-Examples for how to use the package.

## Course editor timing

From the SousVide workspace, open a course in the editor:

```bash
python -m figs.utilities.course_editor --input configs/courses/traverse.json \
    --output configs/courses/my_course.json
```

Add `--gsplat-config path/to/config.yml` to edit against a reconstructed scene
and enable **Run FiGS simulation**. **Solve trajectory** works without a scene.

The **Timing** controls offer three modes:

- **Automatic** (default): initial times are estimated from distances, yaw
  changes, and endpoint stops. The solver chooses arrival times by balancing
  smoothness against total duration. **Aggressiveness** defaults to 1; increasing
  it increases the time penalty and generally favors faster flight. It is not a
  commanded speed or speed limit.
- **Manual waypoint times (advanced)**: edit each keypoint's arrival time.
  Times must start at zero and strictly increase in keypoint order. These times
  are fixed during trajectory optimization. Reordering points does not reorder
  their timestamps; repair the schedule before saving or solving. Switching away
  from manual mode and back restores the manual times within the editor session.
- **Fixed total duration**: set the overall flight duration and let the solver
  allocate time between keypoints. Each segment is bounded to 0.01–30 seconds,
  so the permitted total depends on the number of segments. This mode minimizes
  smoothness cost subject to the specified total, without a time penalty.

After solving or simulating, the timing table distinguishes initial estimates
(or fixed manual times) from solved arrival times. The reference check reports
sampled speed, acceleration, yaw rate, and violations of Viper's thrust/body-rate
limits for the carl frame. Solve-only checks use 100 Hz; simulation checks use the
controller's sampling rate. These checks do not certify continuous-time
feasibility or obstacle clearance. Simulation remains useful for inspecting
tracking behavior and the camera view. Editing the course clears previous solve
and simulation results, including exported-video availability. The simulation's
sampled end time may be rounded up to the next controller time step.

**Save JSON** stores timing settings in `waypoints.timing`, alongside keyframe
times. For example:

```json
"timing": {
  "mode": "automatic",
  "aggressiveness": 1.0,
  "total_duration": 10.0
}
```

`mode` can be `automatic`, `manual`, or `total_duration`; `total_duration` is only
used in the last mode. Explicit course timing overrides the pilot's planning
timing policy in both the editor and downstream rollouts. Automatic mode uses a
linear total-time weight of `10 * aggressiveness`; manual mode disables time
optimization. Courses without `waypoints.timing` retain the existing pilot-driven
behavior when loaded directly by the planner. Opening them in the editor defaults
to automatic timing; select manual mode to recover their original waypoint times
before saving if those times should be fixed.
