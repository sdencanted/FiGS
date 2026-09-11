"""Course timing settings shared by the editor and trajectory planner.

Courses without ``waypoints.timing`` retain the pilot's legacy timing policy.
Explicit timing settings override that policy, including during rollout generation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping


TIMING_MODES = {
    "automatic": "Automatic",
    "manual": "Manual waypoint times (advanced)",
    "total_duration": "Fixed total duration",
}
BASE_TIME_WEIGHT = 10.0
DEFAULT_BOUNDS = (0.01, 30.0)


def timing_settings(value: Mapping | None = None) -> dict:
    """Validate settings, filling defaults for an editor-authored course."""
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError("waypoints.timing must be an object")
    mode = value.get("mode", "automatic")
    if not isinstance(mode, str) or mode not in TIMING_MODES:
        raise ValueError(f"Unknown timing mode: {mode!r}")
    settings = {"mode": mode}
    for name, default in (("aggressiveness", 1.0), ("total_duration", 10.0)):
        number = value.get(name, default)
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or number <= 0:
            raise ValueError(f"Timing {name} must be a finite positive number")
        settings[name] = float(number)
    return settings


def manual_times(keyframes: Mapping) -> list[float]:
    """Require a zero-based, strictly increasing schedule in keyframe order."""
    if len(keyframes) < 2:
        raise ValueError("A trajectory requires at least two keypoints")
    times = []
    for name, frame in keyframes.items():
        value = frame.get("t")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Keypoint {name!r} needs a finite arrival time")
        times.append(float(value))
    if times[0] != 0.0:
        raise ValueError("The first manual arrival time must be 0 seconds")
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("Manual arrival times must increase in keypoint order")
    return times


def estimate_times(keyframes: Mapping, settings: Mapping, bounds=DEFAULT_BOUNDS) -> list[float]:
    """Seed durations using distance, unwrapped yaw, and endpoint stops.

The nominal speed/acceleration are 1 m/s and 1 m/s² at aggressiveness 1.
These estimates initialize optimization; they are not speed constraints.
Unspecified poses inherit previous values for estimation only.
"""
    if len(keyframes) < 2:
        raise ValueError("A trajectory requires at least two keypoints")
    settings = timing_settings(settings)
    if settings["mode"] == "manual":
        return manual_times(keyframes)
    lower, upper = bounds
    frames = list(keyframes.values())
    poses = []
    resolved = [0.0] * 4
    for frame in frames:
        for axis, row in enumerate(frame["fo"]):
            if row[0] is not None:
                resolved[axis] = float(row[0])
        poses.append(resolved.copy())
    # Fixed-total-time initialization is independent of aggressiveness.
    speed = math.sqrt(settings["aggressiveness"]) if settings["mode"] == "automatic" else 1.0
    durations = []
    for index, (start, end) in enumerate(zip(poses, poses[1:])):
        distance = math.dist(start[:3], end[:3])
        duration = max(0.25, distance / speed, abs(end[3] - start[3]) / speed)
        for endpoint in (index, index + 1):
            if endpoint in (0, len(frames) - 1):
                rows = frames[endpoint]["fo"][:3]
                if all(len(row) > 1 and row[1] == 0.0 for row in rows):
                    duration += min(speed, math.sqrt(distance))
        durations.append(min(upper, max(lower, duration)))
    if settings["mode"] == "total_duration":
        total = settings["total_duration"]
        count = len(durations)
        if not count * lower <= total <= count * upper:
            raise ValueError(f"Total duration must be between {count * lower:g} and {count * upper:g} seconds for {count} segments")
        # Find a proportional allocation while respecting every segment bound.
        low, high = 0.0, max(1.0, upper / min(durations))
        for _ in range(70):
            scale = (low + high) / 2.0
            if sum(min(upper, max(lower, duration * scale)) for duration in durations) < total:
                low = scale
            else:
                high = scale
        durations = [min(upper, max(lower, duration * high)) for duration in durations]
    times = [0.0]
    for duration in durations:
        times.append(times[-1] + duration)
    return times
