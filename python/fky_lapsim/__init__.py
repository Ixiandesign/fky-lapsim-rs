"""FKY LAPSIM: Python API over the Rust physics engine.

All functions take and return plain Python objects (dicts / lists / floats),
JSON-compatible. Coordinates are SI, x forward / y left / z up; angles in radians.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from . import _native
from ._native import CancellationToken

__all__ = [
    "call", "CancellationToken", "OptimizationSession",
    "example_project", "formula_car_demo", "defaults", "metric_registry",
    "lap_vehicle_demo", "track_demo", "motion_grid", "result_table",
    "validate", "normalize_project", "simulate", "analyze", "sweep", "ride",
    "linearize_ride", "parameter_registry", "optimize", "candidate_project",
    "evaluate_candidate", "validate_lap_vehicle", "validate_track",
    "validate_lap_request", "run_lap", "validate_lap_optimization_request",
    "run_lap_optimization",
]

JSON = Any


def call(op: str, **fields: JSON) -> JSON:
    """Run native operation `op` with `fields` as its JSON input; return parsed JSON.

    Raises ValueError on invalid input or native failure. See
    `help(fky_lapsim._native.call)` for every op and its input shape.
    """
    return json.loads(_native.call(op, json.dumps(fields)))


# --- demos / registries ---------------------------------------------------
def example_project() -> JSON:
    return call("example_project")


def formula_car_demo() -> JSON:
    """Returns {"project", "request"}."""
    return call("formula_car_demo")


def defaults() -> JSON:
    """Returns {"motion", "ride", "optimization"} default requests."""
    return call("defaults")


def metric_registry() -> JSON:
    return call("metric_registry")


def lap_vehicle_demo() -> JSON:
    return call("lap_vehicle_demo")


def track_demo(shape: str = "oval", radius_m: float = 9.0, straight_m: float = 60.0,
               width_m: float = 8.0, segments: int = 24) -> JSON:
    return call("track_demo", shape=shape, radius_m=radius_m, straight_m=straight_m,
                width_m=width_m, segments=segments)


def motion_grid(request: JSON) -> JSON:
    return call("motion_grid", request=request)


def result_table(kind: str, result: JSON) -> JSON:
    """Returns {"table", "csv"}."""
    return call("result_table", kind=kind, result=result)


# --- suspension project ---------------------------------------------------
def validate(project: JSON) -> JSON:
    return call("validate", project=project)


def normalize_project(project: JSON) -> JSON:
    return call("normalize_project", project=project)


def simulate(project: JSON, motion: JSON) -> JSON:
    return call("simulate", project=project, motion=motion)


def analyze(project: JSON, motion: JSON, step: Optional[float] = None) -> JSON:
    extra = {} if step is None else {"step": step}
    return call("analyze", project=project, motion=motion, **extra)


def sweep(project: JSON, motions: list) -> JSON:
    return call("sweep", project=project, motions=motions)


def ride(project: JSON, request: JSON) -> JSON:
    return call("ride", project=project, request=request)


def linearize_ride(project: JSON, request: JSON) -> JSON:
    return call("linearize_ride", project=project, request=request)


def parameter_registry(project: JSON) -> JSON:
    return call("parameter_registry", project=project)


def optimize(project: JSON, request: JSON) -> JSON:
    """Complete blocking search. Use OptimizationSession for incremental/cancellable."""
    return call("optimize", project=project, request=request)


def candidate_project(project: JSON, request: JSON, values: list) -> JSON:
    return call("candidate_project", project=project, request=request, values=values)


def evaluate_candidate(project: JSON, request: JSON, values: list,
                       validation: bool = False,
                       cancel: Optional[CancellationToken] = None) -> JSON:
    """Score one candidate with the native evaluator (for pymoo/SciPy adapters)."""
    payload = {"project": project, "request": request, "values": values,
               "validation": validation}
    return json.loads(_native.evaluate(json.dumps(payload), cancel or CancellationToken()))


# --- lap simulation -------------------------------------------------------
def validate_lap_vehicle(vehicle: JSON) -> JSON:
    return call("validate_lap_vehicle", vehicle=vehicle)


def validate_track(track: JSON) -> JSON:
    return call("validate_track", track=track)


def validate_lap_request(vehicle: JSON, track: JSON, request: JSON) -> JSON:
    return call("validate_lap_request", vehicle=vehicle, track=track, request=request)


def run_lap(vehicle: JSON, track: JSON, request: JSON) -> JSON:
    return call("run_lap", vehicle=vehicle, track=track, request=request)


def validate_lap_optimization_request(vehicle: JSON, tracks: JSON, request: JSON) -> JSON:
    return call("validate_lap_optimization_request", vehicle=vehicle, tracks=tracks,
                request=request)


def run_lap_optimization(vehicle: JSON, tracks: JSON, request: JSON) -> JSON:
    """Complete blocking multi-track search; cannot be cancelled."""
    return call("run_lap_optimization", vehicle=vehicle, tracks=tracks, request=request)


class OptimizationSession:
    """Incremental, cancellable, checkpointable suspension optimization."""

    def __init__(self, project: JSON, request: JSON, checkpoint: Optional[str] = None):
        self._s = _native.Session(json.dumps(project), json.dumps(request), checkpoint)

    def advance(self, max_work: int, cancel: Optional[CancellationToken] = None) -> JSON:
        return json.loads(self._s.advance(max_work, cancel or CancellationToken()))

    def result(self) -> JSON:
        return json.loads(self._s.result())

    def checkpoint(self) -> str:
        return self._s.checkpoint()

    @property
    def is_finished(self) -> bool:
        return self._s.is_finished
