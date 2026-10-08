//! PyO3 bindings exposing [`fky_lapsim_core`] to Python as the `fky_lapsim._native`
//! extension module.
//!
//! This crate is deliberately thin: instead of mirroring every Rust type as a
//! `#[pyclass]`, it exposes almost the entire surface through one generic
//! JSON-in/JSON-out dispatcher, `call`, tagged by an `op` string. The
//! Python package (`python/fky_lapsim`) owns typed request/response
//! dataclasses and calls `_native.call(op, json_string)`; this crate only
//! needs to (de)serialize [`serde_json::Value`]s into the matching
//! `fky_lapsim_core` types, run the native computation, and serialize the
//! result back. That indirection means the Python surface stays stable
//! while `fky_lapsim_core`'s Rust types evolve, at the cost of `call` being
//! the single place documenting the whole wire contract — see its doc
//! comment for the full list of supported `op` values.
//!
//! Two operations need more than one blocking call and get their own
//! `#[pyclass]` wrappers instead of an `op` string:
//! - `Session` wraps [`opt::OptimizationSession`] so Python can start,
//!   advance incrementally, checkpoint, and resume a suspension
//!   optimization search across multiple calls (and across process
//!   restarts, via the serialized checkpoint string).
//! - `evaluate` runs one optimizer candidate through the controlled,
//!   cancellable evaluator used by external (e.g. pymoo/SciPy) adapters.
//!
//! Both accept a `CancellationToken`: a `Send`-safe flag Python can flip
//! from another thread to cooperatively cancel a running native call.
//! Cancellation is cooperative, not preemptive — an in-flight physics case
//! can still finish after the flag is set.
//!
//! Every `///` comment on a `#[pyfunction]`, `#[pyclass]`, or `#[pymethods]`
//! item below is forwarded by PyO3 into the corresponding Python object's
//! `__doc__`, so it is literally what a Python caller sees from `help(...)`.
//! Keep those comments accurate to the JSON shapes actually read/written,
//! not just accurate to the Rust signature.
use fky_lapsim_core::{
    lap::{optimization as lap_opt, LapVehicle},
    lapsim::{self, vehicle::{QssExtras, QssVehicle}},
    optimize as opt, Motion, Project, RideRequest,
};
use pyo3::{exceptions::PyValueError, prelude::*};
use serde::{de::DeserializeOwned, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

/// Deserialize `v` into `T`, mapping any failure to a display string rather
/// than a typed error — every native operation below reports failures to
/// Python as a plain `PyValueError`, so there is no reason to keep a richer
/// error type alive past this point.
fn parse<T: DeserializeOwned>(v: &Value) -> Result<T, String> {
    serde_json::from_value(v.clone()).map_err(|e| e.to_string())
}

/// Serialize `v` to a JSON string, mapping any failure to a display string.
/// `T`'s own `Serialize` impl is expected to always succeed for the types
/// this crate passes through it (they are plain data, not e.g. `HashMap`s
/// with non-string keys), so a failure here would indicate a bug rather
/// than bad input.
fn encoded<T: Serialize>(v: T) -> Result<String, String> {
    serde_json::to_string(&v).map_err(|e| e.to_string())
}

/// Render any [`std::fmt::Display`] error (typically a `fky_lapsim_core`
/// [`fky_lapsim_core::Error`] or a validation error) as the plain string
/// every operation in [`run`] reports failures with.
fn native_error(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// Set a numeric field addressed by JSON Pointer `path` to `value`, in
/// place. Used by the lap-optimization evaluator (see [`run`]'s
/// `"run_lap_optimization"` case) to apply a candidate's variable values to
/// a cloned vehicle JSON document before re-parsing it as a
/// [`fky_lapsim_core::lap::LapVehicle`].
///
/// # Errors
///
/// Returns an error string if `path` does not resolve to any field in
/// `root`, or resolves to a field that is not a JSON number.
fn set_numeric_path(root: &mut Value, path: &str, value: f64) -> Result<(), String> {
    let slot = root
        .pointer_mut(path)
        .ok_or_else(|| format!("unknown parameter path: {path}"))?;
    if !slot.is_number() {
        return Err(format!("parameter path is not a numeric field: {path}"));
    }
    *slot = json!(value);
    Ok(())
}
/// Add `delta` to a numeric field addressed by JSON Pointer `path`, in
/// place — the optimizer's uncertainty deltas are additive perturbations,
/// applied after a candidate's variables are already set (matching
/// `fky_lapsim_core::optimize`'s `Perturbation` semantics).
///
/// # Errors
///
/// Returns an error string if `path` does not resolve to any field in
/// `root`, or resolves to a field that is not a JSON number.
fn add_numeric_path(root: &mut Value, path: &str, delta: f64) -> Result<(), String> {
    let slot = root
        .pointer_mut(path)
        .ok_or_else(|| format!("unknown parameter path: {path}"))?;
    let base = slot
        .as_f64()
        .ok_or_else(|| format!("parameter path is not a numeric field: {path}"))?;
    *slot = json!(base + delta);
    Ok(())
}
/// Split a lap-vehicle JSON object into the native [`LapVehicle`] and its QSS extras.
///
/// The extras live under the `"qss"` key next to the `LapVehicle` fields, so optimizer parameter
/// paths such as `/qss/unsprung/0/mass_kg` address them like any other numeric leaf.
///
/// # Errors
///
/// Returns an error string when `v` is not an object, has no `"qss"` key, or either part fails
/// to deserialize.
fn split_lap_vehicle(v: &Value) -> Result<(LapVehicle, QssExtras), String> {
    let mut rest = v.clone();
    let qss = rest
        .as_object_mut()
        .ok_or_else(|| "lap vehicle must be a JSON object".to_string())?
        .remove("qss")
        .ok_or_else(|| "lapsim vehicle requires the qss extras (unsprung masses, tyre stiffness, ...)".to_string())?;
    Ok((parse(&rest)?, parse(&qss)?))
}
/// Parse the `{"<track_id>": {"track": Track, "settings": LapRequest}}` map
/// a lap-optimization request's evaluator resolves `TrackCase.id`s against
/// (see [`run`]'s `"run_lap_optimization"` and
/// `"validate_lap_optimization_request"` cases).
///
/// # Errors
///
/// Returns an error string if `v` is not a JSON object with that shape, if
/// any entry's `track` fails to deserialize as a [`fky_lapsim_core::track::Track`]
/// or fails [`fky_lapsim_core::track::Track::validate`], or if any entry's
/// `settings` fails to deserialize as a [`fky_lapsim_core::lapsim::LapRequest`]
/// or fails its own validation.
fn lap_opt_tracks(v: &Value) -> Result<BTreeMap<String, (fky_lapsim_core::track::Track, lapsim::LapRequest)>, String> {
    let map: BTreeMap<String, Value> = parse(v)?;
    map.into_iter()
        .map(|(id, entry)| {
            let track: fky_lapsim_core::track::Track = parse(&entry["track"])?;
            track.validate()?;
            let settings: lapsim::LapRequest = parse(&entry["settings"])?;
            settings.validate().map_err(native_error)?;
            Ok((id, (track, settings)))
        })
        .collect()
}
/// Implementation behind [`call`]: parse `input` as JSON, dispatch on `op`,
/// and return the result re-serialized as a JSON string. See [`call`]'s doc
/// comment for the full, canonical list of supported `op` values and their
/// input/output shapes — this function's `match` arms are the
/// implementation of that contract, not a second copy of it.
///
/// # Errors
///
/// Returns an error string (never panics) for: invalid JSON in `input`; an
/// unrecognized `op`; a `"project"`/`"vehicle"`/`"track"`/`"request"` field
/// that fails to deserialize into its expected `fky_lapsim_core` type or
/// fails that type's own validation; or a native computation error from
/// `fky_lapsim_core` (e.g. an infeasible optimization, a non-finite ride
/// state, or an unresolved JSON-pointer parameter path).
fn run(op: &str, input: &str) -> Result<String, String> {
    let v: Value = serde_json::from_str(input).map_err(native_error)?;
    match op {
        "motion_grid" => {
            return encoded(
                fky_lapsim_core::study::motion_grid(&parse(&v["request"])?).map_err(native_error)?,
            )
        }
        "result_table" => {
            let kind = v["kind"].as_str().ok_or("result kind must be a string")?;
            let table = fky_lapsim_core::results::result_table(kind, &v["result"]).map_err(native_error)?;
            return encoded(json!({"table": table, "csv": table.csv()}));
        }
        "example_project" => return encoded(Project::example()),
        "formula_car_demo" => {
            let (project, request) = fky_lapsim_core::dynamics::formula_car_demo().map_err(native_error)?;
            return encoded(json!({"project":project,"request":request}));
        }
        "defaults" => {
            return encoded(
                json!({"motion":Motion::default(),"ride":RideRequest::default(),"optimization":opt::OptimizationRequest::default()}),
            )
        }
        "metric_registry" => return encoded(opt::metric_registry()),
        "lap_vehicle_demo" => {
            let mut vehicle = serde_json::to_value(LapVehicle::synthetic_demo().map_err(native_error)?).map_err(native_error)?;
            vehicle["qss"] = serde_json::to_value(QssExtras::synthetic_demo()).map_err(native_error)?;
            return encoded(vehicle);
        }
        "track_demo" => {
            let shape = v["shape"].as_str().unwrap_or("oval");
            let radius_m = v["radius_m"].as_f64().unwrap_or(9.);
            let straight_m = v["straight_m"].as_f64().unwrap_or(60.);
            let width_m = v["width_m"].as_f64().unwrap_or(8.);
            let segments = v["segments"].as_u64().unwrap_or(24) as usize;
            let track = match shape {
                "circle" => fky_lapsim_core::track::Track::circle(radius_m, width_m, segments),
                _ => fky_lapsim_core::track::Track::oval(straight_m, radius_m, width_m, segments),
            }
            .map_err(native_error)?;
            return encoded(track);
        }
        "validate_lap_vehicle" => {
            let (vehicle, extras) = split_lap_vehicle(&v["vehicle"])?;
            vehicle.validate().map_err(native_error)?;
            QssVehicle::build(&vehicle, &extras).map_err(native_error)?;
            return encoded(json!({"valid": true}));
        }
        "validate_track" => {
            let track: fky_lapsim_core::track::Track = parse(&v["track"])?;
            track.validate().map_err(native_error)?;
            return encoded(json!({"valid": true}));
        }
        "validate_lap_request" => {
            let (vehicle, extras) = split_lap_vehicle(&v["vehicle"])?;
            let track: fky_lapsim_core::track::Track = parse(&v["track"])?;
            let request: lapsim::LapRequest = parse(&v["request"])?;
            vehicle.validate().map_err(native_error)?;
            QssVehicle::build(&vehicle, &extras).map_err(native_error)?;
            track.validate().map_err(native_error)?;
            request.validate().map_err(native_error)?;
            return encoded(request);
        }
        "run_lap" => {
            let (vehicle, extras) = split_lap_vehicle(&v["vehicle"])?;
            let track: fky_lapsim_core::track::Track = parse(&v["track"])?;
            let request: lapsim::LapRequest = parse(&v["request"])?;
            return encoded(
                lapsim::simulate_lap(&vehicle, &extras, &track, &request).map_err(native_error)?,
            );
        }
        "validate_lap_optimization_request" => {
            let vehicle_json = v["vehicle"].clone();
            let (vehicle, extras) = split_lap_vehicle(&vehicle_json)?;
            vehicle.validate().map_err(native_error)?;
            QssVehicle::build(&vehicle, &extras).map_err(native_error)?;
            let tracks = lap_opt_tracks(&v["tracks"])?;
            let request: lap_opt::OptimizationRequest = parse(&v["request"])?;
            request.validate().map_err(native_error)?;
            for t in &request.tracks {
                tracks.get(&t.id).ok_or_else(|| format!("no track supplied for id {}", t.id))?;
            }
            for path in request
                .variables
                .iter()
                .map(|x| &x.path)
                .chain(request.uncertainty.iter().map(|x| &x.path))
            {
                vehicle_json
                    .pointer(path)
                    .and_then(Value::as_f64)
                    .ok_or_else(|| format!("parameter path is not a numeric field: {path}"))?;
            }
            return encoded(request);
        }
        "run_lap_optimization" => {
            let vehicle_json = v["vehicle"].clone();
            let (base_vehicle, base_extras) = split_lap_vehicle(&vehicle_json)?;
            base_vehicle.validate().map_err(native_error)?;
            QssVehicle::build(&base_vehicle, &base_extras).map_err(native_error)?;
            let tracks = lap_opt_tracks(&v["tracks"])?;
            let request: lap_opt::OptimizationRequest = parse(&v["request"])?;
            let mut evaluator = |values: &[f64],
                                  track: &lap_opt::TrackCase,
                                  sample: &lap_opt::UncertaintySample|
             -> Result<std::collections::BTreeMap<String, f64>, String> {
                let mut candidate = vehicle_json.clone();
                for (variable, value) in request.variables.iter().zip(values) {
                    set_numeric_path(&mut candidate, &variable.path, *value)?;
                }
                for (path, delta) in &sample.deltas {
                    add_numeric_path(&mut candidate, path, *delta)?;
                }
                let (vehicle, extras) = split_lap_vehicle(&candidate)?;
                let (track_json, settings) = tracks
                    .get(&track.id)
                    .ok_or_else(|| format!("no track supplied for id {}", track.id))?;
                let run = lapsim::simulate_lap(&vehicle, &extras, track_json, settings)
                    .map_err(native_error)?;
                if !run.completed {
                    return Err(run.termination);
                }
                Ok(run.metrics)
            };
            return encoded(
                lap_opt::optimize(&request, &mut evaluator, || false).map_err(native_error)?,
            );
        }
        _ => {}
    }
    let p: Project = parse(&v["project"])?;
    p.validate().map_err(native_error)?;
    match op {
        "validate" => encoded(json!({"valid":true})),
        "validate_ride_request" => {
            let r: RideRequest = parse(&v["request"])?;
            fky_lapsim_core::dynamics::validate_request(&r).map_err(native_error)?;
            encoded(r)
        }
        "validate_optimization_request" => {
            let r: opt::OptimizationRequest = parse(&v["request"])?;
            if let Some(ride) = &r.ride_request {
                fky_lapsim_core::dynamics::validate_request(ride).map_err(native_error)?;
            }
            opt::OptimizationSession::start(&p, &r).map_err(native_error)?;
            encoded(r)
        }
        "validate_sweep_request" => {
            let motions: Vec<Motion> = parse(&v["request"])?;
            if motions.len() > 10000 {
                return Err("sweep limit is 10000 motions".into());
            }
            encoded(motions)
        }
        "normalize_project" => encoded(p),
        "simulate" => encoded(fky_lapsim_core::simulate(&p, &parse(&v["motion"])?).map_err(native_error)?),
        "analyze" => encoded(
            fky_lapsim_core::analyze_with_step(
                &p,
                &parse(&v["motion"])?,
                v["step"].as_f64().unwrap_or(0.0002),
            )
            .map_err(native_error)?,
        ),
        "sweep" => encoded(fky_lapsim_core::study::detailed_sweep(
            &p,
            &parse::<Vec<Motion>>(&v["motions"])?,
        )),
        "ride" => encoded(fky_lapsim_core::ride(&p, &parse(&v["request"])?).map_err(native_error)?),
        "linearize_ride" => encoded(fky_lapsim_core::dynamics::linearize_ride(&p, &parse(&v["request"])?).map_err(native_error)?),
        "parameter_registry" => encoded(opt::parameter_registry(&p)),
        "optimize" => encoded(opt::optimize(&p, &parse(&v["request"])?).map_err(native_error)?),
        "candidate_project" => encoded(
            opt::candidate_project(
                &p,
                &parse(&v["request"])?,
                &parse::<Vec<f64>>(&v["values"])?,
            )
            .map_err(native_error)?,
        ),
        _ => Err("unknown native operation".into()),
    }
}
/// Run one native `fky_lapsim_core` operation, named by `op`, against the
/// JSON document `input`, and return its result as a JSON string.
///
/// This is the single entry point almost all of `fky_lapsim`'s Python layer
/// calls through; each `op` below reads specific keys from the `input`
/// object and returns a specific JSON shape. Every case is a thin
/// serialize/deserialize wrapper around a `fky_lapsim_core` function of a
/// similar name — consult that function's own Rust documentation for the
/// physics/numerics it performs; this list only documents the wire shape.
///
/// Released from the GIL for the duration of the native call (via
/// `py.detach`), so it does not block other Python threads.
///
/// Operations that take a `"project"` (a suspension [`Project`]) validate
/// it before dispatching further, so a case-specific error always implies
/// the project itself was already known-valid:
/// - `"validate"` — input `{"project"}`. Returns `{"valid": true}`.
/// - `"normalize_project"` — input `{"project"}`. Returns the project,
///   round-tripped through validation and default-filling.
/// - `"validate_ride_request"` — input `{"project", "request"}` (a
///   [`RideRequest`]). Returns the request, round-tripped and validated
///   against the project's own geometry.
/// - `"validate_optimization_request"` — input `{"project", "request"}` (an
///   [`opt::OptimizationRequest`]). Also validates any nested ride request,
///   and opens (then discards) an [`opt::OptimizationSession`] as part of
///   validating the request against the project. Returns the request,
///   round-tripped.
/// - `"validate_sweep_request"` — input `{"project", "request"}` (a JSON
///   array of up to 10,000 [`Motion`]s; more is rejected). Returns the
///   motion list unchanged.
/// - `"simulate"` — input `{"project", "motion"}` (a [`Motion`]). Returns
///   the solved vehicle state.
/// - `"analyze"` — input `{"project", "motion", "step"}` (`"step"` is the
///   optional derivative-refinement step in metres, default `0.0002`).
///   Returns the geometric analysis (camber/toe gain, motion ratio,
///   projected centers, ...).
/// - `"sweep"` — input `{"project", "motions"}` (a JSON array of
///   [`Motion`]s). Returns one sample per motion; a motion that fails to
///   solve is represented as a failed sample in that array, not as an
///   overall error — callers must inspect each sample rather than assume
///   success.
/// - `"ride"` — input `{"project", "request"}` (a [`RideRequest`]). Runs a
///   full time-domain ride simulation and returns the resulting run
///   (samples, and a termination event if the run stopped early).
/// - `"linearize_ride"` — input `{"project", "request"}`. Returns a
///   linearized-model refinement of the same ride request.
/// - `"parameter_registry"` — input `{"project"}`. Returns the JSON-pointer
///   paths into this project that the native optimizer accepts as
///   variables or uncertainty axes.
/// - `"optimize"` — input `{"project", "request"}` (an
///   [`opt::OptimizationRequest`]). Runs a **complete, blocking** search
///   (see [`Session`] below for an incremental/resumable alternative) and
///   returns the full [`opt::optimize`] result.
/// - `"candidate_project"` — input `{"project", "request", "values"}`
///   (`"values"` a JSON array of `f64`, one per `request`'s declared
///   variable). Returns the concrete project JSON produced by applying
///   those raw values, without evaluating it.
///
/// Operations with no `"project"` key:
/// - `"motion_grid"` — input `{"request"}`. Returns the generated motion
///   grid for that request.
/// - `"result_table"` — input `{"kind", "result"}` (`"kind"` selects which
///   result shape `"result"` is). Returns `{"table", "csv"}`.
/// - `"example_project"` — input ignored. Returns the bundled synthetic
///   example [`Project`].
/// - `"formula_car_demo"` — input ignored. Returns
///   `{"project", "request"}`, a synthetic formula-car project paired with
///   a demo ride request.
/// - `"defaults"` — input ignored. Returns
///   `{"motion", "ride", "optimization"}`, the default value of each
///   corresponding request type.
/// - `"metric_registry"` — input ignored. Returns the metrics the native
///   suspension optimizer can target.
/// - `"lap_vehicle_demo"` — input ignored. Returns a synthetic demo
///   [`fky_lapsim_core::lap::LapVehicle`].
/// - `"track_demo"` — input `{"shape", "radius_m", "straight_m",
///   "width_m", "segments"}`, all optional (`shape` one of `"oval"`
///   (default) or `"circle"`; `straight_m` only affects `"oval"`). Returns
///   a generated synthetic [`fky_lapsim_core::track::Track`].
/// - `"validate_lap_vehicle"` — input `{"vehicle"}`. Returns
///   `{"valid": true}`.
/// - `"validate_track"` — input `{"track"}`. Returns `{"valid": true}`.
/// - `"validate_lap_request"` — input `{"vehicle", "track", "request"}` (a
///   [`fky_lapsim_core::lapsim::LapRequest`]). Validates all three together
///   and returns the request, round-tripped.
/// - `"run_lap"` — input `{"vehicle", "track", "request"}`. Runs a full lap
///   simulation and returns its result (time-series samples, metrics).
/// - `"validate_lap_optimization_request"` — input `{"vehicle", "tracks",
///   "request"}` (`"tracks"` a `{track_id: {"track", "settings"}}` map
///   covering every track id the request references; `"request"` a
///   [`lap_opt::OptimizationRequest`]). Additionally checks that every
///   variable's and uncertainty axis's JSON-pointer path resolves to a
///   numeric field on `vehicle`. Returns the request, round-tripped.
/// - `"run_lap_optimization"` — input shaped like
///   `"validate_lap_optimization_request"`. Runs a **complete, blocking**
///   multi-track stochastic search — every candidate re-solves a full lap
///   simulation per track/uncertainty sample, there is no surrogate model —
///   and returns the [`lap_opt::optimize`] result. Unlike [`Session`] and
///   [`evaluate`], this case has no [`CancellationToken`] hook: once
///   called, it runs to completion or failure with no way to cancel it from
///   Python.
///
/// # Errors
///
/// Returns `Err` (as a Python `ValueError`, prefixed with `op`) for
/// malformed `input` JSON, an unrecognized `op`, a value that fails to
/// deserialize or validate, or a native computation failure. See [`run`].
#[pyfunction]
fn call(py: Python<'_>, op: &str, input: &str) -> PyResult<String> {
    py.detach(|| run(op, input))
        .map_err(|e| PyValueError::new_err(format!("{op}: {e}")))
}
/// A cooperative cancellation flag shared between Python and a running
/// native call.
///
/// Create one, pass it to [`evaluate`] or [`Session::advance`], and call
/// [`CancellationToken::cancel`] from another Python thread to ask the
/// native side to stop. Cancellation is cooperative: `fky_lapsim_core`
/// checks the flag between discrete units of work (e.g. between candidates
/// or generations), so a physics case already in progress when the flag is
/// set can still finish before the call returns.
#[pyclass]
#[derive(Clone)]
struct CancellationToken {
    flag: Arc<AtomicBool>,
}
#[pymethods]
impl CancellationToken {
    /// Create a token that is not cancelled.
    #[new]
    fn new() -> Self {
        Self {
            flag: Arc::new(AtomicBool::new(false)),
        }
    }
    /// Request cancellation. Idempotent, and safe to call from any thread.
    fn cancel(&self) {
        self.flag.store(true, Ordering::Relaxed);
    }
    /// Whether [`CancellationToken::cancel`] has been called on this token
    /// (or a clone of it — cloning shares the same underlying flag).
    #[getter]
    fn cancelled(&self) -> bool {
        self.flag.load(Ordering::Relaxed)
    }
}

/// Evaluate one optimizer candidate: apply `values` to the variables
/// declared in `request` on top of `project`, run the resulting design
/// through the same physics evaluator the native optimizer itself uses,
/// and return the result. Intended for external search algorithms (e.g.
/// Python `pymoo`/SciPy adapters) that want to reuse this crate's physics
/// without reimplementing the native differential-evolution search.
///
/// `input` is a JSON object `{"project", "request", "values", "validation"}`:
/// `"values"` is a JSON array of `f64`, one per variable declared in
/// `request`; `"validation"` (default `false`) selects whether this
/// candidate is scored against `request`'s training cases or its held-out
/// validation cases. Released from the GIL for the duration of the call.
///
/// # Errors
///
/// Returns a Python `ValueError` if `input` is malformed, if `project` or
/// `request` fail to deserialize, or if the underlying evaluation fails
/// (e.g. a values count mismatch, an unresolved parameter path, or a
/// physics case that fails to complete).
#[pyfunction]
fn evaluate(py: Python<'_>, input: &str, cancel: &CancellationToken) -> PyResult<String> {
    py.detach(|| -> Result<String, String> {
        let v: Value = serde_json::from_str(input).map_err(native_error)?;
        let e = opt::evaluate_candidate_controlled(
            &parse(&v["project"])?,
            &parse(&v["request"])?,
            &parse::<Vec<f64>>(&v["values"])?,
            v["validation"].as_bool().unwrap_or(false),
            &|| cancel.flag.load(Ordering::Relaxed),
        )
        .map_err(native_error)?;
        encoded(e)
    })
    .map_err(|e| PyValueError::new_err(format!("evaluate_candidate: {e}")))
}

/// A stateful, checkpointable native suspension-optimization search.
///
/// Unlike [`call`]'s `"optimize"` op, which blocks for the whole search,
/// `Session` lets Python drive the search incrementally: construct one,
/// call [`Session::advance`] repeatedly with a work budget, and inspect
/// [`Session::result`] or [`Session::is_finished`] between calls — useful
/// for reporting progress, enforcing a wall-clock deadline from Python, or
/// stopping early via a [`CancellationToken`]. Serialize
/// [`Session::checkpoint`] to persist a paused search (including across
/// process restarts) and reconstruct a `Session` from it later with the
/// same `project`/`request`; a checkpoint from a different project or
/// request is rejected.
#[pyclass]
struct Session {
    inner: opt::OptimizationSession,
}
#[pymethods]
impl Session {
    /// Start a new search over `project`/`request` (both JSON strings), or
    /// resume one from a previously serialized `checkpoint` — in which case
    /// `project` and `request` must match the ones the checkpoint was taken
    /// with. Released from the GIL for the duration of the call.
    ///
    /// # Errors
    ///
    /// Returns a Python `ValueError` if `project` or `request` fail to
    /// deserialize or validate, or if `checkpoint` is present but does not
    /// match `project`/`request` or fails to parse.
    #[new]
    #[pyo3(signature=(project, request, checkpoint=None))]
    fn new(
        py: Python<'_>,
        project: &str,
        request: &str,
        checkpoint: Option<&str>,
    ) -> PyResult<Self> {
        py.detach(|| -> Result<Self, String> {
            let p = serde_json::from_str(project).map_err(native_error)?;
            let r = serde_json::from_str(request).map_err(native_error)?;
            let inner = match checkpoint {
                Some(c) => opt::OptimizationSession::resume(&p, &r, c),
                None => opt::OptimizationSession::start(&p, &r),
            }
            .map_err(native_error)?;
            Ok(Self { inner })
        })
        .map_err(PyValueError::new_err)
    }
    /// Run up to `max_work` further units of search work (candidate
    /// evaluations), or stop early if `cancel` is cancelled first, and
    /// return the search's current result. Pause time between calls to
    /// `advance` is excluded from the search's own deadline accounting.
    /// Released from the GIL for the duration of the call.
    ///
    /// # Errors
    ///
    /// Returns a Python `ValueError` if the underlying native advance
    /// fails (e.g. a physics case that cannot complete).
    fn advance(
        &mut self,
        py: Python<'_>,
        max_work: usize,
        cancel: &CancellationToken,
    ) -> PyResult<String> {
        py.detach(|| {
            self.inner
                .advance(max_work, Some(&cancel.flag))
                .map_err(native_error)
                .and_then(encoded)
        })
        .map_err(PyValueError::new_err)
    }
    /// The search's current result (best feasible design found so far, and
    /// its status), without advancing it further.
    ///
    /// # Errors
    ///
    /// Returns a Python `ValueError` if the result fails to serialize
    /// (not expected in practice).
    fn result(&self) -> PyResult<String> {
        encoded(self.inner.result()).map_err(PyValueError::new_err)
    }
    /// Serialize this session's current progress (including any partially
    /// completed generation) so it can be resumed later via
    /// [`Session::new`]'s `checkpoint` argument. Released from the GIL for
    /// the duration of the call.
    ///
    /// # Errors
    ///
    /// Returns a Python `ValueError` if checkpointing fails.
    fn checkpoint(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| self.inner.checkpoint())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    /// Whether the search has reached its budget or deadline and will make
    /// no further progress on subsequent [`Session::advance`] calls.
    #[getter]
    fn is_finished(&self) -> bool {
        self.inner.is_finished()
    }
}

/// Register this crate's Python-visible functions and classes as the
/// `fky_lapsim._native` extension module. See the crate-level documentation
/// for how [`call`], [`evaluate`], [`CancellationToken`], and [`Session`]
/// fit together.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(call, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate, m)?)?;
    m.add_class::<CancellationToken>()?;
    m.add_class::<Session>()?;
    Ok(())
}
