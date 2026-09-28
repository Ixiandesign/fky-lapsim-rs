use dw_core::{lap::optimization as lap_opt, optimize as opt, Motion, Project, RideRequest};
use pyo3::{exceptions::PyValueError, prelude::*};
use serde::{de::DeserializeOwned, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
fn parse<T: DeserializeOwned>(v: &Value) -> Result<T, String> {
    serde_json::from_value(v.clone()).map_err(|e| e.to_string())
}
fn encoded<T: Serialize>(v: T) -> Result<String, String> {
    serde_json::to_string(&v).map_err(|e| e.to_string())
}
fn native_error(e: impl std::fmt::Display) -> String {
    e.to_string()
}
/// Set a numeric field addressed by JSON Pointer `path`, in place.
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
/// Add to a numeric field addressed by JSON Pointer `path`, in place — the
/// optimizer's uncertainty deltas are additive perturbations, applied after
/// a candidate's variables are already set (matching dw_core::optimize's
/// Perturbation semantics).
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
/// Parse the `{"<track_id>": {"track": Track, "settings": LapRequest}}` map
/// a lap-optimization request's evaluator resolves `TrackCase.id`s against.
fn lap_opt_tracks(v: &Value) -> Result<BTreeMap<String, (dw_core::track::Track, dw_core::lap::LapRequest)>, String> {
    let map: BTreeMap<String, Value> = parse(v)?;
    map.into_iter()
        .map(|(id, entry)| {
            let track: dw_core::track::Track = parse(&entry["track"])?;
            track.validate()?;
            let settings: dw_core::lap::LapRequest = parse(&entry["settings"])?;
            settings.validate().map_err(native_error)?;
            Ok((id, (track, settings)))
        })
        .collect()
}
fn run(op: &str, input: &str) -> Result<String, String> {
    let v: Value = serde_json::from_str(input).map_err(native_error)?;
    match op {
        "motion_grid" => {
            return encoded(
                dw_core::study::motion_grid(&parse(&v["request"])?).map_err(native_error)?,
            )
        }
        "result_table" => {
            let kind = v["kind"].as_str().ok_or("result kind must be a string")?;
            let table = dw_core::results::result_table(kind, &v["result"]).map_err(native_error)?;
            return encoded(json!({"table": table, "csv": table.csv()}));
        }
        "example_project" => return encoded(Project::example()),
        "formula_car_demo" => {
            let (project, request) = dw_core::dynamics::formula_car_demo().map_err(native_error)?;
            return encoded(json!({"project":project,"request":request}));
        }
        "defaults" => {
            return encoded(
                json!({"motion":Motion::default(),"ride":RideRequest::default(),"optimization":opt::OptimizationRequest::default()}),
            )
        }
        "metric_registry" => return encoded(opt::metric_registry()),
        "lap_vehicle_demo" => return encoded(dw_core::lap::LapVehicle::synthetic_demo().map_err(native_error)?),
        "track_demo" => {
            let shape = v["shape"].as_str().unwrap_or("oval");
            let radius_m = v["radius_m"].as_f64().unwrap_or(9.);
            let straight_m = v["straight_m"].as_f64().unwrap_or(60.);
            let width_m = v["width_m"].as_f64().unwrap_or(8.);
            let segments = v["segments"].as_u64().unwrap_or(24) as usize;
            let track = match shape {
                "circle" => dw_core::track::Track::circle(radius_m, width_m, segments),
                _ => dw_core::track::Track::oval(straight_m, radius_m, width_m, segments),
            }
            .map_err(native_error)?;
            return encoded(track);
        }
        "validate_lap_vehicle" => {
            let vehicle: dw_core::lap::LapVehicle = parse(&v["vehicle"])?;
            vehicle.validate().map_err(native_error)?;
            return encoded(json!({"valid": true}));
        }
        "validate_track" => {
            let track: dw_core::track::Track = parse(&v["track"])?;
            track.validate().map_err(native_error)?;
            return encoded(json!({"valid": true}));
        }
        "validate_lap_request" => {
            let vehicle: dw_core::lap::LapVehicle = parse(&v["vehicle"])?;
            let track: dw_core::track::Track = parse(&v["track"])?;
            let request: dw_core::lap::LapRequest = parse(&v["request"])?;
            vehicle.validate().map_err(native_error)?;
            track.validate().map_err(native_error)?;
            request.validate().map_err(native_error)?;
            return encoded(request);
        }
        "run_lap" => {
            let vehicle: dw_core::lap::LapVehicle = parse(&v["vehicle"])?;
            let track: dw_core::track::Track = parse(&v["track"])?;
            let request: dw_core::lap::LapRequest = parse(&v["request"])?;
            return encoded(
                dw_core::lap::simulate_lap(&vehicle, &track, &request).map_err(native_error)?,
            );
        }
        "validate_lap_optimization_request" => {
            let vehicle_json = v["vehicle"].clone();
            let vehicle: dw_core::lap::LapVehicle = parse(&vehicle_json)?;
            vehicle.validate().map_err(native_error)?;
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
            let base_vehicle: dw_core::lap::LapVehicle = parse(&vehicle_json)?;
            base_vehicle.validate().map_err(native_error)?;
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
                let vehicle: dw_core::lap::LapVehicle = parse(&candidate)?;
                let (track_json, settings) = tracks
                    .get(&track.id)
                    .ok_or_else(|| format!("no track supplied for id {}", track.id))?;
                let run = dw_core::lap::simulate_lap(&vehicle, track_json, settings)
                    .map_err(native_error)?;
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
            dw_core::dynamics::validate_request(&r).map_err(native_error)?;
            encoded(r)
        }
        "validate_optimization_request" => {
            let r: opt::OptimizationRequest = parse(&v["request"])?;
            if let Some(ride) = &r.ride_request {
                dw_core::dynamics::validate_request(ride).map_err(native_error)?;
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
        "simulate" => encoded(dw_core::simulate(&p, &parse(&v["motion"])?).map_err(native_error)?),
        "analyze" => encoded(
            dw_core::analyze_with_step(
                &p,
                &parse(&v["motion"])?,
                v["step"].as_f64().unwrap_or(0.0002),
            )
            .map_err(native_error)?,
        ),
        "sweep" => encoded(dw_core::study::detailed_sweep(
            &p,
            &parse::<Vec<Motion>>(&v["motions"])?,
        )),
        "ride" => encoded(dw_core::ride(&p, &parse(&v["request"])?).map_err(native_error)?),
        "linearize_ride" => encoded(dw_core::dynamics::linearize_ride(&p, &parse(&v["request"])?).map_err(native_error)?),
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
#[pyfunction]
fn call(py: Python<'_>, op: &str, input: &str) -> PyResult<String> {
    py.detach(|| run(op, input))
        .map_err(|e| PyValueError::new_err(format!("{op}: {e}")))
}
#[pyclass]
#[derive(Clone)]
struct CancellationToken {
    flag: Arc<AtomicBool>,
}
#[pymethods]
impl CancellationToken {
    #[new]
    fn new() -> Self {
        Self {
            flag: Arc::new(AtomicBool::new(false)),
        }
    }
    fn cancel(&self) {
        self.flag.store(true, Ordering::Relaxed);
    }
    #[getter]
    fn cancelled(&self) -> bool {
        self.flag.load(Ordering::Relaxed)
    }
}
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
#[pyclass]
struct Session {
    inner: opt::OptimizationSession,
}
#[pymethods]
impl Session {
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
    fn result(&self) -> PyResult<String> {
        encoded(self.inner.result()).map_err(PyValueError::new_err)
    }
    fn checkpoint(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| self.inner.checkpoint())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    #[getter]
    fn is_finished(&self) -> bool {
        self.inner.is_finished()
    }
}
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(call, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate, m)?)?;
    m.add_class::<CancellationToken>()?;
    m.add_class::<Session>()?;
    Ok(())
}
