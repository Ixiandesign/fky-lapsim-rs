use dw_core::{optimize as opt, Motion, Project, RideRequest};
use pyo3::{exceptions::PyValueError, prelude::*};
use serde::{de::DeserializeOwned, Serialize};
use serde_json::{json, Value};
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
        "track_demo" => return encoded(dw_core::track::Track::oval(60., 9., 8., 24).map_err(native_error)?),
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
