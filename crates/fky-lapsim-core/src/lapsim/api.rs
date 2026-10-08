//! Public entry point of the QSS lap simulator: configuration in, one serializable [`LapRun`] out.
use super::channels::{driven_channels, kpis, Channels, KpiSettings, Kpis};
use super::coupled::TyreMode;
use super::err;
use super::solver::{simulate, LapSettings, LapTrace};
use super::thesis::Correlation;
use super::track_model::TrackModel;
use super::vehicle::{DerivedTyre, QssExtras, QssVehicle};
use crate::lap::LapVehicle;
use crate::track::Track;
use crate::Error;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Local re-meshing around tight corners (thesis §4.2.3.2).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct FineMesh {
    /// Corners tighter than this radius get the fine mesh, m.
    pub threshold_radius_m: f64,
    /// Extent of the fine mesh before and after such a corner, m.
    pub span_m: f64,
    /// Section length inside the fine mesh, m.
    pub step_m: f64,
}

impl Default for FineMesh {
    fn default() -> Self {
        Self {
            threshold_radius_m: 35.0,
            span_m: 2.0,
            step_m: 0.1,
        }
    }
}

/// Lap simulation request. Every field has a default, so `{}` is a valid request.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LapRequest {
    /// Flying lap (periodic) instead of a standing start.
    pub flying: bool,
    /// Standing-start initial speed, m/s.
    pub initial_speed_m_s: f64,
    /// Minimum distance between distinct apexes, m.
    pub apex_min_spacing_m: f64,
    /// Track sampling step, m.
    pub resample_step_m: f64,
    /// Fine mesh around tight corners (`None` disables it).
    pub fine_mesh: Option<FineMesh>,
    /// Couple the bike model to the 7×7 geometry matrix (false = thesis-exact bike model).
    pub use_matrix: bool,
    /// Apply longitudinal weight transfer in the bike model.
    pub weight_transfer: bool,
    /// Back-calculate the driven channels and KPIs.
    pub include_channels: bool,
    /// KPI classification thresholds.
    pub kpi: KpiSettings,
}

impl Default for LapRequest {
    fn default() -> Self {
        Self {
            flying: true,
            initial_speed_m_s: 0.0,
            apex_min_spacing_m: 2.0,
            resample_step_m: 0.5,
            fine_mesh: Some(FineMesh::default()),
            use_matrix: true,
            weight_transfer: true,
            include_channels: true,
            kpi: KpiSettings::default(),
        }
    }
}

impl LapRequest {
    /// Check the request.
    ///
    /// # Errors
    /// Returns an error naming the first invalid field.
    pub fn validate(&self) -> Result<(), Error> {
        let positive = |v: f64| v.is_finite() && v > 0.0;
        if !positive(self.resample_step_m) {
            return Err(err("resample_step_m must be positive"));
        }
        if !positive(self.apex_min_spacing_m) {
            return Err(err("apex_min_spacing_m must be positive"));
        }
        if !self.initial_speed_m_s.is_finite() || self.initial_speed_m_s < 0.0 {
            return Err(err("initial_speed_m_s must be nonnegative"));
        }
        if let Some(f) = &self.fine_mesh {
            if !(positive(f.threshold_radius_m) && positive(f.span_m) && positive(f.step_m)) {
                return Err(err("fine_mesh values must be positive"));
            }
        }
        Ok(())
    }
}

/// Outcome of one lap simulation. A physics failure inside the lap (wheel lift-off, coupling
/// non-convergence, …) is reported as an incomplete run, not an error, so callers sweeping over
/// designs can treat the case as infeasible.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LapRun {
    /// Fidelity tag recorded with every result.
    pub model_fidelity: String,
    /// True when a lap time was produced.
    pub completed: bool,
    /// `"completed"`, or the failure message.
    pub termination: String,
    /// Lap time, s.
    pub lap_time_s: Option<f64>,
    /// Scalar metrics (`lap_time_s`, `mean_speed_m_s`, `peak_speed_m_s`, `min_speed_m_s`,
    /// `peak_lateral_acceleration_m_s2`, `track_length_m`).
    pub metrics: BTreeMap<String, f64>,
    /// Per-point lap trace.
    pub trace: Option<LapTrace>,
    /// Driven channels.
    pub channels: Option<Channels>,
    /// Key performance indicators.
    pub kpis: Option<Kpis>,
    /// Friction parameters fitted to the tyres (front, rear), with provenance.
    pub derived_tyres: Vec<DerivedTyre>,
    /// Correlation factors in force.
    pub correlation: Correlation,
}

/// Below this cornering limit a track is declared infeasible for the vehicle, m/s.
const MIN_FEASIBLE_SPEED_M_S: f64 = 0.5;

fn fidelity(v: &QssVehicle, matrix: bool) -> String {
    let model = if matrix {
        "coupled_7x7_geometry"
    } else {
        "thesis_exact"
    };
    let mu = match v.tyre_mode {
        TyreMode::Thesis => "mu_thesis",
        TyreMode::Table(_) => "mu_table",
    };
    let aero = if v.aero_map.is_some() {
        "aeromap"
    } else {
        "const_aero"
    };
    format!("qss_zacharelis2023_bike_{model}_{mu}_{aero}_v1")
}

/// Simulate one lap.
///
/// # Errors
/// Invalid request, vehicle, extras or track. Physics failures during the lap return `Ok` with
/// `completed == false`.
pub fn simulate_lap(
    car: &LapVehicle,
    extras: &QssExtras,
    track: &Track,
    req: &LapRequest,
) -> Result<LapRun, Error> {
    req.validate()?;
    let vehicle = QssVehicle::build(car, extras)?;
    track.validate().map_err(err)?;
    let model = vehicle.step_model(req.use_matrix, req.weight_transfer)?;
    let mut mesh = TrackModel::from_track(track, req.resample_step_m)?;
    if let Some(f) = &req.fine_mesh {
        mesh = mesh.fine_mesh(f.threshold_radius_m, f.span_m, f.step_m)?;
    }
    let settings = LapSettings {
        flying: req.flying,
        initial_speed_m_s: req.initial_speed_m_s,
        apex_min_spacing_m: req.apex_min_spacing_m,
    };
    let mut run = LapRun {
        model_fidelity: fidelity(&vehicle, req.use_matrix),
        completed: false,
        termination: String::new(),
        lap_time_s: None,
        metrics: BTreeMap::new(),
        trace: None,
        channels: None,
        kpis: None,
        derived_tyres: vehicle.derived_tyres.to_vec(),
        correlation: vehicle.correlation,
    };
    let result = match simulate(model.as_ref(), &vehicle.tractive, &mesh, &settings) {
        Ok(r) => r,
        Err(e) => {
            run.termination = e.message;
            return Ok(run);
        }
    };
    let slowest_limit = result
        .trace
        .vmax_m_s
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    if slowest_limit < MIN_FEASIBLE_SPEED_M_S {
        run.termination = format!(
            "infeasible: the vehicle cannot hold more than {slowest_limit:.3} m/s somewhere on the track"
        );
        return Ok(run);
    }
    let mut channels = None;
    if req.include_channels {
        match driven_channels(model.as_ref(), &vehicle.tractive, &result.trace) {
            Ok(c) => channels = Some(c),
            Err(e) => {
                run.termination = e.message;
                return Ok(run);
            }
        }
    }
    let t = result.lap_time_s;
    let speeds = &result.trace.speed_m_s;
    let fold = |init: f64, f: fn(f64, f64) -> f64, xs: &[f64]| xs.iter().copied().fold(init, f);
    let peak = fold(f64::NEG_INFINITY, f64::max, speeds);
    let low = fold(f64::INFINITY, f64::min, speeds);
    let ay = fold(0.0, f64::max, &result.trace.ay_m_s2);
    run.metrics.insert("lap_time_s".into(), t);
    run.metrics
        .insert("mean_speed_m_s".into(), mesh.length_m / t);
    run.metrics.insert("peak_speed_m_s".into(), peak);
    run.metrics.insert("min_speed_m_s".into(), low);
    run.metrics
        .insert("peak_lateral_acceleration_m_s2".into(), ay);
    run.metrics.insert("track_length_m".into(), mesh.length_m);
    run.kpis = channels
        .as_ref()
        .map(|c| kpis(c, t, mesh.length_m, &req.kpi));
    run.channels = channels;
    run.trace = Some(result.trace);
    run.lap_time_s = Some(t);
    run.completed = true;
    run.termination = "completed".into();
    Ok(run)
}
