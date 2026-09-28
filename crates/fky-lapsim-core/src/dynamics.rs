//! Nonlinear sprung-body dynamics on prescribed horizontal supports.
//!
//! q = [heave m, roll rad, pitch rad], rotating Ry(pitch)Rx(roll) about chassis COM.
//! Compression derivatives come from exact geometry closure, with finite-difference
//! refinement. RK4 is conditionally stable: check timestep convergence for each setup.
//! Select retained component inertia explicitly; contact release remains unsupported.
use crate::{AxleInterconnect, Chassis, SpringDamper};
use crate::{CornerId, Error, Project};
use serde::{Deserialize, Serialize};
mod linearization;
pub use linearization::{linearize_ride, RideLinearization, RideModeShape};

/// Fidelity identifier reported in [RideRun::model_fidelity] for [RideMode::Reduced].
pub const MODEL_FIDELITY: &str = "nonlinear_sprung_body_fixed_contact_massless_links";
/// Fidelity identifier reported in [RideRun::model_fidelity] for [RideMode::RetainedComponentInertia].
pub const COMPONENT_MODEL_FIDELITY: &str =
    "nonlinear_rigid_arm_rocker_nonspinning_knuckle_inertia_prescribed_fixed_contact";
/// Ride dynamics fidelity: whether each corner's link/knuckle mass is retained.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RideMode {
    /// Massless wishbones/rocker/knuckle; only the chassis sprung mass, springs, and
    /// dampers carry inertia and weight.
    #[default]
    Reduced,
    /// Include each corner's configured upper/lower arm, rocker, and nonspinning
    /// knuckle/wheel inertia (see [crate::ComponentMasses]). Requires a road input
    /// with continuous velocity; rejects [RoadInput::Histories].
    RetainedComponentInertia,
}
/// Prescribed road height at each corner's wheel, as a function of time.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RoadInput {
    /// Horizontal road at z = 0 for the whole run.
    #[default]
    Flat,
    /// Temporal sinusoid in Hz; phases in radians, ordered like Project.corners.
    Sine {
        /// Road height amplitude, metres.
        amplitude_m: f64,
        /// Oscillation frequency, Hz (must be finite and >= 0).
        frequency_hz: f64,
        /// Per-corner phase offset, radians, ordered like `Project.corners`.
        phases_rad: [f64; 4],
    },
    /// Spatial sinusoid sampled at design wheel x plus speed*time.
    SpatialSine {
        /// Road height amplitude, metres.
        amplitude_m: f64,
        /// Spatial wavelength, metres (must be finite and > 0).
        wavelength_m: f64,
        /// Forward travel speed used to convert wheel x-position into a moving phase, m/s.
        speed_m_s: f64,
        /// Per-corner phase offset, radians, ordered like `Project.corners`.
        phases_rad: [f64; 4],
    },
    /// Continuous piecewise linear heights; times must span the complete run.
    /// Samples are [time seconds, height metres]. Velocity is right-continuous at
    /// interior knots; integration ends the preceding interval with its left velocity.
    Histories {
        /// Per-corner `[time_s, height_m]` samples, ordered like `Project.corners`;
        /// each series must be sorted, span `[0, duration_s]`, and have at least two points.
        corners: [Vec<[f64; 2]>; 4],
    },
}
/// A ride dynamics run: fidelity mode, fixed rack travel, initial state/road, and
/// integration settings. Passed to [ride].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RideRequest {
    /// Dynamics fidelity; see [RideMode].
    pub mode: RideMode,
    /// Fixed rack travel in metres for the complete ride.
    pub rack_front: f64,
    /// Fixed rear rack travel in metres for the complete ride.
    pub rack_rear: f64,
    /// Coarse central difference step, metres for translations/radians for angles.
    /// Mass mode verifies half-step acceleration and support reaction convergence.
    pub derivative_step: f64,
    /// Total simulated duration, seconds.
    pub duration_s: f64,
    /// Fixed integration timestep, seconds; a failed stage halves it down to
    /// `min(dt_s, 1e-5 s)` before the run terminates.
    pub dt_s: f64,
    /// Offsets from equilibrium when solve_equilibrium is true, otherwise absolute.
    pub initial_displacement: [f64; 3],
    /// Heave m/s, roll/pitch rad/s.
    pub initial_velocity: [f64; 3],
    /// Generalized heave force / roll and pitch moments, conjugate to q.
    pub external_force: [f64; 3],
    /// Optional piecewise-linear `[time_s, heave_N, roll_Nm, pitch_Nm]` history.
    /// Replaces `external_force`; at least two finite, strictly increasing rows
    /// must cover `[0, duration_s]`. Equilibrium uses its value at time zero.
    pub external_force_history: Option<Vec<[f64; 4]>>,
    /// Include solved suspension geometry in accepted samples. Defaults to false;
    /// does not request the additional full-vehicle derivative analysis.
    pub report_states: bool,
    /// Solve with the road frozen at t=0 before applying initial offsets.
    pub solve_equilibrium: bool,
    /// Prescribed road input for the run; see [RoadInput].
    pub road: RoadInput,
}
impl Default for RideRequest {
    fn default() -> Self {
        Self {
            mode: RideMode::Reduced,
            rack_front: 0.,
            rack_rear: 0.,
            derivative_step: 0.001,
            duration_s: 1.,
            dt_s: 0.005,
            initial_displacement: [0.; 3],
            initial_velocity: [0.; 3],
            external_force: [0.; 3],
            external_force_history: None,
            report_states: false,
            solve_equilibrium: true,
            road: RoadInput::Flat,
        }
    }
}
/// One axle [crate::AxleInterconnect]'s symmetric (heave) and antisymmetric (roll)
/// channel state at a [RideSample]: compression, compression rate, and combined
/// spring+damper force, in the same conventions as the corner-level fields on
/// [RideSample] (positive compression/velocity in compression; see [spring_force]/
/// [damper_force]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterconnectSample {
    /// Symmetric (heave) channel compression, metres.
    pub heave_compression_m: f64,
    /// Symmetric (heave) channel compression rate, m/s.
    pub heave_velocity_m_s: f64,
    /// Symmetric (heave) channel combined spring+damper force, newtons.
    pub heave_force_n: f64,
    /// Antisymmetric (roll) channel compression, metres.
    pub roll_compression_m: f64,
    /// Antisymmetric (roll) channel compression rate, m/s.
    pub roll_velocity_m_s: f64,
    /// Antisymmetric (roll) channel combined spring+damper force, newtons.
    pub roll_force_n: f64,
}
/// One accepted integration sample of a [RideRun].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RideSample {
    /// Solved geometry at this sample's motion, rack travel and prescribed road.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<crate::VehicleState>,
    /// Simulation time, seconds, since the start of the run.
    pub time_s: f64,
    /// Generalized chassis state `[heave m, roll rad, pitch rad]`.
    pub displacement: [f64; 3],
    /// Generalized chassis rate `[heave m/s, roll rad/s, pitch rad/s]`.
    pub velocity: [f64; 3],
    /// Generalized chassis acceleration `[heave m/s^2, roll rad/s^2, pitch rad/s^2]`.
    pub acceleration: [f64; 3],
    /// Per-corner shock compression relative to design length, metres.
    pub compression_m: [f64; 4],
    /// Per-corner shock compression rate, m/s; positive in compression.
    pub compression_velocity_m_s: [f64; 4],
    /// Per-corner combined spring+damper force, newtons; see [spring_force]/[damper_force].
    pub shock_force_n: [f64; 4],
    /// Vertical prescribed-support reactions, including retained component inertia in mass
    /// mode and any active axle interconnect's contribution (see [InterconnectSample]).
    pub support_reaction_n: [f64; 4],
    /// Front-axle interconnect state, present exactly when [crate::Project::front_interconnect]
    /// is configured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub front_interconnect: Option<InterconnectSample>,
    /// Rear-axle interconnect state, present exactly when [crate::Project::rear_interconnect]
    /// is configured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rear_interconnect: Option<InterconnectSample>,
    /// Total mechanical energy (kinetic + gravitational + spring potential) at this sample, joules.
    pub energy_j: f64,
    /// Positive integral of signed damper force times compression velocity.
    pub dissipated_work_j: f64,
    /// Integral of the selected model's support reactions times prescribed road velocity.
    pub support_work_j: f64,
    /// Integral of the requested generalized external force/moment along the chassis rate.
    pub external_work_j: f64,
    /// `energy_j` minus the initial energy plus dissipated/support/external work; an
    /// energy-conservation residual used as a correctness check, not a physical quantity.
    pub energy_balance_error_j: f64,
}
/// Why a [RideRun] stopped before its requested `duration_s`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RideTermination {
    /// Time of the failed integration-stage or accepted-state evaluation.
    pub time_s: f64,
    /// Time of the last successfully accepted sample, if any.
    pub last_valid_time_s: Option<f64>,
    /// The corner whose geometry or contact condition triggered termination, if identifiable.
    pub corner: Option<CornerId>,
    /// Human-readable termination reason.
    pub reason: String,
}
/// The result of [ride]: fidelity used, retained samples, and any termination event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RideRun {
    /// [MODEL_FIDELITY] or [COMPONENT_MODEL_FIDELITY], matching the request's [RideMode].
    pub model_fidelity: String,
    /// Corner identifiers, in the same order as `compression_m`/`shock_force_n`/etc.
    pub corner_ids: [CornerId; 4],
    /// Solved static-equilibrium `[heave, roll, pitch]`, if `solve_equilibrium` was requested.
    pub equilibrium: Option<[f64; 3]>,
    /// Every accepted sample, including the initial state, in time order.
    pub samples: Vec<RideSample>,
    /// None means the requested duration was completed.
    pub termination: Option<RideTermination>,
}
/// Integrate coupled heave/roll/pitch using RK4 and exact nonlinear ground closure.
/// Reduced mode omits component inertia; retained mode includes configured bodies.
/// Supports are rigid horizontal planes in both modes.
/// Tire compliance/friction/spin, horizontal/yaw motion and contact release are absent.
/// Geometry failure means loss of solver validity, not a calibrated physical bump stop.
/// Negative support below -1e-6 N terminates contact validity. Stage failures are
/// localized by halving the step to min(request.dt_s, 1e-5 s).
/// Invalid input, work limits and failed static equilibrium return Err; a terminal
/// integration event returns Ok with termination populated and the valid prefix.
pub fn ride(p: &Project, r: &RideRequest) -> Result<RideRun, Error> {
    p.validate()?;
    validate_request(r)?;
    let equilibrium = if r.solve_equilibrium {
        Some(equilibrate(p, r)?)
    } else {
        None
    };
    let mut run = RideRun {
        model_fidelity: if r.mode == RideMode::Reduced {
            MODEL_FIDELITY
        } else {
            COMPONENT_MODEL_FIDELITY
        }
        .into(),
        corner_ids: p.corners.each_ref().map(|c| c.id),
        equilibrium,
        samples: Vec::new(),
        termination: None,
    };
    let mut y = [0.; 9];
    for a in 0..3 {
        y[a] = equilibrium.unwrap_or([0.; 3])[a] + r.initial_displacement[a];
        y[a + 3] = r.initial_velocity[a];
    }
    let mut t = 0.;
    let first = match evaluate_reported(p, r, t, y) {
        Ok(s) => s,
        Err(e) => {
            run.termination = Some(event(p, 0., None, e));
            return Ok(run);
        }
    };
    let e0 = first.energy_j;
    run.samples.push(first);
    while t < r.duration_s {
        if run.samples.len() >= MAX_SAMPLES {
            return Err(error("ride sample/work limit exceeded"));
        }
        let mut end = next_knot(&r.road, t, r.duration_s.min(t + r.dt_s));
        if let Some(history) = &r.external_force_history {
            if let Some(row) = history.get(history.partition_point(|row| row[0] <= t)) {
                end = end.min(row[0]);
            }
        }
        let mut dt = end - t;
        loop {
            let next = t + dt;
            let mut failed_time = next;
            let integrated = rk4(y, t, dt, |time, state| {
                failed_time = time;
                let s = evaluate(p, r, time, state, time == next)?;
                let (_, zd) = road_at(p, &r.road, time, time == next);
                let diss = (0..4)
                    .map(|i| {
                        damper_force(&p.corners[i].spring_damper, s.compression_velocity_m_s[i])
                            * s.compression_velocity_m_s[i]
                    })
                    .sum::<f64>()
                    + interconnect_dissipated_power(p, &s);
                let support = (0..4).map(|i| s.support_reaction_n[i] * zd[i]).sum();
                let force = external_force_at(r, time);
                let ext = (0..3).map(|a| force[a] * state[a + 3]).sum();
                Ok([
                    state[3],
                    state[4],
                    state[5],
                    s.acceleration[0],
                    s.acceleration[1],
                    s.acceleration[2],
                    diss,
                    support,
                    ext,
                ])
            })
            .and_then(|state| {
                failed_time = next;
                evaluate_reported(p, r, next, state).map(|s| (state, s))
            });
            match integrated {
                Ok((state, mut sample)) => {
                    sample.energy_balance_error_j =
                        sample.energy_j - e0 + state[6] - state[7] - state[8];
                    y = state;
                    t = next;
                    run.samples.push(sample);
                    break;
                }
                Err(e) => {
                    if dt <= r.dt_s.min(1e-5) || t + dt / 2. == t {
                        run.termination = Some(event(p, failed_time, Some(t), e));
                        return Ok(run);
                    }
                    dt /= 2.;
                }
            }
        }
    }
    Ok(run)
}
/// Evaluate instantaneous ride acceleration and support loads at an absolute
/// generalized state. Does not solve equilibrium or apply initial offsets.
/// Time must lie in `[0, duration_s]`; contact/geometry failures return `Err`.
/// Work integrals and the energy-balance residual are zero (no integration).
/// Geometry is included when `report_states` is true.
pub fn evaluate_ride(
    p: &Project,
    r: &RideRequest,
    time_s: f64,
    displacement: [f64; 3],
    velocity: [f64; 3],
) -> Result<RideSample, Error> {
    p.validate()?;
    validate_request(r)?;
    if !time_s.is_finite()
        || time_s < 0.
        || time_s > r.duration_s
        || !displacement
            .iter()
            .chain(velocity.iter())
            .all(|x| x.is_finite())
    {
        return Err(error(
            "instantaneous ride time/state must be finite and time within the run",
        ));
    }
    let mut y = [0.; 9];
    y[..3].copy_from_slice(&displacement);
    y[3..6].copy_from_slice(&velocity);
    evaluate_reported(p, r, time_s, y)
}
fn evaluate_reported(
    p: &Project,
    r: &RideRequest,
    t: f64,
    y: [f64; 9],
) -> Result<RideSample, Error> {
    let mut sample = evaluate(p, r, t, y, false)?;
    if r.report_states {
        sample.state = Some(crate::simulate_on_road(
            p,
            &crate::Motion {
                heave: y[0],
                roll: y[1],
                pitch: y[2],
                rack_front: r.rack_front,
                rack_rear: r.rack_rear,
            },
            road_at(p, &r.road, t, false).0,
        )?);
    }
    Ok(sample)
}
/// Illustrative 300 kg formula-car-scale sprung mass, not a validated race-car model.
/// Equal preload is derived from the actual symmetric rest-pose compression Jacobian.
pub fn formula_car_demo() -> Result<(Project, RideRequest), Error> {
    let mut p = Project::example();
    p.name = "Formula-car-scale ride demonstration".into();
    p.chassis.sprung_mass = 300.;
    p.chassis.inertia = [100., 250., 300.];
    let (_, j) = compression_map(&p, [0.; 3], [0.; 4])?;
    let preload = 300. * GRAVITY / (-j.iter().map(|row| row[0]).sum::<f64>());
    for c in &mut p.corners {
        c.spring_damper.preload = preload;
    }
    let r = RideRequest {
        initial_displacement: [0.005, 0., 0.],
        ..Default::default()
    };
    Ok((p, r))
}

const GRAVITY: f64 = 9.81;
const MAX_SAMPLES: usize = 20_000;
fn error(message: impl Into<String>) -> Error {
    Error {
        message: message.into(),
    }
}
fn event(p: &Project, time_s: f64, last_valid_time_s: Option<f64>, e: Error) -> RideTermination {
    let corner = p
        .corners
        .iter()
        .find(|c| e.message.contains(&format!("{:?}", c.id)))
        .map(|c| c.id);
    RideTermination {
        time_s,
        last_valid_time_s,
        corner,
        reason: e.message,
    }
}
/// Validate ride input and bounded work without running equilibrium or physics.
pub fn validate_request(r: &RideRequest) -> Result<(), Error> {
    if !r.rack_front.is_finite()
        || !r.rack_rear.is_finite()
        || !r.derivative_step.is_finite()
        || !(0.0001..=0.004).contains(&r.derivative_step)
    {
        return Err(error(
            "invalid rack travel or mass derivative step (allowed 0.0001..0.004)",
        ));
    }
    if r.mode == RideMode::RetainedComponentInertia && matches!(r.road, RoadInput::Histories { .. })
    {
        return Err(error(
            "component inertia requires C1 road: piecewise linear histories have velocity jumps",
        ));
    }
    if ![r.duration_s, r.dt_s]
        .iter()
        .chain(r.initial_displacement.iter())
        .chain(r.initial_velocity.iter())
        .chain(r.external_force.iter())
        .all(|x| x.is_finite())
        || r.duration_s < 0.
        || r.dt_s <= 0.
        || (r.duration_s / r.dt_s).ceil() > (MAX_SAMPLES - 1) as f64
    {
        return Err(error(
            "invalid finite ride duration, timestep, state, force or sample limit",
        ));
    }
    if let Some(history) = &r.external_force_history {
        if history.len() < 2
            || history.len() > MAX_SAMPLES
            || !history.iter().flatten().all(|x| x.is_finite())
            || history[0][0] > 0.
            || history.last().unwrap()[0] < r.duration_s
            || !history.windows(2).all(|w| {
                let dt = w[1][0] - w[0][0];
                dt > 0. && dt.is_finite() && (1..4).all(|i| ((w[1][i] - w[0][i]) / dt).is_finite())
            })
        {
            return Err(error(
                "invalid external force history: finite increasing rows must cover the run",
            ));
        }
    }
    let valid = match &r.road {
        RoadInput::Flat => true,
        RoadInput::Sine {
            amplitude_m,
            frequency_hz,
            phases_rad,
        } => {
            amplitude_m.is_finite()
                && frequency_hz.is_finite()
                && *frequency_hz >= 0.
                && phases_rad.iter().all(|x| x.is_finite())
                && (amplitude_m * frequency_hz * std::f64::consts::TAU).is_finite()
        }
        RoadInput::SpatialSine {
            amplitude_m,
            wavelength_m,
            speed_m_s,
            phases_rad,
        } => {
            amplitude_m.is_finite()
                && wavelength_m.is_finite()
                && *wavelength_m > 0.
                && speed_m_s.is_finite()
                && phases_rad.iter().all(|x| x.is_finite())
                && (amplitude_m * speed_m_s / wavelength_m * std::f64::consts::TAU).is_finite()
        }
        RoadInput::Histories { corners } => corners.iter().all(|v| {
            v.len() >= 2
                && v.len() <= MAX_SAMPLES
                && v.iter().flatten().all(|x| x.is_finite())
                && v[0][0] <= 0.
                && v.last().unwrap()[0] >= r.duration_s
                && v.windows(2).all(|w| {
                    w[1][0] > w[0][0] && ((w[1][1] - w[0][1]) / (w[1][0] - w[0][0])).is_finite()
                })
        }),
    };
    if !valid {
        return Err(error(
            "invalid road input or noncontinuous/unsorted history",
        ));
    }
    {
        let mut knots = vec![0., r.duration_s];
        if let RoadInput::Histories { corners } = &r.road {
            knots.extend(
                corners
                    .iter()
                    .flatten()
                    .map(|p| p[0])
                    .filter(|t| *t > 0. && *t < r.duration_s),
            );
        }
        if let Some(history) = &r.external_force_history {
            knots.extend(
                history
                    .iter()
                    .map(|row| row[0])
                    .filter(|t| *t > 0. && *t < r.duration_s),
            );
        }
        knots.sort_by(f64::total_cmp);
        knots.dedup();
        let steps = knots
            .windows(2)
            .map(|w| ((w[1] - w[0]) / r.dt_s).ceil())
            .sum::<f64>();
        if steps > (MAX_SAMPLES - 1) as f64 {
            return Err(error(
                "combined road/load history knots exceed ride sample/work limit",
            ));
        }
    }
    Ok(())
}
fn external_force_at(r: &RideRequest, t: f64) -> [f64; 3] {
    let Some(history) = &r.external_force_history else {
        return r.external_force;
    };
    let k = history
        .partition_point(|row| row[0] <= t)
        .saturating_sub(1)
        .min(history.len() - 2);
    let a = history[k];
    let b = history[k + 1];
    let fraction = (t - a[0]) / (b[0] - a[0]);
    std::array::from_fn(|i| a[i + 1] + fraction * (b[i + 1] - a[i + 1]))
}
fn next_knot(road: &RoadInput, t: f64, end: f64) -> f64 {
    match road {
        RoadInput::Histories { corners } => corners
            .iter()
            .filter_map(|v| v.get(v.partition_point(|p| p[0] <= t)))
            .map(|p| p[0])
            .fold(end, f64::min),
        _ => end,
    }
}
fn road_at(p: &Project, road: &RoadInput, t: f64, left: bool) -> ([f64; 4], [f64; 4]) {
    let mut z = [0.; 4];
    let mut zd = [0.; 4];
    for i in 0..4 {
        let harmonic = match road {
            RoadInput::Flat => None,
            RoadInput::Sine {
                amplitude_m,
                frequency_hz,
                phases_rad,
            } => Some((
                *amplitude_m,
                std::f64::consts::TAU * frequency_hz,
                phases_rad[i],
            )),
            RoadInput::SpatialSine {
                amplitude_m,
                wavelength_m,
                speed_m_s,
                phases_rad,
            } => Some((
                *amplitude_m,
                std::f64::consts::TAU * speed_m_s / wavelength_m,
                phases_rad[i] + std::f64::consts::TAU * p.corners[i].wheel_center[0] / wavelength_m,
            )),
            RoadInput::Histories { corners } => {
                let v = &corners[i];
                let k = v
                    .partition_point(|s| if left { s[0] < t } else { s[0] <= t })
                    .saturating_sub(1)
                    .min(v.len() - 2);
                zd[i] = (v[k + 1][1] - v[k][1]) / (v[k + 1][0] - v[k][0]);
                z[i] = v[k][1] + zd[i] * (t - v[k][0]);
                None
            }
        };
        if let Some((a, w, phase)) = harmonic {
            z[i] = a * (w * t + phase).sin();
            zd[i] = a * w * (w * t + phase).cos();
        }
    }
    (z, zd)
}
fn compression(
    p: &Project,
    q: [f64; 3],
    z: [f64; 4],
    rack: [f64; 2],
    tight: bool,
) -> Result<([f64; 4], [Option<f64>; 4], [Option<f64>; 4]), Error> {
    let m = crate::Motion {
        heave: q[0],
        roll: q[1],
        pitch: q[2],
        rack_front: rack[0],
        rack_rear: rack[1],
    };
    let corners = crate::study::simulate_on_road_tolerance(
        p,
        &m,
        z,
        false,
        if tight { 1e-12 } else { 1e-8 },
    )?
    .corners;
    Ok((
        corners.each_ref().map(|c| c.metrics.shock_compression_m),
        corners
            .each_ref()
            .map(|c| c.metrics.heave_arm_compression_m),
        corners.each_ref().map(|c| c.metrics.roll_arm_compression_m),
    ))
}
fn compression_map(
    p: &Project,
    q: [f64; 3],
    z: [f64; 4],
) -> Result<([f64; 4], [[f64; 3]; 4]), Error> {
    let m = compression_map_request(p, q, z, &RideRequest::default())?;
    Ok((m.c, m.j))
}
/// Shock compression/Jacobian (`c`/`j`, unchanged from before this type existed) plus, for
/// any corner with the relevant hardpoints configured, the interconnect arm compressions
/// (`heave_arm`/`roll_arm`, `None` where absent) and their chassis-pose Jacobians
/// (`heave_arm_j`/`roll_arm_j`, meaningful only where the compression is `Some`).
struct CompressionMap {
    c: [f64; 4],
    j: [[f64; 3]; 4],
    heave_arm: [Option<f64>; 4],
    heave_arm_j: [[f64; 3]; 4],
    roll_arm: [Option<f64>; 4],
    roll_arm_j: [[f64; 3]; 4],
}
fn compression_map_request(
    p: &Project,
    q: [f64; 3],
    z: [f64; 4],
    r: &RideRequest,
) -> Result<CompressionMap, Error> {
    let rack = [r.rack_front, r.rack_rear];
    let tight = r.mode == RideMode::RetainedComponentInertia;
    let (c, heave_arm, roll_arm) = compression(p, q, z, rack, tight)?;
    let mut j = [[0.; 3]; 4];
    let mut heave_arm_j = [[0.; 3]; 4];
    let mut roll_arm_j = [[0.; 3]; 4];
    // Refined central differences detect reachability failures and nonsmooth branches.
    // Steps exceed closure tolerance; ratios are never silently extrapolated.
    for a in 0..3 {
        let mut previous = [0.; 4];
        let mut previous_heave_arm = [0.; 4];
        let mut previous_roll_arm = [0.; 4];
        for (level, eps) in [0.0002, 0.0001].into_iter().enumerate() {
            let mut qp = q;
            let mut qm = q;
            qp[a] += eps;
            qm[a] -= eps;
            let (cp, heave_arm_p, roll_arm_p) = compression(p, qp, z, rack, tight)?;
            let (cm, heave_arm_m, roll_arm_m) = compression(p, qm, z, rack, tight)?;
            for i in 0..4 {
                let derivative = (cp[i] - cm[i]) / (2. * eps);
                if !derivative.is_finite()
                    || derivative.abs() > 100.
                    || (level == 1
                        && (derivative - previous[i]).abs() > 0.003 * (1. + derivative.abs()))
                {
                    return Err(error(format!(
                        "{:?}: singular or discontinuous compression Jacobian",
                        p.corners[i].id
                    )));
                }
                previous[i] = derivative;
                j[i][a] = derivative;
                // These two blocks are independently no-ops (never entered) for any corner
                // without the relevant interconnect hardpoint, so an arm-less project incurs
                // no extra computation or error surface here.
                if let (Some(hp), Some(hm)) = (heave_arm_p[i], heave_arm_m[i]) {
                    let derivative = (hp - hm) / (2. * eps);
                    if !derivative.is_finite()
                        || derivative.abs() > 100.
                        || (level == 1
                            && (derivative - previous_heave_arm[i]).abs()
                                > 0.003 * (1. + derivative.abs()))
                    {
                        return Err(error(format!(
                            "{:?}: singular or discontinuous heave-arm compression Jacobian",
                            p.corners[i].id
                        )));
                    }
                    previous_heave_arm[i] = derivative;
                    heave_arm_j[i][a] = derivative;
                }
                if let (Some(rp), Some(rm)) = (roll_arm_p[i], roll_arm_m[i]) {
                    let derivative = (rp - rm) / (2. * eps);
                    if !derivative.is_finite()
                        || derivative.abs() > 100.
                        || (level == 1
                            && (derivative - previous_roll_arm[i]).abs()
                                > 0.003 * (1. + derivative.abs()))
                    {
                        return Err(error(format!(
                            "{:?}: singular or discontinuous roll-arm compression Jacobian",
                            p.corners[i].id
                        )));
                    }
                    previous_roll_arm[i] = derivative;
                    roll_arm_j[i][a] = derivative;
                }
            }
        }
    }
    Ok(CompressionMap {
        c,
        j,
        heave_arm,
        heave_arm_j,
        roll_arm,
        roll_arm_j,
    })
}
/// Symmetric (heave) and antisymmetric (roll) interconnect-arm compression and
/// chassis-pose Jacobian for one axle pairing, linearly combined from the two corners'
/// own arm compressions/Jacobians in `cm`. Only called for a [crate::AxleInterconnect]
/// that [Project::validate] has confirmed both `left`/`right` corners carry the relevant
/// hardpoints for, so the arm data is guaranteed `Some`.
struct InterconnectGeometry {
    heave_compression: f64,
    heave_j: [f64; 3],
    roll_compression: f64,
    roll_j: [f64; 3],
}
fn interconnect_geometry(
    p: &Project,
    cm: &CompressionMap,
    left: CornerId,
    right: CornerId,
) -> InterconnectGeometry {
    let li = p.corners.iter().position(|c| c.id == left).unwrap();
    let ri = p.corners.iter().position(|c| c.id == right).unwrap();
    let heave_l = cm.heave_arm[li].expect("axle interconnect requires heave arm data");
    let heave_r = cm.heave_arm[ri].expect("axle interconnect requires heave arm data");
    let roll_l = cm.roll_arm[li].expect("axle interconnect requires roll arm data");
    let roll_r = cm.roll_arm[ri].expect("axle interconnect requires roll arm data");
    InterconnectGeometry {
        heave_compression: (heave_l + heave_r) / 2.,
        heave_j: std::array::from_fn(|a| (cm.heave_arm_j[li][a] + cm.heave_arm_j[ri][a]) / 2.),
        roll_compression: (roll_l - roll_r) / 2.,
        roll_j: std::array::from_fn(|a| (cm.roll_arm_j[li][a] - cm.roll_arm_j[ri][a]) / 2.),
    }
}
/// One configured [crate::AxleInterconnect]'s rate-dependent dynamics at a chassis rate
/// `v`/road velocity `zd`: combined-channel geometry, spring+damper force, corner array
/// indices (for the support-reaction chain-rule correction, which needs each corner's own
/// un-combined arm Jacobian from `cm`, not the combined one here), and the [InterconnectSample]
/// to report.
struct InterconnectDynamics {
    left_index: usize,
    right_index: usize,
    heave_j: [f64; 3],
    roll_j: [f64; 3],
    f_heave: f64,
    f_roll: f64,
    sample: InterconnectSample,
}
fn interconnect_dynamics(
    p: &Project,
    cm: &CompressionMap,
    ai: &AxleInterconnect,
    left: CornerId,
    right: CornerId,
    v: [f64; 3],
    zd: [f64; 4],
) -> InterconnectDynamics {
    let geo = interconnect_geometry(p, cm, left, right);
    let left_index = p.corners.iter().position(|c| c.id == left).unwrap();
    let right_index = p.corners.iter().position(|c| c.id == right).unwrap();
    let zd_heave = (zd[left_index] + zd[right_index]) / 2.;
    let zd_roll = (zd[left_index] - zd[right_index]) / 2.;
    let u_heave = (0..3).map(|a| geo.heave_j[a] * v[a]).sum::<f64>() - geo.heave_j[0] * zd_heave;
    let u_roll = (0..3).map(|a| geo.roll_j[a] * v[a]).sum::<f64>() - geo.roll_j[0] * zd_roll;
    let f_heave = spring_force(&ai.heave, geo.heave_compression) + damper_force(&ai.heave, u_heave);
    let f_roll = spring_force(&ai.roll, geo.roll_compression) + damper_force(&ai.roll, u_roll);
    InterconnectDynamics {
        left_index,
        right_index,
        heave_j: geo.heave_j,
        roll_j: geo.roll_j,
        f_heave,
        f_roll,
        sample: InterconnectSample {
            heave_compression_m: geo.heave_compression,
            heave_velocity_m_s: u_heave,
            heave_force_n: f_heave,
            roll_compression_m: geo.roll_compression,
            roll_velocity_m_s: u_roll,
            roll_force_n: f_roll,
        },
    }
}
/// Sum of [damper_force]`*`velocity across every configured axle interconnect's heave and
/// roll channels at `s`; zero (and no [crate::Project::front_interconnect]/
/// [crate::Project::rear_interconnect] lookups beyond two `None` checks) for a project with
/// neither configured.
fn interconnect_dissipated_power(p: &Project, s: &RideSample) -> f64 {
    let mut total = 0.;
    for (ai, sample) in [
        (&p.front_interconnect, &s.front_interconnect),
        (&p.rear_interconnect, &s.rear_interconnect),
    ] {
        if let (Some(ai), Some(sample)) = (ai, sample) {
            total += damper_force(&ai.heave, sample.heave_velocity_m_s) * sample.heave_velocity_m_s;
            total += damper_force(&ai.roll, sample.roll_velocity_m_s) * sample.roll_velocity_m_s;
        }
    }
    total
}
fn evaluate(
    p: &Project,
    r: &RideRequest,
    t: f64,
    y: [f64; 9],
    left: bool,
) -> Result<RideSample, Error> {
    if !y.iter().all(|x| x.is_finite()) {
        return Err(error("nonfinite integrated state"));
    }
    let q = [y[0], y[1], y[2]];
    let v = [y[3], y[4], y[5]];
    let (z, zd) = road_at(p, &r.road, t, left);
    let cm = compression_map_request(p, q, z, r)?;
    check_limits(p, cm.c)?;
    if r.mode == RideMode::RetainedComponentInertia && crate::mass::active(p) {
        return evaluate_components(p, r, t, y, q, v, z, zd, cm);
    }
    let (c, j) = (cm.c, cm.j);
    let mut u = [0.; 4];
    let mut force = [0.; 4];
    let mut support = [0.; 4];
    let mut generalized = external_force_at(r, t);
    generalized[0] -= p.chassis.sprung_mass * GRAVITY;
    let mut energy =
        chassis_kinetic_energy(&p.chassis, q, v) + p.chassis.sprung_mass * GRAVITY * q[0];
    for i in 0..4 {
        u[i] = (0..3).map(|a| j[i][a] * v[a]).sum::<f64>() - j[i][0] * zd[i];
        let s = &p.corners[i].spring_damper;
        force[i] = spring_force(s, c[i]) + damper_force(s, u[i]);
        support[i] = -force[i] * j[i][0];
        if support[i] < -1e-6 {
            return Err(error(format!(
                "{:?}: fixed contact invalid: negative support reaction {} N",
                p.corners[i].id, support[i]
            )));
        }
        for a in 0..3 {
            generalized[a] -= j[i][a] * force[i];
        }
        energy += spring_energy(s, c[i]);
    }
    let mut front_interconnect = None;
    let mut rear_interconnect = None;
    for (slot, ai, left, right) in [
        (
            &mut front_interconnect,
            &p.front_interconnect,
            CornerId::FrontLeft,
            CornerId::FrontRight,
        ),
        (
            &mut rear_interconnect,
            &p.rear_interconnect,
            CornerId::RearLeft,
            CornerId::RearRight,
        ),
    ] {
        let Some(ai) = ai else { continue };
        let d = interconnect_dynamics(p, &cm, ai, left, right, v, zd);
        for a in 0..3 {
            generalized[a] -= d.heave_j[a] * d.f_heave + d.roll_j[a] * d.f_roll;
        }
        support[d.left_index] -= 0.5
            * (cm.heave_arm_j[d.left_index][0] * d.f_heave
                + cm.roll_arm_j[d.left_index][0] * d.f_roll);
        support[d.right_index] -= 0.5
            * (cm.heave_arm_j[d.right_index][0] * d.f_heave
                - cm.roll_arm_j[d.right_index][0] * d.f_roll);
        for i in [d.left_index, d.right_index] {
            if support[i] < -1e-6 {
                return Err(error(format!(
                    "{:?}: fixed contact invalid: negative support reaction {} N",
                    p.corners[i].id, support[i]
                )));
            }
        }
        energy += spring_energy(&ai.heave, d.sample.heave_compression_m)
            + spring_energy(&ai.roll, d.sample.roll_compression_m);
        *slot = Some(d.sample);
    }
    let acceleration = inertia_acceleration(&p.chassis, q, v, generalized);
    if !energy.is_finite()
        || !acceleration
            .iter()
            .chain(force.iter())
            .chain(u.iter())
            .chain(support.iter())
            .all(|x| x.is_finite())
    {
        return Err(error("nonfinite dynamics force or energy"));
    }
    Ok(RideSample {
        state: None,
        time_s: t,
        displacement: q,
        velocity: v,
        acceleration,
        compression_m: c,
        compression_velocity_m_s: u,
        shock_force_n: force,
        support_reaction_n: support,
        front_interconnect,
        rear_interconnect,
        energy_j: energy,
        dissipated_work_j: y[6],
        support_work_j: y[7],
        external_work_j: y[8],
        energy_balance_error_j: 0.,
    })
}
fn check_limits(p: &Project, c: [f64; 4]) -> Result<(), Error> {
    for (i, corner) in p.corners.iter().enumerate() {
        let rest = (nalgebra::Vector3::from(corner.rocker_shock)
            - nalgebra::Vector3::from(corner.shock_chassis))
        .norm();
        let length = rest - c[i];
        if corner
            .spring_damper
            .min_length_m
            .is_some_and(|min| length < min)
        {
            return Err(error(format!(
                "{:?}: shock minimum length reached",
                corner.id
            )));
        }
        if corner
            .spring_damper
            .max_length_m
            .is_some_and(|max| length > max)
        {
            return Err(error(format!(
                "{:?}: shock maximum length reached",
                corner.id
            )));
        }
    }
    Ok(())
}
fn road_acceleration(road: &RoadInput, z: [f64; 4]) -> [f64; 4] {
    let omega = match road {
        RoadInput::Sine { frequency_hz, .. } => std::f64::consts::TAU * frequency_hz,
        RoadInput::SpatialSine {
            speed_m_s,
            wavelength_m,
            ..
        } => std::f64::consts::TAU * speed_m_s / wavelength_m,
        _ => 0.,
    };
    z.map(|z| -omega * omega * z)
}
#[allow(clippy::too_many_arguments)]
fn evaluate_components(
    p: &Project,
    r: &RideRequest,
    t: f64,
    y: [f64; 9],
    q: [f64; 3],
    v: [f64; 3],
    z: [f64; 4],
    zd: [f64; 4],
    cm: CompressionMap,
) -> Result<RideSample, Error> {
    use crate::mass::{Terms, X};
    let (c, j) = (cm.c, cm.j);
    let x = X::from_row_slice(&[q[0], q[1], q[2], z[0], z[1], z[2], z[3]]);
    let w = X::from_row_slice(&[v[0], v[1], v[2], zd[0], zd[1], zd[2], zd[3]]);
    let zdd = road_acceleration(&r.road, z);
    let u = std::array::from_fn(|i| (0..3).map(|a| j[i][a] * v[a]).sum::<f64>() - j[i][0] * zd[i]);
    let force = std::array::from_fn(|i| {
        spring_force(&p.corners[i].spring_damper, c[i])
            + damper_force(&p.corners[i].spring_damper, u[i])
    });
    // Rate-dependent, so computed once here rather than inside `compute`'s per-refinement-
    // level closure (below) -- geometric compression/Jacobian never depends on `terms`, and
    // recomputing per refinement level could only introduce spurious variation. Both stay
    // `None`, and the correction/energy totals below stay all-zero, for a project with
    // neither axle interconnect configured, so `compute`'s existing arithmetic is unaffected.
    let front_dynamics = p.front_interconnect.as_ref().map(|ai| {
        (
            ai,
            interconnect_dynamics(p, &cm, ai, CornerId::FrontLeft, CornerId::FrontRight, v, zd),
        )
    });
    let rear_dynamics = p.rear_interconnect.as_ref().map(|ai| {
        (
            ai,
            interconnect_dynamics(p, &cm, ai, CornerId::RearLeft, CornerId::RearRight, v, zd),
        )
    });
    let all_dynamics = [&front_dynamics, &rear_dynamics];
    let interconnect_generalized: [f64; 3] = std::array::from_fn(|k| {
        all_dynamics
            .iter()
            .filter_map(|d| d.as_ref())
            .map(|(_, d)| d.heave_j[k] * d.f_heave + d.roll_j[k] * d.f_roll)
            .sum()
    });
    let mut interconnect_support_correction = [0.; 4];
    let mut interconnect_energy = 0.;
    for (ai, d) in all_dynamics.into_iter().filter_map(|d| d.as_ref()) {
        interconnect_support_correction[d.left_index] += 0.5
            * (cm.heave_arm_j[d.left_index][0] * d.f_heave
                + cm.roll_arm_j[d.left_index][0] * d.f_roll);
        interconnect_support_correction[d.right_index] += 0.5
            * (cm.heave_arm_j[d.right_index][0] * d.f_heave
                - cm.roll_arm_j[d.right_index][0] * d.f_roll);
        interconnect_energy += spring_energy(&ai.heave, d.sample.heave_compression_m)
            + spring_energy(&ai.roll, d.sample.roll_compression_m);
    }
    let compute = |mut terms: Terms| -> Result<([f64; 3], [f64; 4], f64), Error> {
        let ch = &p.chassis;
        let a = ch.inertia[1] * q[1].cos().powi(2) + ch.inertia[2] * q[1].sin().powi(2);
        let ap = 2. * (ch.inertia[2] - ch.inertia[1]) * q[1].sin() * q[1].cos();
        terms.mass[(0, 0)] += ch.sprung_mass;
        terms.mass[(1, 1)] += ch.inertia[0];
        terms.mass[(2, 2)] += a;
        terms.gravity[0] += ch.sprung_mass * GRAVITY;
        terms.bias[1] -= 0.5 * ap * v[2] * v[2];
        terms.bias[2] += ap * v[1] * v[2];
        let mut rhs = nalgebra::Vector3::from(external_force_at(r, t));
        for k in 0..3 {
            rhs[k] -= terms.gravity[k]
                + terms.bias[k]
                + (0..4)
                    .map(|i| j[i][k] * force[i] + terms.mass[(k, 3 + i)] * zdd[i])
                    .sum::<f64>()
                + interconnect_generalized[k];
        }
        let acc = terms
            .mass
            .fixed_view::<3, 3>(0, 0)
            .into_owned()
            .cholesky()
            .ok_or_else(|| error("component Mqq is not positive definite"))?
            .solve(&rhs);
        let mut support = std::array::from_fn(|i| {
            terms.gravity[3 + i] + terms.bias[3 + i] - j[i][0] * force[i]
                + (0..3).map(|k| terms.mass[(3 + i, k)] * acc[k]).sum::<f64>()
                + (0..4)
                    .map(|k| terms.mass[(3 + i, 3 + k)] * zdd[k])
                    .sum::<f64>()
        });
        for i in 0..4 {
            support[i] -= interconnect_support_correction[i];
        }
        let energy = 0.5 * w.dot(&(terms.mass * w))
            + terms.potential
            + ch.sprung_mass * GRAVITY * q[0]
            + (0..4)
                .map(|i| spring_energy(&p.corners[i].spring_damper, c[i]))
                .sum::<f64>()
            + interconnect_energy;
        Ok((acc.into(), support, energy))
    };
    // Cache exact stencil positions within this stage only, preserving branch-local solves.
    let cache = std::cell::RefCell::new(std::collections::HashMap::new());
    let provider = |x: X| {
        let key = x.map(f64::to_bits);
        if let Some(v) = cache.borrow().get(&key) {
            return Ok(Vec::clone(v));
        }
        let poses = crate::mass::poses(p, [r.rack_front, r.rack_rear], x)?;
        cache.borrow_mut().insert(key, poses.clone());
        Ok(poses)
    };
    let coarse_terms = crate::mass::terms(provider, x, w, r.derivative_step)?;
    let fine_terms = crate::mass::terms(provider, x, w, r.derivative_step / 2.)?;
    let stable_j = coarse_terms.stable_jacobians(&fine_terms);
    let coarse = compute(coarse_terms)?;
    let close = |a: &([f64; 3], [f64; 4], f64), b: &([f64; 3], [f64; 4], f64)| {
        (0..3).all(|i| (a.0[i] - b.0[i]).abs() <= 1e-4 + 2e-4 * b.0[i].abs())
            && (0..4).all(|i| (a.1[i] - b.1[i]).abs() <= 0.1 + 2e-4 * b.1[i].abs())
    };
    let fine = compute(fine_terms.clone())?;
    let accepted = if stable_j && close(&coarse, &fine) {
        fine
    } else {
        let quarter_terms = crate::mass::terms(provider, x, w, r.derivative_step / 4.)?;
        let stable_j = fine_terms.stable_jacobians(&quarter_terms);
        let quarter = compute(quarter_terms)?;
        if !stable_j || !close(&fine, &quarter) {
            return Err(error(
                "component mass derivative refinement failed acceleration/reaction tolerance",
            ));
        }
        quarter
    };
    let (acceleration, support, energy) = accepted;
    for (i, normal) in support.iter().enumerate() {
        if *normal < -1e-6 {
            return Err(error(format!(
                "{:?}: fixed contact invalid: negative full support reaction {} N",
                p.corners[i].id, support[i]
            )));
        }
    }
    if !energy.is_finite()
        || !acceleration
            .iter()
            .chain(support.iter())
            .all(|v| v.is_finite())
    {
        return Err(error("nonfinite component acceleration/support/energy"));
    }
    Ok(RideSample {
        state: None,
        time_s: t,
        displacement: q,
        velocity: v,
        acceleration,
        compression_m: c,
        compression_velocity_m_s: u,
        shock_force_n: force,
        support_reaction_n: support,
        front_interconnect: front_dynamics.map(|(_, d)| d.sample),
        rear_interconnect: rear_dynamics.map(|(_, d)| d.sample),
        energy_j: energy,
        dissipated_work_j: y[6],
        support_work_j: y[7],
        external_work_j: y[8],
        energy_balance_error_j: 0.,
    })
}
fn static_residual(
    p: &Project,
    r: &RideRequest,
    q: [f64; 3],
) -> Result<nalgebra::Vector3<f64>, Error> {
    let (z, _) = road_at(p, &r.road, 0., false);
    let cm = compression_map_request(p, q, z, r)?;
    let (c, j) = (cm.c, cm.j);
    let component_gravity = if r.mode == RideMode::RetainedComponentInertia
        && crate::mass::active(p)
    {
        let x = crate::mass::X::from_row_slice(&[q[0], q[1], q[2], z[0], z[1], z[2], z[3]]);
        let provider = |x| crate::mass::poses(p, [r.rack_front, r.rack_rear], x);
        let coarse = crate::mass::terms(provider, x, crate::mass::X::zeros(), r.derivative_step)?;
        let fine =
            crate::mass::terms(provider, x, crate::mass::X::zeros(), r.derivative_step / 2.)?;
        if !coarse.stable_jacobians(&fine)
            || (0..7).any(|i| {
                (coarse.gravity[i] - fine.gravity[i]).abs() > 0.001 + 1e-5 * fine.gravity[i].abs()
            })
        {
            return Err(error(
                "component gravity derivative refinement failed at equilibrium",
            ));
        }
        fine.gravity
    } else {
        crate::mass::X::zeros()
    };
    let force = external_force_at(r, 0.);
    let mut residual = nalgebra::Vector3::new(
        p.chassis.sprung_mass * GRAVITY - force[0],
        -force[1],
        -force[2],
    );
    for a in 0..3 {
        residual[a] += component_gravity[a];
    }
    for i in 0..4 {
        for a in 0..3 {
            residual[a] += j[i][a] * spring_force(&p.corners[i].spring_damper, c[i]);
        }
    }
    for (ai, left, right) in [
        (
            &p.front_interconnect,
            CornerId::FrontLeft,
            CornerId::FrontRight,
        ),
        (
            &p.rear_interconnect,
            CornerId::RearLeft,
            CornerId::RearRight,
        ),
    ] {
        if let Some(ai) = ai {
            let geo = interconnect_geometry(p, &cm, left, right);
            let f_heave = spring_force(&ai.heave, geo.heave_compression);
            let f_roll = spring_force(&ai.roll, geo.roll_compression);
            for a in 0..3 {
                residual[a] += geo.heave_j[a] * f_heave + geo.roll_j[a] * f_roll;
            }
        }
    }
    Ok(residual)
}
fn static_stiffness(
    p: &Project,
    r: &RideRequest,
    q: [f64; 3],
) -> Result<nalgebra::Matrix3<f64>, Error> {
    static_stiffness_step(p, r, q, 0.0002)
}
fn static_stiffness_step(
    p: &Project,
    r: &RideRequest,
    q: [f64; 3],
    step: f64,
) -> Result<nalgebra::Matrix3<f64>, Error> {
    let mut h = nalgebra::Matrix3::zeros();
    for a in 0..3 {
        let mut qp = q;
        let mut qm = q;
        qp[a] += step;
        qm[a] -= step;
        h.set_column(
            a,
            &((static_residual(p, r, qp)? - static_residual(p, r, qm)?) / (2. * step)),
        );
    }
    Ok((h + h.transpose()) / 2.)
}
fn equilibrate(p: &Project, r: &RideRequest) -> Result<[f64; 3], Error> {
    let mut q = [0.; 3];
    for _ in 0..30 {
        let residual = static_residual(p, r, q)?;
        let h = static_stiffness(p, r, q)?;
        if residual.amax() < 0.002 {
            if h.symmetric_eigen().eigenvalues.min() <= 0. {
                return Err(error("static equilibrium is unstable"));
            }
            return Ok(q);
        }
        let delta = h
            .lu()
            .solve(&(-residual))
            .ok_or_else(|| error("singular equilibrium stiffness"))?;
        let mut accepted = None;
        for backtrack in 0..16 {
            let trial = std::array::from_fn(|a| q[a] + delta[a] * 0.5_f64.powi(backtrack));
            if let Ok(r2) = static_residual(p, r, trial) {
                if r2.norm() < residual.norm() {
                    accepted = Some(trial);
                    break;
                }
            }
        }
        q = accepted
            .ok_or_else(|| error("equilibrium could not converge within reachable geometry"))?;
    }
    Err(error("equilibrium iteration limit"))
}
/// Bilateral spring force; compression is relative to design shock length.
/// Parameters/tables must be validated through Project::validate or Corner::validate.
pub fn spring_force(s: &SpringDamper, c: f64) -> f64 {
    s.spring_curve
        .as_ref()
        .map_or(s.preload + s.spring_rate * c, |table| table_force(table, c))
}
/// Signed passive force, positive in compression and negative in rebound.
/// Parameters/tables must be validated through Project::validate or Corner::validate.
pub fn damper_force(s: &SpringDamper, u: f64) -> f64 {
    let (table, b) = if u >= 0. {
        (&s.compression_curve, s.compression_damping)
    } else {
        (&s.rebound_curve, s.rebound_damping)
    };
    table
        .as_ref()
        .map_or(u * b, |table| u.signum() * table_force(table, u.abs()))
}
/// Spring potential with arbitrary zero at design compression.
/// Uses the exact piecewise integral for a validated tabulated law.
pub fn spring_energy(s: &SpringDamper, c: f64) -> f64 {
    s.spring_curve
        .as_ref()
        .map_or(s.preload * c + 0.5 * s.spring_rate * c * c, |table| {
            table_integral(table, c) - table_integral(table, 0.)
        })
}
// Public scalar law helpers require validated tables (Project::validate does this for ride).
fn table_force(table: &[[f64; 2]], x: f64) -> f64 {
    if table.len() < 2 {
        return f64::NAN;
    }
    let upper = table.partition_point(|p| p[0] <= x);
    if upper == 0 {
        table[0][1]
    } else if upper == table.len() {
        table[upper - 1][1]
    } else {
        let a = table[upper - 1];
        let b = table[upper];
        a[1] + (b[1] - a[1]) * (x - a[0]) / (b[0] - a[0])
    }
}
fn table_integral(table: &[[f64; 2]], x: f64) -> f64 {
    if table.len() < 2 {
        return f64::NAN;
    }
    if x <= table[0][0] {
        return (x - table[0][0]) * table[0][1];
    }
    let mut integral = 0.;
    for pair in table.windows(2) {
        let a = pair[0];
        let b = pair[1];
        let end = x.min(b[0]);
        let width = end - a[0];
        integral += width * (a[1] + 0.5 * (b[1] - a[1]) * width / (b[0] - a[0]));
        if x <= b[0] {
            return integral;
        }
    }
    let last = table.last().unwrap();
    integral + (x - last[0]) * last[1]
}
/// Tangent wheel stiffness, including preload geometric stiffness.
pub fn wheel_rate(k: f64, force: f64, ratio: f64, gradient: f64) -> f64 {
    k * ratio * ratio + force * gradient
}
/// Exact kinetic energy for R=Ry(pitch)Rx(roll), rotating about chassis COM.
pub fn chassis_kinetic_energy(c: &Chassis, q: [f64; 3], v: [f64; 3]) -> f64 {
    let a = c.inertia[1] * q[1].cos().powi(2) + c.inertia[2] * q[1].sin().powi(2);
    0.5 * (c.sprung_mass * v[0] * v[0] + c.inertia[0] * v[1] * v[1] + a * v[2] * v[2])
}

fn rk4<const N: usize>(
    y: [f64; N],
    t: f64,
    dt: f64,
    mut f: impl FnMut(f64, [f64; N]) -> Result<[f64; N], Error>,
) -> Result<[f64; N], Error> {
    let k1 = f(t, y)?;
    let k2 = f(t + dt / 2., std::array::from_fn(|i| y[i] + dt / 2. * k1[i]))?;
    let k3 = f(t + dt / 2., std::array::from_fn(|i| y[i] + dt / 2. * k2[i]))?;
    let k4 = f(t + dt, std::array::from_fn(|i| y[i] + dt * k3[i]))?;
    Ok(std::array::from_fn(|i| {
        y[i] + dt / 6. * (k1[i] + 2. * k2[i] + 2. * k3[i] + k4[i])
    }))
}
fn inertia_acceleration(c: &Chassis, q: [f64; 3], v: [f64; 3], force: [f64; 3]) -> [f64; 3] {
    let a = c.inertia[1] * q[1].cos().powi(2) + c.inertia[2] * q[1].sin().powi(2);
    let ap = 2. * (c.inertia[2] - c.inertia[1]) * q[1].sin() * q[1].cos();
    [
        force[0] / c.sprung_mass,
        (force[1] + 0.5 * ap * v[2] * v[2]) / c.inertia[0],
        (force[2] - ap * v[1] * v[2]) / a,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    // Breaks on missing RK stages, force-transfer square, damping sign or inertial bias.
    #[test]
    fn rk4_matches_three_analytic_damped_modes() {
        let mu = 0.5_f64;
        let k = 1000.;
        let b = 20.;
        for (mass, lever2) in [(10., 4.), (2., 4. * 0.8 * 0.8), (5., 4. * 1.3 * 1.3)] {
            let stiffness = k * mu * mu * lever2;
            let damping = b * mu * mu * lever2;
            let alpha = damping / (2. * mass);
            let omega = (stiffness / mass - alpha * alpha).sqrt();
            let mut y = [0.01, 0.];
            for step in 0..100 {
                y = rk4(y, step as f64 * 0.001, 0.001, |_, y| {
                    Ok([y[1], (-stiffness * y[0] - damping * y[1]) / mass])
                })
                .unwrap();
            }
            let expected = 0.01
                * (-alpha * 0.1).exp()
                * ((omega * 0.1).cos() + alpha / omega * (omega * 0.1).sin());
            assert!((y[0] - expected).abs() < 1e-10, "{} != {}", y[0], expected);
        }
    }
    #[test]
    fn finite_roll_free_chassis_preserves_energy_and_pitch_momentum() {
        let c = Chassis::default();
        let initial = [0., 0.31, -0.22, 0.2, 0.4, 0.6];
        let mut y = initial;
        for step in 0..100 {
            y = rk4(y, step as f64 * 0.01, 0.01, |_, y| {
                let a = inertia_acceleration(&c, [y[0], y[1], y[2]], [y[3], y[4], y[5]], [0.; 3]);
                Ok([y[3], y[4], y[5], a[0], a[1], a[2]])
            })
            .unwrap();
        }
        assert!(y[1] > 0.7);
        let energy =
            |y: [f64; 6]| chassis_kinetic_energy(&c, [y[0], y[1], y[2]], [y[3], y[4], y[5]]);
        let momentum = |y: [f64; 6]| {
            (c.inertia[1] * y[1].cos().powi(2) + c.inertia[2] * y[1].sin().powi(2)) * y[5]
        };
        assert!((energy(y) - energy(initial)).abs() < 1e-7);
        assert!((momentum(y) - momentum(initial)).abs() < 1e-7);
    }
}
