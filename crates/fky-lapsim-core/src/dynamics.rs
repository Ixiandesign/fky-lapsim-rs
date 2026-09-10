//! Nonlinear sprung-body dynamics on prescribed horizontal supports.
//!
//! q = [heave m, roll rad, pitch rad], rotating Ry(pitch)Rx(roll) about chassis COM.
//! Compression derivatives come from exact geometry closure, with finite-difference
//! refinement. RK4 is conditionally stable: check timestep convergence for each setup.
//! Component/unsprung inertia and contact release require a separate model.
use crate::{Chassis, SpringDamper};
use crate::{CornerId, Error, Project};
use serde::{Deserialize, Serialize};

pub const MODEL_FIDELITY: &str = "nonlinear_sprung_body_fixed_contact_massless_links";
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RoadInput {
    #[default]
    Flat,
    /// Temporal sinusoid in Hz; phases in radians, ordered like Project.corners.
    Sine {
        amplitude_m: f64,
        frequency_hz: f64,
        phases_rad: [f64; 4],
    },
    /// Spatial sinusoid sampled at design wheel x plus speed*time.
    SpatialSine {
        amplitude_m: f64,
        wavelength_m: f64,
        speed_m_s: f64,
        phases_rad: [f64; 4],
    },
    /// Continuous piecewise linear heights; times must span the complete run.
    /// Samples are [time seconds, height metres]. Velocity is right-continuous at
    /// interior knots; integration ends the preceding interval with its left velocity.
    Histories { corners: [Vec<[f64; 2]>; 4] },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RideRequest {
    pub duration_s: f64,
    pub dt_s: f64,
    /// Offsets from equilibrium when solve_equilibrium is true, otherwise absolute.
    pub initial_displacement: [f64; 3],
    /// Heave m/s, roll/pitch rad/s.
    pub initial_velocity: [f64; 3],
    /// Generalized heave force / roll and pitch moments, conjugate to q.
    pub external_force: [f64; 3],
    /// Solve with the road frozen at t=0 before applying initial offsets.
    pub solve_equilibrium: bool,
    pub road: RoadInput,
}
impl Default for RideRequest {
    fn default() -> Self {
        Self {
            duration_s: 1.,
            dt_s: 0.005,
            initial_displacement: [0.; 3],
            initial_velocity: [0.; 3],
            external_force: [0.; 3],
            solve_equilibrium: true,
            road: RoadInput::Flat,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RideSample {
    pub time_s: f64,
    pub displacement: [f64; 3],
    pub velocity: [f64; 3],
    pub acceleration: [f64; 3],
    pub compression_m: [f64; 4],
    pub compression_velocity_m_s: [f64; 4],
    pub shock_force_n: [f64; 4],
    /// Vertical reactions of this reduced prescribed-support model only.
    pub support_reaction_n: [f64; 4],
    pub energy_j: f64,
    /// Positive integral of signed damper force times compression velocity.
    pub dissipated_work_j: f64,
    /// Integral of reduced vertical support reactions times prescribed road velocity.
    pub support_work_j: f64,
    pub external_work_j: f64,
    pub energy_balance_error_j: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RideTermination {
    /// Time of the failed integration-stage or accepted-state evaluation.
    pub time_s: f64,
    pub last_valid_time_s: Option<f64>,
    pub corner: Option<CornerId>,
    pub reason: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RideRun {
    pub model_fidelity: String,
    pub corner_ids: [CornerId; 4],
    pub equilibrium: Option<[f64; 3]>,
    pub samples: Vec<RideSample>,
    /// None means the requested duration was completed.
    pub termination: Option<RideTermination>,
}
/// Integrate coupled heave/roll/pitch using RK4 and exact nonlinear ground closure.
/// Links and unsprung parts are massless; supports are rigid horizontal planes.
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
        model_fidelity: MODEL_FIDELITY.into(),
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
    let first = match evaluate(p, r, t, y, false) {
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
        let end = next_knot(&r.road, t, r.duration_s.min(t + r.dt_s));
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
                    .sum();
                let support = (0..4).map(|i| s.support_reaction_n[i] * zd[i]).sum();
                let ext = (0..3).map(|a| r.external_force[a] * state[a + 3]).sum();
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
                evaluate(p, r, next, state, false).map(|s| (state, s))
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
fn validate_request(r: &RideRequest) -> Result<(), Error> {
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
    if let RoadInput::Histories { corners } = &r.road {
        let mut knots = vec![0., r.duration_s];
        knots.extend(
            corners
                .iter()
                .flatten()
                .map(|p| p[0])
                .filter(|t| *t > 0. && *t < r.duration_s),
        );
        knots.sort_by(f64::total_cmp);
        knots.dedup();
        let steps = knots
            .windows(2)
            .map(|w| ((w[1] - w[0]) / r.dt_s).ceil())
            .sum::<f64>();
        if steps > (MAX_SAMPLES - 1) as f64 {
            return Err(error(
                "combined road history knots exceed ride sample/work limit",
            ));
        }
    }
    Ok(())
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
fn compression(p: &Project, q: [f64; 3], z: [f64; 4]) -> Result<[f64; 4], Error> {
    let m = crate::Motion {
        heave: q[0],
        roll: q[1],
        pitch: q[2],
        ..Default::default()
    };
    Ok(crate::study::simulate_on_road_mode(p, &m, z, false)?
        .corners
        .map(|c| c.metrics.shock_compression_m))
}
fn compression_map(
    p: &Project,
    q: [f64; 3],
    z: [f64; 4],
) -> Result<([f64; 4], [[f64; 3]; 4]), Error> {
    let c = compression(p, q, z)?;
    let mut j = [[0.; 3]; 4];
    // Refined central differences detect reachability failures and nonsmooth branches.
    // Steps exceed closure tolerance; ratios are never silently extrapolated.
    for a in 0..3 {
        let mut previous = [0.; 4];
        for (level, eps) in [0.0002, 0.0001].into_iter().enumerate() {
            let mut qp = q;
            let mut qm = q;
            qp[a] += eps;
            qm[a] -= eps;
            let cp = compression(p, qp, z)?;
            let cm = compression(p, qm, z)?;
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
            }
        }
    }
    Ok((c, j))
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
    let (c, j) = compression_map(p, q, z)?;
    let mut u = [0.; 4];
    let mut force = [0.; 4];
    let mut support = [0.; 4];
    let mut generalized = r.external_force;
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
        time_s: t,
        displacement: q,
        velocity: v,
        acceleration,
        compression_m: c,
        compression_velocity_m_s: u,
        shock_force_n: force,
        support_reaction_n: support,
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
    let (c, j) = compression_map(p, q, z)?;
    let mut residual = nalgebra::Vector3::new(
        p.chassis.sprung_mass * GRAVITY - r.external_force[0],
        -r.external_force[1],
        -r.external_force[2],
    );
    for i in 0..4 {
        for a in 0..3 {
            residual[a] += j[i][a] * spring_force(&p.corners[i].spring_damper, c[i]);
        }
    }
    Ok(residual)
}
fn static_stiffness(
    p: &Project,
    r: &RideRequest,
    q: [f64; 3],
) -> Result<nalgebra::Matrix3<f64>, Error> {
    let mut h = nalgebra::Matrix3::zeros();
    for a in 0..3 {
        let mut qp = q;
        let mut qm = q;
        qp[a] += 0.0002;
        qm[a] -= 0.0002;
        h.set_column(
            a,
            &((static_residual(p, r, qp)? - static_residual(p, r, qm)?) / 0.0004),
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
