//! Isolated scenarios (thesis Ch. 3): steady-state cornering, acceleration, braking.
use super::err;
use super::forces::{max_ay, StepModel};
use super::thesis::BikeParams;
use super::tractive::TractiveTable;
use crate::Error;

/// Front-wheel steer angle and body side-slip from the steering matrix (Eq. 2-53), degrees,
/// signed by the sign of `radius_m` (positive = left). `CF`, `CR` are axle stiffnesses (2× tyre).
///
/// # Errors
/// Returns an error for nonfinite/zero radius, negative speed, or a singular matrix.
pub fn steering(p: &BikeParams, speed_m_s: f64, radius_m: f64) -> Result<(f64, f64), Error> {
    if !speed_m_s.is_finite() || speed_m_s < 0.0 || !radius_m.is_finite() || radius_m == 0.0 {
        return Err(err("invalid speed or radius for steering"));
    }
    let cf = 2.0 * p.front.cornering_stiffness_n_per_deg;
    let cr = 2.0 * p.rear.cornering_stiffness_n_per_deg;
    let (a, b) = (p.a_dist_m(), p.b_dist_m());
    let rhs = p.mass_kg * speed_m_s * speed_m_s / radius_m.abs();
    let (m11, m12, m21, m22) = (cf, cr - cf, a * cf, b * cr - a * cf);
    let det = m11 * m22 - m12 * m21;
    // det = CF·CR·(b − a): the thesis' simplified slip definitions are singular for a = b. Below
    // 0.1 % weight-distribution asymmetry the full steady-state bicycle solution is used instead
    // (see `bicycle_steady_state`); P19 (a/b = 1.04) is far above this and stays thesis-exact.
    if (b - a).abs() < 1e-3 * (a + b) {
        let (delta, beta) = bicycle_steady_state(p, speed_m_s, radius_m.abs())?;
        let sign = radius_m.signum();
        return Ok((sign * delta, sign * beta));
    }
    if det.abs() < 1e-12 {
        return Err(err("singular steering matrix"));
    }
    let sign = radius_m.signum();
    Ok((sign * rhs * m22 / det, sign * (-m21 * rhs) / det))
}

/// Steady-state bicycle-model steer angle and body side-slip (degrees, left turn) for a radius
/// `radius_m > 0`: lateral force balance and yaw-moment balance with the yaw-rate terms of the slip
/// angles `α_f = δ − β − aΩ/v`, `α_r = −β + bΩ/v`. Unlike the thesis' Eq. 2-53 this is well-posed
/// for any weight distribution, so it supplies the yaw angle for the AeroMap and the steer angle of
/// neutrally balanced cars.
///
/// # Errors
/// Returns an error for nonfinite or nonpositive speed/radius or zero cornering stiffness.
pub fn bicycle_steady_state(
    p: &BikeParams,
    speed_m_s: f64,
    radius_m: f64,
) -> Result<(f64, f64), Error> {
    if !speed_m_s.is_finite() || speed_m_s < 0.0 || !radius_m.is_finite() || radius_m <= 0.0 {
        return Err(err("invalid speed or radius for the bicycle steady state"));
    }
    let cf = 2.0 * p.front.cornering_stiffness_n_per_deg;
    let cr = 2.0 * p.rear.cornering_stiffness_n_per_deg;
    let (a, b) = (p.a_dist_m(), p.b_dist_m());
    let k = (1.0 / radius_m).to_degrees(); // Ω/v in deg per metre of travel
    let rhs = [
        p.mass_kg * speed_m_s * speed_m_s / radius_m + (cf * a - cr * b) * k,
        (a * a * cf + b * b * cr) * k,
    ];
    let (m11, m12, m21, m22) = (cf, -(cf + cr), a * cf, b * cr - a * cf);
    let det = m11 * m22 - m12 * m21; // CF·CR·(a + b)
    if det.abs() < 1e-12 {
        return Err(err("singular bicycle steady-state matrix"));
    }
    Ok((
        (rhs[0] * m22 - m12 * rhs[1]) / det,
        (m11 * rhs[1] - m21 * rhs[0]) / det,
    ))
}

/// Result of an isolated steady-state corner.
#[derive(Clone, Debug)]
pub struct CornerResult {
    /// Steady-state speed, m/s.
    pub speed_m_s: f64,
    /// Lateral acceleration, m/s².
    pub ay_m_s2: f64,
    /// Yaw rate `v/R`, rad/s.
    pub yaw_rate_rad_s: f64,
    /// Front-wheel steer angle, degrees (Eq. 2-53).
    pub steer_deg: f64,
    /// Steering-wheel angle, degrees (Eq. 2-54: steer ratio × steer angle).
    pub steer_wheel_deg: f64,
    /// Body side-slip β, degrees.
    pub beta_deg: f64,
}

/// Maximum steady-state cornering speed on a radius (Fig. 3-30): iterate `v = sqrt(ay_max(v) R)`,
/// capped at `top_speed_m_s`.
///
/// # Errors
/// Returns an error for a nonfinite/zero radius, a nonconverging iteration, or any model error.
pub fn corner(
    model: &dyn StepModel,
    top_speed_m_s: f64,
    radius_m: f64,
) -> Result<CornerResult, Error> {
    if !radius_m.is_finite() || radius_m == 0.0 {
        return Err(err("invalid corner radius"));
    }
    let r = radius_m.abs();
    let mut v = 5.0_f64;
    let mut converged = false;
    for _ in 0..1000 {
        // capped inside the loop: with downforce the tyre-limited speed has no fixed point on a
        // very large radius, so the iteration must settle at the cap rather than diverge
        let vn = (max_ay(model, v, 0.0)? * r).sqrt().min(top_speed_m_s);
        if (vn - v).abs() < 1e-10 * v.max(1.0) {
            v = vn;
            converged = true;
            break;
        }
        v = 0.5 * (v + vn);
    }
    if !converged {
        return Err(err("cornering speed did not converge"));
    }
    let v = v.min(top_speed_m_s);
    let p = model.params();
    let (delta, beta) = steering(p, v, radius_m)?;
    Ok(CornerResult {
        speed_m_s: v,
        ay_m_s2: v * v / r,
        yaw_rate_rad_s: v / radius_m,
        steer_deg: delta,
        steer_wheel_deg: delta * p.steer_ratio,
        beta_deg: beta,
    })
}

/// Integration domain of an acceleration/braking scenario (§3.2.1.2).
#[derive(Clone, Copy, Debug)]
pub enum Solver {
    /// Fixed time step, seconds (thesis default 0.001).
    Time {
        /// Step, s.
        dt: f64,
    },
    /// Fixed distance step, metres (thesis default 0.25 accelerating, 0.15 braking).
    Distance {
        /// Step, m.
        dx: f64,
    },
}

/// End condition (§3.2.1.3). `Natural` stops at top speed (accelerating) or standstill (braking).
#[derive(Clone, Copy, Debug)]
pub enum End {
    /// Run to top speed / standstill.
    Natural,
    /// Stop on reaching this speed, m/s.
    TargetSpeed(f64),
    /// Stop on reaching this elapsed time, s.
    TargetTime(f64),
    /// Stop on reaching this distance, m.
    TargetDistance(f64),
}

/// Solver and end condition for a longitudinal scenario.
#[derive(Clone, Copy, Debug)]
pub struct LongSettings {
    /// Integration domain and step.
    pub solver: Solver,
    /// End condition.
    pub end: End,
}

/// Samples of a longitudinal scenario; index 0 is the initial state.
#[derive(Clone, Debug, Default)]
pub struct Trace {
    /// Elapsed time, s.
    pub time_s: Vec<f64>,
    /// Distance travelled, m.
    pub distance_m: Vec<f64>,
    /// Speed, m/s.
    pub speed_m_s: Vec<f64>,
    /// Longitudinal acceleration used for the step starting at this sample, m/s² (negative braking).
    pub ax_m_s2: Vec<f64>,
}

fn run_long(
    model: &dyn StepModel,
    tractive: Option<&TractiveTable>,
    v0: f64,
    s: &LongSettings,
    braking: bool,
) -> Result<Trace, Error> {
    let step_ok = match s.solver {
        Solver::Time { dt } => dt.is_finite() && dt > 0.0,
        Solver::Distance { dx } => dx.is_finite() && dx > 0.0,
    };
    if !step_ok || !v0.is_finite() || v0 < 0.0 {
        return Err(err("invalid scenario step or initial speed"));
    }
    let m = model.params().mass_kg;
    let (mut v, mut t, mut d, mut ax_prev) = (v0, 0.0, 0.0, 0.0);
    let mut out = Trace::default();
    for _ in 0..10_000_000 {
        let f = model.instant(v, ax_prev, 0.0)?.forces;
        let a = if braking {
            -(f.tyres_dec_n() + f.resist_n()) / m
        } else {
            let eng = tractive.expect("tractive table").force_at(v);
            (f.tyres_acc_n().min(eng) - f.resist_n()) / m
        };
        out.time_s.push(t);
        out.distance_m.push(d);
        out.speed_m_s.push(v);
        out.ax_m_s2.push(a);
        if (!braking && a <= 1e-6) || (braking && v <= 1e-6) {
            return Ok(out);
        }
        let (mut dt, mut dx) = match s.solver {
            Solver::Time { dt } => (dt, v * dt + 0.5 * a * dt * dt),
            Solver::Distance { dx } => {
                let v2 = v * v + 2.0 * a * dx;
                let v1 = if v2 > 0.0 { v2.sqrt() } else { 0.0 };
                let dt = if a.abs() > 1e-9 {
                    (v1 - v) / a
                } else {
                    dx / v.max(1e-9)
                };
                (dt, dx)
            }
        };
        let mut v1 = v + a * dt;
        let mut frac = 1.0;
        let mut done = false;
        match s.end {
            End::TargetSpeed(vt) => {
                if (braking && v1 <= vt) || (!braking && v1 >= vt) {
                    frac = ((vt - v) / (v1 - v)).clamp(0.0, 1.0);
                    done = true;
                }
            }
            End::TargetTime(tt) => {
                if t + dt >= tt {
                    frac = ((tt - t) / dt).clamp(0.0, 1.0);
                    done = true;
                }
            }
            End::TargetDistance(dd) => {
                if d + dx >= dd {
                    frac = ((dd - d) / dx).clamp(0.0, 1.0);
                    done = true;
                }
            }
            End::Natural => {
                if braking && v1 <= 0.0 {
                    frac = ((0.0 - v) / (v1 - v)).clamp(0.0, 1.0);
                    done = true;
                }
            }
        }
        if braking && v1 <= 0.0 {
            done = true;
        }
        if frac < 1.0 {
            dt *= frac;
            dx *= frac;
            v1 = v + a * dt;
        }
        t += dt;
        d += dx;
        v = v1.max(0.0);
        ax_prev = a;
        if done {
            out.time_s.push(t);
            out.distance_m.push(d);
            out.speed_m_s.push(v);
            out.ax_m_s2.push(a);
            return Ok(out);
        }
    }
    Err(err("scenario did not terminate"))
}

/// Quasi-steady acceleration from `v0` (§3.2): `ax = (min(engine, tyres) − drag − rolling) / m`
/// (Eqs. 2-44, 4-10), weight transfer from the previous step's `ax`. Build `tractive` with
/// `params.correlation.engine_power` as its engine scalar.
///
/// # Errors
/// Returns an error for a negative/nonfinite `v0`, a nonpositive step, or any model error.
pub fn accelerate(
    model: &dyn StepModel,
    tractive: &TractiveTable,
    v0: f64,
    s: &LongSettings,
) -> Result<Trace, Error> {
    run_long(model, Some(tractive), v0, s, false)
}

/// Quasi-steady braking from `v0` (§3.3): tyre-limited deceleration plus drag and rolling.
///
/// # Errors
/// Returns an error for a negative/nonfinite `v0`, a nonpositive step, or any model error.
pub fn brake(model: &dyn StepModel, v0: f64, s: &LongSettings) -> Result<Trace, Error> {
    run_long(model, None, v0, s, true)
}
