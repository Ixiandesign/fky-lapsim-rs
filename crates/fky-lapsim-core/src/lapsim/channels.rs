//! Driven channels and KPIs (thesis §2.9, §4.4): quantities back-calculated from the lap speed
//! trace that do not feed the equations of motion.
use super::err;
use super::forces::StepModel;
use super::scenarios::{bicycle_steady_state, steering};
use super::solver::LapTrace;
use super::thesis::BikeParams;
use super::tractive::TractiveTable;
use crate::Error;
use serde::{Deserialize, Serialize};

/// Front and rear brake line pressure for a deceleration magnitude, bar (Eqs. 2-55…2-66).
///
/// # Errors
/// Returns an error for a negative or nonfinite deceleration or invalid brake hardware.
pub fn brake_pressure_bar(p: &BikeParams, decel_m_s2: f64) -> Result<(f64, f64), Error> {
    if !decel_m_s2.is_finite() || decel_m_s2 < 0.0 {
        return Err(err("deceleration must be nonnegative"));
    }
    let b = &p.brake;
    let force = p.mass_kg * decel_m_s2; // Eq. 2-55
    let bias = [b.front_bias_pct / 100.0, 1.0 - b.front_bias_pct / 100.0];
    let mut out = [0.0; 2];
    for i in 0..2 {
        let corner = force * bias[i] / 2.0; // Eq. 2-56
        let torque = corner * p.rolling_radius_m; // Eq. 2-57
        let r_eff = b.disc_diameter_m[i] / 2.0 - b.pad_height_m[i] / 2.0; // Eq. 2-58
        if r_eff <= 0.0 || b.pad_mu[i] <= 0.0 || b.pistons[i] == 0 || b.piston_diameter_m[i] <= 0.0
        {
            return Err(err("invalid brake hardware"));
        }
        let clamp = torque / r_eff / b.pad_mu[i]; // Eqs. 2-59, 2-60
        let area =
            f64::from(b.pistons[i]) * std::f64::consts::PI * b.piston_diameter_m[i].powi(2) / 4.0; // Eq. 2-61
        out[i] = clamp / area / 1.0e5; // Eqs. 2-62, 2-63
    }
    Ok((out[0], out[1]))
}

/// Throttle position implied by the lap, percent (Eqs. 2-67/2-68): the share of the engine's
/// tractive force needed to produce `m·ax` against `resist_n`; 0 when not driving, clamped to 100.
///
/// Deviation from the printed Eq. 2-67 (`100·Fx / (F_engine − F_drag)` with `Fx` the net force):
/// that form gives 0 % at constant speed, but the thesis itself reports about 15 % throttle in
/// steady cornering (§3.4.1.5), i.e. it adds the drag-compensating force there. This uses the
/// total tractive force `m·ax + resist` over the engine force, which equals the printed form at
/// full power (100 %) and reproduces the steady-cornering behaviour. Driven channel only.
pub fn throttle_pct(mass_kg: f64, ax: f64, resist_n: f64, engine_force_n: f64) -> f64 {
    let needed = mass_kg * ax + resist_n;
    if needed <= 0.0 || engine_force_n <= 0.0 {
        return 0.0;
    }
    (100.0 * needed / engine_force_n).clamp(0.0, 100.0)
}

/// Per-point driven channels (same indexing as the [`LapTrace`]).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Channels {
    /// Distance, m.
    pub distance_m: Vec<f64>,
    /// Elapsed time, s.
    pub time_s: Vec<f64>,
    /// Speed, km/h.
    pub speed_kmh: Vec<f64>,
    /// Longitudinal acceleration, m/s².
    pub ax_m_s2: Vec<f64>,
    /// Lateral acceleration, m/s² (unsigned).
    pub ay_m_s2: Vec<f64>,
    /// Front lateral force (signed, + left), N.
    pub fy_f_n: Vec<f64>,
    /// Rear lateral force, N.
    pub fy_r_n: Vec<f64>,
    /// Front longitudinal tyre force, N.
    pub fx_f_n: Vec<f64>,
    /// Rear longitudinal tyre force, N.
    pub fx_r_n: Vec<f64>,
    /// Steering-wheel angle, degrees (+ left).
    pub steer_wheel_deg: Vec<f64>,
    /// Body side-slip β, degrees.
    pub beta_deg: Vec<f64>,
    /// Front brake line pressure, bar.
    pub brake_front_bar: Vec<f64>,
    /// Rear brake line pressure, bar.
    pub brake_rear_bar: Vec<f64>,
    /// Throttle position, percent.
    pub tps_pct: Vec<f64>,
    /// Engine speed, rpm.
    pub rpm: Vec<f64>,
    /// Selected gear, zero-based.
    pub gear: Vec<usize>,
    /// Longitudinal weight transfer `m·ax·h/WB`, N.
    pub weight_transfer_n: Vec<f64>,
    /// Effective total downforce coefficient.
    pub cz_total: Vec<f64>,
    /// Effective front aero balance, percent.
    pub aero_balance_front_pct: Vec<f64>,
    /// Effective drag coefficient.
    pub cx_total: Vec<f64>,
    /// Front ride height, mm (0 when the model does not provide it).
    pub front_rh_mm: Vec<f64>,
    /// Rear ride height, mm.
    pub rear_rh_mm: Vec<f64>,
    /// Roll angle, degrees.
    pub roll_deg: Vec<f64>,
    /// Front-left wheel load, N.
    pub fz_fl_n: Vec<f64>,
    /// Front-right wheel load, N.
    pub fz_fr_n: Vec<f64>,
    /// Rear-left wheel load, N.
    pub fz_rl_n: Vec<f64>,
    /// Rear-right wheel load, N.
    pub fz_rr_n: Vec<f64>,
    /// Front-left camber, degrees (outward positive); zero in the bike-only model.
    pub camber_fl_deg: Vec<f64>,
    /// Front-right camber, degrees.
    pub camber_fr_deg: Vec<f64>,
    /// Rear-left camber, degrees.
    pub camber_rl_deg: Vec<f64>,
    /// Rear-right camber, degrees.
    pub camber_rr_deg: Vec<f64>,
    /// True where the point is tyre-grip limited, false where power limited.
    pub grip_limited: Vec<bool>,
}

/// Back-calculate the driven channels for every point of a lap trace.
///
/// # Errors
/// Propagates model and steering errors.
pub fn driven_channels(
    model: &dyn StepModel,
    tractive: &TractiveTable,
    trace: &LapTrace,
) -> Result<Channels, Error> {
    let p = model.params();
    let m = p.mass_kg;
    let mut c = Channels::default();
    for i in 0..trace.speed_m_s.len() {
        let (v, ax, r) = (trace.speed_m_s[i], trace.ax_m_s2[i], trace.radius_m[i]);
        let ay_signed = v * v / r;
        let inst = model.instant(v, ax, ay_signed)?;
        let (f, att) = (&inst.forces, &inst.attitude);
        let fx_total = m * ax + f.resist_n();
        let (fx_f, fx_r) = if fx_total >= 0.0 {
            let acc = f.tyres_acc_n().max(1e-12);
            (
                fx_total * f.tyres_acc_f_n / acc,
                fx_total * f.tyres_acc_r_n / acc,
            )
        } else {
            let dec = f.tyres_dec_n().max(1e-12);
            (
                fx_total * f.tyres_dec_f_n / dec,
                fx_total * f.tyres_dec_r_n / dec,
            )
        };
        // steer angle: the thesis matrix (Eq. 2-53); body slip: the steady-state bicycle value, since
        // the thesis matrix is ill-conditioned for near-neutral weight distributions
        let (steer, beta) = if v > 1e-6 {
            let steer = steering(p, v, r)?.0;
            let (_, beta) = bicycle_steady_state(p, v, r.abs())?;
            (steer, beta * r.signum())
        } else {
            (0.0, 0.0)
        };
        let (bf, br) = if ax < 0.0 {
            brake_pressure_bar(p, -ax)?
        } else {
            (0.0, 0.0)
        };
        let engine = tractive.force_at(v);
        c.distance_m.push(trace.distance_m[i]);
        c.time_s.push(trace.time_s[i]);
        c.speed_kmh.push(v * 3.6);
        c.ax_m_s2.push(ax);
        c.ay_m_s2.push(trace.ay_m_s2[i]);
        c.fy_f_n.push(m * ay_signed * p.b_dist_m() / p.wheelbase_m);
        c.fy_r_n.push(m * ay_signed * p.a_dist_m() / p.wheelbase_m);
        c.fx_f_n.push(fx_f);
        c.fx_r_n.push(fx_r);
        c.steer_wheel_deg.push(steer * p.steer_ratio);
        c.beta_deg.push(beta);
        c.brake_front_bar.push(bf);
        c.brake_rear_bar.push(br);
        c.tps_pct.push(throttle_pct(m, ax, f.resist_n(), engine));
        c.rpm.push(tractive.rpm_at(v));
        c.gear.push(tractive.gear_at(v));
        c.weight_transfer_n
            .push(m * ax * p.cg_height_m / p.wheelbase_m);
        c.cz_total.push(att.cz_total);
        c.aero_balance_front_pct.push(att.aero_balance_front_pct);
        c.cx_total.push(att.cx_total);
        c.front_rh_mm.push(att.front_rh_mm);
        c.rear_rh_mm.push(att.rear_rh_mm);
        c.roll_deg.push(att.roll_deg);
        c.fz_fl_n.push(att.wheel_fz_n[0]);
        c.fz_fr_n.push(att.wheel_fz_n[1]);
        c.fz_rl_n.push(att.wheel_fz_n[2]);
        c.fz_rr_n.push(att.wheel_fz_n[3]);
        c.camber_fl_deg.push(att.camber_deg[0]);
        c.camber_fr_deg.push(att.camber_deg[1]);
        c.camber_rl_deg.push(att.camber_deg[2]);
        c.camber_rr_deg.push(att.camber_deg[3]);
        c.grip_limited.push(!(ax > 0.0 && f.tyres_acc_n() > engine));
    }
    Ok(c)
}

/// Thresholds for the KPI classification (the thesis does not publish its own).
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct KpiSettings {
    /// Below this speed a point counts as low speed, km/h.
    pub low_speed_kmh: f64,
    /// Above this speed a point counts as high speed, km/h.
    pub high_speed_kmh: f64,
    /// Lateral acceleration above which a point counts as cornering, m/s².
    pub cornering_ay_m_s2: f64,
    /// Longitudinal acceleration magnitude above which a point counts as accelerating/decelerating, m/s².
    pub accel_threshold_m_s2: f64,
}

impl Default for KpiSettings {
    fn default() -> Self {
        Self {
            low_speed_kmh: 60.0,
            high_speed_kmh: 100.0,
            cornering_ay_m_s2: 3.0,
            accel_threshold_m_s2: 0.5,
        }
    }
}

/// Key performance indicators of a lap (thesis Table 4-7 subset; time-weighted percentages).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Kpis {
    /// Lap time, s.
    pub lap_time_s: f64,
    /// Track length, m.
    pub track_length_m: f64,
    /// Time at low speed, %.
    pub low_speed_pct: f64,
    /// Time at medium speed, %.
    pub medium_speed_pct: f64,
    /// Time at high speed, %.
    pub high_speed_pct: f64,
    /// Time cornering, %.
    pub cornering_pct: f64,
    /// Time accelerating, %.
    pub accelerating_pct: f64,
    /// Time decelerating, %.
    pub decelerating_pct: f64,
    /// Time grip limited, %.
    pub grip_limited_pct: f64,
    /// Time power limited, %.
    pub power_limited_pct: f64,
    /// Lowest gear used (zero-based).
    pub min_gear: usize,
    /// Highest gear used.
    pub max_gear: usize,
    /// Number of gear changes.
    pub gear_shifts: usize,
    /// Maximum speed, km/h.
    pub max_speed_kmh: f64,
    /// Minimum speed, km/h.
    pub min_speed_kmh: f64,
    /// Median speed (by point), km/h.
    pub median_speed_kmh: f64,
}

/// Compute KPIs from channels.
pub fn kpis(c: &Channels, lap_time_s: f64, track_length_m: f64, s: &KpiSettings) -> Kpis {
    let n = c.speed_kmh.len();
    // time weight of each point: distance to next point over mean speed (closing section wraps)
    let w: Vec<f64> = (0..n)
        .map(|i| {
            let j = (i + 1) % n;
            let ds = if i + 1 < n {
                c.distance_m[j] - c.distance_m[i]
            } else {
                track_length_m - c.distance_m[i]
            };
            2.0 * ds / ((c.speed_kmh[i] + c.speed_kmh[j]) / 3.6).max(1e-9)
        })
        .collect();
    let total: f64 = w.iter().sum::<f64>().max(1e-12);
    let pct = |pred: &dyn Fn(usize) -> bool| -> f64 {
        100.0 * (0..n).filter(|&i| pred(i)).map(|i| w[i]).sum::<f64>() / total
    };
    let mut speeds = c.speed_kmh.clone();
    speeds.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let shifts = c.gear.windows(2).filter(|g| g[0] != g[1]).count();
    Kpis {
        lap_time_s,
        track_length_m,
        low_speed_pct: pct(&|i| c.speed_kmh[i] < s.low_speed_kmh),
        medium_speed_pct: pct(&|i| {
            c.speed_kmh[i] >= s.low_speed_kmh && c.speed_kmh[i] <= s.high_speed_kmh
        }),
        high_speed_pct: pct(&|i| c.speed_kmh[i] > s.high_speed_kmh),
        cornering_pct: pct(&|i| c.ay_m_s2[i] > s.cornering_ay_m_s2),
        accelerating_pct: pct(&|i| c.ax_m_s2[i] > s.accel_threshold_m_s2),
        decelerating_pct: pct(&|i| c.ax_m_s2[i] < -s.accel_threshold_m_s2),
        grip_limited_pct: pct(&|i| c.grip_limited[i]),
        power_limited_pct: pct(&|i| !c.grip_limited[i]),
        min_gear: *c.gear.iter().min().unwrap_or(&0),
        max_gear: *c.gear.iter().max().unwrap_or(&0),
        gear_shifts: shifts,
        max_speed_kmh: *speeds.last().unwrap_or(&0.0),
        min_speed_kmh: *speeds.first().unwrap_or(&0.0),
        median_speed_kmh: speeds.get(n / 2).copied().unwrap_or(0.0),
    }
}
