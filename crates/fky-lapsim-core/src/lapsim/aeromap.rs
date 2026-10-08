//! The thesis AeroMap (§6.2) and the AeroMap-aware step model (§7.2, §7.3: Figs. 7-6, 7-7).
use super::err;
use super::forces::{forces, AeroCoeffs, Attitude, Instant, StepModel};
use super::rates::{rates, Rates, ThesisSuspension};
use super::scenarios::bicycle_steady_state;
use super::thesis::BikeParams;
use super::G;
use crate::Error;
use serde::{Deserialize, Serialize};

/// Percentage change of an aero parameter versus its zero-angle value, as a function of `|angle|`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Sensitivity {
    /// Angles, degrees, strictly increasing, starting at 0.
    pub angle_deg: Vec<f64>,
    /// Percent change of CzT.
    pub czt_pct: Vec<f64>,
    /// Percent change of the aero balance.
    pub ab_pct: Vec<f64>,
    /// Percent change of Cx.
    pub cx_pct: Vec<f64>,
}

/// Aerodynamic map over front/rear ride height with roll and yaw sensitivities (§6.2).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AeroMap {
    /// Front ride-height grid, mm, strictly increasing.
    pub front_rh_mm: Vec<f64>,
    /// Rear ride-height grid, mm, strictly increasing.
    pub rear_rh_mm: Vec<f64>,
    /// Total downforce coefficient `[front index][rear index]`.
    pub czt: Vec<Vec<f64>>,
    /// Front aero balance, percent.
    pub ab_front_pct: Vec<Vec<f64>>,
    /// Drag coefficient (positive).
    pub cx: Vec<Vec<f64>>,
    /// Roll sensitivity.
    pub roll: Sensitivity,
    /// Yaw sensitivity.
    pub yaw: Sensitivity,
}

/// An evaluated aero state.
#[derive(Clone, Copy, Debug)]
pub struct AeroPoint {
    /// Total downforce coefficient.
    pub czt: f64,
    /// Front aero balance, percent.
    pub ab_front_pct: f64,
    /// Drag coefficient.
    pub cx: f64,
    /// True when the ride height was outside the swept range and held at the edge (§6.2.3.2).
    pub clamped: bool,
}

fn increasing(v: &[f64]) -> bool {
    v.len() >= 2 && v.iter().all(|x| x.is_finite()) && v.windows(2).all(|w| w[1] > w[0])
}

fn lookup(grid: &[f64], x: f64) -> (usize, f64, bool) {
    if x <= grid[0] {
        return (0, 0.0, x < grid[0]);
    }
    let last = grid.len() - 1;
    if x >= grid[last] {
        return (last - 1, 1.0, x > grid[last]);
    }
    let i = grid.partition_point(|g| *g <= x) - 1;
    (i, (x - grid[i]) / (grid[i + 1] - grid[i]), false)
}

fn interp(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    let (i, t, _) = lookup(xs, x);
    ys[i] * (1.0 - t) + ys[i + 1] * t
}

impl Sensitivity {
    fn valid(&self) -> bool {
        let n = self.angle_deg.len();
        increasing(&self.angle_deg)
            && self.angle_deg[0] == 0.0
            && [&self.czt_pct, &self.ab_pct, &self.cx_pct]
                .iter()
                .all(|v| v.len() == n && v.iter().all(|x| x.is_finite() && *x > -100.0))
    }
}

impl AeroMap {
    /// Validate grids, table shapes and finite physical values.
    ///
    /// # Errors
    /// Non-increasing grids, wrong table shapes, nonfinite entries, aero balance outside 0–100 %,
    /// negative coefficients, or invalid sensitivities.
    pub fn validate(&self) -> Result<(), Error> {
        let (nf, nr) = (self.front_rh_mm.len(), self.rear_rh_mm.len());
        let shape_ok = |t: &Vec<Vec<f64>>| {
            t.len() == nf
                && t.iter()
                    .all(|r| r.len() == nr && r.iter().all(|x| x.is_finite()))
        };
        if !increasing(&self.front_rh_mm)
            || !increasing(&self.rear_rh_mm)
            || !shape_ok(&self.czt)
            || !shape_ok(&self.ab_front_pct)
            || !shape_ok(&self.cx)
            || self.czt.iter().flatten().any(|x| *x < 0.0)
            || self.cx.iter().flatten().any(|x| *x < 0.0)
            || self
                .ab_front_pct
                .iter()
                .flatten()
                .any(|x| !(0.0..=100.0).contains(x))
            || !self.roll.valid()
            || !self.yaw.valid()
        {
            return Err(err("invalid aero map"));
        }
        Ok(())
    }

    /// Evaluate at a vehicle state: bilinear in ride height (held at the swept edge), then the roll
    /// and yaw sensitivities of `|angle|` applied multiplicatively (thesis Tables 6-3…6-5).
    ///
    /// # Errors
    /// Invalid map or nonfinite inputs.
    pub fn eval(
        &self,
        front_rh_mm: f64,
        rear_rh_mm: f64,
        roll_deg: f64,
        yaw_deg: f64,
    ) -> Result<AeroPoint, Error> {
        self.validate()?;
        if ![front_rh_mm, rear_rh_mm, roll_deg, yaw_deg]
            .iter()
            .all(|x| x.is_finite())
        {
            return Err(err("nonfinite aero state"));
        }
        let (i, ti, ci) = lookup(&self.front_rh_mm, front_rh_mm);
        let (j, tj, cj) = lookup(&self.rear_rh_mm, rear_rh_mm);
        let bil = |t: &Vec<Vec<f64>>| {
            let (a, b) = (
                t[i][j] * (1.0 - tj) + t[i][j + 1] * tj,
                t[i + 1][j] * (1.0 - tj) + t[i + 1][j + 1] * tj,
            );
            a * (1.0 - ti) + b * ti
        };
        let (mut czt, mut ab, mut cx) = (bil(&self.czt), bil(&self.ab_front_pct), bil(&self.cx));
        for (s, angle) in [(&self.roll, roll_deg.abs()), (&self.yaw, yaw_deg.abs())] {
            czt *= 1.0 + interp(&s.angle_deg, &s.czt_pct, angle) / 100.0;
            ab *= 1.0 + interp(&s.angle_deg, &s.ab_pct, angle) / 100.0;
            cx *= 1.0 + interp(&s.angle_deg, &s.cx_pct, angle) / 100.0;
        }
        Ok(AeroPoint {
            czt,
            ab_front_pct: ab.clamp(0.0, 100.0),
            cx,
            clamped: ci || cj,
        })
    }
}

/// The thesis model with the AeroMap and the §7.2/§7.3 ride-height, roll and yaw convergence.
///
/// Ride height follows Eqs. 7-1, 7-8, 7-9 per wheel: `RH = static − 1000·(ΔS_axle/2)/KHeave`,
/// with `ΔS_front = DF_F − WT·(1 − anti_f)` and `ΔS_rear = DF_R + WT·(1 − anti_r)`. (Eq. 7-8 is
/// printed as `WT·AntiFeature`; the surrounding text defines 100 % anti as springs not deflecting,
/// so the spring-borne share is `1 − anti`.) The downforce term reproduces the thesis's skidpad
/// ride heights of Table 7-14 (31.7/40.8 mm).
#[derive(Clone, Debug)]
pub struct ThesisAeroMap {
    /// Vehicle.
    pub params: BikeParams,
    /// Suspension inputs (heave rates, anti-features, roll gradient).
    pub suspension: ThesisSuspension,
    /// Aero map.
    pub map: AeroMap,
    /// Apply longitudinal weight transfer.
    pub weight_transfer: bool,
}

impl ThesisAeroMap {
    fn rates(&self) -> Result<Rates, Error> {
        rates(&self.params, &self.suspension)
    }
}

impl StepModel for ThesisAeroMap {
    fn params(&self) -> &BikeParams {
        &self.params
    }
    fn instant(&self, v: f64, ax: f64, ay: f64) -> Result<Instant, Error> {
        let p = &self.params;
        let s = &self.suspension;
        let r = self.rates()?;
        let ax_wt = if self.weight_transfer { ax } else { 0.0 };
        let wt = p.mass_kg * ax_wt * p.cg_height_m / p.wheelbase_m; // Eq. 7-1
        let (anti_f, anti_r) = if ax_wt < 0.0 {
            (s.anti_dive_front_pct, s.anti_lift_rear_pct)
        } else {
            (s.anti_lift_front_pct, s.anti_squat_rear_pct)
        };
        let roll_deg = r.roll_gradient_deg_per_g * ay / G;
        let beta_deg = if ay.abs() > 1e-9 && v > 1e-6 {
            bicycle_steady_state(p, v, v * v / ay.abs())?.1.abs()
        } else {
            0.0
        };
        let k = p.correlation.aero;
        let (mut frh, mut rrh) = (s.static_rh_mm[0], s.static_rh_mm[1]);
        for it in 1..=100 {
            let a = self.map.eval(frh, rrh, roll_deg, beta_deg)?;
            let coeffs = AeroCoeffs {
                cz_front: a.czt * a.ab_front_pct / 100.0 * k,
                cz_rear: a.czt * (1.0 - a.ab_front_pct / 100.0) * k,
                cx: a.cx * k,
            };
            let f = forces(p, v, ax_wt, coeffs)?;
            let ds_f = f.df_f_n - wt * (1.0 - anti_f / 100.0); // Eq. 7-8 (front)
            let ds_r = f.df_r_n + wt * (1.0 - anti_r / 100.0); // Eq. 7-8 (rear)
            let nf = s.static_rh_mm[0] - 1000.0 * (ds_f / 2.0) / r.kheave_n_m[0]; // Eq. 7-9
            let nr = s.static_rh_mm[1] - 1000.0 * (ds_r / 2.0) / r.kheave_n_m[1];
            if (nf - frh).abs() + (nr - rrh).abs() < 1e-9 {
                let att = Attitude {
                    front_rh_mm: nf,
                    rear_rh_mm: nr,
                    roll_deg,
                    beta_deg: beta_deg * ay.signum(),
                    wheel_fz_n: [
                        f.tot_f_n / 2.0,
                        f.tot_f_n / 2.0,
                        f.tot_r_n / 2.0,
                        f.tot_r_n / 2.0,
                    ],
                    cz_total: coeffs.cz_front + coeffs.cz_rear,
                    aero_balance_front_pct: 100.0 * coeffs.cz_front
                        / (coeffs.cz_front + coeffs.cz_rear).max(1e-12),
                    cx_total: coeffs.cx,
                    iterations: it,
                    converged_aero_clamped: a.clamped,
                    ..Default::default()
                };
                return Ok(Instant {
                    forces: f,
                    attitude: att,
                });
            }
            frh = 0.5 * (frh + nf);
            rrh = 0.5 * (rrh + nr);
        }
        Err(err("AeroMap ride-height convergence failed"))
    }
}
