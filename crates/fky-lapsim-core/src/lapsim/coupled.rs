//! Bike model ↔ 7×7 coupling (spec §6.0): the thesis bike model proposes forces, the full-car
//! matrix distributes them to the four wheels and attitude, and the two are iterated until they
//! agree at every step.
use super::aeromap::AeroMap;
use super::err;
use super::forces::{forces, forces_at_loads, AeroCoeffs, Attitude, Forces, Instant, StepModel};
use super::fullcar::{FullCar, LoadInputs};
use super::scenarios::bicycle_steady_state;
use super::thesis::{AxleTyre, BikeParams};
use super::G;
use crate::powertrain::DrivenAxle;
use crate::Error;

/// Peak friction versus wheel load and camber (built from the Magic Formula tyre in Task 16).
#[derive(Clone, Debug)]
pub struct TyreTable {
    /// Wheel loads, N, strictly increasing.
    pub load_n: Vec<f64>,
    /// Camber angles, degrees (outward positive), strictly increasing.
    pub camber_deg: Vec<f64>,
    /// Peak longitudinal friction `[load][camber]`.
    pub mux: Vec<Vec<f64>>,
    /// Peak lateral friction `[load][camber]`.
    pub muy: Vec<Vec<f64>>,
}

impl TyreTable {
    /// Bilinear `(μx, μy)` at a wheel load and camber.
    ///
    /// # Errors
    /// Returns an error outside the tabulated load or camber range (never extrapolates).
    pub fn mu(&self, fz: f64, camber_deg: f64) -> Result<(f64, f64), Error> {
        let cell = |grid: &[f64], x: f64| -> Result<(usize, f64), Error> {
            if !x.is_finite() || x < grid[0] || x > grid[grid.len() - 1] {
                return Err(err("tyre table query outside its calibrated range"));
            }
            let i = (grid.partition_point(|g| *g <= x).saturating_sub(1)).min(grid.len() - 2);
            Ok((i, (x - grid[i]) / (grid[i + 1] - grid[i])))
        };
        let (i, ti) = cell(&self.load_n, fz)?;
        let (j, tj) = cell(&self.camber_deg, camber_deg)?;
        let bil = |t: &Vec<Vec<f64>>| {
            let a = t[i][j] * (1.0 - tj) + t[i][j + 1] * tj;
            let b = t[i + 1][j] * (1.0 - tj) + t[i + 1][j + 1] * tj;
            a * (1.0 - ti) + b * ti
        };
        Ok((bil(&self.mux), bil(&self.muy)))
    }
}

/// How per-wheel friction is evaluated.
#[derive(Clone, Debug)]
pub enum TyreMode {
    /// The thesis load-sensitivity formula (Eqs. 2-12…2-15) at each wheel's own load.
    Thesis,
    /// Magic-Formula-derived peak friction tables per axle (front, rear), load and camber aware.
    Table([TyreTable; 2]),
}

/// Loop settings.
#[derive(Clone, Copy, Debug)]
pub struct CoupleSettings {
    /// Under-relaxation of the wheel loads between iterations (0, 1].
    pub relax: f64,
    /// Iteration limit.
    pub max_iter: usize,
    /// Convergence tolerance on the largest wheel-load change, N.
    pub load_tol_n: f64,
    /// Convergence tolerance on the ride-height change (front + rear), mm.
    pub rh_tol_mm: f64,
    /// Largest accepted axle-load disagreement between bike model and matrix, fraction of total load.
    pub max_axle_mismatch_frac: f64,
    /// Optional initial wheel loads (testing the fixed point's independence of the guess).
    pub initial_wheel_fz_n: Option<[f64; 4]>,
}

impl Default for CoupleSettings {
    fn default() -> Self {
        Self {
            relax: 0.6,
            max_iter: 100,
            load_tol_n: 1e-4,
            rh_tol_mm: 1e-6,
            max_axle_mismatch_frac: 0.03,
            initial_wheel_fz_n: None,
        }
    }
}

/// The coupled bike ↔ 7×7 model.
#[derive(Clone, Debug)]
pub struct Coupled {
    /// Bike parameters.
    pub params: BikeParams,
    /// The 7×7 full-car model.
    pub car: FullCar,
    /// Optional AeroMap (otherwise constant aero).
    pub aero_map: Option<AeroMap>,
    /// Per-wheel friction evaluation.
    pub tyres: TyreMode,
    /// Loop settings.
    pub settings: CoupleSettings,
}

fn tyre_of(p: &BikeParams, axle: usize) -> &AxleTyre {
    if axle == 0 {
        &p.front
    } else {
        &p.rear
    }
}

impl Coupled {
    /// Per-wheel forces from the matrix wheel loads (axle sums are the totals).
    fn forces_from_wheels(
        &self,
        load: &[f64; 4],
        camber: &[f64; 4],
        df: (f64, f64),
        drag_n: f64,
    ) -> Result<Forces, Error> {
        let p = &self.params;
        if load.iter().any(|f| *f < 0.0) {
            return Err(err("wheel lift-off (negative wheel load)"));
        }
        let c = &p.correlation;
        let (mut acc, mut dec, mut fy) = ([0.0; 2], [0.0; 2], [0.0; 2]);
        let mut fz_axle = [0.0; 2];
        for i in 0..4 {
            let axle = i / 2;
            let t = tyre_of(p, axle);
            let (mux, muy) = match &self.tyres {
                TyreMode::Thesis => (
                    t.mux + t.mux_sens_per_n * (t.mux_norm_kg * G - load[i]),
                    t.muy + t.muy_sens_per_n * (t.muy_norm_kg * G - load[i]),
                ),
                TyreMode::Table(tabs) => tabs[axle].mu(load[i], camber[i])?,
            };
            let driven = match p.drive {
                DrivenAxle::Rear => axle == 1,
                DrivenAxle::Front => axle == 0,
                DrivenAxle::All => true,
            };
            fz_axle[axle] += load[i];
            if driven {
                acc[axle] += mux * load[i];
            }
            dec[axle] += mux * load[i];
            fy[axle] += muy * load[i];
        }
        let mut f = forces_at_loads(p, fz_axle[0] - df.0, fz_axle[1] - df.1, df.0, df.1, drag_n)?;
        f.tyres_acc_f_n = acc[0] * c.mux_accel;
        f.tyres_acc_r_n = acc[1] * c.mux_accel;
        f.tyres_dec_f_n = dec[0] * c.mux_brake;
        f.tyres_dec_r_n = dec[1] * c.mux_brake;
        f.fy_f_n = fy[0] * c.muy;
        f.fy_r_n = fy[1] * c.muy;
        f.mux_f = dec[0] / fz_axle[0].max(1e-12);
        f.mux_r = dec[1] / fz_axle[1].max(1e-12);
        f.muy_f = fy[0] / fz_axle[0].max(1e-12);
        f.muy_r = fy[1] / fz_axle[1].max(1e-12);
        Ok(f)
    }
}

impl StepModel for Coupled {
    fn params(&self) -> &BikeParams {
        &self.params
    }

    fn instant(&self, v: f64, ax: f64, ay: f64) -> Result<Instant, Error> {
        let p = &self.params;
        let s = &self.settings;
        let m = p.mass_kg;
        let guess = forces(p, v, ax, p.const_aero_coeffs())?;
        let mut fz = s.initial_wheel_fz_n.unwrap_or([
            guess.tot_f_n / 2.0,
            guess.tot_f_n / 2.0,
            guess.tot_r_n / 2.0,
            guess.tot_r_n / 2.0,
        ]);
        let (mut frh, mut rrh) = (
            self.car.params.static_rh_mm[0],
            self.car.params.static_rh_mm[1],
        );
        let (mut roll_deg, mut beta_deg) = (0.0, 0.0);
        let k = p.correlation.aero;
        for it in 1..=s.max_iter {
            // 1. aero state at the current attitude
            let (coeffs, clamped) = match &self.aero_map {
                Some(map) => {
                    let a = map.eval(frh, rrh, roll_deg, beta_deg)?;
                    (
                        AeroCoeffs {
                            cz_front: a.czt * a.ab_front_pct / 100.0 * k,
                            cz_rear: a.czt * (1.0 - a.ab_front_pct / 100.0) * k,
                            cx: a.cx * k,
                        },
                        a.clamped,
                    )
                }
                None => (p.const_aero_coeffs(), false),
            };
            // 2. thesis bike-model prediction
            let bike = forces(p, v, ax, coeffs)?;
            // 3. axle forces by statics, split to wheels by current load
            let fy_axle = [
                m * ay * p.b_dist_m() / p.wheelbase_m,
                m * ay * p.a_dist_m() / p.wheelbase_m,
            ];
            // ground force on the car: inertia plus aero drag (rolling resistance is internal to the tyre)
            let fx_total = m * ax + bike.drag_n;
            let mut fy_wheel = [0.0; 4];
            let mut fx_wheel = [0.0; 4];
            let drive_w: [f64; 4] = match p.drive {
                DrivenAxle::Rear => [0.0, 0.0, fz[2], fz[3]],
                DrivenAxle::Front => [fz[0], fz[1], 0.0, 0.0],
                DrivenAxle::All => fz,
            };
            let w_fx = if fx_total >= 0.0 { drive_w } else { fz };
            let w_sum: f64 = w_fx.iter().sum::<f64>().max(1e-12);
            for i in 0..4 {
                let axle = i / 2;
                let pair = (fz[2 * axle] + fz[2 * axle + 1]).max(1e-12);
                fy_wheel[i] = fy_axle[axle] * fz[i] / pair;
                fx_wheel[i] = fx_total * w_fx[i] / w_sum;
            }
            // 4. the matrix check
            let sol = self.car.solve(&LoadInputs {
                ax,
                ay,
                downforce_front_n: bike.df_f_n,
                downforce_rear_n: bike.df_r_n,
                drag_n: bike.drag_n,
                fx_wheel_n: fx_wheel,
                fy_wheel_n: fy_wheel,
            })?;
            // 5. residuals and update
            let d_load = (0..4)
                .map(|i| (sol.wheel_load_n[i] - fz[i]).abs())
                .fold(0.0, f64::max);
            let d_rh = (sol.front_rh_mm - frh).abs() + (sol.rear_rh_mm - rrh).abs();
            if d_load < s.load_tol_n && d_rh < s.rh_tol_mm {
                // the bike model's axle loads plus the drag pitch moment it neglects
                let drag_shift = self.car.drag_transfer_n(bike.drag_n);
                let mismatch = ((sol.wheel_load_n[0] + sol.wheel_load_n[1]
                    - (bike.tot_f_n - drag_shift))
                    .abs()
                    + (sol.wheel_load_n[2] + sol.wheel_load_n[3] - (bike.tot_r_n + drag_shift))
                        .abs())
                    / bike.tot_n().max(1e-12);
                if mismatch > s.max_axle_mismatch_frac {
                    return Err(err(format!(
                        "bike model and 7x7 disagree on axle loads by {:.1} % at v={v:.2} ax={ax:.2} ay={ay:.2}",
                        100.0 * mismatch
                    )));
                }
                let f = self.forces_from_wheels(
                    &sol.wheel_load_n,
                    &sol.camber_deg,
                    (bike.df_f_n, bike.df_r_n),
                    bike.drag_n,
                )?;
                let att = Attitude {
                    front_rh_mm: sol.front_rh_mm,
                    rear_rh_mm: sol.rear_rh_mm,
                    roll_deg: sol.roll_deg,
                    pitch_deg: sol.pitch_deg,
                    beta_deg: beta_deg * ay.signum(),
                    wheel_fz_n: sol.wheel_load_n,
                    camber_deg: sol.camber_deg,
                    toe_deg: sol.toe_deg,
                    cz_total: coeffs.cz_front + coeffs.cz_rear,
                    aero_balance_front_pct: 100.0 * coeffs.cz_front
                        / (coeffs.cz_front + coeffs.cz_rear).max(1e-12),
                    cx_total: coeffs.cx,
                    iterations: it,
                    converged_aero_clamped: clamped,
                };
                return Ok(Instant {
                    forces: f,
                    attitude: att,
                });
            }
            for i in 0..4 {
                fz[i] += s.relax * (sol.wheel_load_n[i] - fz[i]);
            }
            frh += s.relax * (sol.front_rh_mm - frh);
            rrh += s.relax * (sol.rear_rh_mm - rrh);
            roll_deg = sol.roll_deg;
            beta_deg = if ay.abs() > 1e-9 && v > 1e-6 {
                bicycle_steady_state(p, v, v * v / ay.abs())?.1.abs()
            } else {
                0.0
            };
        }
        Err(err(format!(
            "bike/7x7 coupling did not converge in {} iterations at v={v:.2} ax={ax:.2} ay={ay:.2}",
            s.max_iter
        )))
    }
}
