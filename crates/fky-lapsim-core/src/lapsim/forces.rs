//! The thesis Forces Model (§2.8): vertical, longitudinal and lateral axle forces with
//! load-sensitive friction, plus the [`StepModel`] abstraction every solver runs on.
use super::thesis::BikeParams;
use super::{err, G};
use crate::powertrain::DrivenAxle;
use crate::Error;

/// Effective aero coefficients at one instant (constant, or from the AeroMap).
#[derive(Clone, Copy, Debug)]
pub struct AeroCoeffs {
    /// Front downforce coefficient CzF.
    pub cz_front: f64,
    /// Rear downforce coefficient CzR.
    pub cz_rear: f64,
    /// Total drag coefficient Cx (positive).
    pub cx: f64,
}

impl BikeParams {
    /// Constant-aero coefficients (§2.3), scaled by `correlation.aero`.
    pub fn const_aero_coeffs(&self) -> AeroCoeffs {
        let k = self.correlation.aero;
        AeroCoeffs {
            cz_front: self.cz_front() * k,
            cz_rear: self.cz_rear() * k,
            cx: self.aero.cx_total * k,
        }
    }
}

/// One evaluation of the thesis Forces Model. Forces are in newtons; `drag_n` and `rolling_n` are
/// positive magnitudes of resisting forces.
#[derive(Clone, Debug)]
pub struct Forces {
    /// Front mass-related vertical load incl. weight transfer (Eq. 2-26).
    pub mass_f_n: f64,
    /// Rear mass-related vertical load incl. weight transfer (Eq. 2-27).
    pub mass_r_n: f64,
    /// Front downforce (Eq. 2-29).
    pub df_f_n: f64,
    /// Rear downforce (Eq. 2-30).
    pub df_r_n: f64,
    /// Total front vertical load (Eq. 2-32).
    pub tot_f_n: f64,
    /// Total rear vertical load (Eq. 2-33).
    pub tot_r_n: f64,
    /// Front longitudinal friction coefficient (Eq. 2-12).
    pub mux_f: f64,
    /// Rear longitudinal friction coefficient (Eq. 2-13).
    pub mux_r: f64,
    /// Front lateral friction coefficient (Eq. 2-14).
    pub muy_f: f64,
    /// Rear lateral friction coefficient (Eq. 2-15).
    pub muy_r: f64,
    /// Aerodynamic drag (Eq. 2-35).
    pub drag_n: f64,
    /// Rolling resistance (Eq. 2-36).
    pub rolling_n: f64,
    /// Front tyre acceleration force (Eqs. 2-37…2-39), correlation applied.
    pub tyres_acc_f_n: f64,
    /// Rear tyre acceleration force, correlation applied.
    pub tyres_acc_r_n: f64,
    /// Front tyre deceleration force (Eq. 2-41), correlation applied.
    pub tyres_dec_f_n: f64,
    /// Rear tyre deceleration force (Eq. 2-42), correlation applied.
    pub tyres_dec_r_n: f64,
    /// Front lateral force (Eq. 2-46), correlation applied.
    pub fy_f_n: f64,
    /// Rear lateral force (Eq. 2-47), correlation applied.
    pub fy_r_n: f64,
}

impl Forces {
    /// Total vertical load (Eq. 2-34).
    pub fn tot_n(&self) -> f64 {
        self.tot_f_n + self.tot_r_n
    }
    /// Total tyre acceleration force (Eq. 2-40).
    pub fn tyres_acc_n(&self) -> f64 {
        self.tyres_acc_f_n + self.tyres_acc_r_n
    }
    /// Total tyre deceleration force (Eq. 2-43).
    pub fn tyres_dec_n(&self) -> f64 {
        self.tyres_dec_f_n + self.tyres_dec_r_n
    }
    /// Total lateral force (Eq. 2-48).
    pub fn fy_total_n(&self) -> f64 {
        self.fy_f_n + self.fy_r_n
    }
    /// Drag plus rolling resistance (Eq. 2-45, magnitude).
    pub fn resist_n(&self) -> f64 {
        self.drag_n + self.rolling_n
    }
}

/// Vehicle attitude and aero state at one instant (driven channels, and the AeroMap inputs).
#[derive(Clone, Debug, Default)]
pub struct Attitude {
    /// Front ride height, mm (0 when the model does not compute it).
    pub front_rh_mm: f64,
    /// Rear ride height, mm.
    pub rear_rh_mm: f64,
    /// Roll angle, degrees (positive = left side up).
    pub roll_deg: f64,
    /// Pitch angle, degrees (positive = nose down).
    pub pitch_deg: f64,
    /// Chassis side-slip angle β, degrees (steering matrix, Eq. 2-53).
    pub beta_deg: f64,
    /// Per-wheel normal load FL, FR, RL, RR, N (empty-model value: axle load / 2).
    pub wheel_fz_n: [f64; 4],
    /// Camber per wheel, degrees (outward positive); zero when not modelled.
    pub camber_deg: [f64; 4],
    /// Toe per wheel, degrees; zero when not modelled.
    pub toe_deg: [f64; 4],
    /// Effective total downforce coefficient used.
    pub cz_total: f64,
    /// Effective front aero balance used, percent.
    pub aero_balance_front_pct: f64,
    /// Effective drag coefficient used.
    pub cx_total: f64,
    /// Iterations the model needed (0 for closed-form models).
    pub iterations: usize,
    /// True when the AeroMap input was outside its swept range and held at the edge (§6.2.3.2).
    pub converged_aero_clamped: bool,
}

/// Result of [`StepModel::instant`].
#[derive(Clone, Debug)]
pub struct Instant {
    /// Forces at this instant.
    pub forces: Forces,
    /// Attitude/aero state at this instant.
    pub attitude: Attitude,
}

/// A per-step vehicle model: given speed, longitudinal and lateral acceleration, return the
/// thesis forces and the attitude. Implemented by [`ThesisConst`], the AeroMap model and the
/// bike ↔ 7×7 coupled model.
pub trait StepModel {
    /// The bike parameters this model is built on.
    fn params(&self) -> &BikeParams;
    /// Evaluate at speed `v` (m/s), longitudinal acceleration `ax` (m/s², positive accelerating)
    /// and signed lateral acceleration `ay` (m/s², positive = left turn).
    ///
    /// # Errors
    /// Out-of-domain state (negative load, nonconvergence, …).
    fn instant(&self, v: f64, ax: f64, ay: f64) -> Result<Instant, Error>;
}

/// Evaluate the thesis Forces Model (§2.8) for a given speed, longitudinal acceleration and aero.
///
/// # Errors
/// Returns an error for nonfinite input or when either axle's vertical load goes negative (lift-off).
pub fn forces(p: &BikeParams, v: f64, ax: f64, aero: AeroCoeffs) -> Result<Forces, Error> {
    if !v.is_finite() || !ax.is_finite() || v < 0.0 {
        return Err(err("nonfinite or negative speed/acceleration"));
    }
    let m = p.mass_kg;
    let wf_n = m * ax * p.cg_height_m / p.wheelbase_m; // Eq. 2-49 (N)
    let mass_f = m * G * p.wd_front_pct / 100.0 - wf_n;
    let mass_r = m * G * (1.0 - p.wd_front_pct / 100.0) + wf_n;
    let q = 0.5 * p.aero.rho_kg_m3 * p.aero.area_m2 * v * v;
    let df_f = q * aero.cz_front;
    let df_r = q * aero.cz_rear;
    let tot_f = mass_f + df_f;
    let tot_r = mass_r + df_r;
    if tot_f < 0.0 || tot_r < 0.0 {
        return Err(err(format!(
            "axle lift-off at v={v:.2} m/s, ax={ax:.2} m/s² (front {tot_f:.0} N, rear {tot_r:.0} N)"
        )));
    }
    forces_at_loads(p, mass_f, mass_r, df_f, df_r, q * aero.cx)
}

/// The thesis Forces Model for explicit mass-related axle loads and aero loads (used by the
/// coupled model, which takes loads from the 7×7). Axle total loads are `mass + downforce`.
///
/// # Errors
/// Returns an error when either axle's total vertical load is negative (lift-off).
pub fn forces_at_loads(
    p: &BikeParams,
    mass_f_n: f64,
    mass_r_n: f64,
    df_f_n: f64,
    df_r_n: f64,
    drag_n: f64,
) -> Result<Forces, Error> {
    let (tot_f, tot_r) = (mass_f_n + df_f_n, mass_r_n + df_r_n);
    if tot_f < 0.0 || tot_r < 0.0 {
        return Err(err(format!(
            "axle lift-off (front {tot_f:.0} N, rear {tot_r:.0} N)"
        )));
    }
    let mu = |t_mu: f64, norm_kg: f64, sens: f64, tot: f64| t_mu + sens * (norm_kg * G - tot / 2.0);
    let mux_f = mu(
        p.front.mux,
        p.front.mux_norm_kg,
        p.front.mux_sens_per_n,
        tot_f,
    );
    let mux_r = mu(p.rear.mux, p.rear.mux_norm_kg, p.rear.mux_sens_per_n, tot_r);
    let muy_f = mu(
        p.front.muy,
        p.front.muy_norm_kg,
        p.front.muy_sens_per_n,
        tot_f,
    );
    let muy_r = mu(p.rear.muy, p.rear.muy_norm_kg, p.rear.muy_sens_per_n, tot_r);
    let c = &p.correlation;
    let (acc_f, acc_r) = match p.drive {
        DrivenAxle::Rear => (0.0, mux_r * tot_r),
        DrivenAxle::Front => (mux_f * tot_f, 0.0),
        DrivenAxle::All => (mux_f * tot_f, mux_r * tot_r),
    };
    Ok(Forces {
        mass_f_n,
        mass_r_n,
        df_f_n,
        df_r_n,
        tot_f_n: tot_f,
        tot_r_n: tot_r,
        mux_f,
        mux_r,
        muy_f,
        muy_r,
        drag_n,
        rolling_n: p.rolling_resistance_cr * (tot_f + tot_r),
        tyres_acc_f_n: acc_f * c.mux_accel,
        tyres_acc_r_n: acc_r * c.mux_accel,
        tyres_dec_f_n: mux_f * tot_f * c.mux_brake,
        tyres_dec_r_n: mux_r * tot_r * c.mux_brake,
        fy_f_n: muy_f * tot_f * c.muy,
        fy_r_n: muy_r * tot_r * c.muy,
    })
}

/// The thesis model with constant aero coefficients (§2.3, §2.8).
#[derive(Clone, Debug)]
pub struct ThesisConst {
    /// Vehicle.
    pub params: BikeParams,
    /// Apply longitudinal weight transfer (thesis §3.2.2, §4.3.6); when false `ax` is treated as 0.
    pub weight_transfer: bool,
}

impl StepModel for ThesisConst {
    fn params(&self) -> &BikeParams {
        &self.params
    }
    fn instant(&self, v: f64, ax: f64, _ay: f64) -> Result<Instant, Error> {
        let a = self.params.const_aero_coeffs();
        let ax_wt = if self.weight_transfer { ax } else { 0.0 };
        let f = forces(&self.params, v, ax_wt, a)?;
        let attitude = Attitude {
            wheel_fz_n: [
                f.tot_f_n / 2.0,
                f.tot_f_n / 2.0,
                f.tot_r_n / 2.0,
                f.tot_r_n / 2.0,
            ],
            cz_total: a.cz_front + a.cz_rear,
            aero_balance_front_pct: 100.0 * a.cz_front / (a.cz_front + a.cz_rear).max(1e-12),
            cx_total: a.cx,
            ..Default::default()
        };
        Ok(Instant {
            forces: f,
            attitude,
        })
    }
}

/// Steady-state lateral limit `Fy_total / m` at speed `v` and longitudinal acceleration `ax`,
/// iterated to a fixed point in `ay` (models whose forces depend on `ay` converge here).
///
/// # Errors
/// Propagates [`StepModel::instant`] errors, or reports non-convergence after 200 iterations.
pub fn max_ay(model: &dyn StepModel, v: f64, ax: f64) -> Result<f64, Error> {
    let m = model.params().mass_kg;
    let mut ay = 0.0;
    for _ in 0..200 {
        let next = model.instant(v, ax, ay)?.forces.fy_total_n() / m;
        if (next - ay).abs() < 1e-10 * next.abs().max(1.0) {
            return Ok(next);
        }
        ay = 0.5 * (ay + next);
    }
    Err(err("lateral limit did not converge"))
}
