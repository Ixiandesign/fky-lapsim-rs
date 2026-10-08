//! Thesis suspension rates (§5.2, Eqs. 5-1…5-10): wheel/heave/ARB/roll rates and the roll gradient,
//! from spring, anti-roll-bar and tyre stiffnesses and constant motion ratios.
use super::err;
use super::thesis::BikeParams;
use super::G;
use crate::Error;

/// Convert a spring rate from lbs/inch to N/m (Eq. 2-17).
pub fn lbs_in_to_n_m(x: f64) -> f64 {
    175.126835 * x
}

/// Suspension inputs of the thesis rates calculator (index 0 front, 1 rear).
#[derive(Clone, Debug)]
pub struct ThesisSuspension {
    /// Spring rate at the spring, per corner, N/m.
    pub spring_rate_n_m: [f64; 2],
    /// Spring motion ratio, wheel/spring.
    pub spring_mr: [f64; 2],
    /// Anti-roll-bar rate at the bar, N/m.
    pub arb_rate_n_m: [f64; 2],
    /// ARB motion ratio, wheel/bar.
    pub arb_mr: [f64; 2],
    /// Tyre vertical stiffness per corner, N/m.
    pub tyre_stiffness_n_m: [f64; 2],
    /// Roll-centre height per axle, m.
    pub roll_centre_height_m: [f64; 2],
    /// Sprung-mass CG height, m (`CoG_SM`).
    pub cg_sprung_height_m: f64,
    /// Non-suspended mass per axle, kg.
    pub nsm_mass_kg: [f64; 2],
    /// Non-suspended mass CG height per axle, m.
    pub nsm_cg_height_m: [f64; 2],
    /// Front anti-dive under braking, percent.
    pub anti_dive_front_pct: f64,
    /// Rear anti-lift under braking, percent.
    pub anti_lift_rear_pct: f64,
    /// Front anti-lift under acceleration, percent (0 for rear-wheel drive).
    pub anti_lift_front_pct: f64,
    /// Rear anti-squat under acceleration, percent.
    pub anti_squat_rear_pct: f64,
    /// Static ride height per axle, mm.
    pub static_rh_mm: [f64; 2],
    /// Roll inertia of the sprung mass, kg·m².
    pub roll_inertia_kg_m2: f64,
    /// Pitch inertia of the sprung mass, kg·m².
    pub pitch_inertia_kg_m2: f64,
}

impl ThesisSuspension {
    /// P19-Delia (thesis Tables 2-1, 2-4, 7-1).
    pub fn p19() -> Self {
        Self {
            spring_rate_n_m: [lbs_in_to_n_m(350.0), lbs_in_to_n_m(150.0)],
            spring_mr: [1.08, 1.06],
            arb_rate_n_m: [36450.0, 115710.0],
            arb_mr: [1.07, 1.56],
            tyre_stiffness_n_m: [95.0e3, 95.0e3],
            roll_centre_height_m: [0.0295, 0.0690],
            cg_sprung_height_m: 0.324,
            nsm_mass_kg: [20.0, 22.0],
            nsm_cg_height_m: [0.26, 0.28],
            anti_dive_front_pct: 30.3,
            anti_lift_rear_pct: 13.5,
            anti_lift_front_pct: 0.0,
            anti_squat_rear_pct: 20.0,
            static_rh_mm: [35.0, 45.0],
            roll_inertia_kg_m2: 15.0,
            pitch_inertia_kg_m2: 60.0,
        }
    }
}

/// Results of the rates calculator (Table 5-1 subset).
#[derive(Clone, Debug)]
pub struct Rates {
    /// Wheel rate per corner, N/m (Eq. 5-1).
    pub kw_n_m: [f64; 2],
    /// Heave rate per wheel (spring and tyre in series), N/m (Eq. 5-2).
    pub kheave_n_m: [f64; 2],
    /// ARB rate at the wheel, N/m (Eq. 5-6).
    pub kw_arb_n_m: [f64; 2],
    /// Roll stiffness per axle, N·m/deg (Eq. 5-7).
    pub kroll_axle_nm_per_deg: [f64; 2],
    /// Total roll stiffness, N·m/deg (Eq. 5-8).
    pub kroll_nm_per_deg: f64,
    /// Front share of roll stiffness, percent (Eq. 5-9).
    pub mechanical_balance_front_pct: f64,
    /// Roll-axis height under the CG, m (Eq. 2-18, thesis weighting).
    pub roll_axis_height_at_cg_m: f64,
    /// Roll lever arm CG – roll axis, m (Eq. 2-19).
    pub roll_lever_arm_m: f64,
    /// Roll gradient, deg per g (Eq. 5-10, total mass).
    pub roll_gradient_deg_per_g: f64,
}

/// Compute the thesis rates for a vehicle.
///
/// # Errors
/// Returns an error for nonpositive stiffnesses/motion ratios or nonfinite input.
pub fn rates(p: &BikeParams, s: &ThesisSuspension) -> Result<Rates, Error> {
    let positive = s
        .spring_rate_n_m
        .iter()
        .chain(&s.spring_mr)
        .chain(&s.arb_mr)
        .chain(&s.tyre_stiffness_n_m)
        .chain(&s.nsm_mass_kg)
        .all(|v| v.is_finite() && *v > 0.0)
        && s.arb_rate_n_m.iter().all(|v| v.is_finite() && *v >= 0.0)
        && s.cg_sprung_height_m.is_finite()
        && s.cg_sprung_height_m > 0.0;
    if !positive {
        return Err(err("invalid suspension rates input"));
    }
    let tracks = [p.front_track_m, p.rear_track_m];
    let mut kw = [0.0; 2];
    let mut kheave = [0.0; 2];
    let mut kw_arb = [0.0; 2];
    let mut kroll = [0.0; 2];
    for i in 0..2 {
        kw[i] = s.spring_rate_n_m[i] / s.spring_mr[i].powi(2); // Eq. 5-1
        kheave[i] = kw[i] * s.tyre_stiffness_n_m[i] / (kw[i] + s.tyre_stiffness_n_m[i]); // Eq. 5-2
        kw_arb[i] = s.arb_rate_n_m[i] / s.arb_mr[i].powi(2); // Eq. 5-6
        let k = kw[i] + kw_arb[i];
        kroll[i] =
            std::f64::consts::PI / 180.0 * tracks[i].powi(2) / 2.0 * k * s.tyre_stiffness_n_m[i]
                / (k + s.tyre_stiffness_n_m[i]); // Eq. 5-7
    }
    let total = kroll[0] + kroll[1]; // Eq. 5-8
    let wd = p.wd_front_pct / 100.0;
    let rc_cog = (1.0 - wd) * s.roll_centre_height_m[0] + wd * s.roll_centre_height_m[1]; // Eq. 2-18
    let arm = s.cg_sprung_height_m - rc_cog; // Eq. 2-19
    Ok(Rates {
        kw_n_m: kw,
        kheave_n_m: kheave,
        kw_arb_n_m: kw_arb,
        kroll_axle_nm_per_deg: kroll,
        kroll_nm_per_deg: total,
        mechanical_balance_front_pct: 100.0 * kroll[0] / total, // Eq. 5-9
        roll_axis_height_at_cg_m: rc_cog,
        roll_lever_arm_m: arm,
        roll_gradient_deg_per_g: p.mass_kg * G * arm / total, // Eq. 5-10
    })
}
