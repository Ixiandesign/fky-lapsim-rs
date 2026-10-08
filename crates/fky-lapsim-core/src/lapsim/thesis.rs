//! Thesis-level vehicle parameters (Zacharelis 2023, Tables 2-1…2-3, 2-8) and the P19 baseline.
use super::err;
use crate::powertrain::DrivenAxle;
use crate::Error;
use serde::{Deserialize, Serialize};

/// Friction parameters of one axle's tyres (Table 2-3; Eqs. 2-12…2-15).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AxleTyre {
    /// Peak longitudinal friction coefficient at the rated load, dimensionless.
    pub mux: f64,
    /// Peak lateral friction coefficient at the rated load, dimensionless.
    pub muy: f64,
    /// Rated load for `mux`, kg (the thesis `MuX_Norm`; multiplied by g).
    pub mux_norm_kg: f64,
    /// Rated load for `muy`, kg.
    pub muy_norm_kg: f64,
    /// Longitudinal friction gain per newton of half-axle load below rating, 1/N.
    pub mux_sens_per_n: f64,
    /// Lateral friction gain per newton of half-axle load below rating, 1/N.
    pub muy_sens_per_n: f64,
    /// Cornering stiffness of ONE tyre, N/deg (thesis `CF`/`CR`).
    pub cornering_stiffness_n_per_deg: f64,
}

/// Constant-coefficient aerodynamics (thesis Table 2-2, §2.3 simplified model).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConstAero {
    /// Air density, kg/m³.
    pub rho_kg_m3: f64,
    /// Total downforce coefficient CzT, dimensionless (positive = downforce).
    pub cz_total: f64,
    /// Front share of downforce, percent (aero balance).
    pub aero_balance_front_pct: f64,
    /// Total drag coefficient magnitude Cx, dimensionless (positive).
    pub cx_total: f64,
    /// Reference (frontal) area, m².
    pub area_m2: f64,
}

/// Correlation multipliers (thesis Table 4-8). All default to 1 and are recorded on every run.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Correlation {
    /// Multiplies the engine tractive force.
    pub engine_power: f64,
    /// Multiplies both CzT and Cx (constant-aero model) or the AeroMap outputs.
    pub aero: f64,
    /// Multiplies longitudinal tyre force while accelerating.
    pub mux_accel: f64,
    /// Multiplies longitudinal tyre force while braking.
    pub mux_brake: f64,
    /// Multiplies lateral tyre force.
    pub muy: f64,
}

impl Default for Correlation {
    fn default() -> Self {
        Self {
            engine_power: 1.0,
            aero: 1.0,
            mux_accel: 1.0,
            mux_brake: 1.0,
            muy: 1.0,
        }
    }
}

/// Brake hardware for the brake-pressure driven channel (thesis Table 2-8; index 0 front, 1 rear).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BrakeSystem {
    /// Front brake bias, percent of braking force.
    pub front_bias_pct: f64,
    /// Disc diameters, m.
    pub disc_diameter_m: [f64; 2],
    /// Pad heights, m.
    pub pad_height_m: [f64; 2],
    /// Pad friction coefficients.
    pub pad_mu: [f64; 2],
    /// Caliper piston counts.
    pub pistons: [u32; 2],
    /// Caliper piston diameters, m.
    pub piston_diameter_m: [f64; 2],
}

/// The thesis hybrid mass-point / bicycle vehicle (Tables 2-1…2-3, 2-8). Every solver in
/// [`crate::lapsim`] runs on this; it is built either from the thesis tables ([`p19`]) or from a
/// full vehicle configuration (`lapsim::vehicle`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BikeParams {
    /// Total mass including driver, kg.
    pub mass_kg: f64,
    /// Front static weight distribution, percent.
    pub wd_front_pct: f64,
    /// Wheelbase, m.
    pub wheelbase_m: f64,
    /// Front track width, m.
    pub front_track_m: f64,
    /// Rear track width, m.
    pub rear_track_m: f64,
    /// Centre-of-gravity height, m.
    pub cg_height_m: f64,
    /// Steering-wheel to front-wheel ratio.
    pub steer_ratio: f64,
    /// Tyre rolling radius, m.
    pub rolling_radius_m: f64,
    /// Rolling-resistance coefficient magnitude (positive; applied as a resisting force).
    pub rolling_resistance_cr: f64,
    /// Which axle(s) receive drive torque.
    pub drive: DrivenAxle,
    /// Front tyres.
    pub front: AxleTyre,
    /// Rear tyres.
    pub rear: AxleTyre,
    /// Aerodynamics.
    pub aero: ConstAero,
    /// Brake hardware.
    pub brake: BrakeSystem,
    /// Correlation multipliers.
    #[serde(default)]
    pub correlation: Correlation,
}

impl BikeParams {
    /// Front axle to CG distance, m (Eq. 2-1: `(1 - WD/100) * WB`).
    pub fn a_dist_m(&self) -> f64 {
        (1.0 - self.wd_front_pct / 100.0) * self.wheelbase_m
    }
    /// Rear axle to CG distance, m (Eq. 2-2: `WB - a`).
    pub fn b_dist_m(&self) -> f64 {
        self.wheelbase_m - self.a_dist_m()
    }
    /// Front downforce coefficient (Eq. 2-7: `CzT * AB/100`).
    pub fn cz_front(&self) -> f64 {
        self.aero.cz_total * self.aero.aero_balance_front_pct / 100.0
    }
    /// Rear downforce coefficient (Eq. 2-8: `CzT * (1 - AB/100)`).
    pub fn cz_rear(&self) -> f64 {
        self.aero.cz_total * (1.0 - self.aero.aero_balance_front_pct / 100.0)
    }

    /// Reject nonphysical parameters.
    ///
    /// # Errors
    /// Returns an error for any nonfinite or nonpositive mass/geometry/friction/stiffness value, a
    /// weight distribution or aero balance outside 0–100 %, or a negative correlation factor.
    pub fn validate(&self) -> Result<(), Error> {
        let positive = [
            self.mass_kg,
            self.wheelbase_m,
            self.front_track_m,
            self.rear_track_m,
            self.cg_height_m,
            self.steer_ratio,
            self.rolling_radius_m,
            self.aero.rho_kg_m3,
            self.aero.area_m2,
        ];
        let nonneg = [
            self.rolling_resistance_cr,
            self.aero.cz_total,
            self.aero.cx_total,
        ];
        let tyre_ok = |t: &AxleTyre| {
            [
                t.mux,
                t.muy,
                t.mux_norm_kg,
                t.muy_norm_kg,
                t.cornering_stiffness_n_per_deg,
            ]
            .iter()
            .all(|v| v.is_finite() && *v > 0.0)
                && t.mux_sens_per_n.is_finite()
                && t.muy_sens_per_n.is_finite()
        };
        let c = &self.correlation;
        if !positive.iter().all(|v| v.is_finite() && *v > 0.0)
            || !nonneg.iter().all(|v| v.is_finite() && *v >= 0.0)
            || !(0.0..=100.0).contains(&self.wd_front_pct)
            || !(0.0..=100.0).contains(&self.aero.aero_balance_front_pct)
            || !tyre_ok(&self.front)
            || !tyre_ok(&self.rear)
            || ![c.engine_power, c.aero, c.mux_accel, c.mux_brake, c.muy]
                .iter()
                .all(|v| v.is_finite() && *v >= 0.0)
        {
            return Err(err("invalid vehicle parameters"));
        }
        Ok(())
    }
}

/// The thesis baseline vehicle "P19-Delia" (Tables 2-1, 2-2, 2-3, 2-8). Tyre load sensitivity is
/// 1e-4 per N (see the crate plan: the table's "10E-4" reproduces Tables 3-13/3-14 only at 1e-4).
/// Resistance coefficients are stored as positive magnitudes.
pub fn p19() -> BikeParams {
    let tyre = AxleTyre {
        mux: 1.2,
        muy: 1.45,
        mux_norm_kg: 50.0,
        muy_norm_kg: 50.0,
        mux_sens_per_n: 1e-4,
        muy_sens_per_n: 1e-4,
        cornering_stiffness_n_per_deg: 190.0,
    };
    BikeParams {
        mass_kg: 250.0,
        wd_front_pct: 49.0,
        wheelbase_m: 1.53,
        front_track_m: 1.238,
        rear_track_m: 1.15,
        cg_height_m: 0.330,
        steer_ratio: 3.74,
        rolling_radius_m: 0.199,
        rolling_resistance_cr: 0.03,
        drive: DrivenAxle::Rear,
        front: tyre.clone(),
        rear: tyre,
        aero: ConstAero {
            rho_kg_m3: 1.225,
            cz_total: 4.5,
            aero_balance_front_pct: 45.0,
            cx_total: 1.75,
            area_m2: 1.0,
        },
        brake: BrakeSystem {
            front_bias_pct: 60.0,
            disc_diameter_m: [0.185, 0.174],
            pad_height_m: [0.04, 0.04],
            pad_mu: [0.45, 0.45],
            pistons: [4, 2],
            piston_diameter_m: [0.025, 0.025],
        },
        correlation: Correlation::default(),
    }
}
