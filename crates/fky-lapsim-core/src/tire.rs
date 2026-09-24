//! Pacejka, Tire and Vehicle Dynamics, third edition, §4.3.2.
//!
//! Implements 4.E9–4.E67 and 4.E71–4.E78 at nominal inflation pressure,
//! unity scaling factors and no turn slip (zeta=1). Inputs are forward-rolling
//! tire-frame slip coordinates; forces follow the positive-slip convention.
//! These coefficients are NOT interchangeable with a0–a17 or the UKY fit.
use crate::Error;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Exact implemented equation family, serialized with every tire.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TireFormulation {
    /// Third-edition §4.3.2, nominal pressure, unity scaling, no turn slip.
    #[default]
    Pacejka2012NominalPressure,
}
/// Named Magic Formula coefficients. Omitted nonessential coefficients are zero;
/// required shape, peak, stiffness, combined-slip and trail parameters must exist.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TireModel {
    /// Versioned equation family.
    #[serde(default)]
    pub formulation: TireFormulation,
    /// Reference vertical load Fz0, newtons.
    pub reference_load_n: f64,
    /// Unloaded radius R0, metres, for moment scaling.
    pub radius_m: f64,
    /// Uppercase book coefficient names, e.g. PCY1 and PKY4.
    pub coefficients: BTreeMap<String, f64>,
}
/// Tire-frame input. alpha* = tan(alpha), gamma* = sin(gamma).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TireInput {
    /// Positive compression load, N. Zero load produces zero forces.
    pub normal_load_n: f64,
    /// Positive slip angle produces positive lateral force for positive PKY1.
    pub slip_angle_rad: f64,
    /// (Rolling speed - forward contact speed) / forward contact speed.
    pub slip_ratio: f64,
    /// Signed tire-frame inclination angle, radians (book convention).
    pub camber_rad: f64,
    /// Positive forward tire contact velocity, m/s. Standstill/reverse unsupported.
    pub speed_m_s: f64,
}
/// Computed forces and diagnostics; stiffnesses are with respect to dimensionless slip.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TireForces {
    /// Combined longitudinal force, N.
    pub fx_n: f64,
    /// Combined lateral force, N.
    pub fy_n: f64,
    /// Combined self-aligning torque, N m.
    pub mz_nm: f64,
    /// Pure longitudinal force before combined-slip weighting, N.
    pub pure_fx_n: f64,
    /// Pure lateral force before combined-slip weighting, N.
    pub pure_fy_n: f64,
    /// Longitudinal weighting Gxa.
    pub gx: f64,
    /// Lateral weighting Gyk.
    pub gy: f64,
    /// Kxk, N per unit longitudinal slip.
    pub longitudinal_stiffness_n: f64,
    /// Kya, N per unit tan(alpha).
    pub cornering_stiffness_n: f64,
    /// Lateral friction peak coefficient before shifts.
    pub lateral_friction: f64,
    /// Combined-slip pneumatic trail, metres.
    pub pneumatic_trail_m: f64,
}

const NAMES: &str = "PCX1 PDX1 PDX2 PDX3 PEX1 PEX2 PEX3 PEX4 PKX1 PKX2 PKX3 PHX1 PHX2 PVX1 PVX2 PCY1 PDY1 PDY2 PDY3 PEY1 PEY2 PEY3 PEY4 PEY5 PKY1 PKY2 PKY3 PKY4 PKY5 PKY6 PKY7 PHY1 PHY2 PVY1 PVY2 PVY3 PVY4 RBX1 RBX2 RBX3 RCX1 REX1 REX2 RHX1 RBY1 RBY2 RBY3 RBY4 RCY1 REY1 REY2 RHY1 RHY2 RVY1 RVY2 RVY3 RVY4 RVY5 RVY6 QBZ1 QBZ2 QBZ3 QBZ5 QBZ6 QBZ9 QBZ10 QCZ1 QDZ1 QDZ2 QDZ3 QDZ4 QDZ6 QDZ7 QDZ8 QDZ9 QDZ10 QDZ11 QEZ1 QEZ2 QEZ3 QEZ4 QEZ5 QHZ1 QHZ2 QHZ3 QHZ4 SSZ1 SSZ2 SSZ3 SSZ4";
fn error(s: &str) -> Error {
    Error {
        message: format!("tire: {s}"),
    }
}
fn sign(x: f64) -> f64 {
    if x == 0. {
        0.
    } else {
        x.signum()
    }
}
fn phase(b: f64, c: f64, e: f64, x: f64) -> f64 {
    let u = b * x;
    c * (u - e * (u - u.atan())).atan()
}

impl TireModel {
    /// Supported named coefficients, in book-family order.
    pub fn coefficient_names() -> Vec<&'static str> {
        NAMES.split_whitespace().collect()
    }
    /// Validate model version, dimensions and required coefficients. No fitted
    /// parameters are invented; unknown names are errors, including legacy a0.
    pub fn validate(&self) -> Result<(), Error> {
        if !self.reference_load_n.is_finite()
            || self.reference_load_n <= 0.
            || !self.radius_m.is_finite()
            || self.radius_m <= 0.
        {
            return Err(error(
                "reference load and radius must be finite and positive",
            ));
        }
        for (name, value) in &self.coefficients {
            if !value.is_finite() || !NAMES.split_whitespace().any(|s| s == name) {
                return Err(error(&format!("unknown or nonfinite coefficient {name}")));
            }
        }
        for name in [
            "PCX1", "PDX1", "PKX1", "PCY1", "PDY1", "PKY1", "PKY2", "PKY4", "RBX1", "RCX1", "RBY1",
            "RCY1", "QBZ1", "QCZ1",
        ] {
            if !self.coefficients.get(name).is_some_and(|x| *x > 0.) {
                return Err(error(&format!("{name} is required and must be positive")));
            }
        }
        if !self.coefficients.contains_key("QDZ1") || self.p("QDZ1") < 0. {
            return Err(error("QDZ1 is required and must be nonnegative"));
        }
        Ok(())
    }
    fn p(&self, key: &str) -> f64 {
        self.coefficients.get(key).copied().unwrap_or(0.)
    }

    fn lateral(&self, fz: f64, df: f64, alpha: f64, gamma: f64) -> Result<[f64; 6], Error> {
        let p = |s| self.p(s);
        let mu = (p("PDY1") + p("PDY2") * df) * (1. - p("PDY3") * gamma * gamma);
        let denominator = p("PKY2") + p("PKY5") * gamma * gamma;
        let k = p("PKY1")
            * self.reference_load_n
            * (1. - p("PKY3") * gamma.abs())
            * (p("PKY4") * (fz / self.reference_load_n / denominator).atan()).sin();
        let cy = p("PCY1");
        if mu <= 0. || denominator <= 0. || k <= 0. {
            return Err(error(
                "nonpositive lateral peak/stiffness in this load/camber state",
            ));
        }
        let by = k / (cy * mu * fz);
        let sv_gamma = fz * (p("PVY3") + p("PVY4") * df) * gamma;
        let ky_gamma = fz * (p("PKY6") + p("PKY7") * df);
        let sh = p("PHY1") + p("PHY2") * df + (ky_gamma * gamma - sv_gamma) / k;
        let sv = fz * (p("PVY1") + p("PVY2") * df) + sv_gamma;
        let ey = (p("PEY1") + p("PEY2") * df)
            * (1. + p("PEY5") * gamma * gamma - (p("PEY3") + p("PEY4") * gamma) * sign(alpha + sh));
        if ey > 1. {
            return Err(error("lateral curvature factor exceeds one"));
        }
        let fy = mu * fz * phase(by, cy, ey, alpha + sh).sin() + sv;
        Ok([fy, k, mu, by, sh, sv])
    }

    /// Evaluate pure/combined forces and aligning torque using named source equations.
    /// Numerical/physical domain failures return errors; no hidden friction ellipse
    /// or saturation replaces the specified combined-slip weighting functions.
    pub fn evaluate(&self, input: TireInput) -> Result<TireForces, Error> {
        self.validate()?;
        let TireInput {
            normal_load_n: fz,
            slip_angle_rad: alpha,
            slip_ratio: kappa,
            camber_rad: gamma,
            speed_m_s: vx,
        } = input;
        if ![fz, alpha, kappa, gamma, vx].iter().all(|v| v.is_finite())
            || fz < 0.
            || vx <= 0.
            || alpha.abs() >= std::f64::consts::FRAC_PI_2
            || gamma.abs() >= std::f64::consts::FRAC_PI_2
        {
            return Err(error("requires finite slips, nonnegative load, positive forward speed and angles within +/- pi/2"));
        }
        if fz == 0. {
            return Ok(TireForces::default());
        }
        let p = |s| self.p(s);
        let df = (fz - self.reference_load_n) / self.reference_load_n;
        let a = alpha.tan();
        let g = gamma.sin();
        // 4.E9–4.E18, pressure and scaling factors fixed to their nominal values.
        let cx = p("PCX1");
        let dx = fz * (p("PDX1") + p("PDX2") * df) * (1. - p("PDX3") * g * g);
        let kx = fz * (p("PKX1") + p("PKX2") * df) * (p("PKX3") * df).exp();
        let shift_x = p("PHX1") + p("PHX2") * df;
        let ex = (p("PEX1") + p("PEX2") * df + p("PEX3") * df * df)
            * (1. - p("PEX4") * sign(kappa + shift_x));
        if dx <= 0. || kx <= 0. || ex > 1. {
            return Err(error(
                "invalid longitudinal peak, stiffness or curvature in this state",
            ));
        }
        let fx0 = dx * phase(kx / (cx * dx), cx, ex, kappa + shift_x).sin()
            + fz * (p("PVX1") + p("PVX2") * df);
        let [fy0, ky, mu, by, shy, svy] = self.lateral(fz, df, a, g)?;
        let gain = |b: f64, c: f64, e: f64, shift: f64, x: f64| -> Result<f64, Error> {
            let base = phase(b, c, e, shift).cos();
            if b <= 0. || c <= 0. || e > 1. || base <= 1e-10 {
                return Err(error("invalid combined-slip weighting denominator/shape"));
            }
            let v = phase(b, c, e, x + shift).cos() / base;
            if !v.is_finite() || v < 0. {
                return Err(error("combined-slip weighting outside its positive domain"));
            }
            Ok(v)
        };
        let gx = gain(
            (p("RBX1") + p("RBX3") * g * g) * (p("RBX2") * kappa).atan().cos(),
            p("RCX1"),
            p("REX1") + p("REX2") * df,
            p("RHX1"),
            a,
        )?;
        let gy_at = |gamma: f64| {
            gain(
                (p("RBY1") + p("RBY4") * gamma * gamma)
                    * (p("RBY2") * (a - p("RBY3"))).atan().cos(),
                p("RCY1"),
                p("REY1") + p("REY2") * df,
                p("RHY1") + p("RHY2") * df,
                kappa,
            )
        };
        let gy = gy_at(g)?;
        let svyk = mu
            * fz
            * (p("RVY1") + p("RVY2") * df + p("RVY3") * g)
            * (p("RVY4") * a).atan().cos()
            * (p("RVY5") * (p("RVY6") * kappa).atan()).sin();
        let fx = gx * fx0;
        let fy = gy * fy0 + svyk;
        // 4.E31–4.E49 and 4.E71–4.E78. Forward-only cos'(alpha) uses 4.E6a.
        let cos_a = vx / (vx / alpha.cos() + 0.1);
        let bt = (p("QBZ1") + p("QBZ2") * df + p("QBZ3") * df * df)
            * (1. + p("QBZ5") * g.abs() + p("QBZ6") * g * g);
        let ct = p("QCZ1");
        let dt = fz * self.radius_m / self.reference_load_n
            * (p("QDZ1") + p("QDZ2") * df)
            * (1. + p("QDZ3") * g.abs() + p("QDZ4") * g * g);
        let at = a + p("QHZ1") + p("QHZ2") * df + (p("QHZ3") + p("QHZ4") * df) * g;
        let et = (p("QEZ1") + p("QEZ2") * df + p("QEZ3") * df * df)
            * (1.
                + (p("QEZ4") + p("QEZ5") * g) * 2. / std::f64::consts::PI * (bt * ct * at).atan());
        if bt <= 0. || dt < 0. || et > 1. {
            return Err(error("invalid pneumatic trail parameters in this state"));
        }
        let ar = a + shy + svy / ky;
        let equivalent = |slip: f64| (slip * slip + (kx / ky * kappa).powi(2)).sqrt() * sign(slip);
        let trail = dt * phase(bt, ct, et, equivalent(at)).cos() * cos_a;
        let br = p("QBZ9") + p("QBZ10") * by * p("PCY1");
        let dr = fz
            * self.radius_m
            * (p("QDZ6")
                + p("QDZ7") * df
                + ((p("QDZ8") + p("QDZ9") * df) + (p("QDZ10") + p("QDZ11") * df) * g.abs()) * g)
            * cos_a;
        let residual = dr * (br * equivalent(ar)).atan().cos() * cos_a;
        let fy_no_camber = self.lateral(fz, df, a, 0.)?[0] * gy_at(0.)?;
        let arm = self.radius_m
            * (p("SSZ1")
                + p("SSZ2") * fy / self.reference_load_n
                + (p("SSZ3") + p("SSZ4") * df) * g);
        let mz = -trail * fy_no_camber + residual + arm * fx;
        if ![fx, fy, mz, fx0, fy0, kx, ky, mu, trail]
            .iter()
            .all(|v| v.is_finite())
        {
            return Err(error("nonfinite force or moment"));
        }
        Ok(TireForces {
            fx_n: fx,
            fy_n: fy,
            mz_nm: mz,
            pure_fx_n: fx0,
            pure_fy_n: fy0,
            gx,
            gy,
            longitudinal_stiffness_n: kx,
            cornering_stiffness_n: ky,
            lateral_friction: mu,
            pneumatic_trail_m: trail,
        })
    }
}
