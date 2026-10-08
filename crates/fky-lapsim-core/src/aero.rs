//! Aerodynamic body-frame loads (x forward, y left, z up) for the
//! lap-simulation full-car vehicle (`crate::lap`).
//!
//! This is a constant-coefficient planar (CL/CD/COP) model: drag and
//! downforce scale with the square of air-relative body-plane speed through
//! fixed coefficients, with no yaw sensitivity, ground-effect variation with
//! ride height, or stall. See
//! <https://github.com/Ixiandesign/FKY-LAPSIM/blob/main/docs/model-conventions.md>
//! for the crate's coordinate conventions.
use serde::{Deserialize, Serialize};
/// Constant-coefficient planar aerodynamics. No yaw sensitivity or ground effect.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Aero {
    /// Air density, kg/m³. Must be finite and strictly positive.
    pub air_density_kg_m3: f64,
    /// Reference area, m², used with the drag/downforce coefficients to form
    /// dynamic pressure loads. Must be finite and nonnegative; zero disables
    /// aero entirely.
    pub reference_area_m2: f64,
    /// Drag coefficient. Must be finite and nonnegative.
    pub drag_coefficient: f64,
    /// Downforce coefficient. Must be finite and nonnegative; a positive
    /// value produces force toward negative body z (downforce, not lift).
    pub downforce_coefficient: f64,
    /// Body-frame center of pressure relative to the vehicle CG, metres. Must
    /// be finite; otherwise unconstrained.
    pub center_of_pressure_m: [f64; 3],
}
/// Aerodynamic force at the center of pressure, plus that force's equivalent
/// moment about the CG, both in the body frame. Returned by [`Aero::evaluate`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AeroState {
    /// Dynamic pressure from the planar (x-y) air-relative speed, Pa. Always
    /// nonnegative.
    pub dynamic_pressure_pa: f64,
    /// Body-frame aerodynamic force, N. The z component is the downforce
    /// term; drag acts opposite the planar velocity direction.
    pub force_n: [f64; 3],
    /// Body-frame moment about the CG equivalent to `force_n` acting at
    /// [`Aero::center_of_pressure_m`], N·m.
    pub moment_nm: [f64; 3],
}
impl Aero {
    /// Check that all coefficients and the center-of-pressure geometry are
    /// finite and within their physical domains. Called first by
    /// [`Aero::evaluate`].
    ///
    /// # Errors
    ///
    /// Returns an error if `air_density_kg_m3` is nonfinite or not strictly
    /// positive; if `reference_area_m2`, `drag_coefficient` or
    /// `downforce_coefficient` is nonfinite or negative; or if any component
    /// of `center_of_pressure_m` is nonfinite.
    pub fn validate(&self) -> Result<(), String> {
        if !self.air_density_kg_m3.is_finite()
            || self.air_density_kg_m3 <= 0.
            || [
                self.reference_area_m2,
                self.drag_coefficient,
                self.downforce_coefficient,
            ]
            .iter()
            .any(|x| !x.is_finite() || *x < 0.)
            || self.center_of_pressure_m.iter().any(|x| !x.is_finite())
        {
            return Err("invalid aero coefficients or center of pressure".into());
        }
        Ok(())
    }
    /// Evaluate aerodynamic force and moment from an air-relative body
    /// velocity. Dynamic pressure and drag use only the x-y (planar) speed
    /// component; vertical velocity is ignored by this explicitly planar
    /// model. Drag opposes the planar velocity direction; downforce acts
    /// along negative body z regardless of direction of travel. A zero
    /// planar velocity returns all-zero force without dividing by zero.
    ///
    /// # Errors
    ///
    /// Returns an error if [`Aero::validate`] fails, if any component of
    /// `body_velocity_m_s` is nonfinite, or if the resulting dynamic
    /// pressure, force or moment overflows to a nonfinite value.
    pub fn evaluate(&self, body_velocity_m_s: [f64; 3]) -> Result<AeroState, String> {
        self.validate()?;
        if body_velocity_m_s.iter().any(|x| !x.is_finite()) {
            return Err("nonfinite air-relative velocity".into());
        }
        let v = body_velocity_m_s[0].hypot(body_velocity_m_s[1]);
        let dynamic_pressure_pa = 0.5 * self.air_density_kg_m3 * v * v;
        let qa = dynamic_pressure_pa * self.reference_area_m2;
        let drag = qa * self.drag_coefficient;
        let force_n = if v == 0. {
            [0.; 3]
        } else {
            [
                -drag * body_velocity_m_s[0] / v,
                -drag * body_velocity_m_s[1] / v,
                -qa * self.downforce_coefficient,
            ]
        };
        let [x, y, z] = self.center_of_pressure_m;
        let [fx, fy, fz] = force_n;
        let moment_nm = [y * fz - z * fy, z * fx - x * fz, x * fy - y * fx];
        if !dynamic_pressure_pa.is_finite()
            || force_n
                .iter()
                .chain(moment_nm.iter())
                .any(|x| !x.is_finite())
        {
            return Err("aero load overflow".into());
        }
        Ok(AeroState {
            dynamic_pressure_pa,
            force_n,
            moment_nm,
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dynamic_pressure_and_cop_cross_product() {
        let a = Aero {
            air_density_kg_m3: 1.2,
            reference_area_m2: 2.,
            drag_coefficient: 0.5,
            downforce_coefficient: 2.,
            center_of_pressure_m: [1., 0., 0.5],
        };
        let state = a.evaluate([10., 0., 0.]).unwrap();
        assert!((state.force_n[0] + 60.).abs() < 1e-12);
        assert!((state.force_n[2] + 240.).abs() < 1e-12);
        assert!((state.moment_nm[1] - 210.).abs() < 1e-12);
        assert_eq!(a.evaluate([0.; 3]).unwrap().force_n, [0.; 3]);
        let reverse = a.evaluate([-10., 0., 0.]).unwrap();
        assert!((reverse.force_n[0] - 60.).abs() < 1e-12);
        let lateral = a.evaluate([0., 10., 0.]).unwrap();
        assert!((lateral.force_n[1] + 60.).abs() < 1e-12);
        assert!(a.evaluate([f64::NAN, 0., 0.]).is_err());
        let mut invalid = a.clone();
        invalid.downforce_coefficient = -1.;
        assert!(invalid.validate().is_err());
        invalid = a.clone();
        invalid.air_density_kg_m3 = 0.;
        assert!(invalid.validate().is_err());
    }
}
