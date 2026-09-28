//! Aerodynamic body-frame loads (x forward, y left, z up).
use serde::{Deserialize, Serialize};
/// Constant-coefficient planar aerodynamics. No yaw sensitivity or ground effect.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Aero {
    /// Air density, kg/m³ (strictly positive).
    pub air_density_kg_m3: f64,
    /// Reference area, m² (nonnegative permits disabling aero).
    pub reference_area_m2: f64,
    /// Nonnegative drag coefficient.
    pub drag_coefficient: f64,
    /// Nonnegative coefficient; positive means force toward negative body z.
    pub downforce_coefficient: f64,
    /// Body-frame center of pressure relative to CG, meters.
    pub center_of_pressure_m: [f64; 3],
}
/// Force at the center of pressure and its equivalent CG moment.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AeroState {
    /// Planar dynamic pressure, Pa.
    pub dynamic_pressure_pa: f64,
    /// Body force, N.
    pub force_n: [f64; 3],
    /// Body moment about CG, N m.
    pub moment_nm: [f64; 3],
}
impl Aero {
    /// Check coefficient domains and finite geometry.
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
    /// Evaluate air-relative body velocity. Drag opposes its XY component;
    /// vertical velocity is ignored by this explicitly planar model.
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
