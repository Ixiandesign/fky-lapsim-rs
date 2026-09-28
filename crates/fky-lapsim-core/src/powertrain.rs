//! Validated quasi-static drivetrain; SI torque and angular speed, dyno speed in rpm.
use serde::{Deserialize, Serialize};
/// One measured or explicitly synthetic dyno knot.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DynoPoint {
    /// Engine speed, revolutions/minute.
    pub rpm: f64,
    /// Full-throttle shaft torque, N m.
    pub torque_nm: f64,
}
/// Axle receiving equal left/right drive torque.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DrivenAxle {
    /// Front wheels.
    Front,
    /// Rear wheels.
    Rear,
    /// Equal torque at all four wheels.
    All,
}
/// Fixed-ratio transmission with a piecewise-linear positive torque dyno.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Powertrain {
    /// Strictly increasing dyno knots covering idle through redline.
    pub torque_curve: Vec<DynoPoint>,
    /// Minimum allowed engine speed, rpm.
    pub idle_rpm: f64,
    /// Maximum allowed engine speed, rpm.
    pub redline_rpm: f64,
    /// Positive forward ratios; index is the zero-based gear identifier.
    pub gear_ratios: Vec<f64>,
    /// Positive final reduction.
    pub final_drive: f64,
    /// Mechanical efficiency in (0, 1].
    pub efficiency: f64,
    /// Driven wheels.
    pub driven_axle: DrivenAxle,
}
/// Deterministic drivetrain operating point, with locked clutch.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PowertrainState {
    /// Zero-based engaged gear.
    pub gear: usize,
    /// Engine speed, rpm.
    pub engine_rpm: f64,
    /// Shaft torque after throttle, N m.
    pub engine_torque_nm: f64,
    /// Drive torque in FL, FR, RL, RR order, N m.
    pub wheel_torques_nm: [f64; 4],
}
impl Powertrain {
    /// Reject nonphysical parameters and incomplete dyno coverage.
    pub fn validate(&self) -> Result<(), String> {
        if !self.idle_rpm.is_finite()
            || !self.redline_rpm.is_finite()
            || self.idle_rpm <= 0.
            || self.redline_rpm <= self.idle_rpm
            || !self.final_drive.is_finite()
            || self.final_drive <= 0.
            || !self.efficiency.is_finite()
            || self.efficiency <= 0.
            || self.efficiency > 1.
            || self.gear_ratios.is_empty()
            || self.gear_ratios.iter().any(|r| !r.is_finite() || *r <= 0.)
        {
            return Err("invalid drivetrain domain or ratio".into());
        }
        if self.torque_curve.len() < 2
            || self.torque_curve.iter().any(|p| {
                !p.rpm.is_finite() || !p.torque_nm.is_finite() || p.rpm < 0. || p.torque_nm < 0.
            })
            || self.torque_curve.windows(2).any(|p| p[0].rpm >= p[1].rpm)
            || self.torque_curve[0].rpm > self.idle_rpm
            || self.torque_curve.last().unwrap().rpm < self.redline_rpm
        {
            return Err("dyno must be increasing and cover idle through redline".into());
        }
        Ok(())
    }
    /// Interpolate within the explicit running-speed domain, never extrapolate.
    pub fn torque_at_rpm(&self, rpm: f64) -> Result<f64, String> {
        self.validate()?;
        if !rpm.is_finite() || rpm < self.idle_rpm || rpm > self.redline_rpm {
            return Err("engine speed outside idle/redline domain".into());
        }
        let p = self
            .torque_curve
            .windows(2)
            .find(|p| rpm >= p[0].rpm && rpm <= p[1].rpm)
            .ok_or("uncovered dyno speed")?;
        Ok(p[0].torque_nm
            + (p[1].torque_nm - p[0].torque_nm) * (rpm - p[0].rpm) / (p[1].rpm - p[0].rpm))
    }
    /// Evaluate a locked clutch using the driven wheels' average angular speed.
    /// Below idle requires an explicit external launch model. Equal axle torque
    /// represents an ideal open differential; no limited-slip action is assumed.
    pub fn evaluate(
        &self,
        wheel_angular_speed_rad_s: f64,
        gear: usize,
        throttle: f64,
    ) -> Result<PowertrainState, String> {
        self.validate()?;
        if !wheel_angular_speed_rad_s.is_finite()
            || wheel_angular_speed_rad_s < 0.
            || !throttle.is_finite()
            || !(0. ..=1.).contains(&throttle)
        {
            return Err("invalid speed or throttle".into());
        }
        let ratio = self.gear_ratios.get(gear).ok_or("gear out of range")? * self.final_drive;
        let engine_rpm = wheel_angular_speed_rad_s * ratio * 60. / std::f64::consts::TAU;
        let engine_torque_nm = self.torque_at_rpm(engine_rpm)? * throttle;
        let torque = engine_torque_nm * ratio * self.efficiency;
        let wheel_torques_nm = match self.driven_axle {
            DrivenAxle::Front => [torque / 2., torque / 2., 0., 0.],
            DrivenAxle::Rear => [0., 0., torque / 2., torque / 2.],
            DrivenAxle::All => [torque / 4.; 4],
        };
        if !torque.is_finite() {
            return Err("drivetrain load overflow".into());
        }
        Ok(PowertrainState {
            gear,
            engine_rpm,
            engine_torque_nm,
            wheel_torques_nm,
        })
    }
    /// Select maximum full-throttle wheel torque among valid gears; ties use first index.
    pub fn select_gear(&self, wheel_angular_speed_rad_s: f64) -> Result<usize, String> {
        self.validate()?;
        let mut best: Option<(usize, f64)> = None;
        for gear in 0..self.gear_ratios.len() {
            if let Ok(state) = self.evaluate(wheel_angular_speed_rad_s, gear, 1.) {
                let torque = state.wheel_torques_nm.iter().sum();
                if best.is_none_or(|(_, b)| torque > b) {
                    best = Some((gear, torque));
                }
            }
        }
        best.map(|b| b.0)
            .ok_or_else(|| "no gear inside engine speed domain".into())
    }
}
/// Brake allocation; torque magnitudes must oppose each wheel's rotation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BrakeConfig {
    /// Maximum sum of four brake torque magnitudes, N m.
    pub max_total_torque_nm: f64,
    /// Fraction assigned to the front axle in [0,1].
    pub front_fraction: f64,
}
impl BrakeConfig {
    /// Check finite nonnegative capacity and physical bias.
    pub fn validate(&self) -> Result<(), String> {
        if !self.max_total_torque_nm.is_finite()
            || self.max_total_torque_nm < 0.
            || !self.front_fraction.is_finite()
            || !(0. ..=1.).contains(&self.front_fraction)
        {
            return Err("invalid brake capacity or bias".into());
        }
        Ok(())
    }
    /// Return nonnegative torque magnitudes in FL, FR, RL, RR order.
    pub fn wheel_torques(&self, command: f64) -> Result<[f64; 4], String> {
        self.validate()?;
        if !command.is_finite() || !(0. ..=1.).contains(&command) {
            return Err("brake command must be in [0,1]".into());
        }
        let t = self.max_total_torque_nm * command / 2.;
        Ok([
            t * self.front_fraction,
            t * self.front_fraction,
            t * (1. - self.front_fraction),
            t * (1. - self.front_fraction),
        ])
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn strongest_valid_gear_and_brake_split() {
        let p = Powertrain {
            torque_curve: vec![
                DynoPoint {
                    rpm: 1000.,
                    torque_nm: 100.,
                },
                DynoPoint {
                    rpm: 5000.,
                    torque_nm: 100.,
                },
            ],
            idle_rpm: 1000.,
            redline_rpm: 5000.,
            gear_ratios: vec![4., 2.],
            final_drive: 1.,
            efficiency: 1.,
            driven_axle: DrivenAxle::All,
        };
        assert_eq!(p.select_gear(100.).unwrap(), 0);
        assert_eq!(p.select_gear(200.).unwrap(), 1);
        assert!(p.select_gear(0.).is_err());
        assert_eq!(
            BrakeConfig {
                max_total_torque_nm: 1000.,
                front_fraction: 0.6
            }
            .wheel_torques(0.5)
            .unwrap(),
            [150., 150., 100., 100.]
        );
    }
    #[test]
    fn dyno_interpolation_and_gearing_conserve_power_with_efficiency() {
        let p = Powertrain {
            torque_curve: vec![
                DynoPoint {
                    rpm: 1000.,
                    torque_nm: 100.,
                },
                DynoPoint {
                    rpm: 5000.,
                    torque_nm: 200.,
                },
            ],
            idle_rpm: 1000.,
            redline_rpm: 5000.,
            gear_ratios: vec![2.],
            final_drive: 3.,
            efficiency: 0.9,
            driven_axle: DrivenAxle::Rear,
        };
        let s = p
            .evaluate(3000. * std::f64::consts::TAU / 60. / 6., 0, 0.5)
            .unwrap();
        assert!((s.engine_torque_nm - 75.).abs() < 1e-10);
        assert_eq!(s.wheel_torques_nm, [0., 0., 202.5, 202.5]);
        assert!(p.evaluate(1000., 0, 1.).is_err());
        assert_eq!(p.torque_at_rpm(1000.).unwrap(), 100.);
        assert_eq!(p.torque_at_rpm(5000.).unwrap(), 200.);
        assert!(p.torque_at_rpm(999.99).is_err());
        assert!(p.torque_at_rpm(5000.01).is_err());
        assert!(p.evaluate(100., 1, 1.).is_err());
        assert!(p.evaluate(100., 0, f64::NAN).is_err());
        let mut invalid = p.clone();
        invalid.torque_curve[1].rpm = 1000.;
        assert!(invalid.validate().is_err());
        invalid = p.clone();
        invalid.efficiency = 1.01;
        assert!(invalid.validate().is_err());
        let restored: Powertrain =
            serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
        assert_eq!(restored.torque_at_rpm(3000.).unwrap(), 150.);
        assert!(BrakeConfig {
            max_total_torque_nm: 100.,
            front_fraction: 1.1
        }
        .validate()
        .is_err());
    }
}
