//! Validated quasi-static drivetrain and braking model used by the
//! lap-simulation full-car vehicle (`crate::lap`): a piecewise-linear dyno
//! torque curve, fixed gear ratios and final drive, and a fixed front/rear
//! brake torque split. SI units throughout except engine/dyno speed, which is
//! in rpm to match how a dyno curve is normally supplied.
//!
//! "Quasi-static" means a locked clutch evaluated at an assumed wheel
//! angular speed: there is no clutch slip, launch, or engine/driveline
//! inertia model, and below-idle operation requires an explicit external
//! launch model this crate does not provide. See
//! <https://github.com/Ixiandesign/FKY-LAPSIM/blob/main/docs/model-conventions.md>
//! for the crate's broader SI/frame conventions.
use serde::{Deserialize, Serialize};
/// One measured or explicitly synthetic dyno knot: a single full-throttle
/// torque reading at a given engine speed. A [`Powertrain::torque_curve`] is a
/// sequence of these, linearly interpolated between neighboring knots.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DynoPoint {
    /// Engine speed, revolutions/minute. Must be nonnegative and strictly
    /// increasing from one knot to the next within a [`Powertrain`].
    pub rpm: f64,
    /// Full-throttle shaft torque at this engine speed, N m. Must be
    /// nonnegative.
    pub torque_nm: f64,
}
/// Which axle receives drive torque, always split equally left/right.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DrivenAxle {
    /// Front wheels only.
    Front,
    /// Rear wheels only.
    Rear,
    /// Equal torque at all four wheels (e.g. a four-wheel-drive idealization).
    All,
}
/// Fixed-ratio transmission with a piecewise-linear positive-torque dyno
/// curve. [`Powertrain::evaluate`] models an ideal open differential across
/// the driven axle: no limited-slip or torque-vectoring action is assumed.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Powertrain {
    /// Dyno knots, strictly increasing in `rpm` and together covering
    /// `idle_rpm` through `redline_rpm`; see [`Powertrain::torque_at_rpm`].
    pub torque_curve: Vec<DynoPoint>,
    /// Minimum allowed engine speed, rpm. Must be strictly positive.
    pub idle_rpm: f64,
    /// Maximum allowed engine speed, rpm. Must be strictly greater than
    /// `idle_rpm`.
    pub redline_rpm: f64,
    /// Positive forward gear ratios (engine speed / output speed); index is
    /// the zero-based gear identifier passed to [`Powertrain::evaluate`].
    /// Reverse and neutral are not represented.
    pub gear_ratios: Vec<f64>,
    /// Positive final-drive reduction applied after the gearbox.
    pub final_drive: f64,
    /// Mechanical efficiency factor applied to wheel torque, in the range 0
    /// (exclusive) to 1 (inclusive).
    pub efficiency: f64,
    /// Which wheels receive drive torque.
    pub driven_axle: DrivenAxle,
}
/// A deterministic drivetrain operating point computed by
/// [`Powertrain::evaluate`], assuming a locked clutch at the given wheel
/// angular speed.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PowertrainState {
    /// Zero-based engaged gear, indexing [`Powertrain::gear_ratios`].
    pub gear: usize,
    /// Engine speed implied by the wheel angular speed and gearing, rpm.
    pub engine_rpm: f64,
    /// Shaft torque after throttle scaling (before gearing/efficiency), N m.
    pub engine_torque_nm: f64,
    /// Drive torque delivered to each wheel, FL, FR, RL, RR order, N m. Zero
    /// at any wheel on an axle not selected by [`Powertrain::driven_axle`].
    pub wheel_torques_nm: [f64; 4],
}
impl Powertrain {
    /// Reject nonphysical parameters and incomplete dyno coverage. Called
    /// first by every other method on this type.
    ///
    /// # Errors
    ///
    /// Returns an error if `idle_rpm` or `redline_rpm` is nonfinite,
    /// `idle_rpm` is not strictly positive, or `redline_rpm <= idle_rpm`; if
    /// `final_drive` is not finite and strictly positive; if `efficiency` is
    /// not finite and in `(0, 1]`; if `gear_ratios` is empty or contains a
    /// nonfinite or nonpositive ratio; if `torque_curve` has fewer than two
    /// knots, contains a nonfinite or negative `rpm`/`torque_nm`, or is not
    /// strictly increasing in `rpm`; or if the dyno curve's first knot is
    /// above `idle_rpm` or its last knot is below `redline_rpm` (incomplete
    /// coverage).
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
    /// Linearly interpolate full-throttle shaft torque from
    /// [`Powertrain::torque_curve`] at the given engine speed, within the
    /// explicit `idle_rpm`..=`redline_rpm` running-speed domain. Never
    /// extrapolates beyond the dyno knots or the idle/redline domain.
    ///
    /// # Errors
    ///
    /// Returns an error if [`Powertrain::validate`] fails, if `rpm` is
    /// nonfinite or outside `idle_rpm..=redline_rpm`, or in the unreachable
    /// case that no consecutive knot pair brackets a covered `rpm`.
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
    /// Evaluate a locked clutch using the driven wheels' average angular
    /// speed, the engaged `gear` and a `throttle` fraction. Below idle
    /// requires an explicit external launch model this method does not
    /// provide. Equal left/right axle torque represents an ideal open
    /// differential; no limited-slip action is assumed.
    ///
    /// # Errors
    ///
    /// Returns an error if [`Powertrain::validate`] fails; if
    /// `wheel_angular_speed_rad_s` is nonfinite or negative; if `throttle` is
    /// nonfinite or outside `[0, 1]`; if `gear` is not a valid index into
    /// [`Powertrain::gear_ratios`]; if the implied engine speed falls outside
    /// `idle_rpm..=redline_rpm` (see [`Powertrain::torque_at_rpm`]); or if the
    /// resulting wheel torque overflows to a nonfinite value.
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
    /// Search every gear at full throttle and return the index giving the
    /// greatest total wheel torque at this wheel angular speed; ties keep the
    /// lowest (first) index. A gear whose implied engine speed falls outside
    /// `idle_rpm..=redline_rpm` is silently skipped rather than treated as an
    /// error, as long as at least one gear is in range.
    ///
    /// # Errors
    ///
    /// Returns an error if [`Powertrain::validate`] fails, or if no gear's
    /// implied engine speed lies within `idle_rpm..=redline_rpm` at this
    /// wheel speed (see [`Powertrain::evaluate`]) — including when
    /// `wheel_angular_speed_rad_s` itself is nonfinite or negative, since
    /// every gear then fails the same [`Powertrain::evaluate`] check and is
    /// skipped the same way.
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
/// Fixed front/rear brake torque allocation. [`BrakeConfig::wheel_torques`]
/// returns torque *magnitudes* only; the caller is responsible for applying
/// them so as to oppose each wheel's own rotation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BrakeConfig {
    /// Maximum sum of all four brake torque magnitudes at full brake command,
    /// N m. Must be finite and nonnegative.
    pub max_total_torque_nm: f64,
    /// Fraction of total brake torque assigned to the front axle, in the
    /// range 0 to 1 inclusive; the remainder goes to the rear axle, split
    /// evenly left/right on each axle.
    pub front_fraction: f64,
}
impl BrakeConfig {
    /// Check finite nonnegative capacity and a physically meaningful bias.
    ///
    /// # Errors
    ///
    /// Returns an error if `max_total_torque_nm` is nonfinite or negative, or
    /// if `front_fraction` is nonfinite or outside `[0, 1]`.
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
    /// Return nonnegative brake torque magnitudes in FL, FR, RL, RR order for
    /// a given brake `command` fraction, splitting `command *
    /// max_total_torque_nm` between front and rear per `front_fraction` and
    /// evenly left/right on each axle.
    ///
    /// # Errors
    ///
    /// Returns an error if [`BrakeConfig::validate`] fails, or if `command`
    /// is nonfinite or outside `[0, 1]`.
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
