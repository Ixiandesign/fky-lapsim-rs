//! Calibrated tire metadata: a [`crate::tire::TireModel`] paired with its
//! declared calibration domain and required provenance, plus the frame
//! conversion from a suspension-frame contact velocity/camber to the
//! Pacejka book frame those coefficients are defined in.
//!
//! A bare `TireModel` (see [`crate::tire`]) has no notion of where its fit is
//! actually valid. [`ConfiguredTire`] is what a lap vehicle stores instead:
//! it enforces the declared domain by rejecting out-of-domain inputs, never
//! by clamping them, and requires a recorded source so a synthetic
//! demonstration coefficient set can never be silently treated as a measured
//! calibration. This module is unrelated to the rigid disk/cylinder/torus
//! contact envelope used by the suspension-only kinematics model
//! (`Corner::tire_profile` in `crate::model`); see the "Rigid tire envelopes"
//! section of
//! <https://github.com/Ixiandesign/FKY-LAPSIM/blob/main/docs/model-conventions.md>.
use crate::{tire::{TireInput, TireModel, TireForces}, CornerId, Error};
use serde::{Deserialize, Serialize};

/// Origin of a coefficient set, checked by [`ConfiguredTire::validate`].
/// Synthetic data cannot be relabeled as measured without recording its
/// source and fit procedure.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag="kind", rename_all="snake_case", deny_unknown_fields)]
pub enum Provenance {
    /// Analytic demonstration only; not predictive vehicle calibration.
    Synthetic { /// Explanation of the synthetic construction (e.g. which
        /// closed-form shape/curvature choices were made and why). Rejected
        /// by [`ConfiguredTire::validate`] if empty or all whitespace.
        description: String },
    /// User-supplied fit in this exact equation family and sign convention
    /// (see [`crate::tire::TireFormulation`]); not verified independently by
    /// this type beyond requiring both fields below to be recorded.
    Measured { /// Dataset/publication and tire identification. Rejected by
        /// [`ConfiguredTire::validate`] if empty or all whitespace.
        source: String, /// Fit procedure, units and convention conversion
        /// used to produce these coefficients in the book's convention.
        /// Rejected by [`ConfiguredTire::validate`] if empty or all
        /// whitespace.
        fit_notes: String },
}
/// Explicit calibration bounds a [`ConfiguredTire`] declares its coefficients
/// valid over. [`ConfiguredTire::evaluate`] rejects any [`TireInput`] outside
/// these bounds as an error; it never clamps the input into range or
/// silently extrapolates.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TireDomain {
    /// Inclusive load range `[min, max]`, N. `min` must be nonnegative and
    /// `max` strictly greater than `min`.
    pub normal_load_n: [f64; 2],
    /// Maximum absolute book-frame slip angle, rad. Must lie strictly
    /// between 0 and pi/2.
    pub max_abs_slip_angle_rad: f64,
    /// Maximum absolute dimensionless longitudinal slip ratio. Must be
    /// strictly positive.
    pub max_abs_slip_ratio: f64,
    /// Maximum absolute book-frame inclination (camber), rad. Must lie
    /// strictly between 0 and pi/2.
    pub max_abs_camber_rad: f64,
    /// Inclusive positive forward speed range `[min, max]`, m/s. `min` must
    /// be strictly positive and `max` strictly greater than `min`.
    pub speed_m_s: [f64; 2],
}
/// A [`TireModel`] paired with its inseparable declared [`TireDomain`] and
/// [`Provenance`], as saved with every lap vehicle's tire input. Neither the
/// model nor the domain is meaningful alone: the model has no notion of
/// where it is valid, and the domain has no coefficients to bound.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfiguredTire {
    /// The underlying Magic Formula coefficients.
    pub model: TireModel,
    /// The coefficient set's origin; see [`Provenance`].
    pub provenance: Provenance,
    /// The declared domain of valid use; see [`TireDomain`].
    pub domain: TireDomain,
}
fn err(message: &str) -> Error { Error { message: format!("tire configuration: {message}") } }
impl ConfiguredTire {
    /// Validate the underlying [`TireModel`] (see
    /// [`TireModel::validate`]), the [`TireDomain`] bounds, and that
    /// [`Provenance`] carries nonempty text. [`ConfiguredTire::evaluate`]
    /// calls this first on every call.
    ///
    /// # Errors
    ///
    /// Returns an error if the wrapped model fails [`TireModel::validate`];
    /// if any domain bound is nonfinite; if `domain.normal_load_n[0]` is
    /// negative or `domain.normal_load_n[1] <= domain.normal_load_n[0]`; if
    /// `domain.speed_m_s[0]` is not strictly positive or `domain.speed_m_s[1]
    /// <= domain.speed_m_s[0]`; if `domain.max_abs_slip_angle_rad` is not
    /// strictly between 0 and pi/2; if `domain.max_abs_slip_ratio` is not
    /// strictly positive; if `domain.max_abs_camber_rad` is not strictly
    /// between 0 and pi/2; or if the recorded [`Provenance`] description, or
    /// source and fit notes, is empty or all whitespace.
    pub fn validate(&self) -> Result<(), Error> {
        self.model.validate()?;
        let d=&self.domain;
        if !d.normal_load_n.iter().chain(d.speed_m_s.iter()).chain(
            [d.max_abs_slip_angle_rad,d.max_abs_slip_ratio,d.max_abs_camber_rad].iter())
            .all(|v|v.is_finite()) || d.normal_load_n[0] < 0.
            || d.normal_load_n[1] <= d.normal_load_n[0]
            || d.speed_m_s[0] <= 0. || d.speed_m_s[1] <= d.speed_m_s[0]
            || !(0. ..std::f64::consts::FRAC_PI_2).contains(&d.max_abs_slip_angle_rad)
            || d.max_abs_slip_angle_rad == 0. || d.max_abs_slip_ratio <= 0.
            || d.max_abs_camber_rad <= 0. || d.max_abs_camber_rad >= std::f64::consts::FRAC_PI_2 {
            return Err(err("invalid calibration domain"));
        }
        let valid=match &self.provenance {
            Provenance::Synthetic{description}=> !description.trim().is_empty(),
            Provenance::Measured{source,fit_notes}=> !source.trim().is_empty() && !fit_notes.trim().is_empty(),
        };
        if !valid {return Err(err("source and fit description must be recorded"));}
        Ok(())
    }
    /// Evaluate the wrapped [`TireModel`], but only for a [`TireInput`]
    /// inside this configuration's declared [`TireDomain`]. An out-of-domain
    /// input is rejected outright, never clamped into range, so a lap result
    /// can never silently rest on an extrapolated tire fit. Outputs remain in
    /// the book's tire-frame convention; this does not perform the
    /// suspension-frame conversion ([`tire_input`] does that).
    ///
    /// # Errors
    ///
    /// Returns an error if [`ConfiguredTire::validate`] fails, if `input`
    /// falls outside the declared load, speed, slip-angle, slip-ratio or
    /// camber bounds, or if the underlying [`TireModel::evaluate`] itself
    /// errors.
    pub fn evaluate(&self, input:TireInput)->Result<TireForces,Error> {
        self.validate()?;
        let d=&self.domain;
        if input.normal_load_n < d.normal_load_n[0] || input.normal_load_n > d.normal_load_n[1]
            || input.speed_m_s < d.speed_m_s[0] || input.speed_m_s > d.speed_m_s[1]
            || input.slip_angle_rad.abs() > d.max_abs_slip_angle_rad
            || input.slip_ratio.abs() > d.max_abs_slip_ratio
            || input.camber_rad.abs() > d.max_abs_camber_rad {
            return Err(err("input outside declared calibration domain"));
        }
        self.model.evaluate(input)
    }
    /// A smooth, symmetric C=1/E=0 analytic demonstration tire at the given
    /// unloaded radius, deliberately not a measured fit — its [`Provenance`]
    /// is always [`Provenance::Synthetic`]. Useful for tests and as a
    /// placeholder before a vehicle has a calibrated coefficient set; do not
    /// use it to draw conclusions about a real tire's performance.
    pub fn synthetic_demo(radius_m:f64)->Self {
        Self {
            model:TireModel { formulation:Default::default(),reference_load_n:700.,radius_m,
                coefficients:[("PCX1",1.),("PDX1",1.4),("PKX1",20.),
                    ("PCY1",1.),("PDY1",1.4),("PKY1",25.),("PKY2",1.),("PKY4",2.),
                    ("RBX1",10.),("RCX1",1.),("RBY1",10.),("RCY1",1.),
                    ("QBZ1",10.),("QCZ1",1.),("QDZ1",0.05)].into_iter()
                    .map(|(s,v)|(s.to_string(),v)).collect() },
            provenance:Provenance::Synthetic { description:"Analytic C=1, E=0, symmetric demonstration. Not fitted to tire measurements.".into() },
            domain:TireDomain {normal_load_n:[0.,5000.],max_abs_slip_angle_rad:0.6,
                max_abs_slip_ratio:1.,max_abs_camber_rad:0.4,speed_m_s:[0.5,100.]},
        }
    }
}
/// Convert exact contact velocity in the wheel X-forward/Y-left frame and an
/// outward-positive suspension camber to Pacejka's Y-right/Z-down convention.
/// Alpha=atan(Vy_left/Vx); gamma positive means top tilted right viewed from rear.
/// `id` selects the left/right sign flip for camber. Negate returned tire Fy
/// and Mz when converting forces back to the vehicle.
///
/// # Errors
///
/// Returns an error if any input is nonfinite, if
/// `contact_velocity_m_s[0]` (forward speed) is not strictly positive, if
/// `rolling_radius_m` is not strictly positive, or if `angular_speed_rad_s`
/// is negative. Standstill and reverse rolling are unsupported, matching
/// [`TireInput::speed_m_s`].
pub fn tire_input(id:CornerId, normal_load_n:f64, contact_velocity_m_s:[f64;2],
    angular_speed_rad_s:f64, rolling_radius_m:f64, outward_camber_deg:f64)->Result<TireInput,Error> {
    let [vx,vy]=contact_velocity_m_s;
    if ![normal_load_n,vx,vy,angular_speed_rad_s,rolling_radius_m,outward_camber_deg].iter().all(|x|x.is_finite())
        || vx<=0. || rolling_radius_m<=0. || angular_speed_rad_s<0. {
        return Err(err("only finite forward rolling states are supported"));
    }
    let side=match id {CornerId::FrontLeft|CornerId::RearLeft=>1.,_=>-1.};
    Ok(TireInput {normal_load_n,slip_angle_rad:vy.atan2(vx),
        slip_ratio:(rolling_radius_m*angular_speed_rad_s-vx)/vx,
        camber_rad:-side*outward_camber_deg.to_radians(),speed_m_s:vx})
}
