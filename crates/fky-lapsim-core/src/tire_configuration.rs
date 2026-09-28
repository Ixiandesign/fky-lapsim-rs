//! Calibrated tire metadata, declared validity domains and frame conversion.
use crate::{tire::{TireInput, TireModel, TireForces}, CornerId, Error};
use serde::{Deserialize, Serialize};

/// Origin of a coefficient set. Synthetic data cannot be relabeled as measured
/// without recording its source and fit procedure.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag="kind", rename_all="snake_case", deny_unknown_fields)]
pub enum Provenance {
    /// Analytic demonstration only; not predictive vehicle calibration.
    Synthetic { /// Explanation of the synthetic construction.
        description: String },
    /// User-supplied fit in this exact equation/sign convention.
    Measured { /// Dataset/publication and tire identification.
        source: String, /// Fit procedure, units and convention conversion.
        fit_notes: String },
}
/// Explicit calibration bounds. Outside-domain evaluation is an error, not a clamp.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TireDomain {
    /// Inclusive load range, N.
    pub normal_load_n: [f64; 2],
    /// Maximum absolute book slip angle, rad.
    pub max_abs_slip_angle_rad: f64,
    /// Maximum absolute dimensionless longitudinal slip.
    pub max_abs_slip_ratio: f64,
    /// Maximum absolute book inclination, rad.
    pub max_abs_camber_rad: f64,
    /// Inclusive positive forward speed range, m/s.
    pub speed_m_s: [f64; 2],
}
/// Tire model plus inseparable domain/provenance, saved with every lap input.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfiguredTire {
    /// Exact supported Magic Formula coefficients.
    pub model: TireModel,
    /// Coefficient-set origin.
    pub provenance: Provenance,
    /// Declared domain of use.
    pub domain: TireDomain,
}
fn err(message: &str) -> Error { Error { message: format!("tire configuration: {message}") } }
impl ConfiguredTire {
    /// Validate coefficient family, bounds and nonempty provenance.
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
    /// Evaluate only within the recorded domain. Book-frame outputs are unchanged.
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
    /// Smooth, symmetric C=1/E=0 demonstration; deliberately not a measured fit.
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
/// Negate returned tire Fy and Mz when converting forces back to the vehicle.
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
