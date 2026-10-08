//! The full-car configuration the lap simulators run on.
//!
//! [LapVehicle] composes the suspension [crate::Project] with four per-corner tires
//! ([ConfiguredTire]), a [Powertrain], [BrakeConfig] and [Aero] into one description; its numeric
//! fields are the leaves addressed by JSON-pointer paths in multi-track lap optimization (see
//! `crate::lap::optimization`). The quasi-steady-state lap simulator in [crate::lapsim] consumes
//! it together with its own extras ([crate::lapsim::vehicle::QssExtras]).
use crate::{Project,CornerId,Error,tire_configuration::ConfiguredTire,
    powertrain::{Powertrain,BrakeConfig,DrivenAxle,DynoPoint},aero::Aero};
use serde::{Serialize,Deserialize};

const IDS:[CornerId;4]=[CornerId::FrontLeft,CornerId::FrontRight,CornerId::RearLeft,CornerId::RearRight];
fn error(message:impl Into<String>)->Error {Error{message:message.into()}}

/// A tire/contact's physical properties, identified independently of array order.
#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct LapWheel {
    /// Corner identity; matches this wheel to the corresponding suspension
    /// corner and to the per-wheel result arrays, independent of storage
    /// order.
    pub id:CornerId,
    /// Nominal-pressure Magic Formula model, its declared validity bounds,
    /// and data provenance ([ConfiguredTire::evaluate] errors outside that
    /// domain rather than extrapolating).
    pub tire:ConfiguredTire,
    /// Wheel spin inertia about its spin axis, kg m², strictly positive.
    /// Wheel translational mass is lumped into the chassis sprung mass, not
    /// carried here.
    pub spin_inertia_kg_m2:f64,
}
/// Full car: suspension, tires, powertrain, brakes, and aero kept as one
/// deliberately separate composition (distinct from the suspension-only
/// [Project]) so that a suspension-only editor cannot silently discard the
/// lap-specific subsystems.
#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct LapVehicle {
    /// Complete hardpoints, corner/interconnect spring/damper laws, and
    /// total lumped chassis mass/inertia. [LapVehicle::validate] requires
    /// component masses (upper arm, lower arm, knuckle, rocker) to already
    /// be lumped into that chassis mass/inertia; separately retained link
    /// masses are rejected, not silently ignored.
    pub suspension:Project,
    /// Exactly one tire model per corner, matched to `suspension`'s corners
    /// by [CornerId], not by array position; serialized order is arbitrary.
    pub wheels:[LapWheel;4],
    /// Dyno-curve engine, fixed gear ratios/final drive, and an ideal open
    /// differential on one driven axle, with a locked clutch (no
    /// launch/standstill slip model).
    pub powertrain:Powertrain,
    /// Total brake torque capacity and front/rear bias.
    pub brakes:BrakeConfig,
    /// Constant-coefficient drag/downforce/center-of-pressure aero model; no
    /// yaw sensitivity or ground effect.
    pub aero:Aero,
    /// Maximum magnitude of front rack travel, m; must lie in `(0, 0.2]`
    /// (checked by [LapVehicle::validate]).
    pub steering_limit_m:f64,
}
impl LapVehicle {
    /// Validate the full car's declared fidelity, not just its JSON shape:
    /// suspension, powertrain, brakes and aero each pass their own
    /// validation; `steering_limit_m` is in range; `wheels` has exactly one
    /// entry per [CornerId] with a valid tire and positive spin inertia; and
    /// each wheel's tire radius matches its corner's suspension tire
    /// envelope radius exactly. Link/component mass retained separately on a
    /// corner (upper arm, lower arm, knuckle, or rocker) is intentionally
    /// rejected rather than silently omitted — this reduced model requires
    /// all component mass to already be lumped into the chassis sprung
    /// mass/inertia.
    ///
    /// # Errors
    /// Returns an error for the first failing check above: invalid
    /// suspension, powertrain, brakes, or aero configuration;
    /// `steering_limit_m` outside `(0, 0.2]`; a missing, duplicated, or
    /// invalid wheel for some corner; non-finite or non-positive
    /// `spin_inertia_kg_m2`; a tire radius that disagrees with its corner's
    /// suspension envelope by more than 1e-8 m; or any corner carrying
    /// nonzero retained component mass.
    pub fn validate(&self)->Result<(),Error> {
        self.suspension.validate()?;
        self.powertrain.validate().map_err(error)?;self.brakes.validate().map_err(error)?;self.aero.validate().map_err(error)?;
        if !self.steering_limit_m.is_finite() || self.steering_limit_m<=0. || self.steering_limit_m>0.2 {return Err(error("steering limit must be in (0,0.2] m"));}
        for id in IDS {
            if self.wheels.iter().filter(|w|w.id==id).count()!=1 {return Err(error("full car requires one tire per corner"));}
            let w=self.wheels.iter().find(|w|w.id==id).unwrap();w.tire.validate()?;
            if !w.spin_inertia_kg_m2.is_finite() || w.spin_inertia_kg_m2<=0. {return Err(error("wheel spin inertia must be positive"));}
            let c=self.suspension.corners.iter().find(|c|c.id==id).unwrap();
            if (c.tire_radius-w.tire.model.radius_m).abs()>1e-8 {return Err(error("tire force radius must match suspension envelope radius"));}
            let m=&c.component_masses;
            if [&m.upper_arm,&m.lower_arm,&m.knuckle,&m.rocker].iter().any(|b|b.as_ref().is_some_and(|body|body.mass_kg>0.)) {return Err(error("lap reduced model requires component masses lumped into chassis mass/inertia; separately retained link masses are unsupported"));}
        }
        Ok(())
    }
    /// Formula-car-scale illustrative example, built from
    /// [crate::dynamics::formula_car_demo]'s suspension plus an analytic
    /// synthetic tire ([ConfiguredTire::synthetic_demo]) and a plausible but
    /// invented dyno/gearing/brake/aero specification. Never a calibrated
    /// vehicle prediction; intended only for tests, examples, and API
    /// exploration.
    ///
    /// # Errors
    /// Returns an error only if the underlying suspension demo project fails
    /// to build (see [crate::dynamics::formula_car_demo]); the remaining
    /// fields are fixed valid literals.
    pub fn synthetic_demo()->Result<Self,Error> {
        let (suspension,_)=crate::dynamics::formula_car_demo()?;
        let wheels=std::array::from_fn(|i|{let c=suspension.corners.iter().find(|c|c.id==IDS[i]).unwrap();
            LapWheel{id:IDS[i],tire:ConfiguredTire::synthetic_demo(c.tire_radius),spin_inertia_kg_m2:0.8}});
        Ok(Self {suspension,wheels,
            powertrain:Powertrain {torque_curve:vec![DynoPoint{rpm:1000.,torque_nm:35.},DynoPoint{rpm:7000.,torque_nm:45.},DynoPoint{rpm:12000.,torque_nm:30.}],idle_rpm:1000.,redline_rpm:12000.,gear_ratios:vec![3.,2.2,1.6,1.2],final_drive:3.,efficiency:0.92,driven_axle:DrivenAxle::Rear},
            brakes:BrakeConfig{max_total_torque_nm:1800.,front_fraction:0.6},
            aero:Aero {air_density_kg_m3:1.225,reference_area_m2:1.,drag_coefficient:0.8,downforce_coefficient:2.,center_of_pressure_m:[0.,0.,0.]},steering_limit_m:0.04})
    }
}
