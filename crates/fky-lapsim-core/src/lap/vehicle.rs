//! Full-car composition and nonlinear four-contact force evaluation.
use super::{mechanics::{IDS,error,heading},suspension_forces,angular_acceleration,angular_energy,SuspensionForces};
use crate::{Project,CornerId,Error,planar::{PlanarState,contact_velocity},tire_configuration::{ConfiguredTire,tire_input},
    powertrain::{Powertrain,PowertrainState,DrivenAxle,BrakeConfig,DynoPoint},aero::{Aero,AeroState}};
use nalgebra::{UnitQuaternion,Vector3};
use serde::{Serialize,Deserialize};

/// A tire/contact's physical properties, identified independently of array order.
#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct LapWheel {
    /// Corner identity.
    pub id:CornerId,
    /// Nominal-pressure Magic Formula model, validity bounds and provenance.
    pub tire:ConfiguredTire,
    /// Wheel spin inertia, kg m²; wheel translational mass belongs to chassis mass.
    pub spin_inertia_kg_m2:f64,
}
/// Full car: separate composition so suspension-only editors cannot discard it.
#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct LapVehicle {
    /// Complete hardpoints, corner/interconnect laws and total lumped mass/inertia.
    pub suspension:Project,
    /// Exactly one tire model per corner, arbitrary serialized order.
    pub wheels:[LapWheel;4],
    /// Dyno/gears/ideal open differential with locked clutch.
    pub powertrain:Powertrain,
    /// Brake capacity and bias.
    pub brakes:BrakeConfig,
    /// Constant CL/CD/COP model.
    pub aero:Aero,
    /// Maximum magnitude of front rack travel, m.
    pub steering_limit_m:f64,
}
/// State arrays with wheel data always FL,FR,RL,RR.
#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct LapState {
    /// CG in the plane and yaw motion.
    pub planar:PlanarState,
    /// Heave,roll,pitch, m/rad/rad.
    pub displacement:[f64;3],
    /// Their time derivatives, m/s and rad/s.
    pub velocity:[f64;3],
    /// Four forward wheel spin speeds, rad/s.
    pub wheel_angular_speed_rad_s:[f64;4],
}
/// Driver/actuator commands, held or smoothly varied by the integration driver.
#[derive(Debug,Clone,Copy,Default,Serialize,Deserialize)]
pub struct LapControls {
    /// Front rack travel, m.
    pub rack_front_m:f64,
    /// Front rack velocity, m/s, included in contact and damper velocities.
    pub rack_rate_m_s:f64,
    /// Fraction of available engine torque, [0,1].
    pub throttle:f64,
    /// Fraction of configured brake torque, [0,1].
    pub brake:f64,
    /// Zero-based locked gear.
    pub gear:usize,
}
/// State derivative from one coupled force solve.
#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct LapDerivative {
    /// Global X speed, m/s.
    pub x_m_s:f64,
    /// Global Y speed, m/s.
    pub y_m_s:f64,
    /// Heading derivative, rad/s.
    pub heading_rad_s:f64,
    /// Body forward speed derivative, m/s².
    pub u_m_s2:f64,
    /// Body left speed derivative, m/s².
    pub v_m_s2:f64,
    /// Yaw rate derivative, rad/s².
    pub yaw_acceleration_rad_s2:f64,
    /// Generalized suspension velocities.
    pub displacement_velocity:[f64;3],
    /// Heave/roll/pitch accelerations.
    pub displacement_acceleration:[f64;3],
    /// Wheel spin accelerations, rad/s².
    pub wheel_acceleration_rad_s2:[f64;4],
}
/// Per-wheel results in the wheel X-forward/Y-left frame.
#[derive(Debug,Clone,Default,Serialize,Deserialize)]
pub struct WheelResult {
    /// Identity (canonical ordering is also preserved).
    pub id:Option<CornerId>,
    /// Normal force, N, including geometry-derived jacking.
    pub normal_load_n:f64,
    /// Tire input slip angle in the book convention, rad.
    pub slip_angle_rad:f64,
    /// Longitudinal slip ratio.
    pub slip_ratio:f64,
    /// Rolling radius, m; rigid unloaded radius is used.
    pub rolling_radius_m:f64,
    /// Wheel heading counterclockwise from chassis yaw, rad.
    pub steer_rad:f64,
    /// Longitudinal force, N.
    pub fx_n:f64,
    /// Leftward force, N.
    pub fy_n:f64,
    /// Counterclockwise aligning moment, N m.
    pub mz_nm:f64,
    /// Drive torque, N m.
    pub drive_torque_nm:f64,
    /// Brake torque opposing forward rotation, N m.
    pub brake_torque_nm:f64,
    /// Book combined-slip longitudinal weighting.
    pub gx:f64,
    /// Book combined-slip lateral weighting.
    pub gy:f64,
}
/// Complete instantaneous numerical state; accepted integration samples retain this.
#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct LapEvaluation {
    /// Integrated state derivatives.
    pub derivative:LapDerivative,
    /// Individual tire/contact loads and slips.
    pub wheels:[WheelResult;4],
    /// Exact solved suspension and virtual-work forces.
    pub suspension:SuspensionForces,
    /// Aero load in the rolled/pitched chassis frame.
    pub aero:AeroState,
    /// Drivetrain operating point.
    pub powertrain:PowertrainState,
    /// Chassis translation/rotation, wheel spin and spring/gravity energy, J.
    pub mechanical_energy_j:f64,
    /// Implicit jacking/load iterations.
    pub load_iterations:usize,
}
impl LapVehicle {
    /// Validate the declared fidelity, not just JSON shape. Link mass/inertia is
    /// intentionally rejected rather than silently omitted by this reduced model.
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
    /// Formula-scale synthetic example; never a calibrated vehicle prediction.
    pub fn synthetic_demo()->Result<Self,Error> {
        let (suspension,_)=crate::dynamics::formula_car_demo()?;
        let wheels=std::array::from_fn(|i|{let c=suspension.corners.iter().find(|c|c.id==IDS[i]).unwrap();
            LapWheel{id:IDS[i],tire:ConfiguredTire::synthetic_demo(c.tire_radius),spin_inertia_kg_m2:0.8}});
        Ok(Self {suspension,wheels,
            powertrain:Powertrain {torque_curve:vec![DynoPoint{rpm:1000.,torque_nm:35.},DynoPoint{rpm:7000.,torque_nm:45.},DynoPoint{rpm:12000.,torque_nm:30.}],idle_rpm:1000.,redline_rpm:12000.,gear_ratios:vec![3.,2.2,1.6,1.2],final_drive:3.,efficiency:0.92,driven_axle:DrivenAxle::Rear},
            brakes:BrakeConfig{max_total_torque_nm:1800.,front_fraction:0.6},
            aero:Aero {air_density_kg_m3:1.225,reference_area_m2:1.,drag_coefficient:0.8,downforce_coefficient:2.,center_of_pressure_m:[0.,0.,0.]},steering_limit_m:0.04})
    }
    /// Average driven-wheel speed for the ideal open differential.
    pub fn driven_speed(&self,state:&LapState)->f64 {
        let w=state.wheel_angular_speed_rad_s;
        match self.powertrain.driven_axle {DrivenAxle::Front=>(w[0]+w[1])/2.,DrivenAxle::Rear=>(w[2]+w[3])/2.,DrivenAxle::All=>w.iter().sum::<f64>()/4.}
    }
}
impl LapState {
    /// Initial rolling straight-line state at static equilibrium, all contacts
    /// spinning without longitudinal slip. Launch/standstill is not approximated.
    pub fn rolling(car:&LapVehicle,speed:f64)->Result<Self,Error> {
        car.validate()?;
        if !speed.is_finite() || speed<=0. {return Err(error("initial speed must be positive"));}
        let modes=crate::dynamics::linearize_ride(&car.suspension,&crate::RideRequest::default())?;
        Ok(Self {planar:PlanarState{u_m_s:speed,..Default::default()},displacement:modes.equilibrium,
            velocity:[0.;3],wheel_angular_speed_rad_s:std::array::from_fn(|i|speed/car.wheels.iter().find(|w|w.id==IDS[i]).unwrap().tire.model.radius_m)})
    }
}
/// Coupled four-contact force calculation: exact geometry and individual slip,
/// implicit normal-force jacking, nonlinear springs/dampers, Euler inertia and
/// wheel spin. Fixed flat contact, rigid tires, massless links, no wheel gyroscopic
/// torques or tire relaxation. Out-of-domain/lift-off states are explicit errors.
pub fn evaluate_vehicle(car:&LapVehicle,s:&LapState,c:LapControls)->Result<LapEvaluation,Error> {
    car.validate()?;
    if ![c.rack_front_m,c.rack_rate_m_s,c.throttle,c.brake].iter().all(|x|x.is_finite())
        || c.rack_front_m.abs()>car.steering_limit_m || !(0. ..=1.).contains(&c.throttle) || !(0. ..=1.).contains(&c.brake)
        || s.displacement[1].abs()>0.7 || s.displacement[2].abs()>0.7 {return Err(error("vehicle control or attitude outside domain"));}
    let p=&car.suspension;let cg=p.chassis.center_of_mass;
    let suspension=suspension_forces(p,s.displacement,s.velocity,c.rack_front_m,c.rack_rate_m_s)?;
    let powertrain=car.powertrain.evaluate(car.driven_speed(s),c.gear,c.throttle).map_err(error)?;
    let braking=car.brakes.wheel_torques(c.brake).map_err(error)?;
    let rotation=UnitQuaternion::from_axis_angle(&Vector3::y_axis(),s.displacement[2])*
        UnitQuaternion::from_axis_angle(&Vector3::x_axis(),s.displacement[1]);
    let air_body=rotation.inverse()*Vector3::new(s.planar.u_m_s,s.planar.v_m_s,s.velocity[0]);
    let aero=car.aero.evaluate(air_body.into()).map_err(error)?;
    let af=rotation*Vector3::from(aero.force_n);let am=rotation*Vector3::from(aero.moment_nm);
    let rate=[s.velocity[0],s.velocity[1],s.velocity[2],c.rack_rate_m_s];
    let mut wheels:[WheelResult;4]=std::array::from_fn(|_|WheelResult::default());
    let mut normals=suspension.normal_load_n;
    let mut total=[0.;3];let mut qforce=suspension.generalized_force;
    let mut load_iterations=0;
    for iteration in 0..60 {
        total=[af.x,af.y,am.z];
        qforce=[suspension.generalized_force[0]+af.z,
            suspension.generalized_force[1]+s.displacement[2].cos()*am.x-s.displacement[2].sin()*am.z,
            suspension.generalized_force[2]+am.y];
        let mut next=suspension.normal_load_n;
        for i in 0..4 {
            if normals[i]<0. {return Err(error(format!("{:?}: contact lift-off",IDS[i])));}
            let wc=car.wheels.iter().find(|w|w.id==IDS[i]).unwrap();
            let corner=&suspension.state.corners[i];let cp=corner.points.contact_point;
            let pos=[cp[0]-cg[0],cp[1]-cg[1],cp[2]-cg[2]-s.displacement[0]];
            let steer=heading(&suspension.state,i);let (sn,cs)=steer.sin_cos();
            let mut cv=contact_velocity(s.planar,pos,steer)?;
            let dx=(0..4).map(|a|suspension.contact_jacobian[i][a][0]*rate[a]).sum::<f64>();
            let dy=(0..4).map(|a|suspension.contact_jacobian[i][a][1]*rate[a]).sum::<f64>();
            cv[0]+=cs*dx+sn*dy;cv[1]+=-sn*dx+cs*dy;
            let input=tire_input(IDS[i],normals[i],cv,s.wheel_angular_speed_rad_s[i],wc.tire.model.radius_m,corner.metrics.camber_deg)?;
            let f=wc.tire.evaluate(input)?;
            let fx=cs*f.fx_n+sn*f.fy_n;let fy=sn*f.fx_n-cs*f.fy_n;let mz=-f.mz_nm;
            total[0]+=fx;total[1]+=fy;total[2]+=pos[0]*fy-pos[1]*fx+mz;
            for a in 0..3 {
                let work=fx*suspension.contact_jacobian[i][a][0]+fy*suspension.contact_jacobian[i][a][1]+mz*suspension.heading_jacobian[i][a];
                qforce[a]+=work;if a==0 {next[i]+=work;}
            }
            wheels[i]=WheelResult{id:Some(IDS[i]),normal_load_n:normals[i],slip_angle_rad:input.slip_angle_rad,slip_ratio:input.slip_ratio,
                rolling_radius_m:wc.tire.model.radius_m,steer_rad:steer,fx_n:f.fx_n,fy_n:-f.fy_n,mz_nm:mz,
                drive_torque_nm:powertrain.wheel_torques_nm[i],brake_torque_nm:braking[i],gx:f.gx,gy:f.gy};
        }
        load_iterations=iteration+1;
        let residual=(0..4).map(|i|(next[i]-normals[i]).abs()).fold(0_f64,f64::max);
        if residual<1e-5 {break;}
        if iteration==59 {return Err(error("implicit tire/suspension normal loads did not converge"));}
        for i in 0..4 {normals[i]=(normals[i]+next[i])/2.;}
    }
    let angular=angular_acceleration(p.chassis.inertia,[s.displacement[1],s.displacement[2]],
        [s.planar.yaw_rate_rad_s,s.velocity[1],s.velocity[2]],[total[2],qforce[1],qforce[2]])?;
    let (sn,cs)=s.planar.heading_rad.sin_cos();let mass=p.chassis.sprung_mass;
    let derivative=LapDerivative{x_m_s:cs*s.planar.u_m_s-sn*s.planar.v_m_s,y_m_s:sn*s.planar.u_m_s+cs*s.planar.v_m_s,
        heading_rad_s:s.planar.yaw_rate_rad_s,u_m_s2:total[0]/mass+s.planar.yaw_rate_rad_s*s.planar.v_m_s,
        v_m_s2:total[1]/mass-s.planar.yaw_rate_rad_s*s.planar.u_m_s,yaw_acceleration_rad_s2:angular[0],
        displacement_velocity:s.velocity,displacement_acceleration:[qforce[0]/mass,angular[1],angular[2]],
        wheel_acceleration_rad_s2:std::array::from_fn(|i|{let w=&wheels[i];(w.drive_torque_nm-w.brake_torque_nm-w.fx_n*w.rolling_radius_m)/car.wheels.iter().find(|w|w.id==IDS[i]).unwrap().spin_inertia_kg_m2})};
    let mechanical_energy_j=suspension.potential_energy_j+0.5*mass*(s.planar.u_m_s.powi(2)+s.planar.v_m_s.powi(2)+s.velocity[0].powi(2))
        +angular_energy(p.chassis.inertia,[s.displacement[1],s.displacement[2]],[s.planar.yaw_rate_rad_s,s.velocity[1],s.velocity[2]])
        +(0..4).map(|i|0.5*car.wheels.iter().find(|w|w.id==IDS[i]).unwrap().spin_inertia_kg_m2*s.wheel_angular_speed_rad_s[i].powi(2)).sum::<f64>();
    Ok(LapEvaluation {derivative,wheels,suspension,aero,powertrain,mechanical_energy_j,load_iterations})
}
