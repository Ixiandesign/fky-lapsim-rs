//! Explicit driver-policy lap integration, not a global minimum-time racing line.
use super::{LapVehicle,LapState,LapControls,LapDerivative,LapEvaluation,evaluate_vehicle,mechanics::{error,IDS},suspension_forces};
use crate::{Error,track::Track};
use serde::{Serialize,Deserialize};
use std::collections::BTreeMap;

/// Solver and driver policy. Driver acceleration limits constrain its target
/// speed; actual tire forces always come from the four-wheel physical model.
#[derive(Debug,Clone,Serialize,Deserialize)]
#[serde(default,deny_unknown_fields)]
pub struct LapRequest {
    /// RK4 timestep, seconds. Users must verify timestep refinement.
    pub dt_s:f64,
    /// Output interval, seconds (integration is independent).
    pub sample_interval_s:f64,
    /// Safety timeout in simulated seconds.
    pub max_time_s:f64,
    /// Rolling start speed, m/s; locked clutch, no standing-start approximation.
    pub initial_speed_m_s:f64,
    /// Settling laps excluded from reported lap time.
    pub warmup_laps:usize,
    /// Measured laps, reported time is their total.
    pub laps:usize,
    /// Driver's upper speed target, m/s.
    pub max_speed_m_s:f64,
    /// Driver corner-speed policy limit, m/s², not a replacement grip model.
    pub lateral_acceleration_limit_m_s2:f64,
    /// Driver preview braking policy, m/s².
    pub braking_deceleration_m_s2:f64,
    /// Geometric steering preview, m.
    pub lookahead_m:f64,
    /// Proportional speed controller gain, 1/s.
    pub speed_gain_per_s:f64,
    /// Maximum rack actuator speed, m/s.
    pub rack_rate_limit_m_s:f64,
    /// Add detailed alignment/rate analysis to recorded samples (more work).
    pub report_analysis:bool,
}
impl Default for LapRequest {
    fn default()->Self {Self{dt_s:0.005,sample_interval_s:0.05,max_time_s:120.,initial_speed_m_s:10.,warmup_laps:0,laps:1,
        max_speed_m_s:20.,lateral_acceleration_limit_m_s2:6.,braking_deceleration_m_s2:5.,lookahead_m:5.,speed_gain_per_s:1.,rack_rate_limit_m_s:0.08,report_analysis:false}}
}
impl LapRequest {
    /// Bound memory/runtime and reject invalid driver settings.
    pub fn validate(&self)->Result<(),Error> {
        if [self.dt_s,self.sample_interval_s,self.max_time_s,self.initial_speed_m_s,self.max_speed_m_s,
            self.lateral_acceleration_limit_m_s2,self.braking_deceleration_m_s2,self.lookahead_m,self.speed_gain_per_s,self.rack_rate_limit_m_s]
            .iter().any(|x|!x.is_finite() || *x<=0.) || self.dt_s>0.02 || self.dt_s<0.0001
            || self.sample_interval_s<self.dt_s || self.max_time_s/self.dt_s>200000.
            || self.max_time_s/self.sample_interval_s>20000. || self.laps==0 || self.laps+self.warmup_laps>20 {
            return Err(error("invalid lap solver settings or work limit (200000 steps, 20000 samples, 20 laps)"));
        }
        Ok(())
    }
}
/// Recorded accepted state, tied to the complete immutable car/track/request.
#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct LapSample {
    /// Elapsed integration time, s, including warmup.
    pub time_s:f64,
    /// Unwrapped traveled centerline coordinate, m.
    pub distance_m:f64,
    /// Signed distance left of the centerline, m.
    pub lateral_error_m:f64,
    /// Driver target, m/s.
    pub target_speed_m_s:f64,
    /// Actual vehicle state.
    pub state:LapState,
    /// Actual driver commands.
    pub controls:LapControls,
    /// Tire, suspension, drivetrain and force-balance data.
    pub evaluation:LapEvaluation,
    /// Optional expanded suspension alignment/rates.
    pub analysis:Option<crate::Analysis>,
}
/// A completed or explicitly terminated run; incomplete runs have no lap time.
#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct LapRun {
    /// Versioned assumptions, not a claim of measured validation.
    pub model_fidelity:String,
    /// Only true after the requested measured finish crossings.
    pub completed:bool,
    /// Completion/failure/cancellation reason.
    pub termination:String,
    /// Total measured lap time, excluding warmup, only on completion.
    pub lap_time_s:Option<f64>,
    /// Named scalar metrics with explicit units in names.
    pub metrics:BTreeMap<String,f64>,
    /// Accepted samples including final valid sample.
    pub samples:Vec<LapSample>,
}
fn add(s:&LapState,d:&LapDerivative,h:f64)->LapState {
    let mut o=s.clone();let p=&mut o.planar;
    p.x_m+=h*d.x_m_s;p.y_m+=h*d.y_m_s;p.heading_rad+=h*d.heading_rad_s;
    p.u_m_s+=h*d.u_m_s2;p.v_m_s+=h*d.v_m_s2;p.yaw_rate_rad_s+=h*d.yaw_acceleration_rad_s2;
    for i in 0..3 {o.displacement[i]+=h*d.displacement_velocity[i];o.velocity[i]+=h*d.displacement_acceleration[i];}
    for i in 0..4 {o.wheel_angular_speed_rad_s[i]+=h*d.wheel_acceleration_rad_s2[i];}
    o
}
/// One classical RK4 step with a linearly moving front rack. No integration or
/// domain errors are hidden by clamping wheel speed or suspension displacement.
pub fn advance_vehicle(car:&LapVehicle,s:&LapState,c:LapControls,dt:f64)->Result<LapState,Error> {
    if !dt.is_finite() || dt<=0. || dt>0.02 {return Err(error("invalid vehicle integration step"));}
    let command=|t:f64|LapControls{rack_front_m:c.rack_front_m+c.rack_rate_m_s*t,..c};
    let k1=evaluate_vehicle(car,s,c)?.derivative;
    let k2=evaluate_vehicle(car,&add(s,&k1,dt/2.),command(dt/2.))?.derivative;
    let k3=evaluate_vehicle(car,&add(s,&k2,dt/2.),command(dt/2.))?.derivative;
    let k4=evaluate_vehicle(car,&add(s,&k3,dt),command(dt))?.derivative;
    let mut o=add(s,&k1,dt/6.);o=add(&o,&k2,dt/3.);o=add(&o,&k3,dt/3.);o=add(&o,&k4,dt/6.);
    Ok(o)
}
// Nearest segment restricted to the previous progress neighborhood. This avoids
// jumping to the wrong arm of a crossover track. The very first projection is global.
fn project(track:&Track,pos:[f64;2],near:Option<f64>,window:f64)->Result<(f64,f64),Error> {
    let length=track.length_m().map_err(error)?;let p=&track.centerline_m;
    let n=if p.first()==p.last(){p.len()-1}else{p.len()};let mut arc=0.;let mut best=None;
    for i in 0..n {
        let a=p[i];let b=p[(i+1)%n];let dx=b[0]-a[0];let dy=b[1]-a[1];let len=dx.hypot(dy);
        let t=(((pos[0]-a[0])*dx+(pos[1]-a[1])*dy)/(len*len)).clamp(0.,1.);
        let s=arc+t*len;let unwrapped=near.map(|near|s+((near-s)/length).round()*length).unwrap_or(s);
        if near.is_none_or(|near|(unwrapped-near).abs()<=window) {
            let ex=pos[0]-a[0]-t*dx;let ey=pos[1]-a[1]-t*dy;let d=ex*ex+ey*ey;
            if best.is_none_or(|(bd,_,_)|d<bd) {best=Some((d,unwrapped,(-dy*ex+dx*ey)/len));}
        }
        arc+=len;
    }
    best.map(|(_,s,e)|(s,e)).ok_or_else(||error("track projection lost local progress"))
}
fn target_speed(track:&Track,r:&LapRequest,s:f64)->Result<f64,Error> {
    let mut speed=r.max_speed_m_s;
    // A driver preview envelope, not a force capacity approximation. The actual
    // simulation can still understeer, saturate, lock, lift or leave the course.
    let horizon=(r.max_speed_m_s*r.max_speed_m_s/(2.*r.braking_deceleration_m_s2)).max(r.lookahead_m);
    for i in 0..=32 {
        let d=horizon*i as f64/32.;let curvature=track.sample(s+d).map_err(error)?.curvature_per_m.abs();
        if curvature>1e-9 {speed=speed.min((r.lateral_acceleration_limit_m_s2/curvature+2.*r.braking_deceleration_m_s2*d).sqrt());}
    }
    Ok(speed)
}
/// Run one driver-policy lap simulation, checking cancellation every accepted
/// step. Curve speed settings are policy inputs; only successful physical laps
/// yield lap times. This does not optimize the racing line or claim minimum time.
pub fn simulate_lap_controlled(car:&LapVehicle,track:&Track,r:&LapRequest,cancelled:&impl Fn()->bool)->Result<LapRun,Error> {
    car.validate()?;track.validate().map_err(error)?;r.validate()?;
    let mut run=LapRun{model_fidelity:"four_contact_MF2012_nominal_pressure_nonlinear_massless_link_suspension_6DOF_chassis_4wheelspin_locked_open_diff_constant_aero_pure_pursuit_v1".into(),completed:false,termination:"maximum simulated time reached".into(),lap_time_s:None,metrics:BTreeMap::new(),samples:vec![]};
    if cancelled(){run.termination="cancelled".into();return Ok(run);}
    let length=track.length_m().map_err(error)?;
    let start=track.sample(0.).map_err(error)?;
    let mut state=LapState::rolling(car,r.initial_speed_m_s)?;
    state.planar.x_m=start.position_m[0];state.planar.y_m=start.position_m[1];state.planar.heading_rad=start.heading_rad;
    let geom=suspension_forces(&car.suspension,state.displacement,[0.;3],0.,0.)?;
    let steer_gain=(geom.heading_jacobian[0][3]+geom.heading_jacobian[1][3])/2.;
    if steer_gain.abs()<1e-6 {return Err(error("front rack has no usable steering authority"));}
    let wheelbase=(geom.state.corners[0].points.wheel_center[0]+geom.state.corners[1].points.wheel_center[0]-geom.state.corners[2].points.wheel_center[0]-geom.state.corners[3].points.wheel_center[0])/2.;
    if wheelbase<=0. {return Err(error("front axle must be ahead of rear axle"));}
    let mut t=0.;let mut progress=0.;let mut next_record=0.;let mut rack=0.;
    let mut measured_start=if r.warmup_laps==0 {Some(0.)}else{None};
    let mut gear=car.powertrain.select_gear(car.driven_speed(&state)).map_err(error)?;
    let mut next_shift=0.;let mut torque_cut_until=0.;let mut peak_speed=0_f64;let mut peak_lat=0_f64;
    let target_distance=length*(r.warmup_laps+r.laps) as f64;
    while t<r.max_time_s-1e-10 {
        if cancelled(){run.termination="cancelled".into();break;}
        let projection=project(track,[state.planar.x_m,state.planar.y_m],Some(progress),(r.max_speed_m_s*r.dt_s*4.).max(5.));
        let (s,lateral)=match projection {Ok(p)=>p,Err(e)=>{run.termination=e.to_string();break;}};
        progress=s;
        if lateral.abs()>track.width_m/2. {run.termination="vehicle CG left track width".into();break;}
        let target=target_speed(track,r,progress)?;
        let look=track.sample(progress+r.lookahead_m).map_err(error)?;
        let dx=look.position_m[0]-state.planar.x_m;let dy=look.position_m[1]-state.planar.y_m;
        let body_y=-state.planar.heading_rad.sin()*dx+state.planar.heading_rad.cos()*dy;
        let steer=(2.*wheelbase*body_y/(dx*dx+dy*dy).max(0.01)).atan();
        let desired_rack=(steer/steer_gain).clamp(-car.steering_limit_m,car.steering_limit_m);
        let dt=r.dt_s.min(r.max_time_s-t);
        let rack_rate=((desired_rack-rack)/dt).clamp(-r.rack_rate_limit_m_s,r.rack_rate_limit_m_s);
        let driven=car.driven_speed(&state);
        let desired_gear=match car.powertrain.select_gear(driven){Ok(g)=>g,Err(e)=>{run.termination=e;break;}};
        if t>=next_shift && desired_gear!=gear {gear=desired_gear;next_shift=t+0.5;torque_cut_until=t+0.1;}
        // Invalid current gear at a speed boundary must shift rather than exceed redline.
        if car.powertrain.evaluate(driven,gear,0.).is_err(){gear=desired_gear;torque_cut_until=t+0.1;}
        let full=car.powertrain.evaluate(driven,gear,1.).map_err(error)?;
        let tractive=(0..4).map(|i|full.wheel_torques_nm[i]/car.wheels.iter().find(|w|w.id==IDS[i]).unwrap().tire.model.radius_m).sum::<f64>();
        let brake_capacity=car.brakes.wheel_torques(1.).map_err(error)?;
        let braking=(0..4).map(|i|brake_capacity[i]/car.wheels.iter().find(|w|w.id==IDS[i]).unwrap().tire.model.radius_m).sum::<f64>();
        let drag=0.5*car.aero.air_density_kg_m3*car.aero.reference_area_m2*car.aero.drag_coefficient*state.planar.u_m_s.powi(2);
        let demand=car.suspension.chassis.sprung_mass*r.speed_gain_per_s*(target-state.planar.u_m_s)+drag;
        let controls=LapControls{rack_front_m:rack,rack_rate_m_s:rack_rate,gear,
            throttle:if t<torque_cut_until {0.}else{(demand/tractive.max(1.)).clamp(0.,1.)},
            brake:(-demand/braking.max(1.)).clamp(0.,1.)};
        let evaluation=match evaluate_vehicle(car,&state,controls){Ok(e)=>e,Err(e)=>{run.termination=e.to_string();break;}};
        peak_speed=peak_speed.max(state.planar.u_m_s.hypot(state.planar.v_m_s));
        peak_lat=peak_lat.max((evaluation.derivative.v_m_s2+state.planar.yaw_rate_rad_s*state.planar.u_m_s).abs());
        if t>=next_record-1e-10 || progress>=target_distance {
            let analysis=if r.report_analysis {Some(crate::analyze(&car.suspension,&evaluation.suspension.state.motion)?)}else{None};
            run.samples.push(LapSample{time_s:t,distance_m:progress,lateral_error_m:lateral,target_speed_m_s:target,state:state.clone(),controls,evaluation,analysis});
            next_record=t+r.sample_interval_s;
        }
        if progress>=target_distance {run.completed=true;run.termination="completed".into();run.lap_time_s=Some(t-measured_start.unwrap_or(0.));break;}
        let before=progress;
        let advanced=match advance_vehicle(car,&state,controls,dt){Ok(s)=>s,Err(e)=>{run.termination=format!("integration: {e}");break;}};
        let new_progress=project(track,[advanced.planar.x_m,advanced.planar.y_m],Some(progress),(r.max_speed_m_s*dt*4.).max(5.))?.0;
        if new_progress<before-0.02 {run.termination="reverse centerline progress".into();break;}
        if measured_start.is_none() && new_progress>=length*r.warmup_laps as f64 {
            measured_start=Some(t+dt*(length*r.warmup_laps as f64-before)/(new_progress-before).max(1e-12));
        }
        if new_progress>=target_distance {
            let crossing=t+dt*(target_distance-before)/(new_progress-before).max(1e-12);
            run.completed=true;run.termination="completed".into();run.lap_time_s=Some(crossing-measured_start.unwrap_or(0.));
            let end_controls=LapControls{rack_front_m:rack+rack_rate*dt,..controls};
            let end=evaluate_vehicle(car,&advanced,end_controls)?;
            run.samples.push(LapSample{time_s:t+dt,distance_m:new_progress,lateral_error_m:project(track,[advanced.planar.x_m,advanced.planar.y_m],Some(new_progress),5.)?.1,
                target_speed_m_s:target,state:advanced,controls:end_controls,evaluation:end,analysis:None});break;
        }
        state=advanced;t+=dt;rack+=rack_rate*dt;progress=new_progress;
    }
    run.metrics.insert("peak_speed_m_s".into(),peak_speed);run.metrics.insert("peak_lateral_acceleration_m_s2".into(),peak_lat);
    if let Some(time)=run.lap_time_s {run.metrics.insert("lap_time_s".into(),time);run.metrics.insert("mean_speed_m_s".into(),length*r.laps as f64/time);}
    Ok(run)
}
/// Synchronous non-cancellable convenience entry point.
pub fn simulate_lap(car:&LapVehicle,track:&Track,r:&LapRequest)->Result<LapRun,Error> {
    simulate_lap_controlled(car,track,r,&||false)
}
