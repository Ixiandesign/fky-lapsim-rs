//! Suspension virtual-work mapping for the massless-link four-wheel model.
use crate::{Project,VehicleState,Motion,CornerId,Error};
use crate::dynamics::{spring_force,damper_force,spring_energy};
use serde::{Serialize,Deserialize};

pub(crate) const IDS:[CornerId;4]=[CornerId::FrontLeft,CornerId::FrontRight,CornerId::RearLeft,CornerId::RearRight];
pub(crate) fn error(message:impl Into<String>)->Error {Error{message:message.into()}}
fn solve(p:&Project,x:[f64;4])->Result<VehicleState,Error> {
    let mut s=crate::study::simulate_on_road_mode(p,&Motion{heave:x[0],roll:x[1],pitch:x[2],rack_front:x[3],rack_rear:0.},[0.;4],false)?;
    s.corners.sort_by_key(|c|IDS.iter().position(|id|*id==c.id).unwrap());
    Ok(s)
}
fn channel(s:&VehicleState,i:usize,which:usize)->f64 {
    let m=&s.corners[i].metrics;
    match which {0=>m.shock_compression_m,1=>m.heave_arm_compression_m.unwrap_or(0.),_=>m.roll_arm_compression_m.unwrap_or(0.)}
}
pub(crate) fn heading(s:&VehicleState,i:usize)->f64 {
    let side=if i%2==0 {1.} else {-1.};
    -side*s.corners[i].metrics.toe_deg.to_radians()
}
fn wrapped(x:f64)->f64 {(x+std::f64::consts::PI).rem_euclid(std::f64::consts::TAU)-std::f64::consts::PI}
/// Nonlinear spring/damper virtual work. Four-corner arrays always FL, FR, RL, RR.
/// Derivative coordinate order is heave, roll, pitch, front rack (m, rad, rad, m).
#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct SuspensionForces {
    /// Solved pose (same canonical corner order).
    pub state:VehicleState,
    /// Gravity plus spring/damper generalized forces, before tire/aero coupling.
    pub generalized_force:[f64;3],
    /// Normal reactions before tire-force jacking, N.
    pub normal_load_n:[f64;4],
    /// Corner shock force, N.
    pub shock_force_n:[f64;4],
    /// Corner shock compression velocity, m/s, including front-rack motion.
    pub compression_velocity_m_s:[f64;4],
    /// Potential energy of springs and chassis gravity, J, arbitrary datum.
    pub potential_energy_j:f64,
    /// Sum of passive damper force times speed, W.
    pub dissipated_power_w:f64,
    /// Contact-position derivative per wheel, coordinate, xyz.
    pub contact_jacobian:[[[f64;3];4];4],
    /// Wheel heading derivative per wheel and coordinate, radians/unit.
    pub heading_jacobian:[[f64;4];4],
}
/// Exact rigid linkage closure plus central-difference virtual work. Fixed flat
/// contacts, massless links; no empirical lateral load-transfer distribution.
/// Road-height derivative is minus heave derivative for each independent corner.
pub fn suspension_forces(p:&Project,q:[f64;3],v:[f64;3],rack:f64,rack_rate:f64)->Result<SuspensionForces,Error> {
    p.validate()?;
    if !q.iter().chain(v.iter()).chain([rack,rack_rate].iter()).all(|x|x.is_finite()) {
        return Err(error("nonfinite suspension state"));
    }
    let x=[q[0],q[1],q[2],rack]; let rate=[v[0],v[1],v[2],rack_rate];
    let state=solve(p,x)?;
    let mut cj=[[[0.;4];4];3];
    let mut contact_jacobian=[[[0.;3];4];4]; let mut heading_jacobian=[[0.;4];4];
    let h=0.0001;
    for a in 0..4 {
        let mut xp=x;let mut xm=x;xp[a]+=h;xm[a]-=h;
        let plus=solve(p,xp)?;let minus=solve(p,xm)?;
        for i in 0..4 {
            for k in 0..3 {
                cj[k][i][a]=(channel(&plus,i,k)-channel(&minus,i,k))/(2.*h);
                contact_jacobian[i][a][k]=(plus.corners[i].points.contact_point[k]-minus.corners[i].points.contact_point[k])/(2.*h);
            }
            heading_jacobian[i][a]=wrapped(heading(&plus,i)-heading(&minus,i))/(2.*h);
        }
    }
    let mut out=SuspensionForces {state,generalized_force:[-p.chassis.sprung_mass*9.81,0.,0.],normal_load_n:[0.;4],shock_force_n:[0.;4],
        compression_velocity_m_s:[0.;4],potential_energy_j:p.chassis.sprung_mass*9.81*q[0],dissipated_power_w:0.,contact_jacobian,heading_jacobian};
    for i in 0..4 {
        let corner=p.corners.iter().find(|c|c.id==IDS[i]).unwrap();
        let law=&corner.spring_damper;let m=&out.state.corners[i].metrics;
        if law.min_length_m.is_some_and(|min|m.shock_length_m<min) || law.max_length_m.is_some_and(|max|m.shock_length_m>max) {
            return Err(error(format!("{:?}: shock travel limit reached",corner.id)));
        }
        let speed=(0..4).map(|a|cj[0][i][a]*rate[a]).sum();
        let force=spring_force(law,m.shock_compression_m)+damper_force(law,speed);
        out.shock_force_n[i]=force;out.compression_velocity_m_s[i]=speed;
        out.normal_load_n[i]=-force*cj[0][i][0];
        for a in 0..3 {out.generalized_force[a]-=force*cj[0][i][a];}
        out.potential_energy_j+=spring_energy(law,m.shock_compression_m);
        out.dissipated_power_w+=damper_force(law,speed)*speed;
    }
    for (axle,ai) in [&p.front_interconnect,&p.rear_interconnect].iter().enumerate() {
        let Some(ai)=ai else {continue}; let l=2*axle;let r=l+1;
        for (k,law,sign) in [(1,&ai.heave,1.),(2,&ai.roll,-1.)] {
            let c=(channel(&out.state,l,k)+sign*channel(&out.state,r,k))/2.;
            let j:[f64;4]=std::array::from_fn(|a|(cj[k][l][a]+sign*cj[k][r][a])/2.);
            let speed=(0..4).map(|a|j[a]*rate[a]).sum();
            let force=spring_force(law,c)+damper_force(law,speed);
            for a in 0..3 {out.generalized_force[a]-=j[a]*force;}
            out.normal_load_n[l]-=0.5*cj[k][l][0]*force;
            out.normal_load_n[r]-=0.5*sign*cj[k][r][0]*force;
            out.potential_energy_j+=spring_energy(law,c);
            out.dissipated_power_w+=damper_force(law,speed)*speed;
        }
    }
    Ok(out)
}
