use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Metrics {
    pub camber_deg: f64,
    pub toe_deg: f64,
    pub caster_deg: f64,
    pub kpi_deg: f64,
    pub scrub_radius_m: Option<f64>,
    pub mechanical_trail_m: Option<f64>,
    pub shock_length_m: f64,
    pub shock_compression_m: f64,
    /// Shock compression per metre of upward wheel-center motion at fixed chassis/rack.
    /// Central difference at +/-10 micrometres; absent if either perturbation cannot close.
    pub motion_ratio: Option<f64>,
    pub upper_arm_angle_rad: f64,
    pub lower_arm_angle_rad: f64,
    pub rocker_angle_rad: f64,
}
pub(crate) fn measure(c: &crate::Corner, p: &crate::Points, angles: [f64; 3]) -> Metrics {
    use crate::kinematics::v;
    let side = match c.id {
        crate::CornerId::FrontLeft | crate::CornerId::RearLeft => 1.0,
        _ => -1.0,
    };
    let mut a = (v(p.spindle_axis[1]) - v(p.spindle_axis[0])).normalize();
    // Axis endpoints have no prescribed order. Select the outboard direction at the design pose.
    if (c.spindle_axis[1][1] - c.spindle_axis[0][1]) * side < 0.0 {
        a = -a;
    }
    let k = v(p.upper_ball) - v(p.lower_ball);
    let heading = side * a.cross(&nalgebra::Vector3::z());
    let contact = v(p.contact_point);
    let intersection = if k.z.abs() > 1e-12 {
        Some(v(p.lower_ball) + k * ((contact.z - p.lower_ball[2]) / k.z))
    } else {
        None
    };
    let shock = (v(p.rocker_shock) - v(p.shock_chassis)).norm();
    Metrics {
        camber_deg: -a.z.clamp(-1.0, 1.0).asin().to_degrees(),
        toe_deg: -side * heading.y.atan2(heading.x).to_degrees(),
        caster_deg: -k.x.atan2(k.z).to_degrees(),
        kpi_deg: -side * k.y.atan2(k.z).to_degrees(),
        scrub_radius_m: intersection.map(|i| side * (contact.y - i.y)),
        mechanical_trail_m: intersection.map(|i| i.x - contact.x),
        shock_length_m: shock,
        shock_compression_m: (v(c.rocker_shock) - v(c.shock_chassis)).norm() - shock,
        motion_ratio: None,
        upper_arm_angle_rad: angles[0],
        lower_arm_angle_rad: angles[1],
        rocker_angle_rad: angles[2],
    }
}
