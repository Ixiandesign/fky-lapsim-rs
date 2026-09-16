//! Per-corner alignment, rest-length, and linkage-angle metrics computed from a solved pose.
use serde::{Deserialize, Serialize};
/// Alignment, rest-length, and rocker/arm angle metrics for one solved corner. See
/// `docs/model-conventions.md` for sign conventions and the `_deg`/`_rad`/`_m` unit suffixes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Metrics {
    /// Camber, degrees; positive when the top of the tire tilts outward.
    pub camber_deg: f64,
    /// Toe, degrees; positive toe-in, separately signed for left/right wheels.
    pub toe_deg: f64,
    /// Caster, degrees; positive when the upper steering-axis point lies rearward
    /// of the lower point.
    pub caster_deg: f64,
    /// Kingpin inclination, degrees; positive when the upper steering-axis point
    /// lies inward.
    pub kpi_deg: f64,
    /// Signed distance from the steering-axis ground intersection to the contact
    /// point along the horizontal outboard wheel direction. `None` when the
    /// steering axis is (near-)horizontal and has no finite ground intersection.
    pub scrub_radius_m: Option<f64>,
    /// Steering-axis ground intersection minus contact point, projected along the
    /// horizontal wheel rolling heading. `None` under the same conditions as
    /// `scrub_radius_m`.
    pub mechanical_trail_m: Option<f64>,
    /// Exact static (manufactured) length of the upper wishbone's forward leg, metres.
    pub rest_upper_front_leg_m: f64,
    /// Exact static length of the upper wishbone's rearward leg, metres.
    pub rest_upper_rear_leg_m: f64,
    /// Exact static length of the lower wishbone's forward leg, metres.
    pub rest_lower_front_leg_m: f64,
    /// Exact static length of the lower wishbone's rearward leg, metres.
    pub rest_lower_rear_leg_m: f64,
    /// Exact static tie-rod length, metres.
    pub rest_tie_rod_m: f64,
    /// Exact static pushrod length, metres.
    pub rest_pushrod_m: f64,
    /// Exact static shock length (rocker pickup to chassis pickup), metres.
    pub rest_shock_length_m: f64,
    /// Solved shock length at this pose, metres.
    pub shock_length_m: f64,
    /// Static shock length minus solved shock length, metres; positive in compression.
    pub shock_compression_m: f64,
    /// Shock compression per metre of upward wheel-center motion at fixed chassis/rack.
    /// Central difference at +/-10 micrometres; absent if either perturbation cannot close.
    pub motion_ratio: Option<f64>,
    /// Upper wishbone rotation angle about its inner pivot axis, radians, relative
    /// to the design pose.
    pub upper_arm_angle_rad: f64,
    /// Lower wishbone rotation angle about its inner pivot axis, radians, relative
    /// to the design pose.
    pub lower_arm_angle_rad: f64,
    /// Rocker rotation angle about its chassis-fixed axis, radians, relative to the
    /// design pose.
    pub rocker_angle_rad: f64,
}
/// Compute [Metrics] for one corner from its solved [crate::Points] and the three
/// solved link angles `[upper_arm, lower_arm, rocker]`, radians. Called internally by
/// the kinematics solver; not normally invoked directly.
pub fn measure(c: &crate::Corner, p: &crate::Points, angles: [f64; 3]) -> Metrics {
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
        scrub_radius_m: intersection.filter(|_| heading.norm() > 1e-12).map(|i| {
            (contact - i).dot(&(nalgebra::Vector3::z().cross(&heading.normalize()) * side))
        }),
        mechanical_trail_m: intersection
            .filter(|_| heading.norm() > 1e-12)
            .map(|i| (i - contact).dot(&heading.normalize())),
        rest_upper_front_leg_m: (v(c.upper_ball) - v(c.upper_front)).norm(),
        rest_upper_rear_leg_m: (v(c.upper_ball) - v(c.upper_rear)).norm(),
        rest_lower_front_leg_m: (v(c.lower_ball) - v(c.lower_front)).norm(),
        rest_lower_rear_leg_m: (v(c.lower_ball) - v(c.lower_rear)).norm(),
        rest_tie_rod_m: (v(c.steering_outer) - v(c.steering_inner)).norm(),
        rest_pushrod_m: (v(c.pushrod_pickup) - v(c.rocker_pushrod)).norm(),
        rest_shock_length_m: (v(c.rocker_shock) - v(c.shock_chassis)).norm(),
        shock_length_m: shock,
        shock_compression_m: (v(c.rocker_shock) - v(c.shock_chassis)).norm() - shock,
        motion_ratio: None,
        upper_arm_angle_rad: angles[0],
        lower_arm_angle_rad: angles[1],
        rocker_angle_rad: angles[2],
    }
}
