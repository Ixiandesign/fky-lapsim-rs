//! Optional expanded analysis; never called by a dynamics stage.
use crate::{CornerId, Error, Motion, Project, VehicleState};
use serde::{Deserialize, Serialize};
/// A possibly-undefined derived quantity. Serializes as `{value, reason}`. An
/// undefined intersection or failed derivative perturbation is not a zero-valued
/// measurement: `value` is `None` and `reason` explains why.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionalValue<T> {
    /// The computed value, or `None` if undefined/unavailable.
    pub value: Option<T>,
    /// Explanation, present exactly when `value` is `None`.
    pub reason: Option<String>,
}
impl<T> OptionalValue<T> {
    fn known(v: T) -> Self {
        Self {
            value: Some(v),
            reason: None,
        }
    }
    fn absent(s: impl Into<String>) -> Self {
        Self {
            value: None,
            reason: Some(s.into()),
        }
    }
}
/// A projected front-view instantaneous center: the intersection of the projected
/// normals to the upper/lower ball-joint velocities in the world y-z plane, or a
/// direction when both centers project to lateral infinity. See [projected_center].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectedCenter {
    /// Finite intersection point `[y, z]`, metres, when one exists.
    pub point_yz_m: Option<[f64; 2]>,
    /// Direction `[y, z]` toward the center when it lies at infinity (parallel
    /// projected normals).
    pub direction_yz: Option<[f64; 2]>,
    /// Explanation, present whenever the center is undefined or at infinity.
    pub reason: Option<String>,
}
/// One corner's expanded geometry derivatives and projected instant center, from
/// [analyze]/[analyze_with_step].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CornerAnalysis {
    /// This corner's identity.
    pub id: CornerId,
    /// Name of the independent coordinate these derivatives are taken with respect
    /// to: upward world wheel-center z, at fixed chassis pose and rack.
    pub derivative_coordinate: String,
    /// Central-difference step used for these derivatives, metres.
    pub derivative_step_m: f64,
    /// Shock compression derivative with respect to upward world wheel-center
    /// motion, chassis and rack held fixed.
    pub motion_ratio: OptionalValue<f64>,
    /// Second derivative of shock compression with respect to the same coordinate, per metre.
    pub motion_ratio_gradient_per_m: OptionalValue<f64>,
    /// Tangent wheel-rate including preload-dependent geometric stiffness; see
    /// [crate::dynamics::wheel_rate].
    pub spring_wheel_rate_n_per_m: OptionalValue<f64>,
    /// Camber gain, degrees per metre of upward wheel-center motion.
    pub camber_gain_deg_per_m: OptionalValue<f64>,
    /// Toe gain, degrees per metre of upward wheel-center motion.
    pub toe_gain_deg_per_m: OptionalValue<f64>,
    /// Projected front-view instantaneous center for this corner.
    pub projected_front_view_ic: ProjectedCenter,
    /// Set when this corner's tire support point is not a unique/smooth function of
    /// the axle direction at this pose; mirrors [crate::CornerState::contact_ambiguity].
    pub contact_ambiguity: Option<String>,
    /// Scrub radius, metres; undefined under the same conditions as
    /// [crate::Metrics::scrub_radius_m].
    pub scrub_radius_m: OptionalValue<f64>,
    /// Mechanical trail, metres; undefined under the same conditions as
    /// [crate::Metrics::mechanical_trail_m].
    pub mechanical_trail_m: OptionalValue<f64>,
}
/// One axle's track and geometric roll center, from [analyze]/[analyze_with_step].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxleAnalysis {
    /// Signed world-y left-minus-right distance between wheel centers, metres.
    pub wheel_track_m: f64,
    /// Signed world-y left-minus-right distance between contact points, metres.
    pub contact_track_m: f64,
    /// Geometric (not force-based) roll center: intersection of left/right
    /// contact-to-projected-center lines in the world y-z plane. See
    /// [geometric_roll_center].
    pub geometric_roll_center_yz_m: OptionalValue<[f64; 2]>,
}
/// Complete expanded geometry analysis for one [Motion]: the underlying solved
/// [VehicleState] plus per-corner derivatives/centers and per-axle track/roll-center
/// results. Returned by [analyze]/[analyze_with_step].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Analysis {
    /// The solved vehicle state this analysis was computed from.
    pub state: VehicleState,
    /// Per-corner derivatives and projected instant center, in `state.corners` order.
    pub corners: [CornerAnalysis; 4],
    /// Front-axle track and geometric roll center.
    pub front: AxleAnalysis,
    /// Rear-axle track and geometric roll center.
    pub rear: AxleAnalysis,
    /// Signed world-x front-minus-rear distance between left wheel centers, metres.
    pub left_wheelbase_m: f64,
    /// Signed world-x front-minus-rear distance between right wheel centers, metres.
    pub right_wheelbase_m: f64,
    /// Signed world-x front-minus-rear distance between left contact points, metres.
    pub left_contact_wheelbase_m: f64,
    /// Signed world-x front-minus-rear distance between right contact points, metres.
    pub right_contact_wheelbase_m: f64,
}
/// Tangent stiffness including preload-dependent geometry.
pub use crate::dynamics::wheel_rate;
/// Intersection of projected normals to the two ball-joint velocities.
pub fn projected_center(
    upper: [f64; 2],
    lower: [f64; 2],
    vu: [f64; 2],
    vl: [f64; 2],
) -> ProjectedCenter {
    let absent = |s: &str| ProjectedCenter {
        point_yz_m: None,
        direction_yz: None,
        reason: Some(s.into()),
    };
    if !upper
        .iter()
        .chain(lower.iter())
        .chain(vu.iter())
        .chain(vl.iter())
        .all(|v| v.is_finite())
    {
        return absent("nonfinite projected geometry");
    }
    let nu = vu[0].hypot(vu[1]);
    let nl = vl[0].hypot(vl[1]);
    if nu < 1e-10 || nl < 1e-10 {
        return absent("zero projected ball-joint velocity");
    }
    let du = [-vu[1] / nu, vu[0] / nu];
    let dl = [-vl[1] / nl, vl[0] / nl];
    let cross = |a: [f64; 2], b: [f64; 2]| a[0] * b[1] - a[1] * b[0];
    let det = cross(du, dl);
    let delta = [lower[0] - upper[0], lower[1] - upper[1]];
    if det.abs() < 1e-8 {
        if cross(delta, du).abs() < 1e-10 {
            return absent("coincident projected normals: center not unique");
        }
        return ProjectedCenter {
            point_yz_m: None,
            direction_yz: Some(du),
            reason: Some("parallel projected normals: center at infinity".into()),
        };
    }
    let t = cross(delta, dl) / det;
    ProjectedCenter {
        point_yz_m: Some([upper[0] + t * du[0], upper[1] + t * du[1]]),
        direction_yz: None,
        reason: None,
    }
}
/// Geometric intersection of contact-to-IC lines in world front-view coordinates.
pub fn geometric_roll_center(
    left_contact: [f64; 2],
    left: &ProjectedCenter,
    right_contact: [f64; 2],
    right: &ProjectedCenter,
) -> OptionalValue<[f64; 2]> {
    let direction = |c: [f64; 2], ic: &ProjectedCenter| {
        ic.point_yz_m
            .map(|p| [p[0] - c[0], p[1] - c[1]])
            .or(ic.direction_yz)
    };
    let (Some(a), Some(b)) = (
        direction(left_contact, left),
        direction(right_contact, right),
    ) else {
        return OptionalValue::absent("undefined projected instantaneous center");
    };
    let na = a[0].hypot(a[1]);
    let nb = b[0].hypot(b[1]);
    if na < 1e-12 || nb < 1e-12 {
        return OptionalValue::absent("contact coincides with projected center");
    }
    let a = a.map(|x| x / na);
    let b = b.map(|x| x / nb);
    let d = [
        right_contact[0] - left_contact[0],
        right_contact[1] - left_contact[1],
    ];
    let cross = |a: [f64; 2], b: [f64; 2]| a[0] * b[1] - a[1] * b[0];
    let det = cross(a, b);
    if det.abs() < 1e-8 {
        // Both ICs at lateral infinity define the same horizontal line. The axle
        // contact midpoint selects the symmetry-plane intersection of this limit.
        if left.direction_yz.is_some()
            && right.direction_yz.is_some()
            && a[1].abs() < 1e-8
            && b[1].abs() < 1e-8
            && d[1].abs() < 1e-8
        {
            return OptionalValue::known([
                (left_contact[0] + right_contact[0]) / 2.0,
                left_contact[1],
            ]);
        }
        return OptionalValue::absent("parallel or coincident contact-to-center lines");
    }
    let t = cross(d, b) / det;
    OptionalValue::known([left_contact[0] + t * a[0], left_contact[1] + t * a[1]])
}
/// Solve `m` and compute [Analysis]: per-axle track/wheelbase, motion-ratio and
/// camber/toe gradients, wheel rate, and projected instant/roll centers, using a
/// default 0.2 mm wheel-height derivative step. See [analyze_with_step] to control
/// that step explicitly.
pub fn analyze(p: &Project, m: &Motion) -> Result<Analysis, Error> {
    analyze_with_step(p, m, 0.0002)
}
/// Like [analyze], with an explicit wheel-height derivative step `h` (metres, must be
/// finite and in `1e-6..=0.01`) for refinement studies. Smaller values are not
/// automatically more accurate; check sensitivity to `h` before trusting a gradient.
pub fn analyze_with_step(p: &Project, m: &Motion, h: f64) -> Result<Analysis, Error> {
    use crate::kinematics::{at_world_height, err, v, Frame, V};
    use nalgebra::UnitQuaternion;
    if !h.is_finite() || !(1e-6..=0.01).contains(&h) {
        return Err(err("analysis step must be between 1e-6 and 0.01 metres"));
    }
    let mut state = crate::study::simulate_on_road_mode(p, m, [0.0; 4], false)?;
    let rotation = UnitQuaternion::from_axis_angle(&V::y_axis(), m.pitch)
        * UnitQuaternion::from_axis_angle(&V::x_axis(), m.roll);
    let center = v(p.chassis.center_of_mass);
    let frame = Frame {
        rotation,
        translation: center - rotation * center + V::new(0.0, 0.0, m.heave),
    };
    let mut corners = Vec::new();
    for (c, s) in p.corners.iter().zip(&state.corners) {
        let rack = match c.id {
            CornerId::FrontLeft | CornerId::FrontRight => m.rack_front,
            _ => m.rack_rear,
        };
        let mut out = CornerAnalysis {
            id: c.id,
            derivative_coordinate: "upward_world_wheel_center_z_at_fixed_chassis_and_rack".into(),
            derivative_step_m: h,
            motion_ratio: OptionalValue::absent("perturbation failed"),
            motion_ratio_gradient_per_m: OptionalValue::absent("perturbation failed"),
            spring_wheel_rate_n_per_m: OptionalValue::absent("perturbation failed"),
            camber_gain_deg_per_m: OptionalValue::absent("perturbation failed"),
            toe_gain_deg_per_m: OptionalValue::absent("perturbation failed"),
            projected_front_view_ic: ProjectedCenter {
                point_yz_m: None,
                direction_yz: None,
                reason: Some("perturbation failed".into()),
            },
            contact_ambiguity: None,
            scrub_radius_m: s.metrics.scrub_radius_m.map_or_else(
                || {
                    OptionalValue::absent(
                        "steering axis parallel to ground or horizontal wheel heading undefined",
                    )
                },
                OptionalValue::known,
            ),
            mechanical_trail_m: s.metrics.mechanical_trail_m.map_or_else(
                || {
                    OptionalValue::absent(
                        "steering axis parallel to ground or horizontal wheel heading undefined",
                    )
                },
                OptionalValue::known,
            ),
        };
        out.contact_ambiguity = s.contact_ambiguity.clone();
        let z = s.points.wheel_center[2];
        let samples = (
            at_world_height(c, s, &frame, rack, z - h),
            at_world_height(c, s, &frame, rack, z),
            at_world_height(c, s, &frame, rack, z + h),
        );
        match samples {
            (Ok(lo), Ok(mid), Ok(hi)) => {
                let a = lo.metrics.shock_compression_m;
                let b = mid.metrics.shock_compression_m;
                let d = hi.metrics.shock_compression_m;
                let ratio = (d - a) / (2.0 * h);
                let gradient = (d - 2.0 * b + a) / (h * h);
                out.motion_ratio = OptionalValue::known(ratio);
                out.motion_ratio_gradient_per_m = OptionalValue::known(gradient);
                out.camber_gain_deg_per_m = OptionalValue::known(
                    (hi.metrics.camber_deg - lo.metrics.camber_deg) / (2.0 * h),
                );
                let delta =
                    (hi.metrics.toe_deg - lo.metrics.toe_deg + 180.0).rem_euclid(360.0) - 180.0;
                out.toe_gain_deg_per_m = OptionalValue::known(delta / (2.0 * h));
                let spring = &c.spring_damper;
                let tangent = if let Some(curve) = &spring.spring_curve {
                    if curve.iter().any(|point| (point[0] - b).abs() < 1e-10) {
                        None
                    } else {
                        Some(
                            curve
                                .windows(2)
                                .find(|w| b > w[0][0] && b < w[1][0])
                                .map_or(0.0, |w| (w[1][1] - w[0][1]) / (w[1][0] - w[0][0])),
                        )
                    }
                } else {
                    Some(spring.spring_rate)
                };
                out.spring_wheel_rate_n_per_m = match tangent {
                    Some(k) => OptionalValue::known(wheel_rate(
                        k,
                        crate::dynamics::spring_force(spring, b),
                        ratio,
                        gradient,
                    )),
                    None => OptionalValue::absent("spring force table knot: tangent undefined"),
                };
                let yz = |p: crate::Point| [p[1], p[2]];
                let velocity = |a: crate::Point, b: crate::Point| {
                    [(a[1] - b[1]) / (2.0 * h), (a[2] - b[2]) / (2.0 * h)]
                };
                out.projected_front_view_ic = projected_center(
                    yz(mid.points.upper_ball),
                    yz(mid.points.lower_ball),
                    velocity(hi.points.upper_ball, lo.points.upper_ball),
                    velocity(hi.points.lower_ball, lo.points.lower_ball),
                );
            }
            tuple => {
                let reason = [tuple.0.err(), tuple.1.err(), tuple.2.err()]
                    .into_iter()
                    .flatten()
                    .map(|e| e.message)
                    .collect::<Vec<_>>()
                    .join("; ");
                out.motion_ratio.reason = Some(reason.clone());
                out.motion_ratio_gradient_per_m.reason = Some(reason.clone());
                out.spring_wheel_rate_n_per_m.reason = Some(reason.clone());
                out.camber_gain_deg_per_m.reason = Some(reason.clone());
                out.toe_gain_deg_per_m.reason = Some(reason.clone());
                out.projected_front_view_ic.reason = Some(reason);
            }
        }
        corners.push(out);
    }
    for (s, a) in state.corners.iter_mut().zip(&corners) {
        s.metrics.motion_ratio = a.motion_ratio.value;
    }
    let index = |id| p.corners.iter().position(|c| c.id == id).unwrap();
    let fl = index(CornerId::FrontLeft);
    let fr = index(CornerId::FrontRight);
    let rl = index(CornerId::RearLeft);
    let rr = index(CornerId::RearRight);
    let axle = |l: usize, r: usize| {
        let lp = &state.corners[l].points;
        let rp = &state.corners[r].points;
        AxleAnalysis {
            wheel_track_m: lp.wheel_center[1] - rp.wheel_center[1],
            contact_track_m: lp.contact_point[1] - rp.contact_point[1],
            geometric_roll_center_yz_m: geometric_roll_center(
                [lp.contact_point[1], lp.contact_point[2]],
                &corners[l].projected_front_view_ic,
                [rp.contact_point[1], rp.contact_point[2]],
                &corners[r].projected_front_view_ic,
            ),
        }
    };
    let front = axle(fl, fr);
    let rear = axle(rl, rr);
    let left_wheelbase_m =
        state.corners[fl].points.wheel_center[0] - state.corners[rl].points.wheel_center[0];
    let right_wheelbase_m =
        state.corners[fr].points.wheel_center[0] - state.corners[rr].points.wheel_center[0];
    let left_contact_wheelbase_m =
        state.corners[fl].points.contact_point[0] - state.corners[rl].points.contact_point[0];
    let right_contact_wheelbase_m =
        state.corners[fr].points.contact_point[0] - state.corners[rr].points.contact_point[0];
    Ok(Analysis {
        state,
        corners: corners.try_into().unwrap(),
        front,
        rear,
        left_wheelbase_m,
        right_wheelbase_m,
        left_contact_wheelbase_m,
        right_contact_wheelbase_m,
    })
}
