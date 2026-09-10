use crate::{Corner, CornerId, Error, Metrics, Point};
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Points {
    pub upper_front: Point,
    pub upper_rear: Point,
    pub lower_front: Point,
    pub lower_rear: Point,
    pub upper_ball: Point,
    pub lower_ball: Point,
    pub steering_inner: Point,
    pub steering_outer: Point,
    pub wheel_center: Point,
    pub spindle_axis: [Point; 2],
    pub pushrod_pickup: Point,
    pub rocker_axis: [Point; 2],
    pub rocker_pushrod: Point,
    pub rocker_shock: Point,
    pub shock_chassis: Point,
    pub contact_point: Point,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CornerState {
    pub id: CornerId,
    pub points: Points,
    pub orientation: [f64; 4],
    pub contact_ambiguity: Option<String>,
    pub metrics: Metrics,
    pub max_residual_m: f64,
    pub iterations: usize,
}
impl CornerState {
    /// Map a design-pose knuckle attachment into this state's coordinate frame.
    /// `solve_corner` returns chassis coordinates; vehicle studies return world coordinates.
    pub fn transform_knuckle_point(&self, corner: &Corner, point: Point) -> Point {
        let [w, x, y, z] = self.orientation;
        let q = UnitQuaternion::new_normalize(nalgebra::Quaternion::new(w, x, y, z));
        (v(self.points.wheel_center) + q * (v(point) - v(corner.wheel_center))).into()
    }
}
use nalgebra::{SMatrix, SVector, Unit, UnitQuaternion, Vector3};
pub(crate) type V = Vector3<f64>;
type R6 = SVector<f64, 6>;
pub(crate) fn v(p: Point) -> V {
    V::from(p)
}
pub(crate) fn err(s: &str) -> Error {
    Error { message: s.into() }
}
#[derive(Clone)]
pub(crate) struct Frame {
    pub rotation: UnitQuaternion<f64>,
    pub translation: V,
}
impl Frame {
    pub(crate) fn identity() -> Self {
        Self {
            rotation: UnitQuaternion::identity(),
            translation: V::zeros(),
        }
    }
    pub(crate) fn point(&self, p: V) -> V {
        self.rotation * p + self.translation
    }
}
#[derive(Clone)]
struct Pose {
    q: UnitQuaternion<f64>,
    t: V,
}
impl Pose {
    fn point(&self, c: &Corner, p: Point) -> V {
        v(c.wheel_center) + self.t + self.q * (v(p) - v(c.wheel_center))
    }
    fn step(&self, d: R6) -> Self {
        Self {
            t: self.t + d.fixed_rows::<3>(0).into_owned(),
            q: UnitQuaternion::from_scaled_axis(d.fixed_rows::<3>(3).into_owned()) * self.q,
        }
    }
}
#[derive(Clone)]
pub(crate) enum Constraint {
    Jounce(f64),
    Road(Frame, f64),
    WorldHeight(Frame, f64),
}
fn rack_offset(c: &Corner, rack: f64) -> V {
    (v(c.rack_axis[1]) - v(c.rack_axis[0])).normalize() * rack
}
fn contact(center: V, axis: V, c: &Corner) -> V {
    let down = V::z() - axis * axis.z;
    let n = down.norm();
    let radial = if n < 1e-14 && c.tire_profile == crate::TireProfile::Torus {
        axis.cross(&V::y()).normalize()
    } else if n < 1e-14 {
        V::zeros()
    } else {
        down / n
    };
    match c.tire_profile {
        crate::TireProfile::Disk => center - radial * c.tire_radius,
        crate::TireProfile::Cylinder => {
            center
                - radial * c.tire_radius
                - axis
                    * (c.tire_width / 2.0)
                    * if axis.z.abs() < 1e-14 {
                        0.0
                    } else {
                        axis.z.signum()
                    }
        }
        crate::TireProfile::Torus => {
            center - radial * (c.tire_radius - c.tire_width / 2.0) - V::z() * (c.tire_width / 2.0)
        }
    }
}
fn contact_ambiguity(c: &Corner, axis: V) -> Option<String> {
    if axis.z.abs() > 1.0 - 1e-12 {
        Some(
            if c.tire_profile == crate::TireProfile::Torus {
                "vertical axle: torus ring support is nonunique; representative ring point selected"
            } else {
                "vertical axle: radial support is nonunique; center of support set selected"
            }
            .into(),
        )
    } else if c.tire_profile == crate::TireProfile::Cylinder && axis.z.abs() < 1e-12 {
        Some(
            "horizontal cylinder axle: contact line midpoint selected; support has a camber cusp"
                .into(),
        )
    } else {
        None
    }
}
fn residual(c: &Corner, p: &Pose, rack: f64, goal: &Constraint) -> R6 {
    let u = p.point(c, c.upper_ball);
    let l = p.point(c, c.lower_ball);
    let so = p.point(c, c.steering_outer);
    let mut r = R6::zeros();
    for (i, (b, a, rest)) in [
        (u, c.upper_front, c.upper_ball),
        (u, c.upper_rear, c.upper_ball),
        (l, c.lower_front, c.lower_ball),
        (l, c.lower_rear, c.lower_ball),
        (so, c.steering_inner, c.steering_outer),
    ]
    .into_iter()
    .enumerate()
    {
        let moved = v(a)
            + if i == 4 {
                rack_offset(c, rack)
            } else {
                V::zeros()
            };
        r[i] = (b - moved).norm() - (v(rest) - v(a)).norm();
    }
    r[5] = match goal {
        Constraint::Jounce(j) => p.t.z - j,
        Constraint::WorldHeight(f, z) => f.point(p.point(c, c.wheel_center)).z - z,
        Constraint::Road(f, height) => {
            let center = f.point(p.point(c, c.wheel_center));
            let axis = f.rotation * p.q * (v(c.spindle_axis[1]) - v(c.spindle_axis[0])).normalize();
            contact(center, axis, c).z - height
        }
    };
    r
}
fn newton(c: &Corner, pose: &mut Pose, rack: f64, goal: &Constraint) -> Result<usize, Error> {
    newton_tolerance(c, pose, rack, goal, 1e-8)
}
fn newton_tolerance(
    c: &Corner,
    pose: &mut Pose,
    rack: f64,
    goal: &Constraint,
    tolerance: f64,
) -> Result<usize, Error> {
    for iteration in 0..80 {
        let r = residual(c, pose, rack, goal);
        if !r.iter().all(|x| x.is_finite()) {
            return Err(err("nonfinite closure residual"));
        }
        if r.amax() <= tolerance {
            // With an axial inner tie joint, rotation about the kingpin leaves all
            // five link constraints unchanged. If that axis is vertical in the
            // constraint frame, both wheel height and disk support height are
            // unchanged too: this is an actual free DOF, not an isolated toggle.
            let lower = pose.point(c, c.lower_ball);
            let kingpin = (pose.point(c, c.upper_ball) - lower).normalize();
            let inner = v(c.steering_inner) + rack_offset(c, rack);
            let vertical = match goal {
                Constraint::Jounce(_) => V::z(),
                Constraint::Road(frame, _) | Constraint::WorldHeight(frame, _) => {
                    frame.rotation.inverse() * V::z()
                }
            };
            if (inner - lower).cross(&kingpin).norm() < 1e-10
                && kingpin.cross(&vertical).norm() < 1e-10
            {
                return Err(err(
                    "underdetermined steering: free rotation about vertical kingpin",
                ));
            }
            return Ok(iteration);
        }
        let mut j = SMatrix::<f64, 6, 6>::zeros();
        for k in 0..6 {
            let mut d = R6::zeros();
            d[k] = 1e-6;
            j.set_column(
                k,
                &((residual(c, &pose.step(d), rack, goal)
                    - residual(c, &pose.step(-d), rack, goal))
                    / 2e-6),
            );
        }
        let mut accepted = None;
        // SVD gives a minimum-norm Newton step at rank loss. Damped normal steps provide a trust-region fallback.
        for damping in [0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1.0] {
            let delta = if damping == 0.0 {
                j.svd(true, true).solve(&(-r), 1e-12).ok()
            } else {
                (j.transpose() * j + SMatrix::<f64, 6, 6>::identity() * damping)
                    .qr()
                    .solve(&(-j.transpose() * r))
            };
            if let Some(d) = delta {
                for n in 0..16 {
                    let candidate = pose.step(d * 0.5_f64.powi(n));
                    if residual(c, &candidate, rack, goal).norm_squared() < r.norm_squared() {
                        accepted = Some(candidate);
                        break;
                    }
                }
            }
            if accepted.is_some() {
                break;
            }
        }
        if let Some(next) = accepted {
            *pose = next;
        } else {
            return Err(err("kinematic closure stalled or travel unreachable"));
        }
    }
    Err(err("kinematic closure iteration limit"))
}
fn arm_angle(a: Point, b: Point, rest: Point, moved: V) -> f64 {
    let axis = (v(b) - v(a)).normalize();
    let r = v(rest) - v(a);
    let r = r - axis * r.dot(&axis);
    let m = moved - v(a);
    let m = m - axis * m.dot(&axis);
    axis.dot(&r.cross(&m)).atan2(r.dot(&m))
}
fn rotate_axis(p: Point, axis: [Point; 2], angle: f64) -> V {
    v(axis[0])
        + UnitQuaternion::from_axis_angle(&Unit::new_normalize(v(axis[1]) - v(axis[0])), angle)
            * (v(p) - v(axis[0]))
}
fn rocker(c: &Corner, pickup: V, previous: f64) -> Result<f64, Error> {
    let a = v(c.rocker_axis[0]);
    let axis = (v(c.rocker_axis[1]) - a).normalize();
    let r = v(c.rocker_pushrod) - a;
    let parallel = axis * r.dot(&axis);
    let radial = r - parallel;
    let delta = a + parallel - pickup;
    let aa = 2.0 * delta.dot(&radial);
    let bb = 2.0 * delta.dot(&axis.cross(&radial));
    let length = (v(c.rocker_pushrod) - v(c.pushrod_pickup)).norm();
    let cc = length * length - delta.norm_squared() - radial.norm_squared();
    let amplitude = aa.hypot(bb);
    if amplitude < 1e-14 {
        if cc.abs() < 1e-12 {
            return Err(err(
                "underdetermined rocker: pushrod length does not constrain angle",
            ));
        }
        return Err(err("pushrod cannot reach rocker"));
    }
    let ratio = cc / amplitude;
    if ratio.abs() > 1.0 + 1e-10 {
        return Err(err("pushrod cannot reach rocker"));
    }
    let phase = bb.atan2(aa);
    let alpha = ratio.clamp(-1.0, 1.0).acos();
    let tau = std::f64::consts::TAU;
    let nearest = |a: f64| a + ((previous - a) / tau).round() * tau;
    let a = nearest(phase + alpha);
    let b = nearest(phase - alpha);
    Ok(if (a - previous).abs() < (b - previous).abs() {
        a
    } else {
        b
    })
}
fn state(
    c: &Corner,
    p: &Pose,
    rack: f64,
    goal: &Constraint,
    previous: f64,
    iterations: usize,
) -> Result<CornerState, Error> {
    let ua = arm_angle(
        c.upper_front,
        c.upper_rear,
        c.upper_ball,
        p.point(c, c.upper_ball),
    );
    let la = arm_angle(
        c.lower_front,
        c.lower_rear,
        c.lower_ball,
        p.point(c, c.lower_ball),
    );
    let pickup = match c.pushrod_body {
        crate::PushrodBody::UpperArm => {
            rotate_axis(c.pushrod_pickup, [c.upper_front, c.upper_rear], ua)
        }
        crate::PushrodBody::LowerArm => {
            rotate_axis(c.pushrod_pickup, [c.lower_front, c.lower_rear], la)
        }
        crate::PushrodBody::Knuckle => p.point(c, c.pushrod_pickup),
    };
    let ra = rocker(c, pickup, previous)?;
    let identity = Frame::identity();
    let f = match goal {
        Constraint::Jounce(_) => &identity,
        Constraint::Road(f, _) | Constraint::WorldHeight(f, _) => f,
    };
    let fixed = |a: Point| -> Point { f.point(v(a)).into() };
    let moving = |a: Point| -> Point { f.point(p.point(c, a)).into() };
    let mut points = Points {
        upper_front: fixed(c.upper_front),
        upper_rear: fixed(c.upper_rear),
        lower_front: fixed(c.lower_front),
        lower_rear: fixed(c.lower_rear),
        upper_ball: moving(c.upper_ball),
        lower_ball: moving(c.lower_ball),
        steering_inner: f.point(v(c.steering_inner) + rack_offset(c, rack)).into(),
        steering_outer: moving(c.steering_outer),
        wheel_center: moving(c.wheel_center),
        spindle_axis: c.spindle_axis.map(moving),
        pushrod_pickup: f.point(pickup).into(),
        rocker_axis: c.rocker_axis.map(fixed),
        rocker_pushrod: f
            .point(rotate_axis(c.rocker_pushrod, c.rocker_axis, ra))
            .into(),
        rocker_shock: f
            .point(rotate_axis(c.rocker_shock, c.rocker_axis, ra))
            .into(),
        shock_chassis: fixed(c.shock_chassis),
        contact_point: [0.0; 3],
    };
    let axis = (v(points.spindle_axis[1]) - v(points.spindle_axis[0])).normalize();
    points.contact_point = contact(v(points.wheel_center), axis, c).into();
    let metrics = crate::metrics::measure(c, &points, [ua, la, ra]);
    let q = f.rotation * p.q;
    let q = q.quaternion();
    let push_error = ((v(points.pushrod_pickup) - v(points.rocker_pushrod)).norm()
        - (v(c.pushrod_pickup) - v(c.rocker_pushrod)).norm())
    .abs();
    let max_residual_m = residual(c, p, rack, goal).amax().max(push_error);
    if max_residual_m > 1e-8 {
        return Err(err("link closure exceeds tolerance"));
    }
    Ok(CornerState {
        id: c.id,
        contact_ambiguity: contact_ambiguity(c, axis),
        points,
        orientation: [q.w, q.i, q.j, q.k],
        metrics,
        max_residual_m,
        iterations,
    })
}
pub(crate) fn continuation(
    c: &Corner,
    steps: usize,
    goal: impl Fn(f64) -> (f64, Constraint),
) -> Result<CornerState, Error> {
    continuation_mode(c, steps, goal, true)
}
pub(crate) fn continuation_mode(
    c: &Corner,
    steps: usize,
    goal: impl Fn(f64) -> (f64, Constraint),
    measure_motion_ratio: bool,
) -> Result<CornerState, Error> {
    continuation_tolerance(c, steps, goal, measure_motion_ratio, 1e-8)
}
pub(crate) fn continuation_tolerance(
    c: &Corner,
    steps: usize,
    goal: impl Fn(f64) -> (f64, Constraint),
    measure_motion_ratio: bool,
    tolerance: f64,
) -> Result<CornerState, Error> {
    c.validate()?;
    let mut pose = Pose {
        q: UnitQuaternion::identity(),
        t: V::zeros(),
    };
    let mut rocker_angle = 0.0;
    let mut total = 0;
    let mut result = None;
    for i in 0..=steps {
        let (rack, target) = goal(i as f64 / steps as f64);
        total += newton_tolerance(c, &mut pose, rack, &target, tolerance)?;
        let s = state(c, &pose, rack, &target, rocker_angle, total)?;
        rocker_angle = s.metrics.rocker_angle_rad;
        result = Some(s);
    }
    let mut result = result.ok_or_else(|| err("empty continuation"))?;
    if !measure_motion_ratio {
        return Ok(result);
    }
    let (rack, target) = goal(1.0);
    let h = 1e-5;
    let perturbed = |dz: f64| -> Result<f64, Error> {
        let target = match &target {
            Constraint::Jounce(j) => Constraint::Jounce(j + dz),
            Constraint::Road(f, _) | Constraint::WorldHeight(f, _) => {
                Constraint::WorldHeight(f.clone(), result.points.wheel_center[2] + dz)
            }
        };
        let mut pose = pose.clone();
        newton(c, &mut pose, rack, &target)?;
        Ok(state(c, &pose, rack, &target, rocker_angle, 0)?
            .metrics
            .shock_compression_m)
    };
    result.metrics.motion_ratio = match (perturbed(h), perturbed(-h)) {
        (Ok(a), Ok(b)) => Some((a - b) / (2.0 * h)),
        _ => None,
    };
    Ok(result)
}
pub fn solve_corner(c: &Corner, jounce: f64, rack: f64) -> Result<CornerState, Error> {
    if !jounce.is_finite() || !rack.is_finite() {
        return Err(err("motion must be finite"));
    }
    let steps = (jounce.abs().max(rack.abs()) / 0.01).ceil().max(1.0);
    if steps > 10000.0 {
        return Err(err("continuation work limit exceeded"));
    }
    continuation(c, steps as usize, |t| {
        (rack * t, Constraint::Jounce(jounce * t))
    })
}

/// Reclose about an already solved branch, with strict residuals for derivative analysis.
pub(crate) fn at_world_height(
    c: &Corner,
    s: &CornerState,
    frame: &Frame,
    rack: f64,
    z: f64,
) -> Result<CornerState, Error> {
    let [w, x, y, zq] = s.orientation;
    let mut pose = Pose {
        q: frame.rotation.inverse()
            * UnitQuaternion::new_normalize(nalgebra::Quaternion::new(w, x, y, zq)),
        t: frame.rotation.inverse() * (v(s.points.wheel_center) - frame.translation)
            - v(c.wheel_center),
    };
    let goal = Constraint::WorldHeight(frame.clone(), z);
    let iterations = newton_tolerance(c, &mut pose, rack, &goal, 1e-12)?;
    state(
        c,
        &pose,
        rack,
        &goal,
        s.metrics.rocker_angle_rad,
        iterations,
    )
}
