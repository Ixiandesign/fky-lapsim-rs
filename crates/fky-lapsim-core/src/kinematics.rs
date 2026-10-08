//! Quaternion tangent-space Newton-Raphson corner closure: solves one corner's rigid
//! linkage for a commanded jounce/road/world-height goal and reports its solved
//! points, orientation, and [crate::Metrics]. [solve_corner] is the single-corner
//! entry point; vehicle-level studies (`simulate`, `sweep`) drive the same closure
//! per corner with a chassis-relative goal instead of a plain jounce.
//!
//! See <https://github.com/Ixiandesign/FKY-LAPSIM/blob/main/docs/model-conventions.md>
//! for the commanded-motion, sign, and unit conventions used here.
use crate::{Corner, CornerId, Error, Metrics, Point};
use serde::{Deserialize, Serialize};
/// One corner's solved hardpoints, in the coordinate frame of the enclosing
/// [CornerState] (chassis frame for [solve_corner], world frame for vehicle studies).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Points {
    /// Upper wishbone's forward inner pivot (chassis-fixed; unmoved by the solve).
    pub upper_front: Point,
    /// Upper wishbone's rearward inner pivot (chassis-fixed).
    pub upper_rear: Point,
    /// Lower wishbone's forward inner pivot (chassis-fixed).
    pub lower_front: Point,
    /// Lower wishbone's rearward inner pivot (chassis-fixed).
    pub lower_rear: Point,
    /// Upper wishbone's outer ball joint at the solved pose.
    pub upper_ball: Point,
    /// Lower wishbone's outer ball joint at the solved pose.
    pub lower_ball: Point,
    /// Tie-rod's inner pickup at the solved pose, including commanded rack travel.
    pub steering_inner: Point,
    /// Tie-rod's outer, knuckle-side pickup at the solved pose.
    pub steering_outer: Point,
    /// Wheel center at the solved pose.
    pub wheel_center: Point,
    /// Spindle axis endpoints at the solved pose (direction has no sign convention).
    pub spindle_axis: [Point; 2],
    /// Pushrod pickup at the solved pose, on its owning body.
    pub pushrod_pickup: Point,
    /// Rocker's chassis-fixed rotation axis endpoints (unmoved by the solve).
    pub rocker_axis: [Point; 2],
    /// Rocker's pushrod attachment point at the solved rocker angle.
    pub rocker_pushrod: Point,
    /// Rocker's shock attachment point at the solved rocker angle.
    pub rocker_shock: Point,
    /// Shock's chassis-side attachment point (chassis-fixed).
    pub shock_chassis: Point,
    /// Heave-arm tip at the solved rocker angle, when [Corner::rocker_heave_arm] is set.
    pub rocker_heave_arm: Option<Point>,
    /// Heave arm's chassis-fixed virtual anchor, when [Corner::heave_arm_anchor] is set.
    pub heave_arm_anchor: Option<Point>,
    /// Roll-arm tip at the solved rocker angle, when [Corner::rocker_roll_arm] is set.
    pub rocker_roll_arm: Option<Point>,
    /// Roll arm's chassis-fixed virtual anchor, when [Corner::roll_arm_anchor] is set.
    pub roll_arm_anchor: Option<Point>,
    /// Tire's ground/road support (contact representative) point at the solved pose.
    pub contact_point: Point,
}
/// One corner's complete solved state: hardpoints, knuckle orientation, derived
/// [Metrics], and closure diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CornerState {
    /// This corner's identity.
    pub id: CornerId,
    /// Solved hardpoints; chassis frame from [solve_corner], world frame from
    /// vehicle studies (`simulate`/`sweep`).
    pub points: Points,
    /// Knuckle orientation as a unit quaternion `[w, x, y, z]`, relative to the
    /// design pose, in the same frame as `points`.
    pub orientation: [f64; 4],
    /// Set when the tire support point at this pose is not a unique/smooth
    /// function of the axle direction (e.g. a vertical axle or a horizontal
    /// cylinder axle); describes which representative point was selected.
    pub contact_ambiguity: Option<String>,
    /// Alignment, rest-length, and linkage-angle metrics derived from `points`.
    pub metrics: Metrics,
    /// Largest absolute link-length/goal residual at convergence, metres.
    pub max_residual_m: f64,
    /// Newton iterations used by the final continuation step.
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
/// Internal 3-vector representation of a [Point], for nalgebra arithmetic.
pub(crate) type V = Vector3<f64>;
/// Stacked 6-dof residual/step vector: translation (0..3) then rotation tangent (3..6).
type R6 = SVector<f64, 6>;
/// Converts a [Point] to its nalgebra vector form.
pub(crate) fn v(p: Point) -> V {
    V::from(p)
}
/// Builds an [Error] from a message string; the sole error constructor in this module.
pub(crate) fn err(s: &str) -> Error {
    Error { message: s.into() }
}
/// A rigid transform (rotation then translation) from chassis frame into the frame a
/// [CornerState] is ultimately reported in; identity for [solve_corner], the chassis
/// pose for vehicle-level studies.
#[derive(Clone)]
pub(crate) struct Frame {
    /// Rotation applied before translation.
    pub rotation: UnitQuaternion<f64>,
    /// Translation applied after rotation.
    pub translation: V,
}
impl Frame {
    /// The identity transform: no rotation, no translation.
    pub(crate) fn identity() -> Self {
        Self {
            rotation: UnitQuaternion::identity(),
            translation: V::zeros(),
        }
    }
    /// Maps a point through this transform: `rotation * p + translation`.
    pub(crate) fn point(&self, p: V) -> V {
        self.rotation * p + self.translation
    }
}
/// The solved corner's free-body pose relative to the design pose: a rigid rotation
/// `q` and translation `t` of the knuckle/wheel assembly about [Corner::wheel_center].
#[derive(Clone)]
struct Pose {
    q: UnitQuaternion<f64>,
    t: V,
}
impl Pose {
    /// Maps a design-pose point on the moving assembly through this pose.
    fn point(&self, c: &Corner, p: Point) -> V {
        v(c.wheel_center) + self.t + self.q * (v(p) - v(c.wheel_center))
    }
    /// Applies a tangent-space Newton step `d` (translation increment then rotation
    /// increment as a scaled axis) to produce the next candidate pose.
    fn step(&self, d: R6) -> Self {
        Self {
            t: self.t + d.fixed_rows::<3>(0).into_owned(),
            q: UnitQuaternion::from_scaled_axis(d.fixed_rows::<3>(3).into_owned()) * self.q,
        }
    }
}
/// The commanded goal the Newton solve closes onto: a plain chassis-relative jounce,
/// a road-height contact constraint expressed in `Frame`, or a world-height wheel
/// center constraint expressed in `Frame` (used for reclosure/derivative analysis).
#[derive(Clone)]
pub(crate) enum Constraint {
    /// Wheel-center jounce relative to the chassis, metres (see [solve_corner]).
    Jounce(f64),
    /// Tire support point's height, metres, in the given frame.
    Road(Frame, f64),
    /// Wheel-center height, metres, in the given frame.
    WorldHeight(Frame, f64),
}
/// Commanded rack translation of `steering_inner` along the corner's `rack_axis`.
fn rack_offset(c: &Corner, rack: f64) -> V {
    (v(c.rack_axis[1]) - v(c.rack_axis[0])).normalize() * rack
}
/// Tire support (contact representative) point for `c.tire_profile`, given the
/// wheel-center position and unit spindle axis; see `docs/model-conventions.md`
/// "Rigid tire envelopes" for the exact geometry and its ambiguous cases.
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
/// Describes a non-unique/non-smooth tire support case for the given tire profile
/// and unit spindle axis (vertical axle, or a horizontal cylinder axle's camber
/// cusp), or `None` when the support point is well-defined; surfaced as
/// [CornerState::contact_ambiguity].
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
/// The 6 scalar residuals a solved [Pose] must zero: the upper-front, upper-rear,
/// lower-front, and lower-rear wishbone link lengths and the tie-rod length (with
/// commanded rack offset applied to `steering_inner`), each minus its rest length,
/// followed by the commanded `goal` residual in the last slot. Used both to drive
/// Newton iteration and, at convergence, as the reported [CornerState::max_residual_m].
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
/// [newton_tolerance] with the standard 1e-8 m/rad convergence tolerance.
fn newton(c: &Corner, pose: &mut Pose, rack: f64, goal: &Constraint) -> Result<usize, Error> {
    newton_tolerance(c, pose, rack, goal, 1e-8)
}
/// Drives `pose` in place toward zeroing [residual] via damped Gauss-Newton/SVD steps
/// with backtracking line search, returning the iteration count on convergence.
/// Detects and rejects the free-rotation-about-a-vertical-kingpin case (an actual
/// unconstrained DOF, not a numerically isolated toggle) once the link residuals are
/// satisfied. Errors on a non-finite residual, that underdetermined-steering case, or
/// exhausting the 80-iteration budget without a step that reduces the residual norm.
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
/// Signed rotation angle (radians) about the axis `(a, b)` that rigidly carries a
/// point at design position `rest` to solved position `moved`, both projected
/// perpendicular to that axis. Used to recover each wishbone's swept angle from its
/// solved ball-joint position.
fn arm_angle(a: Point, b: Point, rest: Point, moved: V) -> f64 {
    let axis = (v(b) - v(a)).normalize();
    let r = v(rest) - v(a);
    let r = r - axis * r.dot(&axis);
    let m = moved - v(a);
    let m = m - axis * m.dot(&axis);
    axis.dot(&r.cross(&m)).atan2(r.dot(&m))
}
/// Rotates design-pose point `p` by `angle` (radians) about the axis through
/// `axis[0]`/`axis[1]`, in the direction from the first point to the second.
fn rotate_axis(p: Point, axis: [Point; 2], angle: f64) -> V {
    v(axis[0])
        + UnitQuaternion::from_axis_angle(&Unit::new_normalize(v(axis[1]) - v(axis[0])), angle)
            * (v(p) - v(axis[0]))
}
/// Solves the rocker's single rotation angle about `c.rocker_axis` that places
/// `c.rocker_pushrod` at the fixed distance `length` from the given pushrod `pickup`
/// point, picking whichever of the (generally two) solutions is angularly nearest
/// `previous` (radians) for branch continuity. Errors when the pushrod length does
/// not constrain the angle (pickup lies on the rocker axis) or the pushrod cannot
/// reach the rocker at any angle.
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
/// Builds the reported [CornerState] from a converged [Pose]: derives the wishbone
/// and rocker angles, the pushrod pickup (following its owning body), and every
/// solved point/orientation/metric, then re-checks the link and pushrod-length
/// residuals at strict tolerance. `previous` is the prior rocker angle used for
/// branch continuity (see [rocker]); `iterations` is only carried through into the
/// returned [CornerState::iterations].
///
/// # Errors
///
/// Propagates [rocker]'s error; otherwise returns an error if the rebuilt link or
/// pushrod-length residual exceeds 1e-8 m (`"link closure exceeds tolerance"`).
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
        rocker_heave_arm: c
            .rocker_heave_arm
            .map(|p| f.point(rotate_axis(p, c.rocker_axis, ra)).into()),
        heave_arm_anchor: c.heave_arm_anchor.map(fixed),
        rocker_roll_arm: c
            .rocker_roll_arm
            .map(|p| f.point(rotate_axis(p, c.rocker_axis, ra)).into()),
        roll_arm_anchor: c.roll_arm_anchor.map(fixed),
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
/// [continuation_mode] with motion-ratio measurement enabled; the entry point used
/// by [solve_corner].
pub(crate) fn continuation(
    c: &Corner,
    steps: usize,
    goal: impl Fn(f64) -> (f64, Constraint),
) -> Result<CornerState, Error> {
    continuation_mode(c, steps, goal, true)
}
/// [continuation_tolerance] with the standard 1e-8 m/rad Newton tolerance.
pub(crate) fn continuation_mode(
    c: &Corner,
    steps: usize,
    goal: impl Fn(f64) -> (f64, Constraint),
    measure_motion_ratio: bool,
) -> Result<CornerState, Error> {
    continuation_tolerance(c, steps, goal, measure_motion_ratio, 1e-8)
}
/// Validates `c`, then closes the corner in `steps` equal sub-goals from the design
/// pose to `goal(1.0)` (each `goal(t)` giving the rack position and constraint for
/// continuation parameter `t` in `[0, 1]`), reusing each converged [Pose] as the
/// initial guess for the next sub-goal so large commanded travel still converges.
/// When `measure_motion_ratio`, also re-closes the final pose at wheel heights
/// perturbed by ±1e-5 m (chassis/rack held fixed) to fill [crate::Metrics::motion_ratio]
/// by central difference, leaving it `None` rather than propagating an error if
/// either perturbation fails to close.
///
/// # Errors
///
/// Propagates [Corner::validate]'s, [newton_tolerance]'s, and [state]'s errors from
/// any sub-goal step.
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
/// Solve one corner's rigid linkage for commanded wheel-center jounce relative to the
/// chassis (positive raises the wheel center) and rack travel along `c.rack_axis`,
/// both metres. Uses quaternion tangent-space Newton-Raphson with continuation from
/// the design pose for large travel. Returned points are in the chassis frame.
///
/// # Errors
///
/// Returns [Error] if: `jounce` or `rack` is not finite; the commanded travel would
/// require more than 10000 continuation steps (each step subdivides at most 0.01 m/step
/// of the larger of `jounce`/`rack`); [Corner::validate] fails for `c`; a closure
/// residual becomes non-finite; the linkage cannot reach the commanded pose
/// (`"kinematic closure stalled or travel unreachable"`) or fails to converge within
/// the iteration budget; the corner is underdetermined — free rotation about a
/// vertical kingpin with an axial inner tie joint, or a pushrod/rocker geometry whose
/// length does not constrain the rocker angle; the pushrod cannot reach the rocker at
/// the commanded pose; or the converged solution's link-length/goal residual exceeds
/// the solver's tolerance. This function does not panic.
///
/// ```rust
/// use fky_lapsim_core::{solve_corner, Project};
///
/// let project = Project::example();
/// let corner = &project.corners[0];
/// let state = solve_corner(corner, 0.01, 0.0)?;
/// assert!(state.max_residual_m < 1e-6);
/// # Ok::<(), fky_lapsim_core::Error>(())
/// ```
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
