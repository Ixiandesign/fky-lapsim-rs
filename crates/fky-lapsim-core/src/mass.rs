//! Rigid component inertia on seven holonomic coordinates: `x = [heave, roll, pitch,
//! z_fl, z_fr, z_rl, z_rr]`, i.e. [crate::Motion]'s heave/roll/pitch followed by each
//! corner's prescribed road (tire support point) height (in `Project.corners` order),
//! matching the `road_heights` argument of `crate::study::simulate_on_road_tolerance`.
//! This module is internal: it supplies [RideMode::RetainedComponentInertia](crate::RideMode::RetainedComponentInertia)'s
//! generalized mass matrix, gravity generalized force, velocity-product (Christoffel)
//! bias force, and potential energy for whichever of each corner's `component_masses`
//! (upper arm, lower arm, knuckle, rocker) carry positive mass; see
//! <https://github.com/Ixiandesign/FKY-LAPSIM/blob/main/docs/model-conventions.md> for
//! the underlying coordinate/sign conventions and the "Component mass and inertia"
//! section of `docs/rust-guide.md` for how a caller selects this mode.
use crate::Error;
use nalgebra::{Matrix3, SMatrix, SVector, UnitQuaternion, Vector3};
/// The 7 generalized coordinates: heave, roll, pitch, then 4 per-corner prescribed
/// road (tire support point) heights, in that order.
pub(crate) type X = SVector<f64, 7>;
/// A 7-by-7 generalized mass matrix over [X].
pub(crate) type M = SMatrix<f64, 7, 7>;
/// A 3-by-7 Jacobian of a 3-vector (position or angular velocity) with respect to [X].
type J = SMatrix<f64, 3, 7>;
/// One retained rigid body's mass, inertia, and pose at a given generalized
/// coordinate, as produced by [poses].
#[derive(Clone)]
pub(crate) struct BodyPose {
    /// Mass, kilograms; always positive (zero-mass bodies are filtered out earlier).
    pub mass: f64,
    /// Inertia about the body's center of mass, in the body's own (unrotated) axes,
    /// kg m².
    pub inertia: Matrix3<f64>,
    /// World-frame center-of-mass position, metres.
    pub position: Vector3<f64>,
    /// World-frame orientation relating body axes to world axes.
    pub rotation: UnitQuaternion<f64>,
    /// Index into `Project.corners` this body belongs to, when it is a per-corner
    /// link; `None` for a body (if any) not tied to a single corner.
    pub corner: Option<usize>,
}
/// The generalized-coordinate terms [terms] assembles from the retained bodies: the
/// mass matrix, gravity and velocity-product generalized forces, and potential
/// energy, each a function of the coordinate `x` (and, for `bias`, the generalized
/// velocity `w`) passed to [terms].
#[derive(Clone)]
pub(crate) struct Terms {
    /// Generalized mass matrix `M(x)`.
    pub mass: M,
    /// Generalized gravity term `dV/dx` (the potential-energy gradient, `potential`
    /// below); callers subtract it, along with `bias`, from the generalized applied
    /// force before solving for acceleration (`M*qddot = Q_ext - gravity - bias`).
    pub gravity: X,
    /// Velocity-product (Christoffel/Coriolis) generalized bias force at `(x, w)`.
    pub bias: X,
    /// Total gravitational potential energy of the retained bodies at `x`, joules.
    pub potential: f64,
    jacobians: Vec<(J, J)>,
}
impl Terms {
    /// True when this instance's and `finer`'s body-position/orientation-rate
    /// Jacobians agree to within `1e-5 + 2e-4 * |value|`, used to check that the
    /// central-difference step size is small enough for the reported terms to be
    /// trustworthy.
    pub(crate) fn stable_jacobians(&self, finer: &Self) -> bool {
        self.jacobians.len() == finer.jacobians.len()
            && self
                .jacobians
                .iter()
                .zip(&finer.jacobians)
                .all(|((a, b), (c, d))| {
                    a.iter()
                        .chain(b.iter())
                        .zip(c.iter().chain(d.iter()))
                        .all(|(x, y)| (x - y).abs() <= 1e-5 + 2e-4 * y.abs())
                })
    }
}
/// True when any corner has a `component_masses` entry with positive `mass_kg`, i.e.
/// when [RideMode::RetainedComponentInertia](crate::RideMode::RetainedComponentInertia)
/// has actual mass to retain for `p`. Used to decide whether the more expensive
/// retained-inertia dynamics path is needed at all.
pub(crate) fn active(p: &crate::Project) -> bool {
    p.corners.iter().any(|c| {
        [
            &c.component_masses.upper_arm,
            &c.component_masses.lower_arm,
            &c.component_masses.knuckle,
            &c.component_masses.rocker,
        ]
        .iter()
        .any(|b| b.as_ref().is_some_and(|b| b.mass_kg > 0.))
    })
}
/// Closes `p` at generalized coordinate `x` (`rack` giving front/rear rack travel)
/// and returns one [BodyPose] per positive-mass retained component, in a fixed
/// per-corner order (upper arm, lower arm, rocker, then knuckle) repeated for each
/// corner in `p.corners` order. [jacobians] and [terms] rely on this order and count
/// being stable across nearby `x`; a changed body count between calls is itself an
/// error.
///
/// # Errors
///
/// Returns [Error] if the underlying kinematic closure at `x`/`rack` fails, or if a
/// cylinder-tire corner's spindle axis is within 1e-6 (as a unit-vector z-component)
/// of horizontal, where the tire support height has a non-differentiable camber
/// cusp and the component-mass derivative is undefined.
pub(crate) fn poses(p: &crate::Project, rack: [f64; 2], x: X) -> Result<Vec<BodyPose>, Error> {
    let motion = crate::Motion {
        heave: x[0],
        roll: x[1],
        pitch: x[2],
        rack_front: rack[0],
        rack_rear: rack[1],
    };
    let state = crate::study::simulate_on_road_tolerance(
        p,
        &motion,
        [x[3], x[4], x[5], x[6]],
        false,
        1e-12,
    )
    .map_err(|e| Error {
        message: format!("component mass derivative closure: {e}"),
    })?;
    let rc = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), x[2])
        * UnitQuaternion::from_axis_angle(&Vector3::x_axis(), x[1]);
    let cg = Vector3::from(p.chassis.center_of_mass);
    let translation = cg - rc * cg + Vector3::new(0., 0., x[0]);
    let mut out = Vec::new();
    for (i, (c, s)) in p.corners.iter().zip(&state.corners).enumerate() {
        if c.tire_profile == crate::TireProfile::Cylinder
            && (Vector3::from(s.points.spindle_axis[1]) - Vector3::from(s.points.spindle_axis[0]))
                .normalize()
                .z
                .abs()
                < 1e-6
        {
            return Err(Error {
                message: format!(
                    "{:?}: component mass derivative undefined at cylinder camber cusp",
                    c.id
                ),
            });
        }
        let definitions = [
            (
                &c.component_masses.upper_arm,
                [c.upper_front, c.upper_rear],
                s.metrics.upper_arm_angle_rad,
            ),
            (
                &c.component_masses.lower_arm,
                [c.lower_front, c.lower_rear],
                s.metrics.lower_arm_angle_rad,
            ),
            (
                &c.component_masses.rocker,
                c.rocker_axis,
                s.metrics.rocker_angle_rad,
            ),
        ];
        for (body, axis, angle) in definitions {
            if let Some(b) = body.as_ref().filter(|b| b.mass_kg > 0.) {
                let anchor = Vector3::from(axis[0]);
                let hinge = UnitQuaternion::from_axis_angle(
                    &nalgebra::Unit::new_normalize(Vector3::from(axis[1]) - anchor),
                    angle,
                );
                out.push(BodyPose {
                    mass: b.mass_kg,
                    inertia: Matrix3::from_fn(|i, j| b.inertia[i][j]),
                    position: translation
                        + rc * (anchor + hinge * (Vector3::from(b.center_of_mass) - anchor)),
                    rotation: rc * hinge,
                    corner: Some(i),
                });
            }
        }
        if let Some(b) = c
            .component_masses
            .knuckle
            .as_ref()
            .filter(|b| b.mass_kg > 0.)
        {
            let [w, a, y, z] = s.orientation;
            out.push(BodyPose {
                mass: b.mass_kg,
                inertia: Matrix3::from_fn(|i, j| b.inertia[i][j]),
                position: Vector3::from(s.transform_knuckle_point(c, b.center_of_mass)),
                rotation: UnitQuaternion::new_normalize(nalgebra::Quaternion::new(w, a, y, z)),
                corner: Some(i),
            });
        }
    }
    Ok(out)
}
/// Central-difference position and angular-velocity Jacobians (each w.r.t. `x`) for
/// every body `provider` returns at `x`, using step `step`. Only perturbs the first
/// 3 coordinates (heave/roll/pitch) when every body is per-corner (`corner.is_some()`
/// for all of `base`), since a per-corner body's dependence on its own prescribed
/// road height is then filled analytically afterward (columns 3..7, one nonzero
/// column per corner) rather than by finite difference.
///
/// # Errors
///
/// Propagates `provider`'s error, or returns an error if a perturbed call returns a
/// different number of bodies than `base` (the retained-body set must stay stable
/// under the derivative step).
fn jacobians(
    provider: &impl Fn(X) -> Result<Vec<BodyPose>, Error>,
    x: X,
    step: f64,
    base: &[BodyPose],
) -> Result<Vec<(J, J)>, Error> {
    let mut out = vec![(J::zeros(), J::zeros()); base.len()];
    let columns = if base.iter().all(|b| b.corner.is_some()) {
        3
    } else {
        7
    };
    for a in 0..columns {
        let mut xp = x;
        let mut xm = x;
        xp[a] += step;
        xm[a] -= step;
        let plus = provider(xp)?;
        let minus = provider(xm)?;
        if plus.len() != base.len() || minus.len() != base.len() {
            return Err(Error {
                message: "mass derivative body identity changed".into(),
            });
        }
        for i in 0..base.len() {
            out[i]
                .0
                .set_column(a, &((plus[i].position - minus[i].position) / (2. * step)));
            out[i].1.set_column(
                a,
                &((plus[i].rotation * minus[i].rotation.inverse()).scaled_axis() / (2. * step)),
            );
        }
    }
    for (i, b) in base.iter().enumerate() {
        if let Some(c) = b.corner {
            let linear = Vector3::z() - out[i].0.column(0);
            let angular = -out[i].1.column(0);
            out[i].0.set_column(3 + c, &linear);
            out[i].1.set_column(3 + c, &angular);
        }
    }
    Ok(out)
}
/// Assembles [Terms] at generalized coordinate `x` and generalized velocity `w` from
/// the rigid bodies `provider` returns, using central-difference step `step` (and,
/// when `w` is non-negligible, a second central difference along `w` of half-width
/// `min(step/max|w|, 0.05)` to form the velocity-product `bias` term). Pass
/// `w = X::zeros()` to get `mass`/`gravity`/`potential` alone with `bias` left zero.
///
/// # Errors
///
/// Propagates `provider`'s or [jacobians]'s error, or returns an error if any
/// assembled term is non-finite.
pub(crate) fn terms(
    provider: impl Fn(X) -> Result<Vec<BodyPose>, Error>,
    x: X,
    w: X,
    step: f64,
) -> Result<Terms, Error> {
    let poses = provider(x)?;
    let js = jacobians(&provider, x, step, &poses)?;
    let mut t = Terms {
        mass: M::zeros(),
        gravity: X::zeros(),
        bias: X::zeros(),
        potential: 0.,
        jacobians: js.clone(),
    };
    let directional = if w.amax() > 1e-10 {
        let eps = (step / w.amax()).min(0.05);
        let xp = x + eps * w;
        let xm = x - eps * w;
        Some((
            eps,
            jacobians(&provider, xp, step, &provider(xp)?)?,
            jacobians(&provider, xm, step, &provider(xm)?)?,
        ))
    } else {
        None
    };
    for (i, b) in poses.iter().enumerate() {
        let (jv, jw) = &js[i];
        let rotation = b.rotation.to_rotation_matrix();
        let iw = rotation.matrix() * b.inertia * rotation.matrix().transpose();
        t.mass += b.mass * jv.transpose() * jv + jw.transpose() * iw * jw;
        t.gravity += b.mass * 9.81 * jv.row(2).transpose();
        t.potential += b.mass * 9.81 * b.position.z;
        if let Some((eps, plus, minus)) = &directional {
            let a0 = (plus[i].0 - minus[i].0) * w / (2. * eps);
            let alpha = (plus[i].1 - minus[i].1) * w / (2. * eps);
            let omega = jw * w;
            t.bias += jv.transpose() * (b.mass * a0)
                + jw.transpose() * (iw * alpha + omega.cross(&(iw * omega)));
        }
    }
    if !t
        .mass
        .iter()
        .chain(t.gravity.iter())
        .chain(t.bias.iter())
        .all(|v| v.is_finite())
        || !t.potential.is_finite()
    {
        return Err(Error {
            message: "nonfinite component mass derivative".into(),
        });
    }
    Ok(t)
}
#[cfg(test)]
mod tests {
    use super::*;
    fn point(position: Vector3<f64>, mass: f64) -> BodyPose {
        BodyPose {
            mass,
            inertia: Matrix3::zeros(),
            position,
            rotation: UnitQuaternion::identity(),
            corner: None,
        }
    }
    #[test]
    fn affine_mass_and_prescribed_support() {
        let mut a = J::zeros();
        a[(0, 0)] = 2.;
        a[(1, 1)] = -3.;
        a[(2, 0)] = 0.4;
        a[(2, 3)] = 1.;
        let t = terms(
            |x| Ok(vec![point(a * x, 5.)]),
            X::zeros(),
            X::repeat(0.2),
            0.001,
        )
        .unwrap();
        assert!((t.mass - 5. * a.transpose() * a).amax() < 1e-8);
        assert!((t.gravity - 49.05 * a.row(2).transpose()).amax() < 1e-8);
        assert!(t.bias.amax() < 1e-7);
        let t = terms(
            |x| Ok(vec![point(Vector3::new(1., 2., x[3] + 0.3), 7.)]),
            X::zeros(),
            X::repeat(0.2),
            0.001,
        )
        .unwrap();
        assert!((t.mass[(3, 3)] * 2. + t.gravity[3] - 7. * 11.81).abs() < 1e-8);
    }
    #[test]
    fn polar_bias() {
        let mut x = X::zeros();
        x[0] = 1.2;
        x[1] = 0.4;
        let mut w = X::zeros();
        w[0] = 0.3;
        w[1] = 0.7;
        let t = terms(
            |x| {
                Ok(vec![point(
                    Vector3::new(x[0] * x[1].cos(), x[0] * x[1].sin(), 0.),
                    2.,
                )])
            },
            x,
            w,
            0.001,
        )
        .unwrap();
        assert!(
            (t.bias[0] + 2. * 1.2 * 0.7 * 0.7).abs() < 1e-5,
            "{:?}",
            t.bias
        );
        assert!((t.bias[1] - 4. * 1.2 * 0.3 * 0.7).abs() < 1e-5);
    }
    // Deliberately slower independent Christoffel oracle, used only by bounded tests.
    fn christoffel(provider: impl Fn(X) -> Result<Vec<BodyPose>, Error>, x: X, w: X) -> X {
        let d: Vec<M> = (0..7)
            .map(|i| {
                let mut xp = x;
                let mut xm = x;
                xp[i] += 0.002;
                xm[i] -= 0.002;
                (terms(&provider, xp, X::zeros(), 0.0003).unwrap().mass
                    - terms(&provider, xm, X::zeros(), 0.0003).unwrap().mass)
                    / 0.004
            })
            .collect();
        let mut b = X::zeros();
        for i in 0..7 {
            for j in 0..7 {
                for k in 0..7 {
                    b[i] += 0.5 * (d[j][(i, k)] + d[k][(i, j)] - d[i][(j, k)]) * w[j] * w[k];
                }
            }
        }
        b
    }
    #[test]
    fn rigid_attachment_parallel_axis_world_angular_bias() {
        let offset = Vector3::new(0.3, -0.2, 0.5);
        let inertia = Matrix3::from_diagonal(&Vector3::new(0.4, 0.6, 0.8));
        let provider = |x: X| {
            let rotation = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), x[2])
                * UnitQuaternion::from_axis_angle(&Vector3::x_axis(), x[1]);
            Ok(vec![BodyPose {
                mass: 3.,
                inertia,
                position: Vector3::new(0., 0., x[0]) + rotation * offset,
                rotation,
                corner: None,
            }])
        };
        let mut x = X::zeros();
        x[0] = 0.2;
        x[1] = 0.3;
        x[2] = -0.4;
        let mut w = X::zeros();
        w[0] = 0.2;
        w[1] = -0.4;
        w[2] = 0.7;
        let t = terms(provider, x, w, 0.0003).unwrap();
        let pose = &provider(x).unwrap()[0];
        let omega = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), x[2]) * Vector3::x() * w[1]
            + Vector3::y() * w[2];
        let velocity = Vector3::z() * w[0] + omega.cross(&(pose.rotation * offset));
        let body_omega = pose.rotation.inverse() * omega;
        let expected =
            0.5 * (3. * velocity.norm_squared() + body_omega.dot(&(inertia * body_omega)));
        assert!((0.5 * w.dot(&(t.mass * w)) - expected).abs() < 1e-7);
        assert!((t.bias - christoffel(provider, x, w)).amax() < 2e-6);
        let flipped = |x| {
            let mut p = provider(x)?;
            p[0].rotation = UnitQuaternion::new_unchecked(-p[0].rotation.into_inner());
            Ok(p)
        };
        assert!((terms(flipped, x, w, 0.0003).unwrap().mass - t.mass).amax() < 1e-10);
    }
    fn all_bodies() -> crate::Project {
        let mut p = crate::Project::example();
        for c in &mut p.corners {
            let body = |point| {
                Some(crate::BodyMass {
                    mass_kg: 1.,
                    center_of_mass: point,
                    inertia: [[0.01, 0., 0.], [0., 0.02, 0.], [0., 0., 0.025]],
                })
            };
            c.component_masses = crate::ComponentMasses {
                upper_arm: body(c.upper_ball),
                lower_arm: body(c.lower_ball),
                rocker: body(c.rocker_shock),
                knuckle: body(c.steering_outer),
            };
        }
        p
    }
    #[test]
    fn native_owning_transforms_translation_and_christoffel() {
        let p = all_bodies();
        let mut x = X::zeros();
        x[0] = 0.003;
        x[1] = 0.005;
        x[2] = -0.003;
        x[3] = 0.001;
        let state = crate::simulate_on_road(
            &p,
            &crate::Motion {
                heave: x[0],
                roll: x[1],
                pitch: x[2],
                rack_front: 0.001,
                rack_rear: -0.001,
            },
            [x[3], 0., 0., 0.],
        )
        .unwrap();
        let provider = |x| poses(&p, [0.001, -0.001], x);
        let base = provider(x).unwrap();
        for (i, s) in state.corners.iter().enumerate() {
            for (k, point) in [
                s.points.upper_ball,
                s.points.lower_ball,
                s.points.rocker_shock,
                s.points.steering_outer,
            ]
            .iter()
            .enumerate()
            {
                assert!((base[4 * i + k].position - Vector3::from(*point)).norm() < 1e-7);
            }
        }
        let mut shift = X::repeat(1.);
        shift[1] = 0.;
        shift[2] = 0.;
        let moved = provider(x + 0.004 * shift).unwrap();
        for (i, b) in base.iter().enumerate() {
            assert!((moved[i].position - b.position - Vector3::<f64>::z() * 0.004).norm() < 1e-10);
            assert!((moved[i].rotation * b.rotation.inverse()).angle() < 1e-9);
        }
        let js = jacobians(&provider, x, 0.0005, &base).unwrap();
        for (jv, jw) in js {
            assert!((jv * shift - Vector3::z()).norm() < 1e-12);
            assert!((jw * shift).norm() < 1e-12);
        }
        let w = X::from_row_slice(&[0.01, 0.02, -0.01, 0.005, -0.003, 0.004, -0.002]);
        let t = terms(provider, x, w, 0.0005).unwrap();
        assert!((shift.dot(&(t.mass * shift)) - 16.).abs() < 1e-8);
        assert!(
            (t.bias - christoffel(provider, x, w)).amax() < 2e-5,
            "{:?}",
            t.bias - christoffel(provider, x, w)
        );
    }
}
