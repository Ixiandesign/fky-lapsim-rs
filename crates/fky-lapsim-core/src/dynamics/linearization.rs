use super::*;
use nalgebra::{Matrix3, Vector3};

/// One undamped coupled mode about a loaded static equilibrium.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RideModeShape {
    /// Natural frequency in Hz (not damped-response frequency).
    pub frequency_hz: f64,
    /// Relative [heave m, roll rad, pitch rad] amplitudes, largest magnitude one.
    pub shape: [f64; 3],
}
/// Tangent rates and coupled frequencies from the same energy and inertia as ride.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RideLinearization {
    /// Loaded equilibrium [heave m, roll rad, pitch rad].
    pub equilibrium: [f64; 3],
    /// d generalized resisting force / d [heave, roll, pitch]. Diagonal units:
    /// N/m, N m/rad, N m/rad. Off-diagonal units follow their row/column coordinates.
    pub stiffness_matrix: [[f64; 3]; 3],
    /// Generalized mass for the same coordinates, including configured retained bodies.
    pub mass_matrix: [[f64; 3]; 3],
    /// Increasing-frequency modes of K phi = omega² M phi, including coupling.
    pub modes: Vec<RideModeShape>,
    /// Corner identities for reactions.
    pub corner_ids: [CornerId; 4],
    /// Static normal reactions in corner_ids order, N.
    pub support_reaction_n: [f64; 4],
    /// Explicit mass-model identifier.
    pub model_fidelity: String,
}

/// Linearize on a flat stationary road about stable static equilibrium. This is
/// an undamped free-vibration calculation; damping laws do not change these
/// frequencies. Rejects moving-road requests instead of treating a forced state
/// as an equilibrium. Includes geometric/preload and axle-interconnect stiffness.
pub fn linearize_ride(p: &Project, r: &RideRequest) -> Result<RideLinearization, Error> {
    p.validate()?;
    validate_request(r)?;
    if !matches!(r.road, RoadInput::Flat) {
        return Err(error("modal analysis requires a flat stationary road"));
    }
    let q = equilibrate(p, r)?;
    let coarse = static_stiffness_step(p, r, q, 0.0002)?;
    let k = static_stiffness_step(p, r, q, 0.0001)?;
    if coarse
        .iter()
        .zip(k.iter())
        .any(|(a, b)| (a - b).abs() > 0.1 + 0.005 * b.abs())
    {
        return Err(error("modal stiffness derivative refinement failed"));
    }
    let mut mass = Matrix3::from_diagonal(&Vector3::new(
        p.chassis.sprung_mass,
        p.chassis.inertia[0],
        p.chassis.inertia[1] * q[1].cos().powi(2) + p.chassis.inertia[2] * q[1].sin().powi(2),
    ));
    if r.mode == RideMode::RetainedComponentInertia && crate::mass::active(p) {
        let x = crate::mass::X::from_row_slice(&[q[0], q[1], q[2], 0., 0., 0., 0.]);
        let provider = |x| crate::mass::poses(p, [r.rack_front, r.rack_rear], x);
        let coarse = crate::mass::terms(provider, x, crate::mass::X::zeros(), r.derivative_step)?;
        let fine =
            crate::mass::terms(provider, x, crate::mass::X::zeros(), r.derivative_step / 2.)?;
        if !coarse.stable_jacobians(&fine) {
            return Err(error("modal mass derivative refinement failed"));
        }
        for i in 0..3 {
            for j in 0..3 {
                mass[(i, j)] += fine.mass[(i, j)];
            }
        }
    }
    let l = mass
        .cholesky()
        .ok_or_else(|| error("modal mass matrix is not positive definite"))?
        .l();
    let inverse = l
        .try_inverse()
        .ok_or_else(|| error("singular modal mass matrix"))?;
    let operator = inverse * k * inverse.transpose();
    // Machine-epsilon deflation leaves roundoff-sized off-diagonals active and
    // can cause cancellation in the 2x2 eigenvector construction. A relative
    // tolerance below our derivative accuracy avoids that ill-conditioned step.
    let eigen = nalgebra::linalg::SymmetricEigen::try_new(
        (operator + operator.transpose()) / 2.,
        1e-12,
        100,
    )
    .ok_or_else(|| error("modal eigensolver did not converge"))?;
    if eigen.eigenvalues.iter().any(|x| !x.is_finite() || *x <= 0.) {
        return Err(error("modal stiffness is unstable or singular"));
    }
    let mut modes = Vec::new();
    for i in 0..3 {
        let shape = inverse.transpose() * eigen.eigenvectors.column(i);
        let normalized = shape / shape.amax();
        let elastic = k * normalized;
        let inertial = mass * normalized * eigen.eigenvalues[i];
        if (elastic - inertial).norm() > 1e-9 * (elastic.norm() + inertial.norm()).max(1.) {
            return Err(error("modal eigenvector residual check failed"));
        }
        modes.push(RideModeShape {
            frequency_hz: eigen.eigenvalues[i].sqrt() / std::f64::consts::TAU,
            shape: normalized.into(),
        });
    }
    modes.sort_by(|a, b| a.frequency_hz.total_cmp(&b.frequency_hz));
    let mut y = [0.; 9];
    y[..3].copy_from_slice(&q);
    let sample = evaluate(p, r, 0., y, false)?;
    Ok(RideLinearization {
        equilibrium: q,
        stiffness_matrix: std::array::from_fn(|i| std::array::from_fn(|j| k[(i, j)])),
        mass_matrix: std::array::from_fn(|i| std::array::from_fn(|j| mass[(i, j)])),
        modes,
        corner_ids: p.corners.each_ref().map(|c| c.id),
        support_reaction_n: sample.support_reaction_n,
        model_fidelity: if r.mode == RideMode::Reduced {
            MODEL_FIDELITY
        } else {
            COMPONENT_MODEL_FIDELITY
        }
        .into(),
    })
}
