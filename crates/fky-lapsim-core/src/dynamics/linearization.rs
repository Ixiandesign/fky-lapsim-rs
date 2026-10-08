//! Linearized ride-dynamics helpers: tangent stiffness/mass matrices and their
//! coupled undamped vibration modes about a stable static equilibrium, built from the
//! same energy and inertia model that [ride] integrates. See
//! <https://github.com/Ixiandesign/FKY-LAPSIM/blob/main/docs/model-conventions.md>,
//! especially its "Forces and energy" section, for the underlying spring/damper force
//! laws, `AxleInterconnect` heave/roll rate semantics, and generalized force that this
//! module differentiates rather than reintroduces. This is a linearization about
//! equilibrium, not a time-domain integration -- see [linearize_ride] for what it
//! requires and what it includes.
use super::*;
use nalgebra::{Matrix3, Vector3};

/// One undamped coupled vibration mode about a loaded static equilibrium: a natural
/// frequency and its associated mode shape, satisfying
/// `stiffness_matrix * shape = (2*pi*frequency_hz)^2 * mass_matrix * shape` for the
/// [RideLinearization::stiffness_matrix]/[RideLinearization::mass_matrix] it was
/// computed from. "Coupled" means heave, roll, and pitch generally move together in a
/// single mode whenever those matrices have nonzero off-diagonal terms (e.g. from an
/// asymmetric spring or an axle interconnect); a mode is not necessarily pure heave,
/// pure roll, or pure pitch. [RideLinearization::modes] lists modes in
/// increasing-frequency order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RideModeShape {
    /// Natural undamped frequency, Hz. Not a damped-response frequency: damping laws
    /// do not change this value (see [linearize_ride]).
    pub frequency_hz: f64,
    /// Relative `[heave m, roll rad, pitch rad]` mode-shape amplitudes, normalized so
    /// the largest-magnitude entry has absolute value `1.0`; overall sign is
    /// otherwise arbitrary, as for any eigenvector.
    pub shape: [f64; 3],
}
/// The result of [linearize_ride]: tangent stiffness/mass matrices, their coupled
/// vibration modes, and static support reactions, all evaluated at the same loaded
/// static equilibrium and built from the same energy and inertia model as [ride].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RideLinearization {
    /// Loaded static equilibrium `[heave m, roll rad, pitch rad]` this linearization
    /// is about; the same quantity [RideRun::equilibrium] reports for a full run.
    pub equilibrium: [f64; 3],
    /// Tangent stiffness: d(generalized resisting force) / d`[heave, roll, pitch]` at
    /// `equilibrium`, symmetrized (averaged with its own transpose) to cancel
    /// central-difference asymmetry before use in the eigenproblem below. Diagonal
    /// units: N/m, N m/rad, N m/rad; off-diagonal units follow their row/column
    /// coordinates. Includes geometric/preload and axle-interconnect stiffness (see
    /// docs/model-conventions.md's "Forces and energy" section); a nonzero
    /// off-diagonal entry is what makes a mode couple heave, roll, and pitch.
    pub stiffness_matrix: [[f64; 3]; 3],
    /// Generalized mass for the same coordinates: the chassis's own sprung
    /// mass/inertia, plus each retained component body's contribution under
    /// [RideMode::RetainedComponentInertia] (no such contribution under
    /// [RideMode::Reduced]).
    pub mass_matrix: [[f64; 3]; 3],
    /// This system's vibration modes, increasing in frequency, each satisfying
    /// `stiffness_matrix * shape = (2*pi*frequency_hz)^2 * mass_matrix * shape` (see
    /// [RideModeShape]).
    pub modes: Vec<RideModeShape>,
    /// Corner identities, in the same order as `support_reaction_n`.
    pub corner_ids: [CornerId; 4],
    /// Static vertical support reaction at each corner at `equilibrium`, newtons, in
    /// `corner_ids` order -- the same quantity as [RideSample::support_reaction_n]
    /// evaluated at that pose.
    pub support_reaction_n: [f64; 4],
    /// [MODEL_FIDELITY] or [COMPONENT_MODEL_FIDELITY], matching the request's
    /// [RideMode], as for [RideRun::model_fidelity].
    pub model_fidelity: String,
}

/// Linearize the same ride-dynamics equations as [ride] on a flat stationary road
/// about a stable static equilibrium, returning tangent stiffness/mass matrices and
/// their coupled vibration modes.
///
/// This is an undamped free-vibration calculation; damping laws do not change these
/// frequencies. Rejects moving-road requests instead of treating a forced state as an
/// equilibrium. Includes geometric/preload and axle-interconnect stiffness (see
/// docs/model-conventions.md's "Forces and energy" section). `r.mode` selects the
/// same [RideMode::Reduced]/[RideMode::RetainedComponentInertia] fidelity as a full
/// run, and the returned [RideLinearization::model_fidelity] matches [ride]'s.
///
/// # Errors
/// Returns `Err` if `Project::validate` or [validate_request] rejects `p`/`r`; if
/// `r.road` is not [RoadInput::Flat] (this analysis linearizes about a *stationary*
/// equilibrium, not a forced or moving state); if the internal static-equilibrium
/// solve fails (unstable, singular, or non-convergent, as for [ride]); if the
/// stiffness matrix's, or (under [RideMode::RetainedComponentInertia]) the mass
/// matrix's, derivative refinement fails its own convergence check; if the resulting
/// generalized mass matrix is not positive definite or is singular; if the symmetric
/// eigensolver does not converge, returns a non-finite or non-positive eigenvalue (an
/// unstable or singular system), or a mode fails its own
/// `stiffness_matrix * shape = omega^2 * mass_matrix * shape` residual check; or if
/// evaluating the final static support reactions hits one of the same geometry/
/// contact failures [ride] can report.
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
