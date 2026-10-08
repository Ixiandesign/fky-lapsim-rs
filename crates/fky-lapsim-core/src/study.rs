//! Prescribed vehicle-level motion: commands chassis heave/roll/pitch/rack and solves
//! all four corners together, on a flat or per-corner road.
//!
//! This is prescribed-motion kinematics, not equilibrium or dynamics: no forces are
//! solved here, and a returned [VehicleState] only means the commanded pose closed
//! within tolerance, not that it is reachable under load. [simulate]/[simulate_on_road]
//! solve one [Motion]; [sweep] and [detailed_sweep] solve many, retaining every
//! failure rather than filtering it out. [motion_grid] builds bounded, deterministic
//! batches of [Motion] for those sweeps. See the "Commanded motion" section of
//! <https://github.com/Ixiandesign/FKY-LAPSIM/blob/main/docs/model-conventions.md>
//! for the sign conventions and finite-rotation order used below.
pub use crate::motion_grid::{motion_grid, AxisRange, GridMode, MotionGrid};
use crate::{CornerState, Error, Project};
use serde::{Deserialize, Serialize};
/// Commanded chassis/rack motion for [simulate]/[simulate_on_road]/[sweep]. The finite
/// chassis rotation is `Ry(pitch) * Rx(roll)` about the chassis center of mass, so
/// roll and pitch are not interchangeable Euler components. `Motion::default()` is the
/// design pose: zero heave/roll/pitch/rack, identical to the as-supplied static geometry.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Motion {
    /// Chassis heave, metres; positive lifts the chassis.
    pub heave: f64,
    /// Chassis roll, radians; a right-handed rotation about +x.
    pub roll: f64,
    /// Chassis pitch, radians; a right-handed rotation about +y, so positive pitch
    /// lowers the nose.
    pub pitch: f64,
    /// Front-axle rack travel, metres, along each front corner's `rack_axis`.
    pub rack_front: f64,
    /// Rear-axle rack travel, metres, along each rear corner's `rack_axis`.
    pub rack_rear: f64,
}
/// A solved vehicle pose: the commanded [Motion] and all four corners' solved state,
/// in the world frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VehicleState {
    /// The motion this state was solved for.
    pub motion: Motion,
    /// Solved state for all four corners, world frame.
    pub corners: [CornerState; 4],
}
/// One requested [Motion] from a [sweep], with its solved state or failure reason.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sample {
    /// The requested motion.
    pub motion: Motion,
    /// Solved state, or `None` if this motion failed to solve.
    pub state: Option<VehicleState>,
    /// Failure message, or `None` if `state` is present.
    pub error: Option<String>,
}

/// A prescribed-motion sample with expanded geometry analysis. The nested
/// analysis records derivative coordinates, steps, undefined values and reasons.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedSample {
    /// Existing sweep shape, flattened for backward-compatible consumers.
    #[serde(flatten)]
    pub sample: Sample,
    /// Expanded analysis, absent when the requested pose fails.
    pub analysis: Option<crate::Analysis>,
}

/// Analyze every requested motion without discarding failed states. Use [sweep]
/// when only basic geometry is needed; derivatives add closure evaluations. Never
/// errors itself: a motion that fails to solve or analyze is recorded as a
/// [DetailedSample] with `analysis: None` and `sample.error` set, alongside every
/// motion that succeeded, so the returned `Vec` always has one entry per input motion.
pub fn detailed_sweep(p: &Project, motions: &[Motion]) -> Vec<DetailedSample> {
    motions
        .iter()
        .map(|m| match crate::analyze(p, m) {
            Ok(analysis) => DetailedSample {
                sample: Sample {
                    motion: *m,
                    state: Some(analysis.state.clone()),
                    error: None,
                },
                analysis: Some(analysis),
            },
            Err(e) => DetailedSample {
                sample: Sample {
                    motion: *m,
                    state: None,
                    error: Some(e.to_string()),
                },
                analysis: None,
            },
        })
        .collect()
}
/// Solve all four corners for `m` on a flat (zero-height) road. Equivalent to
/// `simulate_on_road(p, m, [0.0; 4])`.
///
/// # Errors
///
/// Returns an [Error] when: `p.validate()` fails (invalid schema, chassis, corner, or
/// interconnect configuration); any component of `m` is not finite; the continuation
/// work implied by `m`'s magnitude exceeds 10,000 steps; or any individual corner
/// fails to close its linkage within tolerance (the message is prefixed with that
/// corner's [crate::CornerId]).
///
/// # Examples
///
/// ```
/// use fky_lapsim_core::{simulate, Motion, Project};
///
/// # fn main() -> Result<(), fky_lapsim_core::Error> {
/// let project = Project::example();
/// project.validate()?;
/// let motion = Motion {
///     heave: 0.01,
///     roll: 1.0_f64.to_radians(),
///     ..Motion::default()
/// };
/// let state = simulate(&project, &motion)?;
/// for corner in &state.corners {
///     println!("{:?}: camber={} deg", corner.id, corner.metrics.camber_deg);
/// }
/// # Ok(())
/// # }
/// ```
pub fn simulate(p: &Project, m: &Motion) -> Result<VehicleState, Error> {
    simulate_on_road(p, m, [0.0; 4])
}
/// Solve all four corners for `m`, closing each corner's tire onto its own horizontal
/// road height (in project corner array order) instead of a single flat road. Chassis
/// motion remains absolute: `road_heights` shifts only where each tire closes, not the
/// commanded chassis pose itself.
///
/// # Errors
///
/// Same conditions as [simulate], plus: any `road_heights` entry is not finite. The
/// continuation work estimate also grows with the largest `road_heights` magnitude.
pub fn simulate_on_road(
    p: &Project,
    m: &Motion,
    road_heights: [f64; 4],
) -> Result<VehicleState, Error> {
    simulate_on_road_mode(p, m, road_heights, true)
}
pub(crate) fn simulate_on_road_mode(
    p: &Project,
    m: &Motion,
    road_heights: [f64; 4],
    measure_motion_ratio: bool,
) -> Result<VehicleState, Error> {
    simulate_on_road_tolerance(p, m, road_heights, measure_motion_ratio, 1e-8)
}
pub(crate) fn simulate_on_road_tolerance(
    p: &Project,
    m: &Motion,
    road_heights: [f64; 4],
    measure_motion_ratio: bool,
    tolerance: f64,
) -> Result<VehicleState, Error> {
    use crate::kinematics::{continuation_tolerance, err, v, Constraint, Frame, V};
    use nalgebra::UnitQuaternion;
    p.validate()?;
    if ![m.heave, m.roll, m.pitch, m.rack_front, m.rack_rear]
        .iter()
        .chain(road_heights.iter())
        .all(|x| x.is_finite())
    {
        return Err(err("motion and road heights must be finite"));
    }
    let max_road = road_heights.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
    let steps = (m
        .heave
        .abs()
        .max(m.rack_front.abs())
        .max(m.rack_rear.abs())
        .max(max_road)
        / 0.01)
        .max(m.roll.abs().max(m.pitch.abs()) / 0.02)
        .ceil()
        .max(1.0);
    if steps > 10000.0 {
        return Err(err("continuation work limit exceeded"));
    }
    let mut corners = Vec::new();
    for (index, c) in p.corners.iter().enumerate() {
        let rack = match c.id {
            crate::CornerId::FrontLeft | crate::CornerId::FrontRight => m.rack_front,
            _ => m.rack_rear,
        };
        corners.push(
            continuation_tolerance(
                c,
                steps as usize,
                |t| {
                    let rotation = UnitQuaternion::from_axis_angle(&V::y_axis(), m.pitch * t)
                        * UnitQuaternion::from_axis_angle(&V::x_axis(), m.roll * t);
                    let center = v(p.chassis.center_of_mass);
                    let translation = center - rotation * center + V::new(0.0, 0.0, m.heave * t);
                    (
                        rack * t,
                        Constraint::Road(
                            Frame {
                                rotation,
                                translation,
                            },
                            road_heights[index] * t,
                        ),
                    )
                },
                measure_motion_ratio,
                tolerance,
            )
            .map_err(|e| err(&format!("{:?}: {}", c.id, e)))?,
        );
    }
    Ok(VehicleState {
        motion: *m,
        corners: corners
            .try_into()
            .map_err(|_| err("expected four corners"))?,
    })
}
/// Solve `simulate` for every requested motion, retaining each result (or error)
/// rather than filtering failures out. Inspect `Sample::error` before assessing a
/// study's feasibility.
///
/// # Examples
///
/// ```
/// use fky_lapsim_core::{sweep, Motion, Project};
///
/// let project = Project::example();
/// let motions = [
///     Motion::default(),
///     Motion { heave: 0.02, ..Motion::default() },
/// ];
/// let samples = sweep(&project, &motions);
/// assert_eq!(samples.len(), motions.len());
/// for sample in &samples {
///     if let Some(state) = &sample.state {
///         println!("{:?}: {} corners solved", sample.motion, state.corners.len());
///     } else {
///         println!("{:?}: {}", sample.motion, sample.error.as_deref().unwrap_or("unknown"));
///     }
/// }
/// ```
pub fn sweep(p: &Project, m: &[Motion]) -> Vec<Sample> {
    m.iter()
        .map(|m| match simulate(p, m) {
            Ok(state) => Sample {
                motion: *m,
                state: Some(state),
                error: None,
            },
            Err(e) => Sample {
                motion: *m,
                state: None,
                error: Some(e.to_string()),
            },
        })
        .collect()
}
