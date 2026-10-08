//! Rigid-body kinematics, geometry analysis, ride dynamics, tire/powertrain/
//! aero modelling, full-car lap simulation, and optimization for
//! double-wishbone suspension systems — the native physics engine behind
//! [FKY-LAPSIM](https://github.com/Ixiandesign/FKY-LAPSIM). Pure Rust, no
//! project-file I/O, no Python or UI dependency; public models and results
//! implement [`serde::Serialize`]/[`serde::Deserialize`].
//!
//! All bodies are rigid except the spring and damper, which are explicit
//! force laws; tire contact is a rigid analytic envelope for suspension
//! kinematics and a Pacejka Magic Formula force model for lap simulation —
//! neither is a deformable/tread model. Coordinates are SI: metres, x
//! forward / y left / z up; commanded rotations and body-angle metrics use
//! radians (metric names ending `_deg` are the exception, in degrees). See
//! [model conventions](https://github.com/Ixiandesign/FKY-LAPSIM/blob/main/docs/model-conventions.md)
//! for the full sign/frame conventions and the precise, deliberately
//! hedged meaning of terms like "undefined", "failed", and "rejected"
//! throughout this crate's diagnostics — this project treats those as
//! load-bearing distinctions, not filler words.
//!
//! # Layout
//!
//! - [`model`] — the data model: [`Project`], [`Corner`], [`Chassis`],
//!   [`Point`], [`SpringDamper`], [`BodyMass`]/[`ComponentMasses`],
//!   [`PushrodBody`], [`AxleInterconnect`], [`TireProfile`], and [`Error`].
//! - [`kinematics`] — single-corner rigid-linkage kinematics
//!   ([`solve_corner`]) via quaternion tangent-space Newton–Raphson.
//! - [`study`] — whole-vehicle prescribed-motion simulation and sweeps
//!   ([`simulate`], [`simulate_on_road`], [`sweep`]).
//! - [`analysis`] — geometric derivatives separate from simulation:
//!   camber/toe gain, motion ratio, projected instant and roll centers
//!   ([`analyze`], [`analyze_with_step`]).
//! - [`metrics`] — per-corner solved metrics ([`Metrics`]: camber, toe,
//!   caster, KPI, scrub, trail, ...).
//! - [`dynamics`] — nonlinear time-domain ride simulation, reduced or
//!   retained-component-inertia fidelity ([`ride`]).
//! - [`optimize`] — the native, parallel, checkpointable/resumable
//!   differential-evolution search over suspension geometry and
//!   spring/damper variables.
//! - [`tire`], [`tire_configuration`], [`powertrain`], [`aero`], [`track`] —
//!   vehicle-component models shared by the suspension-only model (rigid
//!   tire envelope) and the full-car lap simulator (Pacejka tire force,
//!   powertrain, aero, track geometry).
//! - [`lapsim`] — the quasi-steady-state lap simulator after Zacharelis (2023): a thesis-style
//!   bike model coupled to a 7×7 full-car matrix built from the suspension geometry; the
//!   public entry point is [`lapsim::simulate_lap`].
//! - [`lap`] — the full-car configuration ([`lap::LapVehicle`]) shared by the lap simulator and
//!   the multi-track stochastic lap optimizer ([`lap::optimization`]).
//! - [`results`] — exportable, units-labeled result tables.
//!
//! # Example
//!
//! ```
//! use fky_lapsim_core::{analyze, simulate, Motion, Project};
//!
//! let project = Project::example();
//! project.validate()?;
//!
//! let motion = Motion {
//!     heave: 0.01,
//!     roll: 1.0_f64.to_radians(),
//!     ..Motion::default()
//! };
//! let state = simulate(&project, &motion)?;
//! for corner in &state.corners {
//!     println!(
//!         "{:?}: camber={:.3} deg, toe={:.3} deg",
//!         corner.id, corner.metrics.camber_deg, corner.metrics.toe_deg
//!     );
//! }
//!
//! let analysis = analyze(&project, &motion)?;
//! println!("Front track: {:.3} m", analysis.front.wheel_track_m);
//! # Ok::<(), fky_lapsim_core::Error>(())
//! ```
#![warn(missing_docs)]

pub mod dynamics;
pub mod kinematics;
pub mod metrics;
pub mod model;
mod motion_grid;
pub mod results;
pub mod tire;
pub mod tire_configuration;
pub mod powertrain;
pub mod aero;
pub mod track;
pub mod lap;
pub mod lapsim;
pub mod study;
pub use dynamics::{
    ride, InterconnectSample, RideMode, RideRequest, RideRun, RideSample, RideTermination,
    RoadInput,
};
pub use kinematics::{solve_corner, CornerState, Points};
pub use metrics::Metrics;
pub use model::{
    AxleInterconnect, BodyMass, Chassis, ComponentMasses, Corner, CornerId, Error, Point, Project,
    PushrodBody, SpringDamper, TireProfile,
};
pub use study::{simulate, simulate_on_road, sweep, Motion, Sample, VehicleState};
pub mod analysis;

pub use analysis::{
    analyze, analyze_with_step, Analysis, AxleAnalysis, CornerAnalysis, OptionalValue,
    ProjectedCenter,
};
mod mass;
pub mod optimize;
