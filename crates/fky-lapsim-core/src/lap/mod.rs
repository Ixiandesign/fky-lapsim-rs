//! Full-car configuration and multi-track lap optimization.
//!
//! A [LapVehicle] pairs a suspension [crate::Project] with four configured tires
//! ([crate::tire_configuration::ConfiguredTire]), a [crate::powertrain::Powertrain], brakes and
//! constant-coefficient aero. It is the car description shared by the lap simulator
//! ([crate::lapsim], the quasi-steady-state model after Zacharelis, 2023) and by the `optimization`
//! submodule, a multi-track stochastic search that scores candidates through an evaluator callback
//! (the platform passes [crate::lapsim::simulate_lap]).
//!
//! See <https://github.com/Ixiandesign/FKY-LAPSIM/blob/main/docs/model-conventions.md> for the
//! shared sign/unit/frame conventions (metres; x forward, y left, z up).
//! [LapVehicle::synthetic_demo] is a formula-car-scale illustrative example only, never a
//! calibrated vehicle prediction. Everything here is reached as `fky_lapsim_core::lap::...`; it is
//! not re-exported at the crate root.
pub mod optimization;
mod vehicle;
pub use vehicle::*;
