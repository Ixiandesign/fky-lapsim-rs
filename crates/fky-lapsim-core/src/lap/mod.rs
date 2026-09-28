//! Four-wheel transient simulation on a closed planar track.
mod inertia;
pub use inertia::{angular_acceleration, angular_energy};
mod mechanics;
pub use mechanics::{suspension_forces, SuspensionForces};
pub mod optimization;
mod vehicle;
pub use vehicle::*;
mod runner;
pub use runner::*;
