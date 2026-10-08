//! Quasi-steady-state lap simulation after T. Zacharelis, *Vehicle Dynamics and Performance
//! Simulation* (NTUA MSc thesis, 2023), coupled to a full-car 7×7 suspension matrix built from the
//! vehicle's real geometry. Equation numbers in doc comments refer to that thesis.
//!
//! Units are SI; axes follow `docs/model-conventions.md` (X forward, Y left, Z up).
pub mod aeromap;
pub mod api;
pub mod apex;
pub mod channels;
pub mod coupled;
pub mod forces;
pub mod fullcar;
pub mod ggv;
pub mod rates;
pub mod scenarios;
pub mod solver;
pub mod thesis;
pub mod track_model;
pub mod tractive;
pub mod vehicle;

pub use api::{simulate_lap, LapRequest, LapRun};

/// Standard gravity used by the thesis (m/s²).
pub const G: f64 = 9.81;

/// Build a crate [`Error`](crate::Error) with a `lapsim:` prefix.
pub(crate) fn err(message: impl Into<String>) -> crate::Error {
    crate::Error {
        message: format!("lapsim: {}", message.into()),
    }
}
