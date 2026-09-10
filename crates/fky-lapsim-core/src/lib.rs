//! Native suspension model and numerical engine. Coordinates and forces use SI units.
pub mod kinematics;
pub mod metrics;
pub mod model;
pub mod study;
pub use kinematics::{solve_corner, CornerState, Points};
pub use metrics::Metrics;
pub use model::{Chassis, Corner, CornerId, Error, Point, Project, PushrodBody, SpringDamper};
pub use study::{simulate, simulate_on_road, sweep, Motion, Sample, VehicleState};
