//! Native suspension model and numerical engine. Coordinates and forces use SI units.
pub mod dynamics;
pub mod kinematics;
pub mod metrics;
pub mod model;
pub mod study;
pub use dynamics::{ride, RideMode, RideRequest, RideRun, RideSample, RideTermination, RoadInput};
pub use kinematics::{solve_corner, CornerState, Points};
pub use metrics::Metrics;
pub use model::{
    BodyMass, Chassis, ComponentMasses, Corner, CornerId, Error, Point, Project, PushrodBody,
    SpringDamper, TireProfile,
};
pub use study::{simulate, simulate_on_road, sweep, Motion, Sample, VehicleState};
pub mod analysis;

pub use analysis::{
    analyze, analyze_with_step, Analysis, AxleAnalysis, CornerAnalysis, OptionalValue,
    ProjectedCenter,
};
mod mass;
pub mod optimize;
