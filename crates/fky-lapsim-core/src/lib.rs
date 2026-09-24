//! Native suspension model and numerical engine. Coordinates and forces use SI units.
#![warn(missing_docs)]

pub mod dynamics;
pub mod kinematics;
pub mod metrics;
pub mod model;
mod motion_grid;
pub mod results;
pub mod tire;
pub mod planar;
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
