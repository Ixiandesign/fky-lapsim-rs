use crate::{CornerState, Error, Project};
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Motion {
    pub heave: f64,
    pub roll: f64,
    pub pitch: f64,
    pub rack_front: f64,
    pub rack_rear: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VehicleState {
    pub motion: Motion,
    pub corners: [CornerState; 4],
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sample {
    pub motion: Motion,
    pub state: Option<VehicleState>,
    pub error: Option<String>,
}
pub fn simulate(p: &Project, m: &Motion) -> Result<VehicleState, Error> {
    simulate_on_road(p, m, [0.0; 4])
}
/// Horizontal road heights in project corner array order. Chassis motion remains absolute.
pub fn simulate_on_road(
    p: &Project,
    m: &Motion,
    road_heights: [f64; 4],
) -> Result<VehicleState, Error> {
    use crate::kinematics::{continuation, err, v, Constraint, Frame, V};
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
        corners.push(continuation(c, steps as usize, |t| {
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
        })?);
    }
    Ok(VehicleState {
        motion: *m,
        corners: corners
            .try_into()
            .map_err(|_| err("expected four corners"))?,
    })
}
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
