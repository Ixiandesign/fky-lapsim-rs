//! Four-contact planar rigid-body force balance (Pacejka §1.3.1).
//!
//! X forward, Y left, Z up. Exact steer rotations and individual contact velocity
//! offsets are retained. This primitive has three planar degrees of freedom; it
//! does not itself supply suspension loads, roll dynamics, wheel spin or a driver.
//! Book-frame tire outputs (Y right, Z down) require explicit Fy/Mz sign reversal.
use crate::Error;
use serde::{Deserialize, Serialize};

/// Vehicle CG pose and velocity in a level plane.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct PlanarState {
    /// World X position, metres.
    pub x_m: f64,
    /// World Y position, metres.
    pub y_m: f64,
    /// Counterclockwise heading from world X, radians.
    pub heading_rad: f64,
    /// Body forward velocity, m/s.
    pub u_m_s: f64,
    /// Body leftward velocity, m/s.
    pub v_m_s: f64,
    /// Counterclockwise yaw velocity, rad/s.
    pub yaw_rate_rad_s: f64,
}
/// Force at one wheel, expressed in the wheel's X-forward/Y-left frame.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct WheelForce {
    /// Contact point relative to CG in body coordinates, metres.
    pub position_m: [f64; 3],
    /// Wheel heading relative to body X, counterclockwise radians.
    pub steer_rad: f64,
    /// Wheel longitudinal force, N.
    pub fx_n: f64,
    /// Wheel leftward lateral force, N.
    pub fy_n: f64,
    /// Wheel self-aligning moment, positive counterclockwise, N m.
    pub mz_nm: f64,
}
/// Time derivatives and force sums, including rotating-frame transport terms.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PlanarDerivative {
    /// Global X velocity, m/s.
    pub x_m_s: f64,
    /// Global Y velocity, m/s.
    pub y_m_s: f64,
    /// Heading derivative, rad/s.
    pub heading_rad_s: f64,
    /// Body-coordinate longitudinal velocity derivative, m/s².
    pub u_m_s2: f64,
    /// Body-coordinate lateral velocity derivative, m/s².
    pub v_m_s2: f64,
    /// Yaw acceleration, rad/s².
    pub yaw_acceleration_rad_s2: f64,
    /// Total body force Fx, Fy and yaw moment Mz, N/N/N m.
    pub generalized_force: [f64; 3],
}
fn finite_state(s: PlanarState) -> bool {
    [
        s.x_m,
        s.y_m,
        s.heading_rad,
        s.u_m_s,
        s.v_m_s,
        s.yaw_rate_rad_s,
    ]
    .iter()
    .all(|x| x.is_finite())
}
fn error() -> Error {
    Error {
        message: "planar balance requires finite values and positive mass/inertia".into(),
    }
}

/// Rigid contact velocity in the steered wheel frame, including the left/right
/// yaw offset. Does not include suspension-point velocity or wheel spin.
pub fn contact_velocity(
    s: PlanarState,
    position_m: [f64; 3],
    steer_rad: f64,
) -> Result<[f64; 2], Error> {
    if !finite_state(s) || !position_m.iter().all(|x| x.is_finite()) || !steer_rad.is_finite() {
        return Err(error());
    }
    let vx = s.u_m_s - s.yaw_rate_rad_s * position_m[1];
    let vy = s.v_m_s + s.yaw_rate_rad_s * position_m[0];
    let (sn, cs) = steer_rad.sin_cos();
    let out = [cs * vx + sn * vy, -sn * vx + cs * vy];
    if !out.iter().all(|v| v.is_finite()) {
        return Err(error());
    }
    Ok(out)
}

/// Newton–Euler planar balance for four independent wheel contacts. `external`
/// is body Fx/Fy/Mz, for example aero drag and its CG-relative moment.
pub fn balance(
    mass_kg: f64,
    yaw_inertia_kg_m2: f64,
    s: PlanarState,
    wheels: &[WheelForce; 4],
    external: [f64; 3],
) -> Result<PlanarDerivative, Error> {
    if !finite_state(s)
        || !mass_kg.is_finite()
        || mass_kg <= 0.
        || !yaw_inertia_kg_m2.is_finite()
        || yaw_inertia_kg_m2 <= 0.
        || !external.iter().all(|v| v.is_finite())
    {
        return Err(error());
    }
    let mut total = external;
    for w in wheels {
        if !w
            .position_m
            .iter()
            .chain([w.steer_rad, w.fx_n, w.fy_n, w.mz_nm].iter())
            .all(|v| v.is_finite())
        {
            return Err(error());
        }
        let (sn, cs) = w.steer_rad.sin_cos();
        let fx = cs * w.fx_n - sn * w.fy_n;
        let fy = sn * w.fx_n + cs * w.fy_n;
        total[0] += fx;
        total[1] += fy;
        total[2] += w.position_m[0] * fy - w.position_m[1] * fx + w.mz_nm;
    }
    let (sn, cs) = s.heading_rad.sin_cos();
    let out = PlanarDerivative {
        x_m_s: cs * s.u_m_s - sn * s.v_m_s,
        y_m_s: sn * s.u_m_s + cs * s.v_m_s,
        heading_rad_s: s.yaw_rate_rad_s,
        u_m_s2: total[0] / mass_kg + s.yaw_rate_rad_s * s.v_m_s,
        v_m_s2: total[1] / mass_kg - s.yaw_rate_rad_s * s.u_m_s,
        yaw_acceleration_rad_s2: total[2] / yaw_inertia_kg_m2,
        generalized_force: total,
    };
    if ![
        out.x_m_s,
        out.y_m_s,
        out.u_m_s2,
        out.v_m_s2,
        out.yaw_acceleration_rad_s2,
    ]
    .iter()
    .chain(total.iter())
    .all(|v| v.is_finite())
    {
        return Err(error());
    }
    Ok(out)
}
