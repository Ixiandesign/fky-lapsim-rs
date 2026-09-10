use serde::{Deserialize, Serialize};
use std::fmt;

/// Coordinates in metres: x forward, y left, z up.
pub type Point = [f64; 3];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CornerId {
    FrontLeft,
    FrontRight,
    RearLeft,
    RearRight,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PushrodBody {
    UpperArm,
    LowerArm,
    Knuckle,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpringDamper {
    /// Linear spring rate in N/m.
    pub spring_rate: f64,
    /// Positive compression preload in N at the design pose.
    pub preload: f64,
    /// Viscous damping in N s/m.
    pub compression_damping: f64,
    pub rebound_damping: f64,
    /// Optional [compression m, total spring force N] curve replaces rate/preload.
    /// Piecewise linear, with constant force beyond endpoints; bilateral forces allowed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spring_curve: Option<Vec<[f64; 2]>>,
    /// Optional [speed m/s, force magnitude N] curves replace viscous coefficients.
    /// Start at [0,0], interpolate linearly, hold the last magnitude above the table.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compression_curve: Option<Vec<[f64; 2]>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rebound_curve: Option<Vec<[f64; 2]>>,
}
impl Default for SpringDamper {
    fn default() -> Self {
        Self {
            spring_rate: 30000.0,
            preload: 0.0,
            compression_damping: 1500.0,
            rebound_damping: 2000.0,
            spring_curve: None,
            compression_curve: None,
            rebound_curve: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Chassis {
    pub sprung_mass: f64,
    pub center_of_mass: Point,
    /// Principal inertia about the center of mass in kg m², aligned to chassis axes.
    pub inertia: Point,
}
impl Default for Chassis {
    fn default() -> Self {
        Self {
            sprung_mass: 1000.0,
            center_of_mass: [0.0, 0.0, 0.5],
            inertia: [400.0, 1200.0, 1400.0],
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TireProfile {
    #[default]
    Disk,
    Cylinder,
    Torus,
}
fn default_rack_axis() -> [Point; 2] {
    [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Corner {
    pub id: CornerId,
    pub upper_front: Point,
    pub upper_rear: Point,
    pub lower_front: Point,
    pub lower_rear: Point,
    pub upper_ball: Point,
    pub lower_ball: Point,
    pub steering_inner: Point,
    pub steering_outer: Point,
    pub wheel_center: Point,
    /// Two distinct points define the spindle axis (direction has no sign convention).
    pub spindle_axis: [Point; 2],
    pub tire_radius: f64,
    #[serde(default)]
    pub tire_profile: TireProfile,
    #[serde(default = "default_rack_axis")]
    pub rack_axis: [Point; 2],
    /// Full width for cylinder/torus envelopes.
    pub tire_width: f64,
    pub pushrod_body: PushrodBody,
    pub pushrod_pickup: Point,
    pub rocker_axis: [Point; 2],
    pub rocker_pushrod: Point,
    pub rocker_shock: Point,
    pub shock_chassis: Point,
    #[serde(default)]
    pub spring_damper: SpringDamper,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Project {
    pub schema_version: u32,
    pub name: String,
    #[serde(default)]
    pub chassis: Chassis,
    /// Exactly one of each CornerId; array order does not identify a corner.
    pub corners: [Corner; 4],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    pub message: String,
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.message.fmt(f)
    }
}
impl std::error::Error for Error {}

impl Project {
    pub fn validate(&self) -> Result<(), Error> {
        require(self.schema_version == 1, "unsupported schema version")?;
        require(
            self.chassis.sprung_mass.is_finite() && self.chassis.sprung_mass > 0.0,
            "sprung mass must be finite and positive",
        )?;
        require(
            self.chassis.center_of_mass.iter().all(|x| x.is_finite()),
            "nonfinite chassis center of mass",
        )?;
        let inertia = self.chassis.inertia;
        require(
            inertia.iter().all(|x| x.is_finite() && *x > 0.0),
            "principal inertia must be finite and positive",
        )?;
        for i in 0..3 {
            require(
                inertia[i] / 2.0 <= inertia[(i + 1) % 3] / 2.0 + inertia[(i + 2) % 3] / 2.0,
                "principal inertia violates triangle inequality",
            )?;
        }
        let mut ids = std::collections::HashSet::new();
        for corner in &self.corners {
            require(
                ids.insert(corner.id),
                "duplicate corner ID: all four distinct corner IDs are required",
            )?;
            corner.validate()?;
        }
        Ok(())
    }

    /// Equal-length parallel wishbones; a matching tie rod preserves upright orientation.
    /// Left/right geometry is mirrored in y, front/rear geometry translated in x.
    pub fn example() -> Self {
        let corner = |id, x: f64, side: f64| {
            let p = |dx: f64, y: f64, z: f64| [x + dx, side * y, z];
            Corner {
                id,
                upper_front: p(0.2, 0.4, 0.6),
                upper_rear: p(-0.2, 0.4, 0.6),
                lower_front: p(0.2, 0.4, 0.3),
                lower_rear: p(-0.2, 0.4, 0.3),
                upper_ball: p(0.0, 0.8, 0.6),
                lower_ball: p(0.0, 0.8, 0.3),
                steering_inner: p(-0.15, 0.4, 0.45),
                steering_outer: p(-0.15, 0.8, 0.45),
                wheel_center: p(0.0, 0.9, 0.3),
                spindle_axis: [p(0.0, 0.8, 0.3), p(0.0, 1.0, 0.3)],
                tire_radius: 0.3,
                tire_width: 0.2,
                tire_profile: TireProfile::Disk,
                rack_axis: default_rack_axis(),
                pushrod_body: PushrodBody::LowerArm,
                pushrod_pickup: p(0.0, 0.7, 0.3),
                rocker_axis: [p(-0.1, 0.4, 0.8), p(0.1, 0.4, 0.8)],
                rocker_pushrod: p(0.0, 0.55, 0.8),
                rocker_shock: p(0.0, 0.4, 0.95),
                shock_chassis: p(0.0, 0.15, 0.8),
                spring_damper: SpringDamper::default(),
            }
        };
        Self {
            schema_version: 1,
            name: "Analytic parallelogram".into(),
            chassis: Chassis::default(),
            corners: [
                corner(CornerId::FrontLeft, 1.3, 1.0),
                corner(CornerId::FrontRight, 1.3, -1.0),
                corner(CornerId::RearLeft, -1.3, 1.0),
                corner(CornerId::RearRight, -1.3, -1.0),
            ],
        }
    }
}

fn require(condition: bool, message: impl Into<String>) -> Result<(), Error> {
    if condition {
        Ok(())
    } else {
        Err(Error {
            message: message.into(),
        })
    }
}
fn sub(a: Point, b: Point) -> Point {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn norm(a: Point) -> f64 {
    a[0].hypot(a[1]).hypot(a[2])
}
fn distinct(a: Point, b: Point) -> bool {
    let n = norm(sub(a, b));
    n.is_finite() && n > 1e-10
}
fn triangle(a: Point, b: Point, c: Point) -> bool {
    let u = sub(b, a);
    let v = sub(c, a);
    let un = norm(u);
    let vn = norm(v);
    if !un.is_finite() || !vn.is_finite() || un <= 1e-10 || vn <= 1e-10 {
        return false;
    }
    let u = u.map(|x| x / un);
    let v = v.map(|x| x / vn);
    norm([
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ]) > 1e-10
}

impl Corner {
    pub fn validate(&self) -> Result<(), Error> {
        let c = self;
        let points = [
            c.rack_axis[0],
            c.rack_axis[1],
            c.upper_front,
            c.upper_rear,
            c.lower_front,
            c.lower_rear,
            c.upper_ball,
            c.lower_ball,
            c.steering_inner,
            c.steering_outer,
            c.wheel_center,
            c.spindle_axis[0],
            c.spindle_axis[1],
            c.pushrod_pickup,
            c.rocker_axis[0],
            c.rocker_axis[1],
            c.rocker_pushrod,
            c.rocker_shock,
            c.shock_chassis,
        ];
        require(
            points.iter().flatten().all(|x| x.is_finite()),
            format!("{:?}: nonfinite hardpoint", c.id),
        )?;
        for (label, a, b) in [
            ("rack axis", c.rack_axis[0], c.rack_axis[1]),
            ("upper axis", c.upper_front, c.upper_rear),
            ("lower axis", c.lower_front, c.lower_rear),
            ("spindle axis", c.spindle_axis[0], c.spindle_axis[1]),
            ("rocker axis", c.rocker_axis[0], c.rocker_axis[1]),
            ("kingpin", c.upper_ball, c.lower_ball),
            ("tie rod", c.steering_inner, c.steering_outer),
            ("pushrod", c.pushrod_pickup, c.rocker_pushrod),
            ("shock", c.rocker_shock, c.shock_chassis),
        ] {
            require(distinct(a, b), format!("{:?}: degenerate {label}", c.id))?;
        }
        for (label, a, b, d) in [
            ("upper wishbone", c.upper_front, c.upper_rear, c.upper_ball),
            ("lower wishbone", c.lower_front, c.lower_rear, c.lower_ball),
            (
                "upright steering",
                c.upper_ball,
                c.lower_ball,
                c.steering_outer,
            ),
            (
                "rocker pushrod lever",
                c.rocker_axis[0],
                c.rocker_axis[1],
                c.rocker_pushrod,
            ),
            (
                "rocker shock lever",
                c.rocker_axis[0],
                c.rocker_axis[1],
                c.rocker_shock,
            ),
        ] {
            require(triangle(a, b, d), format!("{:?}: degenerate {label}", c.id))?;
        }
        for (label, v) in [
            ("tire radius", c.tire_radius),
            ("tire width", c.tire_width),
            ("spring rate", c.spring_damper.spring_rate),
        ] {
            require(
                v.is_finite() && v > 0.0,
                format!("{:?}: {label} must be finite and positive", c.id),
            )?;
        }
        require(
            c.tire_profile != TireProfile::Torus || c.tire_width / 2.0 < c.tire_radius,
            "torus requires half width smaller than outer radius",
        )?;
        for (label, v) in [
            ("preload", c.spring_damper.preload),
            ("compression damping", c.spring_damper.compression_damping),
            ("rebound damping", c.spring_damper.rebound_damping),
        ] {
            require(
                v.is_finite() && v >= 0.0,
                format!("{:?}: {label} must be finite and nonnegative", c.id),
            )?;
        }
        for (name, curve, damper) in [
            ("spring", &c.spring_damper.spring_curve, false),
            (
                "compression damper",
                &c.spring_damper.compression_curve,
                true,
            ),
            ("rebound damper", &c.spring_damper.rebound_curve, true),
        ] {
            if let Some(curve) = curve {
                require(
                    curve.len() >= 2
                        && curve.len() <= 10000
                        && curve.iter().flatten().all(|x| x.is_finite())
                        && curve.windows(2).all(|w| {
                            w[1][0] > w[0][0]
                                && ((w[1][1] - w[0][1]) / (w[1][0] - w[0][0])).is_finite()
                        }),
                    format!("{:?}: invalid {name} force table", c.id),
                )?;
                if damper {
                    require(
                        curve[0] == [0.0, 0.0] && curve.iter().all(|p| p[0] >= 0.0 && p[1] >= 0.0),
                        format!(
                            "{:?}: {name} table must start at zero and remain passive",
                            c.id
                        ),
                    )?;
                }
            }
        }
        Ok(())
    }
}
