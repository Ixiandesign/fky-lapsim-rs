//! Bounded, deterministic prescribed-motion grids in SI units.
use crate::{Error, Motion};
use serde::{Deserialize, Serialize};

/// Inclusive range for one motion coordinate. A fixed coordinate has count 1
/// and equal endpoints. Translational axes use metres; angles use radians.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AxisRange {
    /// First coordinate value.
    pub start: f64,
    /// Last coordinate value; may be smaller than start.
    pub end: f64,
    /// Number of endpoint-inclusive samples.
    pub count: usize,
}
impl Default for AxisRange {
    fn default() -> Self {
        Self {
            start: 0.,
            end: 0.,
            count: 1,
        }
    }
}
/// Whether independently sampled axes form a grid or move together along a path.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GridMode {
    /// Every combination, with rear rack varying fastest and heave slowest.
    #[default]
    Cartesian,
    /// All varying axes share a count and advance together. Fixed axes remain fixed.
    Linked,
}
/// A motion study that can be saved without allocating its expanded samples.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MotionGrid {
    /// Combination rule.
    pub mode: GridMode,
    /// Chassis vertical displacement in metres.
    pub heave: AxisRange,
    /// Chassis roll in radians.
    pub roll: AxisRange,
    /// Chassis pitch in radians.
    pub pitch: AxisRange,
    /// Front rack travel in metres.
    pub rack_front: AxisRange,
    /// Rear rack travel in metres.
    pub rack_rear: AxisRange,
}
/// Generate up to 10,000 finite motions, validating counts before allocating.
pub fn motion_grid(g: &MotionGrid) -> Result<Vec<Motion>, Error> {
    let axes = [g.heave, g.roll, g.pitch, g.rack_front, g.rack_rear];
    let fail = |message: &str| Error {
        message: message.into(),
    };
    for a in &axes {
        if !a.start.is_finite()
            || !a.end.is_finite()
            || !(a.end - a.start).is_finite()
            || a.count == 0
            || a.count > 10_000
            || (a.count == 1 && a.start != a.end)
        {
            return Err(fail("each axis needs finite endpoints and 1..10000 samples; one sample requires equal endpoints"));
        }
    }
    let count = match g.mode {
        GridMode::Cartesian => axes.iter().try_fold(1usize, |n, a| {
            n.checked_mul(a.count)
                .filter(|n| *n <= 10_000)
                .ok_or_else(|| fail("combined motion grid exceeds 10000 samples"))
        })?,
        GridMode::Linked => {
            let n = axes.iter().map(|a| a.count).max().unwrap_or(1);
            if axes.iter().any(|a| a.count != 1 && a.count != n) {
                return Err(fail("linked varying axes must have the same sample count"));
            }
            n
        }
    };
    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let mut indices = [0; 5];
        let mut remainder = i;
        for k in (0..5).rev() {
            indices[k] = match g.mode {
                GridMode::Cartesian => {
                    let j = remainder % axes[k].count;
                    remainder /= axes[k].count;
                    j
                }
                GridMode::Linked => {
                    if axes[k].count == 1 {
                        0
                    } else {
                        i
                    }
                }
            };
        }
        let values: [f64; 5] = std::array::from_fn(|k| {
            let a = axes[k];
            if a.count == 1 {
                a.start
            } else if indices[k] == a.count - 1 {
                a.end
            } else {
                a.start + (a.end - a.start) * (indices[k] as f64 / (a.count - 1) as f64)
            }
        });
        result.push(Motion {
            heave: values[0],
            roll: values[1],
            pitch: values[2],
            rack_front: values[3],
            rack_rear: values[4],
        });
    }
    Ok(result)
}
