//! Radius-and-length track model (thesis §4.2): the quantity the lap solver needs.
use super::err;
use crate::track::Track;
use crate::Error;
use serde::{Deserialize, Serialize};

/// Radius assigned to straight sections (thesis: "exceptionally high, e.g. 10E+5").
pub const STRAIGHT_RADIUS_M: f64 = 1.0e5;

/// A closed lap as points with a signed radius (positive = left) and the distance of each point
/// along the lap. Section `i` runs from point `i` to point `i+1` (the last wraps to the first).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrackModel {
    /// Distance of each point along the lap, m, strictly increasing from the start.
    pub distance_m: Vec<f64>,
    /// Signed corner radius at each point, m (`±STRAIGHT_RADIUS_M` for straights).
    pub radius_m: Vec<f64>,
    /// Total lap length, m (greater than the last point's distance).
    pub length_m: f64,
}

fn radius_from_curvature(c: f64) -> f64 {
    if c.abs() < 1.0 / STRAIGHT_RADIUS_M {
        STRAIGHT_RADIUS_M
    } else {
        1.0 / c
    }
}

impl TrackModel {
    /// Validate and build a track model.
    ///
    /// # Errors
    /// Fewer than 3 points, mismatched lengths, nonfinite/zero radius, non-increasing distance, or
    /// `length_m` not beyond the last point.
    pub fn new(distance_m: Vec<f64>, radius_m: Vec<f64>, length_m: f64) -> Result<Self, Error> {
        let n = distance_m.len();
        let ok = n >= 3
            && radius_m.len() == n
            && length_m.is_finite()
            && distance_m.iter().all(|d| d.is_finite())
            && radius_m.iter().all(|r| r.is_finite() && *r != 0.0)
            && distance_m.windows(2).all(|w| w[1] > w[0])
            && length_m > distance_m[n - 1];
        if !ok {
            return Err(err("invalid track model"));
        }
        Ok(Self {
            distance_m,
            radius_m,
            length_m,
        })
    }

    /// Build from `(signed radius, section length)` rows (thesis Table 4-1).
    ///
    /// # Errors
    /// As [`Self::new`], plus nonpositive section lengths.
    pub fn from_radius_length(rows: &[(f64, f64)]) -> Result<Self, Error> {
        if rows.iter().any(|(_, l)| !l.is_finite() || *l <= 0.0) {
            return Err(err("nonpositive section length"));
        }
        let (mut d, mut distance, mut radius) = (0.0, vec![], vec![]);
        for (r, l) in rows {
            distance.push(d);
            radius.push(*r);
            d += l;
        }
        Self::new(distance, radius, d)
    }

    /// Sample a [`Track`] polygon every `step_m` metres and convert curvature to radius.
    ///
    /// # Errors
    /// Invalid track or nonpositive `step_m`.
    pub fn from_track(track: &Track, step_m: f64) -> Result<Self, Error> {
        if !step_m.is_finite() || step_m <= 0.0 {
            return Err(err("nonpositive step"));
        }
        let length = track.length_m().map_err(err)?;
        let n = ((length / step_m).round() as usize).max(3);
        let ds = length / n as f64;
        let (mut distance, mut radius) = (vec![], vec![]);
        for k in 0..n {
            let s = k as f64 * ds;
            let sample = track.sample(s).map_err(err)?;
            distance.push(s);
            radius.push(radius_from_curvature(sample.curvature_per_m));
        }
        Self::new(distance, radius, length)
    }

    /// Number of points.
    pub fn len(&self) -> usize {
        self.distance_m.len()
    }
    /// True when there are no points (never for a validated track).
    pub fn is_empty(&self) -> bool {
        self.distance_m.is_empty()
    }
    /// Length of the section starting at point `i`, m (the last wraps to the first).
    pub fn section_m(&self, i: usize) -> f64 {
        if i + 1 < self.len() {
            self.distance_m[i + 1] - self.distance_m[i]
        } else {
            self.length_m - self.distance_m[i]
        }
    }
    fn curvature(&self, i: usize) -> f64 {
        1.0 / self.radius_m[i]
    }

    /// Resample sections that touch a tight corner (§4.2.3.2): every point with
    /// `|radius| < threshold_radius_m`, extended by `span_m` before and after, has its following
    /// section subdivided to at most `step_m`, interpolating curvature linearly.
    ///
    /// # Errors
    /// Nonpositive parameters.
    pub fn fine_mesh(
        &self,
        threshold_radius_m: f64,
        span_m: f64,
        step_m: f64,
    ) -> Result<Self, Error> {
        if ![threshold_radius_m, span_m, step_m]
            .iter()
            .all(|v| v.is_finite() && *v > 0.0)
        {
            return Err(err("invalid fine-mesh parameters"));
        }
        let n = self.len();
        let tight: Vec<bool> = (0..n)
            .map(|i| self.radius_m[i].abs() < threshold_radius_m)
            .collect();
        let mut mark = tight.clone();
        for i in 0..n {
            if !tight[i] {
                continue;
            }
            let (mut acc, mut j) = (0.0, i);
            while acc < span_m {
                j = (j + n - 1) % n;
                acc += self.section_m(j);
                mark[j] = true;
                if j == i {
                    break;
                }
            }
            let (mut acc, mut j) = (0.0, i);
            while acc < span_m {
                acc += self.section_m(j);
                j = (j + 1) % n;
                mark[j] = true;
                if j == i {
                    break;
                }
            }
        }
        let (mut distance, mut radius) = (vec![], vec![]);
        for i in 0..n {
            if !mark[i] {
                distance.push(self.distance_m[i]);
                radius.push(self.radius_m[i]);
                continue;
            }
            let sec = self.section_m(i);
            let k = (sec / step_m).ceil().max(1.0) as usize;
            let (c0, c1) = (self.curvature(i), self.curvature((i + 1) % n));
            for j in 0..k {
                let x = j as f64 / k as f64;
                distance.push(self.distance_m[i] + x * sec);
                radius.push(radius_from_curvature(c0 + (c1 - c0) * x));
            }
        }
        Self::new(distance, radius, self.length_m)
    }

    /// Moving average of curvature over `window` points (even values are raised to the next odd),
    /// periodic (§4.2.3.1). Distances and length are unchanged.
    pub fn moving_average_curvature(&self, window: usize) -> Self {
        let n = self.len();
        let w = if window % 2 == 0 { window + 1 } else { window }.max(1);
        let half = (w / 2) as isize;
        let radius = (0..n)
            .map(|i| {
                let s: f64 = (-half..=half)
                    .map(|k| self.curvature(((i as isize + k).rem_euclid(n as isize)) as usize))
                    .sum();
                radius_from_curvature(s / w as f64)
            })
            .collect();
        Self {
            distance_m: self.distance_m.clone(),
            radius_m: radius,
            length_m: self.length_m,
        }
    }
}
