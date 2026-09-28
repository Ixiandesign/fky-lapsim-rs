//! Closed planar centerline geometry in meters.
use serde::{Deserialize, Serialize};
/// Closed polygonal path with a continuous, interpolated curvature estimate.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Track {
    /// Ordered XY centerline points, meters; closure is implicit. An identical
    /// final copy of the first point is accepted and ignored.
    pub centerline_m: Vec<[f64; 2]>,
    /// Constant full track width, meters.
    pub width_m: f64,
}
/// Periodic centerline sample.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrackSample {
    /// Wrapped polygon arc length, meters.
    pub s_m: f64,
    /// Linear interpolation along the polygon, meters.
    pub position_m: [f64; 2],
    /// Unit tangent to the containing polygon segment.
    pub tangent: [f64; 2],
    /// Segment heading counterclockwise from +X, radians.
    pub heading_rad: f64,
    /// Signed three-point circumcircle curvature interpolated between vertices,
    /// 1/m. Positive is a left bend. This estimates the sampled smooth path,
    /// rather than the distributional curvature of the literal polygon.
    pub curvature_per_m: f64,
    /// Full width, meters.
    pub width_m: f64,
}
fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}
impl Track {
    fn points(&self) -> &[[f64; 2]] {
        let p = &self.centerline_m;
        if p.len() > 1 && p.first() == p.last() {
            &p[..p.len() - 1]
        } else {
            p
        }
    }
    /// Validate closure, finite geometry, positive width and nonzero edges.
    /// Self intersections are permitted (e.g. a mapped crossover course).
    pub fn validate(&self) -> Result<(), String> {
        let p = self.points();
        if !self.width_m.is_finite()
            || self.width_m <= 0.
            || p.len() < 3
            || p.iter().flatten().any(|v| !v.is_finite())
        {
            return Err("track needs finite points and positive width".into());
        }
        let mut length = 0.;
        for i in 0..p.len() {
            let d = distance(p[i], p[(i + 1) % p.len()]);
            if !d.is_finite() || d <= 1e-9 {
                return Err("track contains a zero-length or invalid segment".into());
            }
            let a = p[(i + p.len() - 1) % p.len()];
            let b = p[i];
            let c = p[(i + 1) % p.len()];
            if distance(a, c) <= 1e-9 {
                return Err("track reverses along the same segment".into());
            }
            let cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
            let dot = (b[0] - a[0]) * (c[0] - b[0]) + (b[1] - a[1]) * (c[1] - b[1]);
            if !cross.is_finite() || !dot.is_finite() || (cross.abs() < 1e-12 && dot < 0.) {
                return Err("track has an undefined turnaround".into());
            }
            length += d;
        }
        if !length.is_finite() {
            return Err("track length overflow".into());
        }
        Ok(())
    }
    /// Closed polygon length, meters.
    pub fn length_m(&self) -> Result<f64, String> {
        self.validate()?;
        let p = self.points();
        Ok((0..p.len())
            .map(|i| distance(p[i], p[(i + 1) % p.len()]))
            .sum())
    }
    fn curvature(p: &[[f64; 2]], i: usize) -> f64 {
        let a = p[(i + p.len() - 1) % p.len()];
        let b = p[i];
        let c = p[(i + 1) % p.len()];
        let ab = distance(a, b);
        let bc = distance(b, c);
        let ac = distance(a, c);
        2. * (((b[0] - a[0]) / ab) * ((c[1] - b[1]) / bc)
            - ((b[1] - a[1]) / ab) * ((c[0] - b[0]) / bc))
            / ac
    }
    /// Sample any finite arc length, wrapping both positive and negative laps.
    pub fn sample(&self, s_m: f64) -> Result<TrackSample, String> {
        let length = self.length_m()?;
        if !s_m.is_finite() {
            return Err("nonfinite track coordinate".into());
        }
        let s_m = s_m.rem_euclid(length);
        let p = self.points();
        let mut remaining = s_m;
        for i in 0..p.len() {
            let j = (i + 1) % p.len();
            let d = distance(p[i], p[j]);
            if remaining < d || i == p.len() - 1 {
                let t = (remaining / d).clamp(0., 1.);
                let tangent = [(p[j][0] - p[i][0]) / d, (p[j][1] - p[i][1]) / d];
                let position_m = [
                    p[i][0] + t * (p[j][0] - p[i][0]),
                    p[i][1] + t * (p[j][1] - p[i][1]),
                ];
                let curvature_per_m = (1. - t) * Self::curvature(p, i) + t * Self::curvature(p, j);
                return Ok(TrackSample {
                    s_m,
                    position_m,
                    tangent,
                    heading_rad: tangent[1].atan2(tangent[0]),
                    curvature_per_m,
                    width_m: self.width_m,
                });
            }
            remaining -= d;
        }
        Err("could not locate track segment".into())
    }
    /// Synthetic counterclockwise circular course, centered at the origin.
    pub fn circle(radius_m: f64, width_m: f64, segments: usize) -> Result<Self, String> {
        if !radius_m.is_finite() || radius_m <= 0. || !(3..=1_000_000).contains(&segments) {
            return Err("invalid circle radius or segment count".into());
        }
        let centerline_m = (0..segments)
            .map(|i| {
                let a = std::f64::consts::TAU * i as f64 / segments as f64;
                [radius_m * a.cos(), radius_m * a.sin()]
            })
            .collect();
        let track = Self {
            centerline_m,
            width_m,
        };
        track.validate()?;
        Ok(track)
    }
    /// Synthetic counterclockwise stadium: two straights of the specified
    /// length and two semicircles. Straight segments use comparable spacing.
    pub fn oval(
        straight_m: f64,
        radius_m: f64,
        width_m: f64,
        segments_per_arc: usize,
    ) -> Result<Self, String> {
        if !straight_m.is_finite()
            || straight_m < 0.
            || !radius_m.is_finite()
            || radius_m <= 0.
            || !(2..=100_000).contains(&segments_per_arc)
        {
            return Err("invalid oval geometry or segment count".into());
        }
        if straight_m == 0. {
            return Self::circle(radius_m, width_m, 2 * segments_per_arc);
        }
        let n = (straight_m / (std::f64::consts::PI * radius_m / segments_per_arc as f64)).ceil();
        if !n.is_finite() || n > 100_000. {
            return Err("oval requires too many straight segments".into());
        }
        let n = (n as usize).max(2);
        let h = straight_m / 2.;
        let mut centerline_m = Vec::new();
        for i in 0..n {
            centerline_m.push([-h + straight_m * i as f64 / n as f64, -radius_m]);
        }
        for i in 0..segments_per_arc {
            let a = -std::f64::consts::FRAC_PI_2
                + std::f64::consts::PI * i as f64 / segments_per_arc as f64;
            centerline_m.push([h + radius_m * a.cos(), radius_m * a.sin()]);
        }
        for i in 0..n {
            centerline_m.push([h - straight_m * i as f64 / n as f64, radius_m]);
        }
        for i in 0..segments_per_arc {
            let a = std::f64::consts::FRAC_PI_2
                + std::f64::consts::PI * i as f64 / segments_per_arc as f64;
            centerline_m.push([-h + radius_m * a.cos(), radius_m * a.sin()]);
        }
        let track = Self {
            centerline_m,
            width_m,
        };
        track.validate()?;
        Ok(track)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn oval_has_straights_and_closure_and_rejects_degeneracy() {
        let t = Track::oval(40., 10., 4., 128).unwrap();
        assert!((t.length_m().unwrap() - (80. + 20. * std::f64::consts::PI)).abs() < 0.002);
        assert!(t.sample(10.).unwrap().curvature_per_m.abs() < 1e-12);
        let mut duplicated = t.clone();
        duplicated.centerline_m.push(t.centerline_m[0]);
        assert_eq!(t.length_m().unwrap(), duplicated.length_m().unwrap());
        duplicated
            .centerline_m
            .insert(1, duplicated.centerline_m[0]);
        assert!(duplicated.validate().is_err());
        assert!(Track::circle(f64::NAN, 3., 32).is_err());
        assert!(t.sample(f64::INFINITY).is_err());
    }
    #[test]
    fn circle_length_converges_and_curvature_is_continuous() {
        let coarse = Track::circle(10., 3., 16).unwrap();
        let fine = Track::circle(10., 3., 128).unwrap();
        let exact = std::f64::consts::TAU * 10.;
        assert!(
            (fine.length_m().unwrap() - exact).abs()
                < (coarse.length_m().unwrap() - exact).abs() / 50.
        );
        for s in [0., 1., 5., 50., 63.] {
            assert!((fine.sample(s).unwrap().curvature_per_m - 0.1).abs() < 1e-10);
        }
        assert_eq!(
            fine.sample(0.).unwrap().position_m,
            fine.sample(fine.length_m().unwrap()).unwrap().position_m
        );
    }
}
