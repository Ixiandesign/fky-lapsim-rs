//! Apex-corrected maximum cornering speed (thesis §4.2.4.3, Fig. 4-43) and apex finding.
use super::err;
use super::forces::{max_ay, StepModel};
use super::scenarios::corner;
use super::track_model::{TrackModel, STRAIGHT_RADIUS_M};
use super::tractive::TractiveTable;
use crate::Error;
use std::collections::HashMap;

/// Highest speed at which `min(engine, tyre acceleration force) ≥ drag + rolling resistance`.
///
/// # Errors
/// Propagates model errors.
pub fn top_speed(model: &dyn StepModel, tractive: &TractiveTable) -> Result<f64, Error> {
    let net = |v: f64| -> Result<f64, Error> {
        let f = model.instant(v, 0.0, 0.0)?.forces;
        Ok(f.tyres_acc_n().min(tractive.force_at(v)) - f.resist_n())
    };
    let hi = tractive.top_speed_m_s();
    if net(hi)? >= 0.0 {
        return Ok(hi);
    }
    let (mut lo, mut hi) = (0.0, hi);
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if net(mid)? > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(lo)
}

/// Corrected apex speed on `radius_m` (Fig. 4-43): at an apex ΣFx = 0, so tyre longitudinal
/// grip must cancel drag + rolling, leaving `ay_lim·sqrt(1 − (ax_needed/ax_tyre)²)` for cornering.
///
/// # Errors
/// Nonfinite/zero radius, nonconvergence, or model errors.
pub fn vmax_point(
    model: &dyn StepModel,
    tractive: &TractiveTable,
    radius_m: f64,
) -> Result<f64, Error> {
    if !radius_m.is_finite() || radius_m == 0.0 {
        return Err(err("invalid radius"));
    }
    let top = top_speed(model, tractive)?;
    let r = radius_m.abs();
    if r >= STRAIGHT_RADIUS_M * 0.1 {
        return Ok(top);
    }
    let m = model.params().mass_kg;
    let mut v = corner(model, top, radius_m)?.speed_m_s.min(top);
    for _ in 0..200 {
        let ay_need = v * v / r * radius_m.signum();
        let f = model.instant(v, 0.0, ay_need)?.forces;
        let ax_needed = f.resist_n() / m;
        let ax_tyre = f.tyres_acc_n() / m;
        let ay_lim = max_ay(model, v, 0.0)?;
        let ay_remain = if ax_tyre > ax_needed && ax_tyre > 0.0 {
            ay_lim * (1.0 - (ax_needed / ax_tyre).powi(2)).sqrt()
        } else {
            0.0
        };
        let vn = (ay_remain * r).sqrt().min(top);
        if (vn - v).abs() < 1e-9 * v.max(1.0) {
            return Ok(vn);
        }
        v = 0.5 * (v + vn);
    }
    Err(err("apex speed did not converge"))
}

/// Apex-corrected maximum speed at every track point (memoised by radius).
///
/// # Errors
/// Propagates [`vmax_point`] errors.
pub fn vmax_profile(
    model: &dyn StepModel,
    tractive: &TractiveTable,
    track: &TrackModel,
) -> Result<Vec<f64>, Error> {
    let mut cache: HashMap<u64, f64> = HashMap::new();
    let mut out = Vec::with_capacity(track.len());
    for &r in &track.radius_m {
        let key = r.to_bits();
        if let Some(v) = cache.get(&key) {
            out.push(*v);
        } else {
            let v = vmax_point(model, tractive, r)?;
            cache.insert(key, v);
            out.push(v);
        }
    }
    Ok(out)
}

fn periodic_gap(a: f64, b: f64, length: f64) -> f64 {
    let d = (a - b).abs();
    d.min(length - d)
}

/// Apex indices of `vmax` (see [`smooth_clusters`]) without modifying the input.
pub fn find_apexes(
    vmax: &[f64],
    distance_m: &[f64],
    length_m: f64,
    min_spacing_m: f64,
) -> Vec<usize> {
    let mut v = vmax.to_vec();
    smooth_clusters(&mut v, distance_m, length_m, min_spacing_m)
}

/// In-place cluster smoothing (§4.2.4.3): find local minima, merge those within `min_spacing_m`
/// keeping the lowest, and replace the points between the first and last merged candidate with a
/// piecewise-linear interpolation through the real apex. Returns the apex indices in order. A
/// profile with no local minimum (flat) has a single apex at its first lowest point.
pub fn smooth_clusters(
    vmax: &mut [f64],
    distance_m: &[f64],
    length_m: f64,
    min_spacing_m: f64,
) -> Vec<usize> {
    let n = vmax.len();
    let top = vmax.iter().cloned().fold(f64::MIN, f64::max);
    // A constant-radius corner is a plateau of equal values; its apex is the middle of the plateau
    // (thesis §4.2.4.1: "the middle of the corner for a corner with constant radius"). Treat each
    // maximal run of equal values as one candidate, located at the run's midpoint.
    let eq = |a: f64, b: f64| (a - b).abs() <= 1e-9 * b.abs().max(1.0);
    let mut cands: Vec<usize> = vec![];
    if let Some(s) = (0..n).find(|&i| !eq(vmax[i], vmax[(i + n - 1) % n])) {
        let mut i = 0;
        while i < n {
            let val = vmax[(s + i) % n];
            let mut len = 1;
            while i + len < n && eq(vmax[(s + i + len) % n], val) {
                len += 1;
            }
            let prev = vmax[(s + i + n - 1) % n];
            let next = vmax[(s + i + len) % n];
            if val < top - 1e-9 && prev > val && next > val {
                cands.push((s + i + len / 2) % n);
            }
            i += len;
        }
    }
    if cands.is_empty() {
        let (imin, _) =
            vmax.iter()
                .enumerate()
                .fold((0, f64::MAX), |a, (i, &v)| if v < a.1 { (i, v) } else { a });
        return vec![imin];
    }
    cands.sort_unstable();
    let mut groups: Vec<Vec<usize>> = vec![vec![cands[0]]];
    for &c in &cands[1..] {
        let last = *groups.last().unwrap().last().unwrap();
        if periodic_gap(distance_m[c], distance_m[last], length_m) < min_spacing_m {
            groups.last_mut().unwrap().push(c);
        } else {
            groups.push(vec![c]);
        }
    }
    let mut apexes = vec![];
    for g in groups {
        let apex = *g
            .iter()
            .min_by(|&&a, &&b| vmax[a].partial_cmp(&vmax[b]).unwrap())
            .unwrap();
        let (first, last) = (g[0], *g.last().unwrap());
        if g.len() > 1 {
            let (va, vl, vp) = (vmax[first], vmax[last], vmax[apex]);
            for i in first..=apex {
                let t = if apex == first {
                    0.0
                } else {
                    (i - first) as f64 / (apex - first) as f64
                };
                vmax[i] = va + (vp - va) * t;
            }
            for i in apex..=last {
                let t = if last == apex {
                    0.0
                } else {
                    (i - apex) as f64 / (last - apex) as f64
                };
                vmax[i] = vp + (vl - vp) * t;
            }
        }
        apexes.push(apex);
    }
    apexes.sort_unstable();
    apexes
}
