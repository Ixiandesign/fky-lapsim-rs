//! The thesis lap algorithm (§4.3): accelerate from every apex, brake to every apex, take the
//! minimum, re-insert the exact braking points.
use super::apex::{smooth_clusters, top_speed, vmax_profile};
use super::err;
use super::forces::StepModel;
use super::ggv::ellipse_remaining;
use super::track_model::TrackModel;
use super::tractive::TractiveTable;
use crate::Error;
use serde::{Deserialize, Serialize};

/// Lap simulation settings.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct LapSettings {
    /// Flying lap (periodic, no initial speed) or standing start.
    pub flying: bool,
    /// Standing-start initial speed at the first point, m/s.
    pub initial_speed_m_s: f64,
    /// Minimum distance between distinct apexes, m (thesis: 2 m Formula Student, >5 m F1).
    pub apex_min_spacing_m: f64,
}

impl Default for LapSettings {
    fn default() -> Self {
        Self {
            flying: true,
            initial_speed_m_s: 0.0,
            apex_min_spacing_m: 2.0,
        }
    }
}

/// Per-point lap results (after re-inserting braking points).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LapTrace {
    /// Distance along the lap, m.
    pub distance_m: Vec<f64>,
    /// Signed radius at each point, m.
    pub radius_m: Vec<f64>,
    /// Apex-corrected maximum speed, m/s.
    pub vmax_m_s: Vec<f64>,
    /// Final speed, m/s.
    pub speed_m_s: Vec<f64>,
    /// Longitudinal acceleration along the next section, m/s².
    pub ax_m_s2: Vec<f64>,
    /// Lateral acceleration `v²/|R|`, m/s².
    pub ay_m_s2: Vec<f64>,
    /// Elapsed time at each point, s.
    pub time_s: Vec<f64>,
    /// Indices of the apexes in this trace.
    pub apex_index: Vec<usize>,
    /// True for points inserted by braking-point re-processing.
    pub inserted: Vec<bool>,
}

/// Lap result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LapResult {
    /// Lap time, s.
    pub lap_time_s: f64,
    /// Per-point channels.
    pub trace: LapTrace,
}

fn forward(
    model: &dyn StepModel,
    tr: &TractiveTable,
    vmax: &[f64],
    track: &TrackModel,
    start: usize,
    v_start: f64,
    wrap: bool,
) -> Result<Vec<(usize, f64)>, Error> {
    let (n, m) = (track.len(), model.params().mass_kg);
    let (mut out, mut v, mut i, mut ax_prev) = (vec![(start, v_start)], v_start, start, 0.0);
    for _ in 0..n {
        let j = (i + 1) % n;
        if j == 0 && !wrap {
            break;
        }
        let ds = track.section_m(i);
        let ay = v * v / track.radius_m[i];
        let f = model.instant(v, ax_prev, ay)?.forces;
        let ax_tyre = ellipse_remaining(f.tyres_acc_n() / m, f.fy_total_n() / m, ay.abs());
        let ax_vehicle = ax_tyre.min(tr.force_at(v) / m) - f.resist_n() / m; // Eq. 4-10
        let ax_limited = (vmax[j] * vmax[j] - v * v) / (2.0 * ds); // Eq. 4-11
        let ax = ax_vehicle.min(ax_limited); // Eq. 4-12
                                             // At the corrected apex speed ΣFx = 0 by construction, so `ax` is zero up to the apex
                                             // iteration tolerance; only a clearly negative value means braking is needed (§4.3.2.2).
        if ax < -1e-5 {
            break;
        }
        let ax = ax.max(0.0);
        v = (v * v + 2.0 * ax * ds).max(0.0).sqrt().min(vmax[j]);
        ax_prev = ax;
        i = j;
        out.push((i, v));
    }
    Ok(out)
}

/// Braking trace from `apex` backwards. Returns the envelope points (where the trace is below the
/// cap) and the points where the cap binds, with the envelope value that the cap cut off (needed to
/// locate exact braking points). The trace runs through binding caps, carrying the cap speed on,
/// so a cap that rises more slowly than braking allows (a constant-radius corner, or a steep step
/// in `vmax` such as a corner entry) still constrains the points before it. It stops when the
/// envelope reaches the top speed or meets one already found from another apex (`known`).
fn backward(
    model: &dyn StepModel,
    vmax: &[f64],
    track: &TrackModel,
    apex: usize,
    top: f64,
    wrap: bool,
    known: &[f64],
) -> Result<(Vec<(usize, f64)>, Vec<(usize, f64)>), Error> {
    let (n, m) = (track.len(), model.params().mass_kg);
    let (mut out, mut v, mut i, mut ax_prev) = (vec![(apex, vmax[apex])], vmax[apex], apex, 0.0);
    let mut capped = Vec::new();
    for _ in 0..n {
        let k = if i == 0 {
            if !wrap {
                break;
            }
            n - 1
        } else {
            i - 1
        };
        let ds = track.section_m(k);
        let ay = v * v / track.radius_m[i];
        let f = model.instant(v, ax_prev, ay)?.forces;
        let dec_tyre = ellipse_remaining(f.tyres_dec_n() / m, f.fy_total_n() / m, ay.abs());
        let decel = dec_tyre + f.resist_n() / m;
        let vk = (v * v + 2.0 * decel * ds).sqrt();
        if vk >= top || vk >= known[k] {
            break;
        }
        if vk >= vmax[k] {
            // the cap binds here: record the cut-off envelope value and carry the cap speed on
            capped.push((k, vk));
            ax_prev = -decel;
            v = vmax[k];
            i = k;
            continue;
        }
        ax_prev = -decel;
        v = vk;
        i = k;
        out.push((i, v));
    }
    Ok((out, capped))
}

/// Run the thesis lap algorithm on `track`.
///
/// # Errors
/// Invalid settings or model/convergence errors.
pub fn simulate(
    model: &dyn StepModel,
    tractive: &TractiveTable,
    track: &TrackModel,
    s: &LapSettings,
) -> Result<LapResult, Error> {
    if !s.initial_speed_m_s.is_finite()
        || s.initial_speed_m_s < 0.0
        || !(s.apex_min_spacing_m.is_finite() && s.apex_min_spacing_m > 0.0)
    {
        return Err(err("invalid lap settings"));
    }
    let n = track.len();
    let raw = vmax_profile(model, tractive, track)?;
    let mut vmax = raw.clone();
    let apexes = smooth_clusters(
        &mut vmax,
        &track.distance_m,
        track.length_m,
        s.apex_min_spacing_m,
    );
    let top = top_speed(model, tractive)?;
    let wrap = s.flying;
    let mut v_acc = vec![f64::INFINITY; n];
    let mut v_dec = vec![f64::INFINITY; n];
    let mut v_over = vec![f64::INFINITY; n];
    let mut seeds: Vec<(usize, f64)> = apexes.iter().map(|&a| (a, vmax[a])).collect();
    if !s.flying {
        seeds.push((0, s.initial_speed_m_s.min(vmax[0])));
    }
    for (a, va) in seeds {
        for (i, v) in forward(model, tractive, &vmax, track, a, va, wrap)? {
            v_acc[i] = v_acc[i].min(v);
        }
    }
    for &a in &apexes {
        let (points, capped) = backward(model, &vmax, track, a, top, wrap, &v_dec)?;
        for (i, v) in points {
            v_dec[i] = v_dec[i].min(v);
        }
        for (k, vk) in capped {
            v_over[k] = v_over[k].min(vk);
        }
    }
    let speed: Vec<f64> = (0..n)
        .map(|i| vmax[i].min(v_acc[i]).min(v_dec[i]))
        .collect();
    // Re-process braking points (§4.3.5): the acceleration trace meets the braking trace
    // between points i and i+1; v² is linear in distance for each, so solve for the crossing.
    let (mut d, mut r, mut vm, mut sp, mut ins, mut apex_out) =
        (vec![], vec![], vec![], vec![], vec![], vec![]);
    for i in 0..n {
        if apexes.contains(&i) {
            apex_out.push(d.len());
        }
        d.push(track.distance_m[i]);
        r.push(track.radius_m[i]);
        vm.push(raw[i]);
        sp.push(speed[i]);
        ins.push(false);
        let j = (i + 1) % n;
        if j == 0 && !wrap {
            continue;
        }
        // L = the limit the car can reach going forward (acceleration trace or the vmax cap);
        // D = the braking trace including its overshoot point past the cap
        let lim = |k: usize| vmax[k].min(v_acc[k]);
        let dec = |k: usize| v_dec[k].min(v_over[k]);
        let acc_i = lim(i) <= dec(i);
        let acc_j = lim(j) <= dec(j);
        if acc_i && !acc_j && dec(j).is_finite() {
            let (a0, a1) = (lim(i).powi(2), lim(j).powi(2));
            let (b0, b1) = (dec(i).min(1.0e6).powi(2), dec(j).powi(2));
            let denom = (a1 - a0) - (b1 - b0);
            if denom.abs() > 1e-12 {
                let x = (b0 - a0) / denom;
                if x > 1e-6 && x < 1.0 - 1e-6 {
                    let sec = track.section_m(i);
                    let (c0, c1) = (1.0 / track.radius_m[i], 1.0 / track.radius_m[j]);
                    d.push(track.distance_m[i] + x * sec);
                    r.push(1.0 / (c0 + (c1 - c0) * x));
                    vm.push(raw[i] + (raw[j] - raw[i]) * x);
                    sp.push((a0 + (a1 - a0) * x).sqrt());
                    ins.push(true);
                }
            }
        }
    }
    let m = d.len();
    let (mut time, mut ax, mut lap) = (vec![0.0; m], vec![0.0; m], 0.0);
    for i in 0..m {
        let ds = if i + 1 < m {
            d[i + 1] - d[i]
        } else {
            track.length_m - d[i]
        };
        let v_end = if i + 1 < m || s.flying {
            sp[(i + 1) % m]
        } else {
            sp[i]
        };
        let dt = 2.0 * ds / (sp[i] + v_end).max(1e-9);
        ax[i] = (v_end * v_end - sp[i] * sp[i]) / (2.0 * ds);
        if i + 1 < m {
            time[i + 1] = time[i] + dt;
        }
        lap += dt;
    }
    let ay: Vec<f64> = (0..m).map(|i| sp[i] * sp[i] / r[i].abs()).collect();
    Ok(LapResult {
        lap_time_s: lap,
        trace: LapTrace {
            distance_m: d,
            radius_m: r,
            vmax_m_s: vm,
            speed_m_s: sp,
            ax_m_s2: ax,
            ay_m_s2: ay,
            time_s: time,
            apex_index: apex_out,
            inserted: ins,
        },
    })
}
