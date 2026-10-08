//! Friction ellipse and the speed-dependent GGV performance envelope (thesis §2.10, §4.3.1.2).
use super::err;
use super::forces::{max_ay, StepModel};
use super::tractive::TractiveTable;
use crate::Error;

/// Remaining longitudinal capability on the friction ellipse (Eqs. 4-8, 4-9):
/// `max_long * sqrt(1 − (lat_used / max_lat)²)`; 0 when the lateral demand reaches `max_lat`.
pub fn ellipse_remaining(max_long: f64, max_lat: f64, lat_used: f64) -> f64 {
    if max_lat <= 0.0 || lat_used.abs() >= max_lat {
        return 0.0;
    }
    max_long * (1.0 - (lat_used / max_lat).powi(2)).sqrt()
}

/// GGV map: per speed, the acceleration envelope boundary.
#[derive(Clone, Debug)]
pub struct GgvMap {
    /// Speeds, m/s.
    pub speed_m_s: Vec<f64>,
    /// Lateral acceleration of each boundary point per speed, m/s²: the accelerating half runs
    /// from +max to −max, then the braking half runs from −max back to +max.
    pub ay_m_s2: Vec<Vec<f64>>,
    /// Longitudinal acceleration of each boundary point, m/s² (negative braking).
    pub ax_m_s2: Vec<Vec<f64>>,
}

/// Build the GGV map (§2.10.2.2): at each speed take the pure-lateral limit, then trace the
/// combined-slip boundary with the friction ellipse; accelerating is capped by the engine and
/// reduced by drag + rolling, braking is tyre-limited and helped by drag + rolling.
///
/// # Errors
/// Returns an error for empty `speeds`, fewer than 3 `n_angles`, or any model error.
pub fn ggv(
    model: &dyn StepModel,
    tractive: &TractiveTable,
    speeds: &[f64],
    n_angles: usize,
) -> Result<GgvMap, Error> {
    if speeds.is_empty() || n_angles < 3 {
        return Err(err("GGV needs at least one speed and 3 angles"));
    }
    let m = model.params().mass_kg;
    let mut out = GgvMap {
        speed_m_s: speeds.to_vec(),
        ay_m_s2: vec![],
        ax_m_s2: vec![],
    };
    for &v in speeds {
        let ay_max = max_ay(model, v, 0.0)?;
        let f = model.instant(v, 0.0, ay_max)?.forces;
        let ax_acc = f.tyres_acc_n() / m;
        let ax_dec = f.tyres_dec_n() / m;
        let eng = tractive.force_at(v) / m;
        let drag = f.resist_n() / m;
        let (mut ays, mut axs) = (vec![], vec![]);
        for upper in [true, false] {
            for k in 0..n_angles {
                let theta = std::f64::consts::PI * k as f64 / (n_angles - 1) as f64;
                let ay = if upper {
                    ay_max * theta.cos()
                } else {
                    -ay_max * theta.cos()
                };
                ays.push(ay);
                axs.push(if upper {
                    ellipse_remaining(ax_acc, ay_max, ay).min(eng) - drag
                } else {
                    -(ellipse_remaining(ax_dec, ay_max, ay) + drag)
                });
            }
        }
        out.ay_m_s2.push(ays);
        out.ax_m_s2.push(axs);
    }
    Ok(out)
}
