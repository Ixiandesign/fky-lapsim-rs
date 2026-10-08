//! Engine tractive force versus vehicle speed (thesis §2.6.3–2.7).
use super::err;
use crate::powertrain::Powertrain;
use crate::Error;

/// Tractive force at the wheels versus speed, with the selected gear and engine speed.
#[derive(Clone, Debug)]
pub struct TractiveTable {
    /// Vehicle speed grid, m/s (uniform, starting at 0).
    pub speed_m_s: Vec<f64>,
    /// Tractive force per speed, N (Eq. 2-25), engine scalar applied.
    pub force_n: Vec<f64>,
    /// Selected gear (zero-based) per speed, after the shift filter.
    pub gear: Vec<usize>,
    /// Engine speed per speed, rpm.
    pub rpm: Vec<f64>,
}

const RAD_PER_RPM: f64 = 2.0 * std::f64::consts::PI / 60.0;

impl TractiveTable {
    /// Build the table (Eqs. 2-21…2-25 plus the §2.6.4 shift filter).
    ///
    /// # Errors
    /// Returns an error for an invalid powertrain, nonpositive `rolling_radius_m`/`dv`, or a negative
    /// `engine_scalar`.
    pub fn build(
        pt: &Powertrain,
        rolling_radius_m: f64,
        engine_scalar: f64,
        dv: f64,
    ) -> Result<Self, Error> {
        pt.validate().map_err(err)?;
        if !(rolling_radius_m.is_finite() && rolling_radius_m > 0.0)
            || !(dv.is_finite() && dv > 0.0)
            || !(engine_scalar.is_finite() && engine_scalar >= 0.0)
        {
            return Err(err("invalid tractive table inputs"));
        }
        let r = rolling_radius_m;
        let ratio = |g: usize| pt.gear_ratios[g] * pt.final_drive;
        let rpm_at = |v: f64, g: usize| v / r * ratio(g) / RAD_PER_RPM;
        let force = |rpm: f64, g: usize| -> Result<f64, Error> {
            Ok(pt.torque_at_rpm(rpm).map_err(err)? * ratio(g) * pt.efficiency * engine_scalar / r)
        };
        let top_gear = pt.gear_ratios.len() - 1;
        let v_top = pt.redline_rpm * RAD_PER_RPM * r / ratio(top_gear);
        let v_first = pt.idle_rpm * RAD_PER_RPM * r / ratio(0);
        let n = (v_top / dv).floor() as usize + 1;
        let mut speed = Vec::with_capacity(n);
        let mut f_out = Vec::with_capacity(n);
        let mut g_out: Vec<usize> = Vec::with_capacity(n);
        let mut rpm_out = Vec::with_capacity(n);
        for i in 0..n {
            let v = i as f64 * dv;
            let vq = v.max(v_first);
            let mut best: Option<(usize, f64, f64)> = None;
            for g in 0..pt.gear_ratios.len() {
                let rpm = rpm_at(vq, g);
                if rpm < pt.idle_rpm - 1e-9 || rpm > pt.redline_rpm + 1e-9 {
                    continue;
                }
                let rpm = rpm.clamp(pt.idle_rpm, pt.redline_rpm);
                let f = force(rpm, g)?;
                if best.is_none_or(|(_, bf, _)| f > bf) {
                    best = Some((g, f, rpm));
                }
            }
            let (mut g, mut f, mut rpm) = best.ok_or_else(|| err("no gear covers this speed"))?;
            if let Some(&prev) = g_out.last() {
                if g < prev {
                    let prpm = rpm_at(vq, prev);
                    if prpm >= pt.idle_rpm && prpm <= pt.redline_rpm {
                        g = prev;
                        rpm = prpm;
                        f = force(rpm, g)?;
                    }
                }
            }
            speed.push(v);
            f_out.push(f);
            g_out.push(g);
            rpm_out.push(rpm);
        }
        Ok(Self {
            speed_m_s: speed,
            force_n: f_out,
            gear: g_out,
            rpm: rpm_out,
        })
    }

    fn locate(&self, v: f64) -> (usize, f64) {
        let dv = self.speed_m_s[1] - self.speed_m_s[0];
        let x = (v / dv).clamp(0.0, (self.speed_m_s.len() - 1) as f64);
        let i = (x.floor() as usize).min(self.speed_m_s.len() - 2);
        (i, x - i as f64)
    }
    /// Highest speed the table covers (redline in the top gear), m/s.
    pub fn top_speed_m_s(&self) -> f64 {
        *self.speed_m_s.last().unwrap()
    }
    /// Tractive force at `v` by linear interpolation, N; 0 above [`Self::top_speed_m_s`].
    pub fn force_at(&self, v: f64) -> f64 {
        if v > self.top_speed_m_s() {
            return 0.0;
        }
        let (i, t) = self.locate(v);
        self.force_n[i] * (1.0 - t) + self.force_n[i + 1] * t
    }
    /// Selected gear (zero-based) at `v`.
    pub fn gear_at(&self, v: f64) -> usize {
        let (i, t) = self.locate(v);
        if t < 0.5 {
            self.gear[i]
        } else {
            self.gear[i + 1]
        }
    }
    /// Engine speed at `v`, rpm.
    pub fn rpm_at(&self, v: f64) -> f64 {
        let (i, t) = self.locate(v);
        self.rpm[i] * (1.0 - t) + self.rpm[i + 1] * t
    }
}
