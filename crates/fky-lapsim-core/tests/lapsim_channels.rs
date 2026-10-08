mod common;
use common::{p19_powertrain, stadium_rows, within};
use fky_lapsim_core::lapsim::channels::{
    brake_pressure_bar, driven_channels, kpis, throttle_pct, KpiSettings,
};
use fky_lapsim_core::lapsim::forces::ThesisConst;
use fky_lapsim_core::lapsim::solver::{simulate, LapResult, LapSettings};
use fky_lapsim_core::lapsim::thesis::p19;
use fky_lapsim_core::lapsim::track_model::TrackModel;
use fky_lapsim_core::lapsim::tractive::TractiveTable;

#[test]
fn brake_pressure_matches_thesis_tables_2_10_and_2_11() {
    // 2 g deceleration on a 250 kg car (Table 2-9: 19.62 m/s², 4905 N)
    let (front, rear) = brake_pressure_bar(&p19(), 19.62).unwrap();
    within(front, 45.7, 0.005);
    within(rear, 66.0, 0.005);
}

#[test]
fn brake_pressure_is_linear_in_deceleration_eq_2_66() {
    let p = p19();
    let (f1, r1) = brake_pressure_bar(&p, 5.0).unwrap();
    let (f2, r2) = brake_pressure_bar(&p, 10.0).unwrap();
    within(f2, 2.0 * f1, 1e-9);
    within(r2, 2.0 * r1, 1e-9);
    assert!(brake_pressure_bar(&p, -1.0).is_err());
}

#[test]
fn throttle_covers_resistance_and_saturates() {
    // steady state: tractive force just balances resistance
    within(throttle_pct(250.0, 0.0, 300.0, 1200.0), 25.0, 1e-12);
    // full power-limited acceleration
    within(
        throttle_pct(250.0, (1200.0 - 300.0) / 250.0, 300.0, 1200.0),
        100.0,
        1e-9,
    );
    // braking and coasting: no throttle
    assert_eq!(throttle_pct(250.0, -5.0, 300.0, 1200.0), 0.0);
    // more than the engine can give clamps to 100 (grip would limit first)
    assert_eq!(throttle_pct(250.0, 10.0, 300.0, 1200.0), 100.0);
}

fn lap() -> (ThesisConst, TractiveTable, LapResult) {
    let m = ThesisConst {
        params: p19(),
        weight_transfer: true,
    };
    let tr = TractiveTable::build(&p19_powertrain(), 0.199, 1.0, 0.05).unwrap();
    let t = TrackModel::from_radius_length(&stadium_rows(120.0, 12.0, 0.5)).unwrap();
    let r = simulate(&m, &tr, &t, &LapSettings::default()).unwrap();
    (m, tr, r)
}

#[test]
fn driven_channels_are_consistent_with_the_trace() {
    let (m, tr, r) = lap();
    let c = driven_channels(&m, &tr, &r.trace).unwrap();
    let n = r.trace.speed_m_s.len();
    assert_eq!(c.speed_kmh.len(), n);
    assert_eq!(c.gear.len(), n);
    for i in 0..n {
        within(c.speed_kmh[i], r.trace.speed_m_s[i] * 3.6, 1e-12);
        // braking only where decelerating, throttle only where thrust is needed
        if r.trace.ax_m_s2[i] < -0.1 {
            assert!(c.brake_front_bar[i] > 0.0);
            assert_eq!(c.tps_pct[i], 0.0);
        } else {
            assert_eq!(c.brake_front_bar[i], 0.0);
        }
        // total lateral force equals mass * ay (statics)
        within(
            c.fy_f_n[i] + c.fy_r_n[i],
            250.0 * r.trace.ay_m_s2[i] * r.trace.radius_m[i].signum(),
            1e-6,
        );
        assert!((0.0..=100.0).contains(&c.tps_pct[i]));
    }
    // left-hand corners need a positive steering angle (Eq. 2-53 sign convention)
    let corner = (0..n)
        .find(|&i| r.trace.radius_m[i] > 0.0 && r.trace.radius_m[i] < 20.0)
        .unwrap();
    assert!(c.steer_wheel_deg[corner] > 0.0);
}

#[test]
fn kpis_are_internally_consistent() {
    let (m, tr, r) = lap();
    let c = driven_channels(&m, &tr, &r.trace).unwrap();
    let k = kpis(
        &c,
        r.lap_time_s,
        2.0 * 120.0 + 2.0 * std::f64::consts::PI * 12.0,
        &KpiSettings::default(),
    );
    within(
        k.low_speed_pct + k.medium_speed_pct + k.high_speed_pct,
        100.0,
        1e-9,
    );
    within(k.grip_limited_pct + k.power_limited_pct, 100.0, 1e-9);
    assert!(k.min_speed_kmh < k.median_speed_kmh && k.median_speed_kmh <= k.max_speed_kmh);
    assert!(k.cornering_pct > 10.0 && k.cornering_pct < 90.0);
    assert!(k.accelerating_pct > 0.0 && k.decelerating_pct > 0.0);
    assert!(k.max_gear >= k.min_gear);
}

#[test]
fn body_slip_is_the_physical_steady_state_value_not_the_ill_conditioned_thesis_one() {
    // the thesis Eq. 2-53 beta for P19 near its skidpad speed is hundreds of degrees
    let (m, tr, r) = lap();
    let c = driven_channels(&m, &tr, &r.trace).unwrap();
    assert!(
        c.beta_deg.iter().all(|b| b.abs() < 10.0),
        "max |beta| {:.1}",
        c.beta_deg.iter().fold(0.0_f64, |a, b| a.max(b.abs()))
    );
}

#[test]
fn per_wheel_loads_and_camber_channels_follow_the_attitude() {
    let (m, tr, r) = lap();
    let c = driven_channels(&m, &tr, &r.trace).unwrap();
    let n = c.speed_kmh.len();
    for ch in [
        &c.fz_fl_n,
        &c.fz_fr_n,
        &c.fz_rl_n,
        &c.fz_rr_n,
        &c.camber_fl_deg,
        &c.camber_fr_deg,
        &c.camber_rl_deg,
        &c.camber_rr_deg,
    ] {
        assert_eq!(ch.len(), n);
    }
    // the bike-only model splits each axle evenly
    for i in 0..n {
        within(c.fz_fl_n[i], c.fz_fr_n[i], 1e-9);
        assert!(c.fz_fl_n[i] > 0.0 && c.fz_rr_n[i] > 0.0);
    }
}
