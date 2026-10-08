mod common;
use common::{stadium_rows, within};
use fky_lapsim_core::lap::LapVehicle;
use fky_lapsim_core::lapsim::api::{simulate_lap, LapRequest};
use fky_lapsim_core::lapsim::vehicle::QssExtras;
use fky_lapsim_core::track::Track;

fn stadium_track() -> Track {
    // build a closed stadium polygon (two straights joined by two semicircles) in metres
    let (straight, r) = (60.0_f64, 15.0_f64);
    let mut pts = vec![];
    pts.push([0.0, 0.0]);
    pts.push([straight, 0.0]);
    let n = 24;
    for k in 1..n {
        let a = -std::f64::consts::FRAC_PI_2 + std::f64::consts::PI * k as f64 / n as f64;
        pts.push([straight + r * a.cos(), r + r * a.sin()]);
    }
    pts.push([straight, 2.0 * r]);
    pts.push([0.0, 2.0 * r]);
    for k in 1..n {
        let a = std::f64::consts::FRAC_PI_2 + std::f64::consts::PI * k as f64 / n as f64;
        pts.push([r * a.cos(), r + r * a.sin()]);
    }
    Track {
        centerline_m: pts,
        width_m: 4.0,
    }
}

#[test]
fn demo_car_completes_a_lap_with_the_expected_metrics() {
    let car = LapVehicle::synthetic_demo().unwrap();
    let r = simulate_lap(
        &car,
        &QssExtras::synthetic_demo(),
        &stadium_track(),
        &LapRequest::default(),
    )
    .unwrap();
    assert!(r.completed, "{}", r.termination);
    let t = r.lap_time_s.unwrap();
    assert!(t > 5.0 && t < 60.0, "lap time {t}");
    for key in [
        "lap_time_s",
        "mean_speed_m_s",
        "peak_speed_m_s",
        "min_speed_m_s",
        "peak_lateral_acceleration_m_s2",
        "track_length_m",
    ] {
        assert!(r.metrics.contains_key(key), "missing metric {key}");
    }
    within(r.metrics["lap_time_s"], t, 1e-12);
    within(
        r.metrics["mean_speed_m_s"],
        r.metrics["track_length_m"] / t,
        1e-9,
    );
    assert!(r.metrics["peak_speed_m_s"] > r.metrics["min_speed_m_s"]);
    assert!(r.channels.is_some() && r.kpis.is_some() && r.trace.is_some());
    assert!(r
        .model_fidelity
        .starts_with("qss_zacharelis2023_bike_coupled_7x7_geometry"));
}

#[test]
fn thesis_exact_mode_is_selectable_and_labelled() {
    let car = LapVehicle::synthetic_demo().unwrap();
    let req = LapRequest {
        use_matrix: false,
        ..LapRequest::default()
    };
    let r = simulate_lap(&car, &QssExtras::synthetic_demo(), &stadium_track(), &req).unwrap();
    assert!(r.completed);
    assert!(r.model_fidelity.contains("thesis_exact"));
}

#[test]
fn the_matrix_changes_the_result_in_the_expected_direction() {
    let car = LapVehicle::synthetic_demo().unwrap();
    let e = QssExtras::synthetic_demo();
    let exact = simulate_lap(
        &car,
        &e,
        &stadium_track(),
        &LapRequest {
            use_matrix: false,
            ..LapRequest::default()
        },
    )
    .unwrap();
    let full = simulate_lap(&car, &e, &stadium_track(), &LapRequest::default()).unwrap();
    // lateral transfer and camber cost grip: the full model is not faster than thesis-exact (+1 %)
    assert!(full.lap_time_s.unwrap() >= exact.lap_time_s.unwrap() * 0.99);
}

#[test]
fn results_are_deterministic_and_serialize() {
    let car = LapVehicle::synthetic_demo().unwrap();
    let e = QssExtras::synthetic_demo();
    let a = simulate_lap(&car, &e, &stadium_track(), &LapRequest::default()).unwrap();
    let b = simulate_lap(&car, &e, &stadium_track(), &LapRequest::default()).unwrap();
    assert_eq!(a.lap_time_s, b.lap_time_s);
    let json = serde_json::to_string(&a).unwrap();
    let back: fky_lapsim_core::lapsim::api::LapRun = serde_json::from_str(&json).unwrap();
    assert_eq!(back.lap_time_s, a.lap_time_s);
}

#[test]
fn invalid_requests_are_errors_and_physics_failures_are_incomplete_runs() {
    let car = LapVehicle::synthetic_demo().unwrap();
    let e = QssExtras::synthetic_demo();
    let bad = LapRequest {
        resample_step_m: 0.0,
        ..LapRequest::default()
    };
    assert!(simulate_lap(&car, &e, &stadium_track(), &bad).is_err());
    // an impossible vehicle: zero friction scaling makes every corner unachievable
    let mut weak = e.clone();
    weak.correlation.muy = 0.0;
    let r = simulate_lap(&car, &weak, &stadium_track(), &LapRequest::default()).unwrap();
    assert!(!r.completed);
    assert!(r.lap_time_s.is_none());
    assert!(!r.termination.is_empty());
    let _ = stadium_rows; // keep the shared helper import exercised by this file
}

#[test]
fn unknown_request_fields_are_rejected_not_ignored() {
    // a legacy driver-policy key must not be silently dropped
    assert!(serde_json::from_str::<LapRequest>(r#"{"dt_s": 0.01}"#).is_err());
    assert!(serde_json::from_str::<LapRequest>("{}").is_ok());
    let r: LapRequest = serde_json::from_str(r#"{"flying": false}"#).unwrap();
    assert!(!r.flying && r.use_matrix);
}
