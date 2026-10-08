//! Sanity checks of the QSS lap on a circular course, where the answer is known in closed form:
//! a steady-state lap at constant radius has lap time x speed = circumference, and the speed is the
//! cornering limit of the vehicle model.
use fky_lapsim_core::lap::LapVehicle;
use fky_lapsim_core::lapsim::api::{simulate_lap, LapRequest};
use fky_lapsim_core::lapsim::vehicle::QssExtras;
use fky_lapsim_core::track::Track;

#[test]
fn a_circle_is_a_steady_state_at_the_lateral_limit() {
    let car = LapVehicle::synthetic_demo().unwrap();
    let circle = Track::circle(30.0, 6.0, 360).unwrap();
    let qss = simulate_lap(
        &car,
        &QssExtras::synthetic_demo(),
        &circle,
        &LapRequest::default(),
    )
    .unwrap();
    assert!(qss.completed, "{}", qss.termination);
    let t = qss.lap_time_s.unwrap();
    let v = qss.metrics["mean_speed_m_s"];
    let length = qss.metrics["track_length_m"];
    assert!((t * v - length).abs() < 1e-6 * length);
    // every point runs at the same limit speed (within the apex tolerance) and at 1+ g
    let peak = qss.metrics["peak_speed_m_s"];
    let low = qss.metrics["min_speed_m_s"];
    assert!((peak - low) / peak < 0.01, "speed spread {low}..{peak}");
    assert!(qss.metrics["peak_lateral_acceleration_m_s2"] > 9.81);
}

#[test]
fn the_thesis_exact_and_coupled_models_agree_on_the_circle_within_the_load_transfer_effect() {
    let car = LapVehicle::synthetic_demo().unwrap();
    let circle = Track::circle(30.0, 6.0, 360).unwrap();
    let extras = QssExtras::synthetic_demo();
    let exact = simulate_lap(
        &car,
        &extras,
        &circle,
        &LapRequest {
            use_matrix: false,
            ..Default::default()
        },
    )
    .unwrap();
    let full = simulate_lap(&car, &extras, &circle, &LapRequest::default()).unwrap();
    let (a, b) = (exact.lap_time_s.unwrap(), full.lap_time_s.unwrap());
    // lateral transfer costs grip through load sensitivity: not faster, and within 15 %
    assert!(
        b >= a * 0.999 && b < a * 1.15,
        "exact {a:.3} s vs coupled {b:.3} s"
    );
}
