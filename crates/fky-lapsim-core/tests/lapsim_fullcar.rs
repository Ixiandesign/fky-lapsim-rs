mod common;
use common::within;
use fky_lapsim_core::lapsim::fullcar::{thesis_params, FullCar, LoadInputs};
use fky_lapsim_core::lapsim::rates::{rates, ThesisSuspension};
use fky_lapsim_core::lapsim::thesis::p19;
use fky_lapsim_core::lapsim::G;

fn car() -> FullCar {
    FullCar::from_rates(thesis_params(&p19(), &ThesisSuspension::p19()).unwrap()).unwrap()
}

fn none() -> LoadInputs {
    LoadInputs {
        ax: 0.0,
        ay: 0.0,
        downforce_front_n: 0.0,
        downforce_rear_n: 0.0,
        drag_n: 0.0,
        fx_wheel_n: [0.0; 4],
        fy_wheel_n: [0.0; 4],
    }
}

#[test]
fn stiffness_is_symmetric_and_positive_definite() {
    let k = car().stiffness;
    for i in 0..7 {
        for j in 0..7 {
            assert!((k[(i, j)] - k[(j, i)]).abs() < 1e-6 * k[(i, i)].max(1.0));
        }
    }
    let eig = k.symmetric_eigen().eigenvalues;
    assert!(eig.iter().all(|e| *e > 0.0), "eigenvalues {eig:?}");
}

#[test]
fn mass_matrix_is_the_thesis_diagonal() {
    let c = car();
    let p = thesis_params(&p19(), &ThesisSuspension::p19()).unwrap();
    within(c.mass[(0, 0)], p.sprung_mass_kg, 1e-12);
    within(c.mass[(1, 1)], 15.0, 1e-12);
    within(c.mass[(2, 2)], 60.0, 1e-12);
    within(c.mass[(3, 3)], 10.0, 1e-12);
    within(c.mass[(6, 6)], 11.0, 1e-12);
    assert_eq!(c.mass[(0, 1)], 0.0);
}

#[test]
fn roll_gradient_from_the_matrix_matches_eq_5_10_and_table_5_1() {
    let c = car();
    let r = rates(&p19(), &ThesisSuspension::p19()).unwrap();
    let mut l = none();
    l.ay = G; // 1 g left
    let s = c.solve(&l).unwrap();
    within(s.roll_deg, r.roll_gradient_deg_per_g, 2e-3);
    assert!(s.roll_deg > 0.0);
}

#[test]
fn heave_under_downforce_matches_the_heave_rate_eq_5_2() {
    let c = car();
    let r = rates(&p19(), &ThesisSuspension::p19()).unwrap();
    let mut l = none();
    l.downforce_front_n = 400.0;
    l.downforce_rear_n = 500.0;
    let s = c.solve(&l).unwrap();
    within(s.front_rh_mm, 35.0 - 1000.0 * 200.0 / r.kheave_n_m[0], 1e-6);
    within(s.rear_rh_mm, 45.0 - 1000.0 * 250.0 / r.kheave_n_m[1], 1e-6);
    // total vertical load rises by the downforce
    let sum: f64 = s.wheel_load_n.iter().sum();
    let static_sum: f64 = c.params.static_wheel_load_n.iter().sum();
    within(sum, static_sum + 900.0, 1e-9);
}

#[test]
fn longitudinal_weight_transfer_loads_and_ride_height_follow_eqs_7_1_7_8_7_9() {
    let c = car();
    let r = rates(&p19(), &ThesisSuspension::p19()).unwrap();
    let mut l = none();
    l.ax = -12.0; // braking
    let s = c.solve(&l).unwrap();
    let wt = 250.0 * -12.0 * 0.33 / 1.53; // negative: load moves forward
                                          // tyre loads carry the full transfer (springs + geometric)
    let front: f64 = s.wheel_load_n[0] + s.wheel_load_n[1];
    let static_front: f64 = c.params.static_wheel_load_n[0] + c.params.static_wheel_load_n[1];
    within(front - static_front, -wt, 1e-9);
    // ride height sees only the spring-borne share (1 - anti)
    let anti_dive = 0.303;
    within(
        s.front_rh_mm,
        35.0 - 1000.0 * (-wt * (1.0 - anti_dive) / 2.0) / r.kheave_n_m[0],
        1e-6,
    );
}

#[test]
fn lateral_geometric_transfer_is_fy_times_roll_centre_height_over_track() {
    let c = car();
    let mut l = none();
    l.fy_wheel_n = [400.0, 400.0, 500.0, 500.0]; // left turn: left-directed tyre forces
    let s = c.solve(&l).unwrap(); // no inertial roll moment: only the kinematic transfer
    let t_f = 1.238;
    let geo_front = 800.0 * 0.0295 / t_f;
    // outside (right) wheel gains, inside (left) loses, springs untouched
    within(
        s.wheel_load_n[1] - c.params.static_wheel_load_n[1],
        geo_front,
        1e-9,
    );
    within(
        s.wheel_load_n[0] - c.params.static_wheel_load_n[0],
        -geo_front,
        1e-9,
    );
    assert!(s.roll_deg.abs() < 1e-12);
}

#[test]
fn rejects_invalid_parameters() {
    let mut p = thesis_params(&p19(), &ThesisSuspension::p19()).unwrap();
    p.tyre_rate_n_m[2] = 0.0;
    assert!(FullCar::from_rates(p).is_err());
    let mut p = thesis_params(&p19(), &ThesisSuspension::p19()).unwrap();
    p.sprung_mass_kg = -1.0;
    assert!(FullCar::from_rates(p).is_err());
}
