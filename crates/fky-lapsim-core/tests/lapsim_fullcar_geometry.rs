mod common;
use common::within;
use fky_lapsim_core::dynamics::{formula_car_demo, linearize_ride, RideRequest};
use fky_lapsim_core::lapsim::fullcar::{FullCar, GeometryInputs, LoadInputs};
use fky_lapsim_core::lapsim::G;
use fky_lapsim_core::{Motion, Project};

fn demo() -> Project {
    formula_car_demo().unwrap().0
}

fn inputs(p: &Project, kt: f64) -> GeometryInputs<'_> {
    GeometryInputs {
        project: p,
        unsprung_mass_kg: [8.0, 8.0, 9.0, 9.0],
        unsprung_cg_height_m: [0.25, 0.25, 0.26, 0.26],
        tyre_rate_n_m: [kt; 4],
        static_rh_mm: [35.0, 45.0],
        aero_cop_offset_m: [0.0, 0.0, 0.1],
    }
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
    let p = demo();
    let car = FullCar::from_geometry(&inputs(&p, 95.0e3)).unwrap();
    let k = car.stiffness;
    for i in 0..7 {
        for j in 0..7 {
            assert!((k[(i, j)] - k[(j, i)]).abs() <= 1e-6 * k[(i, i)].abs().max(k[(j, j)].abs()));
        }
    }
    let eig = k.symmetric_eigen().eigenvalues;
    assert!(eig.iter().all(|e| *e > 0.0), "eigenvalues {eig:?}");
}

#[test]
fn rigid_tyre_limit_recovers_the_existing_3x3_ride_stiffness() {
    let p = demo();
    let car = FullCar::from_geometry(&inputs(&p, 1.0e12)).unwrap();
    // Schur complement of the unsprung rows: Kqq - Kqu Kuu^-1 Kuq
    let k = car.stiffness;
    let kqq = k.fixed_view::<3, 3>(0, 0).into_owned();
    let kqu = k.fixed_view::<3, 4>(0, 3).into_owned();
    let kuu = k.fixed_view::<4, 4>(3, 3).into_owned();
    let schur = kqq - kqu * kuu.try_inverse().unwrap() * kqu.transpose();
    let lin = linearize_ride(&p, &RideRequest::default())
        .unwrap()
        .stiffness_matrix;
    for i in 0..3 {
        within(schur[(i, i)], lin[i][i], 0.03);
    }
}

#[test]
fn vertical_loads_sum_to_static_plus_downforce() {
    let p = demo();
    let car = FullCar::from_geometry(&inputs(&p, 95.0e3)).unwrap();
    let mut l = none();
    l.downforce_front_n = 300.0;
    l.downforce_rear_n = 500.0;
    let s = car.solve(&l).unwrap();
    let sum: f64 = s.wheel_load_n.iter().sum();
    let stat: f64 = car.params.static_wheel_load_n.iter().sum();
    within(sum, stat + 800.0, 1e-3);
    assert!(s.heave_m < 0.0 && s.front_rh_mm < 35.0 && s.rear_rh_mm < 45.0);
}

#[test]
fn symmetric_loads_do_not_roll_a_symmetric_car() {
    let p = demo();
    let car = FullCar::from_geometry(&inputs(&p, 95.0e3)).unwrap();
    let mut l = none();
    l.downforce_front_n = 300.0;
    l.downforce_rear_n = 500.0;
    let s = car.solve(&l).unwrap();
    assert!(s.roll_rad.abs() < 1e-6);
    within(s.wheel_load_n[0], s.wheel_load_n[1], 1e-4);
    within(s.wheel_load_n[2], s.wheel_load_n[3], 1e-4);
}

#[test]
fn lateral_load_transfer_obeys_the_moment_balance_independent_of_geometry() {
    let p = demo();
    let inp = inputs(&p, 95.0e3);
    let car = FullCar::from_geometry(&inp).unwrap();
    let total: f64 = car.params.static_wheel_load_n.iter().sum::<f64>() / G;
    let ay = 1.2 * G;
    let (a, b) = (car.params.a_m, car.params.b_m);
    let wb = a + b;
    // tyre lateral forces split by the yaw balance, equally left/right
    let fy_f = total * ay * b / wb / 2.0;
    let fy_r = total * ay * a / wb / 2.0;
    let mut l = none();
    l.ay = ay;
    l.fy_wheel_n = [fy_f, fy_f, fy_r, fy_r];
    let s = car.solve(&l).unwrap();
    // sum of y_i * dFz_i = -(m_s * ay * h_cg + sum m_u * ay * h_u)   (left turn: right wheels gain)
    let y = [
        car.params.half_track_m[0],
        -car.params.half_track_m[0],
        car.params.half_track_m[1],
        -car.params.half_track_m[1],
    ];
    let moment: f64 = (0..4)
        .map(|i| y[i] * (s.wheel_load_n[i] - car.params.static_wheel_load_n[i]))
        .sum();
    let q0 = linearize_ride(&p, &RideRequest::default())
        .unwrap()
        .equilibrium;
    let h_cg = p.chassis.center_of_mass[2] + q0[0];
    let expected = car.params.sprung_mass_kg * ay * h_cg
        + (0..4)
            .map(|i| inp.unsprung_mass_kg[i] * ay * inp.unsprung_cg_height_m[i])
            .sum::<f64>();
    // the rolled body's CG also shifts outboard, adding m_s g h phi to the transfer
    let expected = expected + car.params.sprung_mass_kg * G * h_cg * s.roll_rad;
    within(-moment, expected, 0.01);
    assert!(s.wheel_load_n[1] > s.wheel_load_n[0] && s.wheel_load_n[3] > s.wheel_load_n[2]);
    assert!(s.roll_deg > 0.0);
}

#[test]
fn braking_transfers_load_forward_and_pitches_the_nose_down() {
    let p = demo();
    let car = FullCar::from_geometry(&inputs(&p, 95.0e3)).unwrap();
    let total: f64 = car.params.static_wheel_load_n.iter().sum::<f64>() / G;
    let ax = -1.0 * G;
    let mut l = none();
    l.ax = ax;
    l.fx_wheel_n = [total * ax / 4.0; 4];
    let s = car.solve(&l).unwrap();
    let front: f64 = s.wheel_load_n[0] + s.wheel_load_n[1];
    let stat_front: f64 = car.params.static_wheel_load_n[0] + car.params.static_wheel_load_n[1];
    let q0 = linearize_ride(&p, &RideRequest::default())
        .unwrap()
        .equilibrium;
    let expected =
        total * -ax * (p.chassis.center_of_mass[2] + q0[0]) / (car.params.a_m + car.params.b_m);
    within(front - stat_front, expected, 0.05);
    assert!(s.pitch_deg > 0.0, "nose should dive, pitch {}", s.pitch_deg);
}

#[test]
fn camber_follows_the_kinematic_gain() {
    let p = demo();
    let car = FullCar::from_geometry(&inputs(&p, 95.0e3)).unwrap();
    let mut l = none();
    l.ay = 1.0 * G;
    let total: f64 = car.params.static_wheel_load_n.iter().sum::<f64>() / G;
    l.fy_wheel_n = [total * G / 4.0; 4];
    let s = car.solve(&l).unwrap();
    // reference: solve the kinematics at the matrix's own roll
    let q0 = linearize_ride(&p, &RideRequest::default())
        .unwrap()
        .equilibrium;
    let base = fky_lapsim_core::simulate(
        &p,
        &Motion {
            heave: q0[0],
            roll: q0[1],
            pitch: q0[2],
            ..Motion::default()
        },
    )
    .unwrap();
    let rolled = fky_lapsim_core::simulate(
        &p,
        &Motion {
            heave: q0[0] + s.heave_m,
            roll: q0[1] + s.roll_rad,
            pitch: q0[2] + s.pitch_rad,
            ..Motion::default()
        },
    )
    .unwrap();
    for c in &rolled.corners {
        let i = match c.id {
            fky_lapsim_core::CornerId::FrontLeft => 0,
            fky_lapsim_core::CornerId::FrontRight => 1,
            fky_lapsim_core::CornerId::RearLeft => 2,
            _ => 3,
        };
        let b = base.corners.iter().find(|x| x.id == c.id).unwrap();
        let delta = c.metrics.camber_deg - b.metrics.camber_deg;
        assert!(
            (s.camber_deg[i] - (b.metrics.camber_deg + delta)).abs() < 0.05 + 0.15 * delta.abs(),
            "wheel {i}: {} vs {}",
            s.camber_deg[i],
            b.metrics.camber_deg + delta
        );
    }
}

#[test]
fn rejects_invalid_geometry_inputs() {
    let p = demo();
    let mut bad = inputs(&p, 95.0e3);
    bad.tyre_rate_n_m[2] = 0.0;
    assert!(FullCar::from_geometry(&bad).is_err());
    let mut bad = inputs(&p, 95.0e3);
    bad.unsprung_mass_kg = [100.0; 4]; // heavier than the whole car
    assert!(FullCar::from_geometry(&bad).is_err());
}
