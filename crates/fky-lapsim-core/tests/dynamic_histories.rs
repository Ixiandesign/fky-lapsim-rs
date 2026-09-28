use dw_core::dynamics::*;

#[test]
fn legacy_request_defaults_and_combined_knot_work_limit() {
    let legacy: RideRequest = serde_json::from_str(r#"{"external_force":[-12,3,4]}"#).unwrap();
    assert!(legacy.external_force_history.is_none());
    assert!(!legacy.report_states);
    assert_eq!(legacy.external_force, [-12., 3., 4.]);
    let r = RideRequest {
        dt_s: 1. / 19998.,
        external_force_history: Some(vec![
            [0.; 4],
            [0.00001, 0., 0., 0.],
            [0.00002, 0., 0., 0.],
            [1., 0., 0., 0.],
        ]),
        ..Default::default()
    };
    assert!(validate_request(&r).is_err());
}

#[test]
fn retained_component_acceleration_and_reaction_use_instantaneous_load() {
    let (mut p, mut r) = formula_car_demo().unwrap();
    for c in &mut p.corners {
        c.component_masses.knuckle = Some(dw_core::BodyMass {
            mass_kg: 8.,
            center_of_mass: c.wheel_center,
            inertia: [[0.1, 0., 0.], [0., 0.1, 0.], [0., 0., 0.1]],
        });
    }
    r.mode = RideMode::RetainedComponentInertia;
    r.external_force_history = Some(vec![[0., 0., 0., 0.], [1., -80., 2., 4.]]);
    let varying = evaluate_ride(&p, &r, 0.5, [0.; 3], [0.; 3]).unwrap();
    r.external_force_history = None;
    r.external_force = [-40., 1., 2.];
    let constant = evaluate_ride(&p, &r, 0.5, [0.; 3], [0.; 3]).unwrap();
    assert_eq!(varying.acceleration, constant.acceleration);
    assert_eq!(varying.support_reaction_n, constant.support_reaction_n);
}

#[test]
fn ramp_load_knots_are_samples_and_external_work_balances_energy() {
    let (mut p, mut r) = formula_car_demo().unwrap();
    for c in &mut p.corners {
        c.spring_damper.compression_damping = 0.;
        c.spring_damper.rebound_damping = 0.;
    }
    r.duration_s = 0.02;
    r.dt_s = 0.004;
    r.initial_velocity = [0.01, 0., 0.];
    r.external_force_history = Some(vec![
        [0., 0., 0., 0.],
        [0.007, -80., 3., -2.],
        [0.02, -20., 0., 1.],
    ]);
    let run = ride(&p, &r).unwrap();
    assert!(run.termination.is_none(), "{:?}", run.termination);
    assert!(run.samples.iter().any(|s| s.time_s == 0.007));
    let last = run.samples.last().unwrap();
    assert!(last.external_work_j.abs() > 1e-5);
    assert!(
        last.energy_balance_error_j.abs() < 1e-4,
        "{}",
        last.energy_balance_error_j
    );
    // At a frozen state, interpolated generalized force must produce precisely
    // the same acceleration as an explicitly supplied constant force.
    let sample = &run.samples[0];
    let dynamic = evaluate_ride(&p, &r, 0.0035, sample.displacement, sample.velocity).unwrap();
    r.external_force_history = None;
    r.external_force = [-40., 1.5, -1.];
    let constant = evaluate_ride(&p, &r, 0.0035, sample.displacement, sample.velocity).unwrap();
    assert_eq!(dynamic.acceleration, constant.acceleration);
}

#[test]
fn reported_geometry_matches_instantaneous_motion_road_and_racks() {
    let (p, mut r) = formula_car_demo().unwrap();
    r.duration_s = 0.;
    r.rack_front = 0.001;
    r.rack_rear = -0.001;
    r.road = RoadInput::Sine {
        amplitude_m: 0.001,
        frequency_hz: 1.,
        phases_rad: [0.4; 4],
    };
    let ordinary = ride(&p, &r).unwrap();
    assert!(ordinary.samples[0].state.is_none());
    r.report_states = true;
    let run = ride(&p, &r).unwrap();
    let s = &run.samples[0];
    let state = s.state.as_ref().unwrap();
    let expected = dw_core::simulate_on_road(
        &p,
        &dw_core::Motion {
            heave: s.displacement[0],
            roll: s.displacement[1],
            pitch: s.displacement[2],
            rack_front: r.rack_front,
            rack_rear: r.rack_rear,
        },
        [0.001 * 0.4_f64.sin(); 4],
    )
    .unwrap();
    assert_eq!(
        serde_json::to_value(state).unwrap(),
        serde_json::to_value(expected).unwrap()
    );
    let instant = evaluate_ride(&p, &r, 0., s.displacement, s.velocity).unwrap();
    assert_eq!(instant.acceleration, s.acceleration);
    assert_eq!(
        serde_json::to_value(instant.state).unwrap(),
        serde_json::to_value(&s.state).unwrap()
    );
    assert!(evaluate_ride(&p, &r, 1., s.displacement, s.velocity).is_err());
}

#[test]
fn malformed_load_histories_are_rejected_before_physics() {
    for history in [
        vec![],
        vec![[0.; 4]],
        vec![[0.; 4], [0.; 4]],
        vec![[0.1, 0., 0., 0.], [1., 0., 0., 0.]],
        vec![[0.; 4], [0.5, 0., 0., 0.]],
        vec![[0.; 4], [1., f64::NAN, 0., 0.]],
        vec![[0., -f64::MAX, 0., 0.], [1., f64::MAX, 0., 0.]],
    ] {
        let r = RideRequest {
            external_force_history: Some(history),
            ..Default::default()
        };
        assert!(validate_request(&r).is_err());
    }
}

#[test]
fn constant_load_history_matches_constant_force_and_equilibrium() {
    let (p, mut r) = formula_car_demo().unwrap();
    r.duration_s = 0.01;
    r.external_force = [-40., 2., -3.];
    let constant = ride(&p, &r).unwrap();
    r.external_force_history = Some(vec![[0., -40., 2., -3.], [0.01, -40., 2., -3.]]);
    r.external_force = [999.; 3];
    let history = ride(&p, &r).unwrap();
    assert!(constant.termination.is_none());
    assert!(history.termination.is_none());
    assert_eq!(
        serde_json::to_value(constant).unwrap(),
        serde_json::to_value(history).unwrap()
    );
}
