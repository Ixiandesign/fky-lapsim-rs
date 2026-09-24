use dw_core::{
    dynamics::{formula_car_demo, linearize_ride},
    RideMode,
};

#[test]
fn ride_modes_match_generalized_eigenproblem_and_static_weight() {
    let (p, r) = formula_car_demo().unwrap();
    let out = linearize_ride(&p, &r).unwrap();
    assert!((out.support_reaction_n.iter().sum::<f64>() - 2943.).abs() < 0.01);
    assert_eq!(out.mass_matrix[0][0], 300.);
    assert_eq!(out.mass_matrix[1][1], 100.);
    for mode in &out.modes {
        assert!(mode.frequency_hz > 0.);
        let omega2 = (mode.frequency_hz * std::f64::consts::TAU).powi(2);
        for i in 0..3 {
            let residual = (0..3)
                .map(|j| {
                    (out.stiffness_matrix[i][j] - omega2 * out.mass_matrix[i][j]) * mode.shape[j]
                })
                .sum::<f64>();
            assert!(residual.abs() < 1e-6, "{residual}: {out:#?}");
        }
    }
    // An independently evaluated spring-energy second difference checks the
    // reported heave stiffness, including geometric/preload effects.
    let energy = |h: f64| {
        let state = dw_core::simulate(
            &p,
            &dw_core::Motion {
                heave: out.equilibrium[0] + h,
                roll: out.equilibrium[1],
                pitch: out.equilibrium[2],
                ..Default::default()
            },
        )
        .unwrap();
        state
            .corners
            .iter()
            .zip(&p.corners)
            .map(|(s, c)| {
                let x = s.metrics.shock_compression_m;
                c.spring_damper.preload * x + 0.5 * c.spring_damper.spring_rate * x * x
            })
            .sum::<f64>()
            + p.chassis.sprung_mass * 9.81 * (out.equilibrium[0] + h)
    };
    let expected = (energy(0.0001) + energy(-0.0001) - 2. * energy(0.)) / 1e-8;
    assert!((expected - out.stiffness_matrix[0][0]).abs() < 2.);
}

#[test]
fn asymmetric_springs_produce_coupled_modes_with_small_residuals() {
    let (mut p, r) = formula_car_demo().unwrap();
    p.corners[0].spring_damper.spring_rate *= 1.7;
    let out = linearize_ride(&p, &r).unwrap();
    assert!(out.stiffness_matrix[0][1].abs() > 1000.);
    for mode in &out.modes {
        let omega2 = (mode.frequency_hz * std::f64::consts::TAU).powi(2);
        for i in 0..3 {
            let residual = (0..3)
                .map(|j| {
                    (out.stiffness_matrix[i][j] - omega2 * out.mass_matrix[i][j]) * mode.shape[j]
                })
                .sum::<f64>();
            assert!(residual.abs() < 1e-5, "{residual}");
        }
    }
}

#[test]
fn retained_mass_changes_modal_mass_and_road_motion_is_explicitly_rejected() {
    let (mut p, mut r) = formula_car_demo().unwrap();
    let mass = dw_core::BodyMass {
        mass_kg: 2.,
        center_of_mass: p.corners[0].wheel_center,
        inertia: [[0.01, 0., 0.], [0., 0.01, 0.], [0., 0., 0.01]],
    };
    p.corners[0].component_masses.knuckle = Some(mass);
    r.mode = RideMode::RetainedComponentInertia;
    let out = linearize_ride(&p, &r).unwrap();
    assert!(out.mass_matrix[1][1] > 100.);
    r.road = dw_core::RoadInput::Sine {
        amplitude_m: 0.01,
        frequency_hz: 1.,
        phases_rad: [0.; 4],
    };
    assert!(linearize_ride(&p, &r).is_err());
}
