use fky_lapsim_core::{dynamics::formula_car_demo, ride, Project, RideRequest};
fn massive() -> (Project, RideRequest) {
    let (p, r) = formula_car_demo().unwrap();
    let mut value = serde_json::to_value(p).unwrap();
    for c in value["corners"].as_array_mut().unwrap() {
        c["component_masses"] = serde_json::json!({"knuckle":{"mass_kg":8.,"center_of_mass":c["wheel_center"],"inertia":[[0.1,0.,0.],[0.,0.1,0.],[0.,0.,0.1]]}});
    }
    let mut request = serde_json::to_value(r).unwrap();
    request["mode"] = serde_json::json!("retained_component_inertia");
    request["duration_s"] = serde_json::json!(0.);
    request["initial_displacement"] = serde_json::json!([0., 0., 0.]);
    (
        serde_json::from_value(value).unwrap(),
        serde_json::from_value(request).unwrap(),
    )
}
#[test]
fn real_vertical_knuckle_support() {
    let (p, r) = massive();
    let run = ride(&p, &r).unwrap();
    assert!(run.termination.is_none(), "{:?}", run.termination);
    let s = &run.samples[0];
    for n in s.support_reaction_n {
        assert!((n - (300. / 4. + 8.) * 9.81).abs() < 0.03, "{n}");
    }
}
#[test]
fn reject_bad_component_inertia() {
    let (p, _) = massive();
    let mut v = serde_json::to_value(p).unwrap();
    v["corners"][0]["component_masses"]["knuckle"]["inertia"] =
        serde_json::json!([[10., 0., 0.], [0., 1., 0.], [0., 0., 1.]]);
    assert!(serde_json::from_value::<Project>(v)
        .unwrap()
        .validate()
        .is_err());
}
fn massive_with_interconnect() -> (Project, RideRequest) {
    let mut p = Project::example_with_interconnect();
    p.chassis.sprung_mass = 300.;
    p.chassis.inertia = [100., 250., 300.];
    let mu = 0.1875 / 0.085_f64.sqrt();
    for c in &mut p.corners {
        c.spring_damper.preload = 300. * 9.81 / (4. * mu);
    }
    let mut value = serde_json::to_value(p).unwrap();
    for c in value["corners"].as_array_mut().unwrap() {
        c["component_masses"] = serde_json::json!({"knuckle":{"mass_kg":8.,"center_of_mass":c["wheel_center"],"inertia":[[0.1,0.,0.],[0.,0.1,0.],[0.,0.,0.1]]}});
    }
    let p: Project = serde_json::from_value(value).unwrap();
    let r = RideRequest {
        mode: fky_lapsim_core::RideMode::RetainedComponentInertia,
        duration_s: 0.,
        solve_equilibrium: false,
        initial_displacement: [0., 0., 0.],
        ..Default::default()
    };
    (p, r)
}
#[test]
fn support_reaction_sums_to_weight_with_interconnect_in_mass_mode() {
    // Static only (duration_s stays 0, no initial displacement/velocity): a dynamic,
    // moving run's knuckle articulation contributes its own inertial "extra" term to
    // support_reaction_n even on a flat road with no interconnect at all (see
    // moving_knuckle_inertial_support's explicit "extra" term above) -- that confound is
    // orthogonal to what this test checks, so it isolates the static case, matching
    // real_vertical_knuckle_support's proven pattern.
    let (mut p, mut r) = massive_with_interconnect();
    p.front_interconnect.as_mut().unwrap().heave.preload = 500.;
    r.solve_equilibrium = true;
    let run = ride(&p, &r).unwrap();
    assert!(run.termination.is_none(), "{:?}", run.termination);
    let total_mass = 300. + 4. * 8.;
    let s = &run.samples[0];
    assert!(
        (s.support_reaction_n.iter().sum::<f64>() - total_mass * (9.81 + s.acceleration[0]))
            .abs()
            < 0.02,
        "{:?}",
        s.support_reaction_n
    );
    let ic = s.front_interconnect.as_ref().unwrap();
    assert!(
        ic.heave_force_n.abs() > 1.,
        "heave channel should carry real load"
    );
}
#[test]
fn interconnect_mode_decoupling_holds_in_mass_mode() {
    let (p, mut r) = massive_with_interconnect();
    r.solve_equilibrium = false;
    r.initial_displacement = [0., 0.01, 0.];
    let plus = ride(&p, &r).unwrap();
    r.initial_displacement = [0., -0.01, 0.];
    let minus = ride(&p, &r).unwrap();
    assert!(plus.termination.is_none() && minus.termination.is_none());
    let a = plus.samples[0].front_interconnect.as_ref().unwrap();
    let b = minus.samples[0].front_interconnect.as_ref().unwrap();
    assert!((a.heave_force_n - b.heave_force_n).abs() < 1e-3);
    assert!((a.roll_force_n + b.roll_force_n).abs() < 1e-3);
}
#[test]
fn shock_length_limit_is_named() {
    let (p, mut r) = formula_car_demo().unwrap();
    let mut v = serde_json::to_value(p).unwrap();
    v["corners"][0]["spring_damper"]["min_length_m"] = serde_json::json!(0.5);
    let p = serde_json::from_value(v).unwrap();
    r.solve_equilibrium = false;
    r.duration_s = 0.;
    let run = ride(&p, &r).unwrap();
    assert!(run
        .termination
        .unwrap()
        .reason
        .contains("shock minimum length"));
}
fn all_mass() -> (Project, RideRequest) {
    let (mut p, mut r) = massive();
    for c in &mut p.corners {
        let body = |point| {
            Some(fky_lapsim_core::BodyMass {
                mass_kg: 1.5,
                center_of_mass: point,
                inertia: [[0.01, 0., 0.], [0., 0.02, 0.], [0., 0., 0.025]],
            })
        };
        c.component_masses.upper_arm = body(c.upper_ball);
        c.component_masses.lower_arm = body(c.lower_ball);
        c.component_masses.rocker = body(c.rocker_shock);
    }
    r.duration_s = 0.12;
    r.initial_displacement = [0.003, 0.001, -0.0005];
    (p, r)
}
fn finish(p: &Project, r: &RideRequest) -> fky_lapsim_core::RideRun {
    let run = ride(p, r).unwrap();
    assert!(run.termination.is_none(), "{:?}", run.termination);
    run
}
#[test]
fn moving_knuckle_inertial_support() {
    let (p, mut r) = massive();
    r.solve_equilibrium = false;
    r.road = fky_lapsim_core::RoadInput::Sine {
        amplitude_m: 0.002,
        frequency_hz: 3.,
        phases_rad: [std::f64::consts::FRAC_PI_2; 4],
    };
    let run = finish(&p, &r);
    let s = &run.samples[0];
    let extra = s.support_reaction_n.iter().sum::<f64>() - 300. * (9.81 + s.acceleration[0]);
    let expected = 32. * (9.81 - 0.002 * (std::f64::consts::TAU * 3.).powi(2));
    assert!((extra - expected).abs() < 0.002, "{extra} {expected}");
}
#[test]
fn asymmetric_mass_changes_equilibrium_and_zero_limit() {
    let (mut p, mut r) = all_mass();
    r.duration_s = 0.;
    r.initial_displacement = [0.; 3];
    p.corners[0]
        .component_masses
        .upper_arm
        .as_mut()
        .unwrap()
        .mass_kg = 8.;
    let mass = finish(&p, &r);
    r.mode = fky_lapsim_core::RideMode::Reduced;
    let reduced = finish(&p, &r);
    assert!((mass.equilibrium.unwrap()[1] - reduced.equilibrium.unwrap()[1]).abs() > 1e-5);
    r.mode = fky_lapsim_core::RideMode::RetainedComponentInertia;
    for c in &mut p.corners {
        for b in [
            &mut c.component_masses.upper_arm,
            &mut c.component_masses.lower_arm,
            &mut c.component_masses.rocker,
            &mut c.component_masses.knuckle,
        ]
        .into_iter()
        .flatten()
        {
            b.mass_kg = 0.;
        }
    }
    let zero = finish(&p, &r);
    for i in 0..3 {
        assert!(
            (zero.samples[0].acceleration[i] - reduced.samples[0].acceleration[i]).abs() < 1e-7
        );
    }
}
#[test]
fn energy_timestep_derivative_and_runtime_gates() {
    let (mut p, mut r) = all_mass();
    let start = std::time::Instant::now();
    for damped in [false, true] {
        for c in &mut p.corners {
            c.spring_damper.compression_damping = if damped { 1500. } else { 0. };
            c.spring_damper.rebound_damping = if damped { 2000. } else { 0. };
        }
        for moving in [false, true] {
            r.road = if moving {
                fky_lapsim_core::RoadInput::Sine {
                    amplitude_m: 0.001,
                    frequency_hz: 2.,
                    phases_rad: [0., 0.3, 0.1, -0.2],
                }
            } else {
                fky_lapsim_core::RoadInput::Flat
            };
            let mut errors = Vec::new();
            let mut final_states = Vec::new();
            for dt in [0.01, 0.005, 0.0025] {
                r.dt_s = dt;
                let run = finish(&p, &r);
                let first = &run.samples[0];
                let last = run.samples.last().unwrap();
                errors.push(last.energy_balance_error_j.abs());
                final_states.push(last.displacement);
                if damped && !moving {
                    assert!(last.energy_j < first.energy_j);
                }
            }
            eprintln!("damped={damped} moving={moving} dt errors J={errors:?}");
            assert!(errors[2] < 0.0002, "{errors:?}");
            assert!(errors[2] <= errors[0] + 1e-7);
            let delta =
                |a: [f64; 3], b: [f64; 3]| (0..3).map(|i| (a[i] - b[i]).abs()).fold(0., f64::max);
            assert!(
                delta(final_states[1], final_states[2])
                    < delta(final_states[0], final_states[1]) + 1e-9
            );
            r.derivative_step = 0.0005;
            let refined = finish(&p, &r);
            let first = &refined.samples[0];
            let last = refined.samples.last().unwrap();
            let mut reference = r.clone();
            reference.duration_s = 0.;
            reference.initial_displacement = [0.; 3];
            reference.initial_velocity = [0.; 3];
            let equilibrium_energy = finish(&p, &reference).samples[0].energy_j;
            let dynamic_scale = refined
                .samples
                .iter()
                .map(|s| {
                    (s.energy_j - first.energy_j).abs()
                        + s.dissipated_work_j.abs()
                        + s.support_work_j.abs()
                        + s.external_work_j.abs()
                })
                .fold((first.energy_j - equilibrium_energy).abs(), f64::max);
            eprintln!(
                "half derivative step residual={} J, dynamic scale={} J, normalized={}",
                last.energy_balance_error_j.abs(),
                dynamic_scale,
                last.energy_balance_error_j.abs() / dynamic_scale.max(1e-6)
            );
            assert!(
                delta(
                    refined.samples.last().unwrap().displacement,
                    final_states[2]
                ) < 1e-7
            );
            assert!(refined.samples.last().unwrap().energy_balance_error_j.abs() < 0.0002);
            r.derivative_step = 0.001;
        }
    }
    eprintln!("all-component 16 runs total wall {:?}", start.elapsed());
}
#[test]
fn mass_road_history_rejected_and_cylinder_cusp_diagnostic() {
    let (mut p, mut r) = massive();
    r.road = fky_lapsim_core::RoadInput::Histories {
        corners: std::array::from_fn(|_| vec![[0., 0.], [1., 0.01]]),
    };
    assert!(ride(&p, &r).unwrap_err().message.contains("C1 road"));
    r.road = fky_lapsim_core::RoadInput::Flat;
    r.solve_equilibrium = false;
    for c in &mut p.corners {
        c.tire_profile = fky_lapsim_core::TireProfile::Cylinder;
    }
    let run = ride(&p, &r).unwrap();
    assert!(run.termination.is_some());
}
#[test]
fn torus_fixed_steering_and_permutation() {
    let (mut p, mut r) = all_mass();
    r.duration_s = 0.01;
    r.rack_front = 0.001;
    r.rack_rear = -0.0005;
    for c in &mut p.corners {
        c.tire_profile = fky_lapsim_core::TireProfile::Torus;
    }
    let run = finish(&p, &r);
    p.corners.swap(0, 3);
    let permuted = finish(&p, &r);
    for (a, b) in run.samples.iter().zip(permuted.samples) {
        for i in 0..3 {
            assert!((a.displacement[i] - b.displacement[i]).abs() < 1e-8);
        }
        assert!((a.support_reaction_n[0] - b.support_reaction_n[3]).abs() < 0.01);
    }
}

#[test]
fn reflected_symmetry_and_explicit_failure_events() {
    let (mut p, mut r) = all_mass();
    r.duration_s = 0.02;
    let left = finish(&p, &r);
    r.initial_displacement[1] *= -1.;
    let right = finish(&p, &r);
    let a = left.samples.last().unwrap();
    let b = right.samples.last().unwrap();
    assert!((a.displacement[0] - b.displacement[0]).abs() < 1e-8);
    assert!((a.displacement[1] + b.displacement[1]).abs() < 1e-8);
    assert!((a.support_reaction_n[0] - b.support_reaction_n[1]).abs() < 0.01);
    r.solve_equilibrium = false;
    r.duration_s = 0.;
    r.initial_velocity = [3., 0., 0.];
    let invalid = ride(&p, &r).unwrap();
    assert!(invalid
        .termination
        .unwrap()
        .reason
        .contains("negative full support reaction"));
    r.initial_velocity = [0.; 3];
    r.initial_displacement = [1., 0., 0.];
    assert!(ride(&p, &r).unwrap().termination.is_some());
    r.initial_displacement = [0.; 3];
    p.corners[0].spring_damper.max_length_m = Some(0.1);
    assert!(ride(&p, &r)
        .unwrap()
        .termination
        .unwrap()
        .reason
        .contains("shock maximum length"));
    p.corners[0].spring_damper.min_length_m = Some(0.2);
    assert!(p.validate().is_err());
}
