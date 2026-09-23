use dw_core::dynamics::*;
use dw_core::{Project, SpringDamper};

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "{a} != {b} (tolerance {tol})");
}

#[test]
fn force_energy_chain_rule_and_signed_passive_damping() {
    let s = SpringDamper {
        spring_rate: 1000.,
        preload: 100.,
        compression_damping: 20.,
        rebound_damping: 30.,
        ..Default::default()
    };
    close(wheel_rate(1000., 100., 0.5, 2.), 450., 1e-12);
    close(spring_force(&s, 0.02), 120., 1e-12);
    close(damper_force(&s, 0.5), 10., 1e-12);
    close(damper_force(&s, -0.5), -15., 1e-12);
    for c in [-0.2, 0., 0.03] {
        close(
            (spring_energy(&s, c + 1e-6) - spring_energy(&s, c - 1e-6)) / 2e-6,
            spring_force(&s, c),
            1e-7,
        );
    }
    for u in [-2., -0.1, 0., 0.2, 2.] {
        assert!(u * damper_force(&s, u) >= 0.);
    }
}

#[test]
fn tabulated_laws_integrate_exactly_and_validate_passivity() {
    let mut p = demo();
    let s = &mut p.corners[0].spring_damper;
    s.spring_curve = Some(vec![[-0.1, -20.], [0., 100.], [0.1, 300.], [0.2, 400.]]);
    s.compression_curve = Some(vec![[0., 0.], [0.1, 20.], [0.5, 50.]]);
    s.rebound_curve = Some(vec![[0., 0.], [0.2, 60.], [0.6, 80.]]);
    close(spring_force(s, 0.05), 200., 1e-12);
    close(spring_force(s, 0.3), 400., 1e-12);
    close(spring_energy(s, 0.2), 55., 1e-12);
    close(spring_energy(s, -0.1), -4., 1e-12);
    close(spring_energy(s, 0.3), 95., 1e-12);
    close(damper_force(s, 0.3), 35., 1e-12);
    close(damper_force(s, -0.1), -30., 1e-12);
    close(damper_force(s, -2.), -80., 1e-12);
    for c in [-0.2, -0.1, -0.05, 0., 0.05, 0.1, 0.15, 0.2, 0.3] {
        close(
            (spring_energy(s, c + 1e-7) - spring_energy(s, c - 1e-7)) / 2e-7,
            spring_force(s, c),
            0.0001,
        );
    }
    p.validate().unwrap();
    for bad in [
        vec![[0., 0.], [0., 2.]],
        vec![[0., 0.], [1., -2.]],
        vec![[0.1, 0.], [1., 2.]],
        vec![[0., 0.], [f64::NAN, 2.]],
    ] {
        p.corners[0].spring_damper.rebound_curve = Some(bad);
        assert!(p.validate().is_err());
    }
}

#[test]
fn finite_roll_inertia_matches_world_rotation_and_free_motion_conserves_momentum() {
    use nalgebra::{UnitQuaternion, Vector3};
    let p = Project::example();
    let q = [0.01, 0.31, -0.22];
    let v = [0.2, 0.4, 0.6];
    let rot = |r: f64, p: f64| {
        UnitQuaternion::from_axis_angle(&Vector3::y_axis(), p)
            * UnitQuaternion::from_axis_angle(&Vector3::x_axis(), r)
    };
    let rotation = rot(q[1], q[2]);
    let eps = 1e-6;
    let omega = (rot(q[1] + eps * v[1], q[2] + eps * v[2])
        * rot(q[1] - eps * v[1], q[2] - eps * v[2]).inverse())
    .scaled_axis()
        / (2. * eps);
    let body = rotation.inverse() * omega;
    let expected = 0.5 * p.chassis.sprung_mass * v[0] * v[0]
        + 0.5
            * (0..3)
                .map(|i| p.chassis.inertia[i] * body[i] * body[i])
                .sum::<f64>();
    close(chassis_kinetic_energy(&p.chassis, q, v), expected, 1e-7);
}

fn demo() -> Project {
    let mut p = Project::example();
    p.chassis.sprung_mass = 300.;
    p.chassis.inertia = [100., 250., 300.];
    let mu = 0.1875 / 0.085_f64.sqrt();
    for c in &mut p.corners {
        c.spring_damper.preload = 300. * 9.81 / (4. * mu);
    }
    p
}

#[test]
fn derived_demo_equilibrates_and_heave_preserves_symmetry_and_dissipates_energy() {
    let (p, mut request) = formula_car_demo().unwrap();
    close(p.chassis.sprung_mass, 300., 1e-10);
    let mu = 0.1875 / 0.085_f64.sqrt();
    close(
        p.corners[0].spring_damper.preload,
        300. * 9.81 / (4. * mu),
        0.1,
    );
    request.duration_s = 0.15;
    request.initial_displacement = [0.001, 0., 0.];
    let run = ride(&p, &request).unwrap();
    assert!(run.termination.is_none(), "{:?}", run.termination);
    assert_eq!(run.model_fidelity, MODEL_FIDELITY);
    let eq = run.equilibrium.unwrap();
    for x in eq {
        close(x, 0., 1e-5);
    }
    let first = &run.samples[0];
    let last = run.samples.last().unwrap();
    close(last.time_s, request.duration_s, 1e-12);
    assert!(last.energy_j < first.energy_j);
    assert!(last.dissipated_work_j > 0.);
    assert!(last.displacement[0].abs() < 0.001);
    for s in &run.samples {
        close(s.displacement[1], 0., 1e-8);
        close(s.displacement[2], 0., 1e-8);
        close(
            s.support_reaction_n.iter().sum(),
            300. * (9.81 + s.acceleration[0]),
            1e-7,
        );
        assert!(s.energy_balance_error_j.abs() < 1e-4);
    }
}

#[test]
fn coupled_force_matches_independent_potential_gradient_and_road_work() {
    let p = demo();
    let q = [0.002, 0.003, -0.004];
    let v = [0.01, 0.02, -0.03];
    let phases = [0.2, -0.4, 0.6, -0.8];
    let amp = 0.001;
    let omega = std::f64::consts::TAU * 2.;
    let req = RideRequest {
        duration_s: 0.,
        solve_equilibrium: false,
        initial_displacement: q,
        initial_velocity: v,
        road: RoadInput::Sine {
            amplitude_m: amp,
            frequency_hz: 2.,
            phases_rad: phases,
        },
        ..Default::default()
    };
    let run = ride(&p, &req).unwrap();
    let s = &run.samples[0];
    let z = phases.map(|phase: f64| amp * phase.sin());
    let zd = phases.map(|phase: f64| amp * omega * phase.cos());
    let compression = |q: [f64; 3], z| {
        let m = dw_core::Motion {
            heave: q[0],
            roll: q[1],
            pitch: q[2],
            ..Default::default()
        };
        dw_core::simulate_on_road(&p, &m, z)
            .unwrap()
            .corners
            .map(|c| c.metrics.shock_compression_m)
    };
    let eps = 0.0001;
    let cp = compression(
        std::array::from_fn(|a| q[a] + eps * v[a]),
        std::array::from_fn(|i| z[i] + eps * zd[i]),
    );
    let cm = compression(
        std::array::from_fn(|a| q[a] - eps * v[a]),
        std::array::from_fn(|i| z[i] - eps * zd[i]),
    );
    for i in 0..4 {
        close(
            s.compression_velocity_m_s[i],
            (cp[i] - cm[i]) / (2. * eps),
            2e-5,
        );
    }
    let mut force = [0.; 3];
    for a in 0..3 {
        let mut qp = q;
        let mut qm = q;
        qp[a] += eps;
        qm[a] -= eps;
        let cp = compression(qp, z);
        let cm = compression(qm, z);
        force[a] = -(0..4)
            .map(|i| s.shock_force_n[i] * (cp[i] - cm[i]) / (2. * eps))
            .sum::<f64>();
    }
    let a = p.chassis.inertia[1] * q[1].cos().powi(2) + p.chassis.inertia[2] * q[1].sin().powi(2);
    let ap = 2. * (p.chassis.inertia[2] - p.chassis.inertia[1]) * q[1].sin() * q[1].cos();
    close(s.acceleration[0], force[0] / 300. - 9.81, 2e-4);
    close(
        s.acceleration[1],
        (force[1] + 0.5 * ap * v[2] * v[2]) / 100.,
        2e-4,
    );
    close(s.acceleration[2], (force[2] - ap * v[1] * v[2]) / a, 2e-4);
}

fn demo_with_interconnect() -> Project {
    let mut p = Project::example_with_interconnect();
    p.chassis.sprung_mass = 300.;
    p.chassis.inertia = [100., 250., 300.];
    let mu = 0.1875 / 0.085_f64.sqrt();
    for c in &mut p.corners {
        c.spring_damper.preload = 300. * 9.81 / (4. * mu);
    }
    p
}
fn single_sample(p: &Project, displacement: [f64; 3]) -> RideSample {
    let r = RideRequest {
        duration_s: 0.,
        solve_equilibrium: false,
        initial_displacement: displacement,
        ..Default::default()
    };
    let run = ride(p, &r).unwrap();
    assert!(run.termination.is_none(), "{:?}", run.termination);
    run.samples.into_iter().next().unwrap()
}
#[test]
fn ride_sample_has_no_interconnect_fields_without_configuration() {
    let (p, r) = formula_car_demo().unwrap();
    let run = ride(&p, &r).unwrap();
    assert!(run.termination.is_none(), "{:?}", run.termination);
    for s in &run.samples {
        assert!(s.front_interconnect.is_none());
        assert!(s.rear_interconnect.is_none());
    }
}
#[test]
fn pure_heave_excites_only_the_heave_interconnect_channel() {
    let p = demo_with_interconnect();
    let s = single_sample(&p, [0.005, 0., 0.]);
    let ic = s.front_interconnect.expect("front interconnect configured");
    // roll_arm_compression is an exact function of the (mirror-symmetric) rocker angle,
    // and pure heave gives both corners of the pair the identical rocker angle -- so the
    // roll channel's compression, and hence its zero-preload force, is exactly zero at any
    // heave magnitude, not just to first order.
    close(ic.roll_compression_m, 0., 1e-9);
    close(ic.roll_force_n, 0., 1e-4);
    assert!(ic.heave_force_n.abs() > 1., "heave channel should carry real load");
}
#[test]
fn pure_roll_mirrors_the_heave_channel_and_flips_the_roll_channel() {
    let p = demo_with_interconnect();
    let plus = single_sample(&p, [0., 0.01, 0.]);
    let minus = single_sample(&p, [0., -0.01, 0.]);
    let ic_plus = plus.front_interconnect.expect("front interconnect configured");
    let ic_minus = minus.front_interconnect.expect("front interconnect configured");
    // Distance is invariant under the left/right mirror reflection that relates a +roll
    // and -roll chassis pose, so heave_arm_compression (built from tip-to-anchor distances)
    // is an exact even function of roll: this is the mode-decoupling guarantee itself, not
    // a small-signal approximation, and holds at this (moderate) 0.01 rad magnitude.
    close(ic_plus.heave_compression_m, ic_minus.heave_compression_m, 1e-9);
    close(ic_plus.heave_force_n, ic_minus.heave_force_n, 1e-4);
    // roll_arm_compression is correspondingly an exact odd function of roll, and this
    // fixture's roll spring has zero preload, so its force flips sign exactly too.
    close(ic_plus.roll_compression_m, -ic_minus.roll_compression_m, 1e-9);
    close(ic_plus.roll_force_n, -ic_minus.roll_force_n, 1e-4);
    assert!(ic_plus.roll_force_n.abs() > 1., "roll channel should carry real load");
}
#[test]
fn support_reaction_sums_to_weight_with_interconnect_active() {
    let mut p = demo_with_interconnect();
    p.front_interconnect.as_mut().unwrap().heave.preload = 500.;
    let r = RideRequest {
        duration_s: 0.05,
        initial_displacement: [0.001, 0.0005, 0.],
        ..Default::default()
    };
    let run = ride(&p, &r).unwrap();
    assert!(run.termination.is_none(), "{:?}", run.termination);
    for s in &run.samples {
        close(
            s.support_reaction_n.iter().sum(),
            300. * (9.81 + s.acceleration[0]),
            1e-6,
        );
        let ic = s.front_interconnect.as_ref().unwrap();
        assert!(ic.heave_force_n.abs() > 1., "heave channel should carry real load");
    }
}
#[test]
fn energy_balances_with_interconnect_springs_and_dampers_active() {
    let p = demo_with_interconnect();
    let r = RideRequest {
        duration_s: 0.1,
        solve_equilibrium: true,
        initial_displacement: [0.006, 0.003, 0.],
        ..Default::default()
    };
    let run = ride(&p, &r).unwrap();
    assert!(run.termination.is_none(), "{:?}", run.termination);
    let last = run.samples.last().unwrap();
    assert!(last.energy_balance_error_j.abs() < 1e-3);
    assert!(last.dissipated_work_j > 0.);
}
#[test]
fn invalid_contact_and_unreachable_geometry_are_diagnostic_not_success() {
    let p = demo();
    let req = RideRequest {
        duration_s: 0.1,
        solve_equilibrium: false,
        initial_velocity: [2., 0., 0.],
        ..Default::default()
    };
    let run = ride(&p, &req).unwrap();
    assert!(run.termination.unwrap().reason.contains("contact"));
    assert!(run.samples.is_empty());
    let req = RideRequest {
        initial_displacement: [2., 0., 0.],
        initial_velocity: [0.; 3],
        ..req
    };
    assert!(ride(&p, &req).unwrap().termination.is_some());
    for req in [
        RideRequest {
            dt_s: 0.,
            ..Default::default()
        },
        RideRequest {
            duration_s: f64::NAN,
            ..Default::default()
        },
        RideRequest {
            duration_s: 1000.,
            dt_s: 1e-9,
            ..Default::default()
        },
    ] {
        assert!(ride(&p, &req).is_err());
    }
}

#[test]
fn free_and_forced_ride_converge_in_motion_and_energy_balance() {
    for (damped, forced) in [(false, false), (true, false), (true, true)] {
        let mut p = demo();
        if !damped {
            for c in &mut p.corners {
                c.spring_damper.compression_damping = 0.;
                c.spring_damper.rebound_damping = 0.;
            }
        }
        let mut endpoints = Vec::new();
        let mut balances = Vec::new();
        for dt in [0.02, 0.01, 0.005, 0.0025] {
            let req = RideRequest {
                duration_s: 0.4,
                dt_s: dt,
                solve_equilibrium: false,
                initial_displacement: [0.008, 0.001, -0.001],
                road: if forced {
                    RoadInput::Sine {
                        amplitude_m: 0.002,
                        frequency_hz: 3.,
                        phases_rad: [0., 0.1, 0.3, 0.4],
                    }
                } else {
                    RoadInput::Flat
                },
                ..Default::default()
            };
            let run = ride(&p, &req).unwrap();
            assert!(run.termination.is_none(), "{:?}", run.termination);
            let s = run.samples.last().unwrap();
            endpoints.push(s.displacement);
            balances.push(s.energy_balance_error_j.abs());
            if damped {
                assert!(s.dissipated_work_j > 0.);
            } else {
                close(s.dissipated_work_j, 0., 1e-12);
            }
            if forced {
                assert!(s.support_work_j.abs() > 0.01);
            }
        }
        let distance =
            |a: [f64; 3], b: [f64; 3]| (0..3).map(|i| (a[i] - b[i]).powi(2)).sum::<f64>().sqrt();
        let errors = [
            distance(endpoints[0], endpoints[3]),
            distance(endpoints[1], endpoints[3]),
            distance(endpoints[2], endpoints[3]),
        ];
        eprintln!("damped={damped} forced={forced} errors={errors:?}, balance={balances:?}");
        assert!(errors[1] < errors[0] && errors[2] < errors[1]);
        assert!(balances[1] < balances[0] && balances[2] < balances[1]);
        assert!(balances[3] < 0.0005);
    }
}

#[test]
fn history_knots_are_sampled_and_corner_order_does_not_change_motion() {
    let p = demo();
    let histories = std::array::from_fn(|i| {
        vec![
            [0., 0.],
            [0.013, 0.0001 * (i + 1) as f64],
            [0.031, -0.0001],
            [0.07, 0.],
        ]
    });
    let r = RideRequest {
        duration_s: 0.07,
        dt_s: 0.01,
        solve_equilibrium: false,
        road: RoadInput::Histories {
            corners: histories.clone(),
        },
        ..Default::default()
    };
    let a = ride(&p, &r).unwrap();
    assert!(a.termination.is_none());
    for knot in [0.013, 0.031, 0.07] {
        assert!(a.samples.iter().any(|s| (s.time_s - knot).abs() < 1e-12));
    }
    let mut shuffled = p.clone();
    shuffled.corners.swap(0, 3);
    let mut histories = histories;
    histories.swap(0, 3);
    let b = ride(
        &shuffled,
        &RideRequest {
            road: RoadInput::Histories { corners: histories },
            ..r.clone()
        },
    )
    .unwrap();
    assert!(b.termination.is_none());
    assert_eq!(a.samples.len(), b.samples.len());
    for (a, b) in a.samples.iter().zip(&b.samples) {
        for i in 0..3 {
            close(a.displacement[i], b.displacement[i], 1e-10);
        }
        close(a.shock_force_n[0], b.shock_force_n[3], 1e-6);
    }
    assert!(a.samples.last().unwrap().energy_balance_error_j.abs() < 0.0001);
    let invalid = RoadInput::Histories {
        corners: std::array::from_fn(|_| vec![[0., 0.], [0., 0.01], [1., 0.]]),
    };
    assert!(ride(&p, &RideRequest { road: invalid, ..r }).is_err());
}

#[test]
fn moving_road_velocity_matches_direct_derivatives_at_tilted_pose() {
    let p = demo();
    let q = [0.004, 0.009, -0.007];
    let z = [0.001, 0.002, -0.001, -0.002];
    let state = |q: [f64; 3], z: [f64; 4]| {
        dw_core::simulate_on_road(
            &p,
            &dw_core::Motion {
                heave: q[0],
                roll: q[1],
                pitch: q[2],
                ..Default::default()
            },
            z,
        )
        .unwrap()
        .corners
        .map(|c| c.metrics.shock_compression_m)
    };
    let c = state(q, z);
    let translated = state([q[0] + 0.003, q[1], q[2]], z.map(|z| z + 0.003));
    for i in 0..4 {
        close(c[i], translated[i], 1e-8);
    }
    let eps = 0.0001;
    let hp = state([q[0] + eps, q[1], q[2]], z);
    let hm = state([q[0] - eps, q[1], q[2]], z);
    for i in 0..4 {
        let mut zp = z;
        let mut zm = z;
        zp[i] += eps;
        zm[i] -= eps;
        let cp = state(q, zp);
        let cm = state(q, zm);
        close(cp[i] - cm[i], hm[i] - hp[i], 1e-8);
    }
}

#[test]
fn equilibrium_solves_preload_deficit_and_external_work_is_accounted() {
    let mut p = demo();
    for c in &mut p.corners {
        c.spring_damper.preload *= 0.9;
    }
    let run = ride(
        &p,
        &RideRequest {
            duration_s: 0.,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(run.termination.is_none());
    assert!(run.equilibrium.unwrap()[0] < -0.001);
    for a in run.samples[0].acceleration {
        close(a, 0., 0.00002);
    }
    let r = RideRequest {
        duration_s: 0.1,
        external_force: [100., 5., -3.],
        initial_velocity: [0.01, 0.002, -0.003],
        ..Default::default()
    };
    let run = ride(&p, &r).unwrap();
    assert!(run.termination.is_none());
    let first = run.samples.first().unwrap();
    let last = run.samples.last().unwrap();
    let external = (0..3)
        .map(|a| r.external_force[a] * (last.displacement[a] - first.displacement[a]))
        .sum::<f64>();
    close(last.external_work_j, external, 1e-10);
    assert!(last.energy_balance_error_j.abs() < 1e-5);
}

#[test]
fn spatial_road_matches_equivalent_temporal_sine_and_request_round_trips() {
    let p = demo();
    let phases = [0.1, 0.2, 0.3, 0.4];
    let speed = 5.;
    let wavelength = 2.;
    let r = RideRequest {
        duration_s: 0.03,
        solve_equilibrium: false,
        road: RoadInput::SpatialSine {
            amplitude_m: 0.001,
            wavelength_m: wavelength,
            speed_m_s: speed,
            phases_rad: phases,
        },
        ..Default::default()
    };
    let encoded = serde_json::to_string(&r).unwrap();
    let decoded = serde_json::from_str(&encoded).unwrap();
    let a = ride(&p, &decoded).unwrap();
    let r = RideRequest {
        road: RoadInput::Sine {
            amplitude_m: 0.001,
            frequency_hz: speed / wavelength,
            phases_rad: std::array::from_fn(|i| {
                phases[i] + std::f64::consts::TAU * p.corners[i].wheel_center[0] / wavelength
            }),
        },
        ..r
    };
    let b = ride(&p, &r).unwrap();
    assert!(a.termination.is_none() && b.termination.is_none());
    for (a, b) in a.samples.iter().zip(&b.samples) {
        for i in 0..3 {
            close(a.displacement[i], b.displacement[i], 1e-12);
        }
    }
}

#[test]
fn roll_mirrors_and_late_contact_loss_reports_last_valid_state() {
    let p = demo();
    let r = RideRequest {
        duration_s: 0.08,
        solve_equilibrium: false,
        initial_displacement: [0., 0.004, 0.],
        ..Default::default()
    };
    let a = ride(&p, &r).unwrap();
    let b = ride(
        &p,
        &RideRequest {
            initial_displacement: [0., -0.004, 0.],
            ..r
        },
    )
    .unwrap();
    assert!(a.termination.is_none() && b.termination.is_none());
    for (a, b) in a.samples.iter().zip(&b.samples) {
        close(a.displacement[1], -b.displacement[1], 1e-8);
        close(a.shock_force_n[0], b.shock_force_n[1], 1e-4);
    }
    let run = ride(
        &p,
        &RideRequest {
            duration_s: 1.,
            solve_equilibrium: false,
            external_force: [8000., 0., 0.],
            ..Default::default()
        },
    )
    .unwrap();
    let event = run.termination.unwrap();
    assert!(event.reason.contains("contact"));
    let last = run.samples.last().unwrap();
    close(event.last_valid_time_s.unwrap(), last.time_s, 1e-12);
    assert!(event.time_s >= last.time_s && event.time_s < 1.);
    assert!(event.time_s - last.time_s <= 1e-5);
    assert!(last.support_reaction_n.iter().all(|n| *n >= -1e-6));
}

#[test]
fn combined_history_knots_respect_request_work_limit_before_solving() {
    let p = demo();
    let corners = std::array::from_fn(|i| {
        let mut v = vec![[0., 2.]];
        for k in 1..6000 {
            v.push([(k * 4 + i) as f64 / 24000., 2.]);
        }
        v.push([1., 2.]);
        v
    });
    let r = RideRequest {
        duration_s: 1.,
        dt_s: 1.,
        solve_equilibrium: false,
        road: RoadInput::Histories { corners },
        ..Default::default()
    };
    assert!(ride(&p, &r).is_err());
}
