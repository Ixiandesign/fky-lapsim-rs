use dw_core::{optimize::*, Project};
fn request() -> OptimizationRequest {
    OptimizationRequest {
        variables: vec![Variable {
            path: "/corners/0/shock_chassis/1".into(),
            kind: VariableKind::Continuous {
                lower: 0.05,
                upper: 0.25,
            },
        }],
        targets: vec![Target {
            corner: None,
            metric: "project:/corners/0/shock_chassis/1".into(),
            value: 0.21,
            values: None,
            scale: 1.,
            weight: 1.,
            aggregation: Aggregation::MeanSquared,
        }],
        max_evaluations: 100,
        generations: 12,
        population_size: 6,
        ..Default::default()
    }
}
#[test]
fn reachable_and_deterministic() {
    let p = Project::example();
    let r = request();
    let a = optimize(&p, &r).unwrap();
    let b = optimize(&p, &r).unwrap();
    assert!(a.best_feasible.as_ref().unwrap().evaluation.score.unwrap() < 1e-6);
    assert_eq!(
        serde_json::to_value(&a.best_feasible).unwrap(),
        serde_json::to_value(&b.best_feasible).unwrap()
    );
    assert_eq!(a.status, "validated");
    assert!(a.candidate_attempts <= 100);
}
#[test]
fn zero_budget() {
    let mut r = request();
    r.max_evaluations = 0;
    let a = optimize(&Project::example(), &r).unwrap();
    assert_eq!(a.candidate_attempts, 0);
    assert!(a.best_feasible.is_none());
}
#[test]
fn invalid_contract() {
    let p = Project::example();
    let mut r = request();
    r.variables[0].path = "/schema_version".into();
    assert!(optimize(&p, &r).is_err());
    r = request();
    r.targets[0].scale = 0.;
    assert!(optimize(&p, &r).is_err());
}
#[test]
fn feasibility_and_score_oracle() {
    let p = Project::example();
    let mut r = request();
    r.constraints.push(Constraint {
        corner: None,
        metric: r.targets[0].metric.clone(),
        min: None,
        max: Some(0.18),
        scale: 0.01,
    });
    let a = evaluate_candidate(&p, &r, &[0.15]).unwrap();
    let b = evaluate_candidate(&p, &r, &[0.21]).unwrap();
    assert!(a.feasible);
    assert!(!b.feasible);
    assert!((a.score.unwrap() - 0.0036).abs() < 1e-12);
    assert!((b.violation - 3.).abs() < 1e-12);
    assert!(a.better_than(&b));
    assert!(evaluate_candidate(&p, &r, &[0.18]).unwrap().feasible);
}
#[test]
fn resume_mid_generation() {
    let p = Project::example();
    let r = request();
    let mut s = OptimizationSession::start(&p, &r).unwrap();
    s.advance(4, None).unwrap();
    let json = s.checkpoint().unwrap();
    let mut t = OptimizationSession::resume(&p, &r, &json).unwrap();
    while !t.is_finished() {
        t.advance(3, None).unwrap();
    }
    let a = optimize(&p, &r).unwrap();
    assert_eq!(
        serde_json::to_value(a.best_feasible).unwrap(),
        serde_json::to_value(t.result().best_feasible).unwrap()
    );
    let mut changed = r.clone();
    changed.seed += 1;
    assert!(OptimizationSession::resume(&p, &changed, &json).is_err());
}
#[test]
fn fixed_design_has_no_fake_population() {
    let mut r = request();
    r.variables[0].kind = VariableKind::Continuous {
        lower: 0.15,
        upper: 0.15,
    };
    let out = optimize(&Project::example(), &r).unwrap();
    assert_eq!(
        out.candidate_attempts, 3,
        "baseline + one design + independent validation"
    );
    assert_eq!(out.generation, 0);
    assert_eq!(
        out.physics_cases_completed, 0,
        "project property reads are not physics calls"
    );
}
#[test]
fn real_geometry_reachable_shock_target() {
    let p = Project::example();
    let mut r = request();
    r.scenarios = vec![dw_core::Motion::default()];
    r.targets[0].metric = "shock_length_m".into();
    r.targets[0].corner = Some(dw_core::CornerId::FrontLeft);
    r.targets[0].value = (0.19_f64.powi(2) + 0.15_f64.powi(2)).sqrt();
    r.validation_samples = 0;
    r.max_seconds = 120.;
    let out = optimize(&p, &r).unwrap();
    assert!(
        out.best_feasible
            .as_ref()
            .unwrap()
            .evaluation
            .score
            .unwrap()
            < 1e-7
    );
    assert!(
        out.best_feasible
            .as_ref()
            .unwrap()
            .evaluation
            .score
            .unwrap()
            < out.baseline.unwrap().evaluation.score.unwrap() / 100.
    );
    assert_eq!(out.status, "validated");
}
#[test]
fn held_out_failure_and_retained_cases() {
    let p = Project::example();
    let mut r = request();
    r.variables[0].kind = VariableKind::Continuous {
        lower: 0.15,
        upper: 0.15,
    };
    r.constraints.push(Constraint {
        corner: None,
        metric: r.targets[0].metric.clone(),
        min: Some(0.14),
        max: Some(0.16),
        scale: 0.01,
    });
    r.uncertainty.push(Perturbation {
        path: r.variables[0].path.clone(),
        half_range: 0.04,
        factor: None,
    });
    r.validation_samples = 8;
    let out = optimize(&p, &r).unwrap();
    assert!(out.best_feasible.is_some());
    assert_eq!(out.status, "validation_failed");
    assert!(!out.validation.unwrap().feasible);
    r.scenarios = vec![
        dw_core::Motion {
            heave: 1000.,
            ..Default::default()
        },
        dw_core::Motion::default(),
    ];
    let e = evaluate_candidate(&p, &r, &[0.15]).unwrap();
    assert!(!e.feasible);
    assert_eq!(e.requested_cases, 3);
    assert_eq!(e.completed_cases, 3);
    assert_eq!(e.failed_cases, 1);
    assert_eq!(e.failures.len(), 1);
    assert_eq!(e.physics_cases_completed, 2);
    r.scenarios.clear();
    r.constraints.clear();
    r.uncertainty = vec![Perturbation {
        path: "/chassis/inertia/0".into(),
        half_range: 100000.,
        factor: None,
    }];
    let out = optimize(&p, &r).unwrap();
    assert_eq!(out.status, "validation_failed");
    assert!(out.validation.unwrap().failed_samples > 0);
}
#[test]
fn budget_reserve_and_worker_determinism() {
    let p = Project::example();
    let mut r = request();
    r.max_evaluations = 10;
    let a = optimize(&p, &r).unwrap();
    assert_eq!(a.candidate_attempts, 10);
    r.workers = 2;
    let b = optimize(&p, &r).unwrap();
    assert_eq!(b.candidate_attempts, 10);
    assert_eq!(
        serde_json::to_value(a.best_feasible).unwrap(),
        serde_json::to_value(b.best_feasible).unwrap()
    );
    r.max_evaluations = 1;
    let a = optimize(&p, &r).unwrap();
    assert!(a.best_feasible.is_none());
    assert!(a.validation.is_none());
    assert_eq!(a.candidate_attempts, 1);
}
#[test]
fn target_curve_squares_before_aggregation() {
    let p = Project::example();
    let mut r = request();
    r.scenarios = vec![dw_core::Motion::default(); 2];
    let y = (0.25_f64.powi(2) + 0.15_f64.powi(2)).sqrt();
    r.targets[0].corner = Some(dw_core::CornerId::FrontLeft);
    r.targets[0].metric = "shock_length_m".into();
    r.targets[0].values = Some(vec![y - 0.01, y + 0.01]);
    r.targets[0].scale = 0.01;
    r.targets[0].weight = 3.;
    let e = evaluate_candidate(&p, &r, &[0.15]).unwrap();
    assert!((e.score.unwrap() - 3.).abs() < 1e-10);
    r.targets[0].values = Some(vec![y - 0.01, y + 0.02]);
    assert!((evaluate_candidate(&p, &r, &[0.15]).unwrap().score.unwrap() - 7.5).abs() < 1e-10);
    r.targets[0].aggregation = Aggregation::WorstSquared;
    assert!((evaluate_candidate(&p, &r, &[0.15]).unwrap().score.unwrap() - 12.).abs() < 1e-10);
    r.targets[0].values = Some(vec![y]);
    assert!(evaluate_candidate(&p, &r, &[0.15]).is_err());
}
#[test]
fn invalid_project_attempt_has_no_phantom_physics() {
    let p = Project::example();
    let mut r = request();
    r.variables[0] = Variable {
        path: "/corners/0/tire_radius".into(),
        kind: VariableKind::Continuous {
            lower: -1.,
            upper: 1.,
        },
    };
    r.scenarios = vec![dw_core::Motion::default()];
    let e = evaluate_candidate(&p, &r, &[-0.5]).unwrap();
    assert_eq!(e.failed_cases, 2);
    assert_eq!(e.physics_cases_completed, 0);
}
#[test]
fn failed_candidate_checkpoint_is_resumable() {
    let p = Project::example();
    let mut r = request();
    r.variables[0] = Variable {
        path: "/corners/0/tire_radius".into(),
        kind: VariableKind::Continuous {
            lower: -1.,
            upper: -0.1,
        },
    };
    let mut s = OptimizationSession::start(&p, &r).unwrap();
    s.advance(3, None).unwrap();
    assert!(OptimizationSession::resume(&p, &r, &s.checkpoint().unwrap()).is_ok());
}
fn ride_project() -> Project {
    let mut p = Project::example();
    p.chassis.sprung_mass = 300.;
    p.chassis.inertia = [100., 250., 300.];
    let ratio = 0.1875 / 0.085_f64.sqrt();
    for c in &mut p.corners {
        c.spring_damper.preload = 300. * 9.81 / (4. * ratio);
    }
    p
}
#[test]
fn real_ride_spring_and_damper_objectives() {
    let p = ride_project();
    let mut r = request();
    r.variables[0] = Variable {
        path: "/corners/0/spring_damper/spring_rate".into(),
        kind: VariableKind::Continuous {
            lower: 10000.,
            upper: 70000.,
        },
    };
    r.ride_request = Some(dw_core::RideRequest {
        duration_s: 0.01,
        dt_s: 0.005,
        solve_equilibrium: false,
        initial_displacement: [0.005, 0., 0.],
        initial_velocity: [0.1, 0., 0.],
        ..Default::default()
    });
    r.targets[0].metric = "ride.rms_heave_acceleration_m_s2".into();
    r.targets[0].value = 0.;
    r.validation_samples = 0;
    r.max_seconds = 300.;
    let a = evaluate_candidate(&p, &r, &[10000.]).unwrap();
    let b = evaluate_candidate(&p, &r, &[70000.]).unwrap();
    assert!(a.feasible && b.feasible, "{a:?} {b:?}");
    assert!((a.score.unwrap() - b.score.unwrap()).abs() > 1e-4);
    r.ride_request.as_mut().unwrap().initial_velocity = [-0.1, 0., 0.];
    r.variables[0] = Variable {
        path: "/corners/0/spring_damper/compression_damping".into(),
        kind: VariableKind::Continuous {
            lower: 100.,
            upper: 4000.,
        },
    };
    let a = evaluate_candidate(&p, &r, &[100.]).unwrap();
    let b = evaluate_candidate(&p, &r, &[4000.]).unwrap();
    assert!(a.feasible && b.feasible);
    assert!((a.score.unwrap() - b.score.unwrap()).abs() > 1e-4);
    r.targets[0].value = evaluate_candidate(&p, &r, &[3000.])
        .unwrap()
        .score
        .unwrap()
        .sqrt();
    r.max_evaluations = 24;
    r.generations = 3;
    let out = optimize(&p, &r).unwrap();
    assert_eq!(out.status, "validated");
    assert!(
        out.best_feasible
            .as_ref()
            .unwrap()
            .evaluation
            .score
            .unwrap()
            < out.baseline.as_ref().unwrap().evaluation.score.unwrap() / 10.
    );
    assert!(out
        .best_feasible
        .unwrap()
        .evaluation
        .model_fidelity
        .is_some());
}
fn synthetic_ride() -> dw_core::RideRun {
    let samples = [0., 0.25, 1.]
        .into_iter()
        .map(|t| dw_core::RideSample {
            time_s: t,
            displacement: [-2. * t, 3. * t, -4. * t],
            velocity: [0.; 3],
            acceleration: [t, 0., 0.],
            compression_m: [0.; 4],
            compression_velocity_m_s: [0.; 4],
            shock_force_n: [0.; 4],
            support_reaction_n: [0.; 4],
            front_interconnect: None,
            rear_interconnect: None,
            energy_j: 0.,
            dissipated_work_j: 0.,
            support_work_j: 0.,
            external_work_j: 0.,
            energy_balance_error_j: 0.,
        })
        .collect();
    dw_core::RideRun {
        model_fidelity: "fixture".into(),
        corner_ids: Project::example().corners.map(|c| c.id),
        equilibrium: None,
        samples,
        termination: None,
    }
}
#[test]
fn uneven_ride_quadrature_and_partial_termination() {
    let mut run = synthetic_ride();
    let m = ride_summary(&run, 1.).unwrap();
    let integral = 0.25 * 0.0625 / 2. + 0.75 * (0.0625 + 1.) / 2.;
    assert!((m["ride.rms_heave_acceleration_m_s2"] - f64::sqrt(integral)).abs() < 1e-14);
    assert_eq!(m["ride.peak_heave_m"], 2.);
    assert_eq!(m["ride.peak_pitch_rad"], 4.);
    for s in &mut run.samples {
        s.acceleration[0] = 2.;
    }
    assert_eq!(
        ride_summary(&run, 1.).unwrap()["ride.rms_heave_acceleration_m_s2"],
        2.
    );
    run.termination = Some(dw_core::RideTermination {
        time_s: 1.,
        last_valid_time_s: Some(1.),
        corner: None,
        reason: "failed stage".into(),
    });
    assert!(ride_summary(&run, 1.).is_err());
    run.termination = None;
    run.samples.pop();
    assert!(ride_summary(&run, 1.).is_err());
}
#[test]
fn scheduled_work_cap_is_checked() {
    let p = Project::example();
    let mut r = request();
    r.max_evaluations = 1_000_000;
    r.training_samples = 256;
    r.scenarios = vec![dw_core::Motion::default(); 128];
    assert!(OptimizationSession::start(&p, &r).is_err());
}

#[test]
fn cancelled_case_is_incomplete_not_survivor_average() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let p = Project::example();
    let mut r = request();
    r.scenarios = vec![dw_core::Motion::default(); 2];
    let polls = AtomicUsize::new(0);
    let e = evaluate_candidate_controlled(&p, &r, &[0.15], false, &|| {
        polls.fetch_add(1, Ordering::Relaxed) >= 2
    })
    .unwrap();
    assert!(!e.complete && !e.feasible);
    assert_eq!(e.completed_cases, 2);
    assert_eq!(e.requested_cases, 3);
    assert!(e.score.is_none());
    let token = std::sync::atomic::AtomicBool::new(true);
    let mut s = OptimizationSession::start(&p, &r).unwrap();
    let out = s.advance(10, Some(&token)).unwrap();
    assert_eq!(out.candidate_attempts, 0);
    assert_eq!(out.status, "cancelled_not_validated");
    r.max_seconds = 0.;
    let e = evaluate_candidate(&p, &r, &[0.15]).unwrap();
    assert_eq!(e.completed_cases, 0);
    assert!(!e.complete);
}
#[test]
fn forged_cached_winner_is_recomputed() {
    let p = Project::example();
    let mut r = request();
    r.max_evaluations = 3;
    let mut s = OptimizationSession::start(&p, &r).unwrap();
    s.advance(2, None).unwrap();
    let (_, payload): (u64, String) = serde_json::from_str(&s.checkpoint().unwrap()).unwrap();
    let mut v: serde_json::Value = serde_json::from_str(&payload).unwrap();
    v["result"]["best_feasible"]["evaluation"]["score"] = serde_json::json!(0.);
    v["result"]["best_feasible"]["evaluation"]["contributions"] = serde_json::json!([0.]);
    let payload = serde_json::to_string(&v).unwrap();
    let sum = payload.bytes().fold(0xcbf29ce484222325u64, |a, b| {
        (a ^ u64::from(b)).wrapping_mul(0x100000001b3)
    });
    let forged = serde_json::to_string(&(sum, payload)).unwrap();
    let mut s = OptimizationSession::resume(&p, &r, &forged).unwrap();
    s.advance(1, None).unwrap();
    let best = s.result().best_feasible.unwrap();
    assert!((best.evaluation.score.unwrap() - 0.0036).abs() < 1e-12);
}
#[test]
fn global_vehicle_metrics_use_no_corner_selector() {
    let p = Project::example();
    let mut r = request();
    r.scenarios = vec![dw_core::Motion::default()];
    r.targets[0].metric = "vehicle.left_wheelbase_m".into();
    r.targets[0].value = 2.6;
    let e = evaluate_candidate(&p, &r, &[0.15]).unwrap();
    assert!(e.feasible);
    assert!(e.score.unwrap() < 1e-20);
    r.targets[0].corner = Some(dw_core::CornerId::FrontLeft);
    assert!(evaluate_candidate(&p, &r, &[0.15]).is_err());
}
#[test]
fn relations_discrete_and_correlated_uncertainty() {
    let p = Project::example();
    let mut r = request();
    r.relations = vec![Relation {
        source: r.variables[0].path.clone(),
        destination: "/corners/1/shock_chassis/1".into(),
        factor: -1.,
        offset: 0.,
    }];
    let q = candidate_project(&p, &r, &[0.2]).unwrap();
    assert_eq!(q.corners[1].shock_chassis[1], -0.2);
    r.targets.push(Target {
        corner: None,
        metric: "project:/corners/1/shock_chassis/1".into(),
        value: -0.21,
        values: None,
        scale: 1.,
        weight: 1.,
        aggregation: Aggregation::MeanSquared,
    });
    r.training_samples = 12;
    r.uncertainty = vec![Perturbation {
        path: r.variables[0].path.clone(),
        half_range: 0.02,
        factor: None,
    }];
    let a = evaluate_candidate(&p, &r, &[0.21]).unwrap();
    assert_eq!(a.contributions[0], a.contributions[1]);
    assert_eq!(
        serde_json::to_value(&a).unwrap(),
        serde_json::to_value(evaluate_candidate(&p, &r, &[0.21]).unwrap()).unwrap()
    );
    r.relations.clear();
    r.uncertainty = vec![
        Perturbation {
            path: "/corners/0/tire_radius".into(),
            half_range: 0.02,
            factor: Some("batch".into()),
        },
        Perturbation {
            path: "/corners/1/tire_radius".into(),
            half_range: 0.02,
            factor: Some("batch".into()),
        },
    ];
    for (i, t) in r.targets.iter_mut().enumerate() {
        t.metric = format!("project:/corners/{i}/tire_radius");
        t.value = 0.3;
    }
    let a = evaluate_candidate(&p, &r, &[0.21]).unwrap();
    assert_eq!(a.contributions[0], a.contributions[1]);
    assert!(a.worst_squared_residual <= 0.02_f64.powi(2));
    r.variables = vec![Variable {
        path: "/corners/0/tire_profile".into(),
        kind: VariableKind::Discrete {
            choices: vec!["disk".into(), "torus".into()],
        },
    }];
    assert_eq!(
        candidate_project(&p, &r, &[1.]).unwrap().corners[0].tire_profile,
        dw_core::TireProfile::Torus
    );
    assert!(candidate_project(&p, &r, &[0.5]).is_err());
    r.relations = vec![
        Relation {
            source: "/corners/0/tire_radius".into(),
            destination: "/corners/1/tire_radius".into(),
            factor: 1.,
            offset: 0.,
        },
        Relation {
            source: "/corners/1/tire_radius".into(),
            destination: "/corners/0/tire_radius".into(),
            factor: 1.,
            offset: 0.,
        },
    ];
    assert!(evaluate_candidate(&p, &r, &[0.]).is_err());
}
#[test]
fn shuffled_corner_ids_keep_geometry_objective() {
    let p = Project::example();
    let mut q = p.clone();
    q.corners.swap(0, 3);
    let mut r = request();
    r.targets[0].metric = "shock_length_m".into();
    r.targets[0].corner = Some(dw_core::CornerId::FrontLeft);
    r.scenarios = vec![dw_core::Motion::default()];
    let a = evaluate_candidate(&p, &r, &[0.21]).unwrap();
    r.variables[0].path = "/corners/3/shock_chassis/1".into();
    let b = evaluate_candidate(&q, &r, &[0.21]).unwrap();
    assert_eq!(a.score, b.score);
}
#[test]
fn malformed_checkpoint_population_is_rejected() {
    let p = Project::example();
    let r = request();
    let s = OptimizationSession::start(&p, &r).unwrap();
    let (_, payload): (u64, String) = serde_json::from_str(&s.checkpoint().unwrap()).unwrap();
    let mut v: serde_json::Value = serde_json::from_str(&payload).unwrap();
    v["phase"] = serde_json::json!("evolution");
    let payload = serde_json::to_string(&v).unwrap();
    let sum = payload.bytes().fold(0xcbf29ce484222325u64, |a, b| {
        (a ^ u64::from(b)).wrapping_mul(0x100000001b3)
    });
    assert!(
        OptimizationSession::resume(&p, &r, &serde_json::to_string(&(sum, payload)).unwrap())
            .is_err()
    );
}
#[test]
fn truncated_failure_details_preserve_each_case_status() {
    let p = Project::example();
    let mut r = request();
    r.scenarios = vec![
        dw_core::Motion {
            heave: 1000.,
            ..Default::default()
        };
        40
    ];
    let e = evaluate_candidate(&p, &r, &[0.15]).unwrap();
    assert_eq!(e.failed_cases, 40);
    assert_eq!(e.failures.len(), 32);
    assert!(e.failures_truncated);
    assert_eq!(e.case_status(0, 0), Some(CaseStatus::Success));
    assert_eq!(e.case_status(0, 40), Some(CaseStatus::Failure));
    assert_eq!(e.case_status(0, 41), None);
}
#[test]
fn overflow_is_a_retained_failure_not_nonfinite_json() {
    let p = Project::example();
    let mut r = request();
    r.targets[0].weight = f64::MAX;
    r.targets[0].value = 10.;
    let e = evaluate_candidate(&p, &r, &[0.15]).unwrap();
    assert!(!e.feasible);
    assert!(e.score.is_none());
    assert_eq!(e.failed_samples, 1);
    assert_eq!(e.case_status(0, 0), Some(CaseStatus::Failure));
    assert!(!serde_json::to_string(&e.contributions)
        .unwrap()
        .contains("null"));
}
fn edited_checkpoint(json: &str, edit: impl FnOnce(&mut serde_json::Value)) -> String {
    let (_, payload): (u64, String) = serde_json::from_str(json).unwrap();
    let mut state: serde_json::Value = serde_json::from_str(&payload).unwrap();
    edit(&mut state);
    let payload = serde_json::to_string(&state).unwrap();
    let sum = payload.bytes().fold(0xcbf29ce484222325u64, |a, b| {
        (a ^ u64::from(b)).wrapping_mul(0x100000001b3)
    });
    serde_json::to_string(&(sum, payload)).unwrap()
}
#[test]
fn checkpoint_rejects_fabricated_terminal_success() {
    let p = Project::example();
    let r = request();
    let s = OptimizationSession::start(&p, &r).unwrap();
    let forged = edited_checkpoint(&s.checkpoint().unwrap(), |v| {
        v["phase"] = serde_json::json!("done");
        v["result"]["status"] = serde_json::json!("validated");
    });
    assert!(OptimizationSession::resume(&p, &r, &forged).is_err());
}
#[test]
fn checkpoint_checks_baseline_and_both_validation_reports() {
    let p = Project::example();
    let mut r = request();
    r.max_evaluations = 3;
    let mut s = OptimizationSession::start(&p, &r).unwrap();
    while !s.is_finished() {
        s.advance(3, None).unwrap();
    }
    let json = s.checkpoint().unwrap();
    assert!(OptimizationSession::resume(&p, &r, &json)
        .unwrap()
        .is_finished());
    for path in [
        "/result/baseline/evaluation/completed_cases",
        "/result/training_revalidation/completed_cases",
        "/result/validation/completed_cases",
    ] {
        let forged = edited_checkpoint(&json, |v| {
            *v.pointer_mut(path).unwrap() = serde_json::json!(999999);
        });
        assert!(
            OptimizationSession::resume(&p, &r, &forged).is_err(),
            "accepted {path}"
        );
    }
    for path in ["/result/validation", "/result/best_feasible"] {
        let forged = edited_checkpoint(&json, |v| {
            *v.pointer_mut(path).unwrap() = serde_json::Value::Null;
        });
        assert!(
            OptimizationSession::resume(&p, &r, &forged).is_err(),
            "accepted missing {path}"
        );
    }
}
#[test]
fn ride_summary_rejects_nonfinite_duration() {
    let run = synthetic_ride();
    for duration in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        assert!(ride_summary(&run, duration).is_err(), "accepted {duration}");
    }
}
#[test]
fn combined_case_and_aggregate_failure_checkpoint_round_trips() {
    let p = Project::example();
    let mut r = request();
    r.max_evaluations = 3;
    r.targets[0].value = f64::MAX;
    let mut aggregate = r.targets[0].clone();
    aggregate.value = 10.;
    aggregate.weight = f64::MAX;
    r.targets.push(aggregate);
    let mut s = OptimizationSession::start(&p, &r).unwrap();
    s.advance(2, None).unwrap();
    let json = s.checkpoint().unwrap();
    let mut resumed = OptimizationSession::resume(&p, &r, &json)
        .expect("optimizer must resume its own failed-candidate checkpoint");
    let e = resumed.result().best_infeasible.unwrap().evaluation;
    assert_eq!(e.completed_cases, 1);
    assert_eq!(e.failed_cases, 1);
    assert_eq!(e.failed_samples, 1);
    assert_eq!(e.failures.len(), 1);
    assert!(e.failures[0]
        .reason
        .contains("nonfinite objective aggregate"));
    assert_eq!(e.case_status(0, 0), Some(CaseStatus::Failure));
    resumed.advance(1, None).unwrap();
    assert!(resumed.is_finished());
    assert!(OptimizationSession::resume(&p, &r, &resumed.checkpoint().unwrap()).is_ok());
}
#[test]
fn interconnect_parameter_registry_only_when_configured() {
    let registry = parameter_registry(&Project::example());
    assert!(!registry.iter().any(|s| s.starts_with("/front_interconnect")));
    assert!(!registry.iter().any(|s| s.starts_with("/rear_interconnect")));
    assert!(!registry.iter().any(|s| s.contains("rocker_heave_arm")));

    let registry = parameter_registry(&Project::example_with_interconnect());
    assert!(registry.contains(&"/front_interconnect/heave/spring_rate".to_string()));
    assert!(registry.contains(&"/front_interconnect/roll/spring_rate".to_string()));
    assert!(registry.contains(&"/front_interconnect/heave/preload".to_string()));
    // example_with_interconnect() only configures front_interconnect.
    assert!(!registry.iter().any(|s| s.starts_with("/rear_interconnect")));
    // Every corner gets the hardpoints in this fixture, even ones without an active
    // interconnect -- the per-corner registry entry follows the hardpoint, not the axle.
    for i in 0..4 {
        assert!(registry.contains(&format!("/corners/{i}/rocker_heave_arm/0")));
    }
}
#[test]
fn interconnect_metric_registry_includes_axle_wheel_rate_and_corner_compression() {
    let registry = metric_registry();
    assert!(registry.contains(&"vehicle.front.heave_wheel_rate_n_per_m".to_string()));
    assert!(registry.contains(&"vehicle.rear.roll_wheel_rate_n_per_m".to_string()));
    // Picked up automatically from crate::Metrics's serialized field names.
    assert!(registry.contains(&"heave_arm_compression_m".to_string()));
    assert!(registry.contains(&"roll_arm_compression_m".to_string()));
}
#[test]
fn optimize_targets_interconnect_heave_wheel_rate() {
    let p = Project::example_with_interconnect();
    let known_rate = 55_000.0;
    let mut known_project = p.clone();
    known_project
        .front_interconnect
        .as_mut()
        .unwrap()
        .heave
        .spring_rate = known_rate;
    let known_metric = dw_core::analyze(&known_project, &dw_core::Motion::default())
        .unwrap()
        .front
        .heave_wheel_rate_n_per_m
        .value
        .expect("front heave interconnect configured");
    let r = OptimizationRequest {
        variables: vec![Variable {
            path: "/front_interconnect/heave/spring_rate".into(),
            kind: VariableKind::Continuous {
                lower: 10_000.,
                upper: 80_000.,
            },
        }],
        targets: vec![Target {
            corner: None,
            metric: "vehicle.front.heave_wheel_rate_n_per_m".into(),
            value: known_metric,
            values: None,
            scale: known_metric.max(1.0),
            weight: 1.,
            aggregation: Aggregation::MeanSquared,
        }],
        scenarios: vec![dw_core::Motion::default()],
        max_evaluations: 400,
        generations: 30,
        population_size: 10,
        ..Default::default()
    };
    let a = optimize(&p, &r).unwrap();
    let best = a
        .best_feasible
        .expect("optimizer should find a feasible candidate for a linear, reachable target");
    assert!(
        best.evaluation.score.unwrap() < 1e-4,
        "{:?}",
        best.evaluation.score
    );
}
