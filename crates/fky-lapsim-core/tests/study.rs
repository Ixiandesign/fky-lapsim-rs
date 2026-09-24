use dw_core::*;
#[test]
fn heave_changes_every_shock() {
    let p = Project::example();
    let a = simulate(&p, &Motion::default()).unwrap();
    let b = simulate(
        &p,
        &Motion {
            heave: 0.02,
            ..Motion::default()
        },
    )
    .unwrap();
    for i in 0..4 {
        assert!(
            (a.corners[i].metrics.shock_length_m - b.corners[i].metrics.shock_length_m).abs()
                > 1e-4
        );
    }
}
#[test]
fn rest_recovery_and_failed_samples() {
    let p = Project::example();
    let s = simulate(&p, &Motion::default()).unwrap();
    for (c, s) in p.corners.iter().zip(s.corners) {
        for k in 0..3 {
            assert!((c.wheel_center[k] - s.points.wheel_center[k]).abs() < 1e-8);
        }
        assert!(s.metrics.shock_compression_m.abs() < 1e-8);
    }
    let samples = sweep(
        &p,
        &[
            Motion::default(),
            Motion {
                heave: 1.0,
                ..Motion::default()
            },
            Motion {
                heave: 0.01,
                ..Motion::default()
            },
        ],
    );
    assert_eq!(samples.len(), 3);
    assert!(samples[0].state.is_some());
    assert!(samples[1].state.is_none());
    assert!(samples[1].error.is_some());
    assert!(samples[2].state.is_some());
}

#[test]
fn detailed_sweep_retains_derivatives_and_failed_rows() {
    let rows = dw_core::study::detailed_sweep(
        &Project::example(),
        &[
            Motion::default(),
            Motion {
                heave: 1.,
                ..Default::default()
            },
        ],
    );
    assert_eq!(rows.len(), 2);
    assert!(rows[0].analysis.as_ref().unwrap().corners[0]
        .caster_gain_deg_per_m
        .value
        .is_some());
    assert!(rows[0].sample.state.is_some());
    assert!(rows[1].analysis.is_none());
    assert!(rows[1].sample.state.is_none());
    assert!(rows[1].sample.error.is_some());
}
#[test]
fn finite_roll_pitch_contact_and_chassis_transform() {
    let p = Project::example();
    let m = Motion {
        heave: 0.012,
        roll: 0.035,
        pitch: 0.025,
        rack_front: 0.002,
        rack_rear: 0.0,
    };
    let s = simulate(&p, &m).unwrap();
    for (c, s) in p.corners.iter().zip(s.corners) {
        assert!(s.points.contact_point[2].abs() < 1e-8);
        assert!(s.max_residual_m <= 1e-8);
        let [x, y, z] = std::array::from_fn(|i| c.upper_front[i] - p.chassis.center_of_mass[i]);
        let yr = y * m.roll.cos() - z * m.roll.sin();
        let zr = y * m.roll.sin() + z * m.roll.cos();
        let expected = [
            x * m.pitch.cos() + zr * m.pitch.sin() + p.chassis.center_of_mass[0],
            yr + p.chassis.center_of_mass[1],
            -x * m.pitch.sin() + zr * m.pitch.cos() + p.chassis.center_of_mass[2] + m.heave,
        ];
        for (expected, actual) in expected.iter().zip(s.points.upper_front) {
            assert!((expected - actual).abs() < 1e-10);
        }
        for (expected, actual) in s
            .points
            .upper_ball
            .iter()
            .zip(s.transform_knuckle_point(c, c.upper_ball))
        {
            assert!((expected - actual).abs() < 1e-10);
        }
    }
}
#[test]
fn roll_mirror_and_road_relative_camber() {
    let p = Project::example();
    let a = simulate(
        &p,
        &Motion {
            roll: 0.03,
            ..Motion::default()
        },
    )
    .unwrap();
    let b = simulate(
        &p,
        &Motion {
            roll: -0.03,
            ..Motion::default()
        },
    )
    .unwrap();
    assert!(
        (a.corners[0].metrics.shock_length_m - b.corners[1].metrics.shock_length_m).abs() < 1e-8
    );
    assert!((a.corners[0].metrics.camber_deg + 0.03_f64.to_degrees()).abs() < 1e-5);
    assert!((a.corners[0].metrics.camber_deg - b.corners[1].metrics.camber_deg).abs() < 1e-5);
}
#[test]
fn equal_road_and_chassis_translation_preserves_mechanism() {
    let p = Project::example();
    let s = simulate_on_road(
        &p,
        &Motion {
            heave: 0.03,
            ..Motion::default()
        },
        [0.03; 4],
    )
    .unwrap();
    for (c, s) in p.corners.iter().zip(s.corners) {
        assert!((s.points.wheel_center[2] - c.wheel_center[2] - 0.03).abs() < 1e-8);
        assert!((s.points.contact_point[2] - 0.03).abs() < 1e-8);
        assert!(s.metrics.shock_compression_m.abs() < 1e-8);
        assert!(s.metrics.motion_ratio.unwrap() > 0.6);
    }
}
#[test]
fn nonfinite_and_work_limit_fail_explicitly() {
    let p = Project::example();
    assert!(simulate(
        &p,
        &Motion {
            roll: f64::INFINITY,
            ..Motion::default()
        }
    )
    .is_err());
    assert!(simulate(
        &p,
        &Motion {
            heave: 1e12,
            ..Motion::default()
        }
    )
    .is_err());
    assert!(simulate_on_road(&p, &Motion::default(), [f64::NAN; 4]).is_err());
}
