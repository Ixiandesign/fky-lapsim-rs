use dw_core::*;
fn configured(profile: &str) -> Corner {
    let mut c = serde_json::to_value(&Project::example().corners[0]).unwrap();
    c["tire_profile"] = profile.into();
    c["tire_radius"] = 0.25.into();
    c["spindle_axis"] = serde_json::json!([[1.3, 0.9, 0.3], [1.3, 1.7660254037844386, 0.8]]);
    serde_json::from_value(c).unwrap()
}
#[test]
fn analytic_tire_supports() {
    for (profile, height) in [
        ("disk", 0.21650635094610965),
        ("cylinder", 0.26650635094610965),
        ("torus", 0.2299038105676658),
    ] {
        let s = solve_corner(&configured(profile), 0.0, 0.0).unwrap();
        assert!(
            (s.points.wheel_center[2] - s.points.contact_point[2] - height).abs() < 1e-10,
            "{profile}"
        );
    }
}
#[test]
fn rack_axis_moves_inner_point() {
    let mut value = serde_json::to_value(&Project::example().corners[0]).unwrap();
    value["rack_axis"] = serde_json::json!([[0, 0, 0], [1, 0, 0]]);
    let c: Corner = serde_json::from_value(value).unwrap();
    let s = solve_corner(&c, 0.0, 0.002).unwrap();
    assert!((s.points.steering_inner[0] - c.steering_inner[0] - 0.002).abs() < 1e-12);
}
#[test]
fn finite_steering_scrub_uses_wheel_axes() {
    let c = Project::example().corners[0].clone();
    let mut p = solve_corner(&c, 0.0, 0.0).unwrap().points;
    let a = 0.7_f64;
    let rotate = |p: Point| {
        [
            a.cos() * p[0] - a.sin() * p[1],
            a.sin() * p[0] + a.cos() * p[1],
            p[2],
        ]
    };
    p.upper_ball = rotate([0.2, 0.3, 0.6]);
    p.lower_ball = rotate([0.2, 0.3, 0.3]);
    p.contact_point = rotate([0.1, 0.5, 0.0]);
    p.spindle_axis = [rotate([0.0, 0.0, 0.0]), rotate([0.0, 1.0, 0.0])];
    // Public alignment helper permits independent rigid orientation oracles.
    let m = dw_core::metrics::measure(&c, &p, [0.0; 3]);
    assert!((m.scrub_radius_m.unwrap() - 0.2).abs() < 1e-12);
    assert!((m.mechanical_trail_m.unwrap() - 0.1).abs() < 1e-12);
}
#[test]
fn preload_geometry_stiffness_oracle() {
    assert_eq!(
        dw_core::analysis::wheel_rate(1000.0, 100.0, 0.5, 2.0),
        450.0
    );
}
#[test]
fn planar_instant_center_oracle() {
    let ic = dw_core::analysis::projected_center([0.8, 0.6], [0.8, 0.3], [0.2, 0.8], [0.5, 0.8]);
    let p = ic.point_yz_m.expect("finite center");
    assert!((p[0] - 0.0).abs() < 1e-12 && (p[1] - 0.8).abs() < 1e-12);
}
#[test]
fn planar_vehicle_analysis_and_parallel_limit() {
    let p = Project::example();
    let a = analyze(&p, &Motion::default()).unwrap();
    assert!((a.front.wheel_track_m - 1.8).abs() < 1e-12);
    assert!((a.left_wheelbase_m - 2.6).abs() < 1e-12);
    assert!(a.front.geometric_roll_center_yz_m.value.unwrap()[1].abs() < 1e-9);
    assert!(a.corners[0].projected_front_view_ic.direction_yz.is_some());
    let mut p = p;
    for c in &mut p.corners {
        c.upper_front[2] = 0.7;
        c.upper_rear[2] = 0.7;
        c.lower_front[2] = 0.55;
        c.lower_rear[2] = 0.55;
    }
    let a = analyze(&p, &Motion::default()).unwrap();
    let ic = a.corners[0].projected_front_view_ic.point_yz_m.unwrap();
    assert!(ic[0].abs() < 1e-6 && (ic[1] - 0.8).abs() < 1e-6, "{ic:?}");
    let rc = a.front.geometric_roll_center_yz_m.value.unwrap();
    assert!(rc[0].abs() < 1e-6 && (rc[1] - 0.8).abs() < 1e-6, "{rc:?}");
}
#[test]
fn profiles_close_world_road_and_validate() {
    for profile in [TireProfile::Disk, TireProfile::Cylinder, TireProfile::Torus] {
        let mut p = Project::example();
        for c in &mut p.corners {
            c.tire_profile = profile;
        }
        let state = simulate(
            &p,
            &Motion {
                roll: 0.03,
                pitch: 0.01,
                ..Motion::default()
            },
        )
        .unwrap();
        for s in state.corners {
            assert!(s.points.contact_point[2].abs() < 1e-8);
        }
    }
    let mut p = Project::example();
    p.corners[0].rack_axis = [[0.0; 3]; 2];
    assert!(p.validate().is_err());
    p = Project::example();
    p.corners[0].tire_profile = TireProfile::Torus;
    p.corners[0].tire_width = 0.6;
    assert!(p.validate().is_err());
}
#[test]
fn rack_axis_rotates_with_chassis() {
    let mut p = Project::example();
    for c in &mut p.corners {
        c.rack_axis = [[0.0; 3], [1.0, 2.0, 0.5]];
    }
    let m = Motion {
        rack_front: 0.002,
        roll: 0.03,
        pitch: 0.02,
        ..Motion::default()
    };
    let state = simulate(&p, &m).unwrap();
    let q = nalgebra::UnitQuaternion::from_axis_angle(&nalgebra::Vector3::y_axis(), m.pitch)
        * nalgebra::UnitQuaternion::from_axis_angle(&nalgebra::Vector3::x_axis(), m.roll);
    let v = nalgebra::Vector3::from;
    let com = v(p.chassis.center_of_mass);
    let expected =
        com + q * (v(p.corners[0].steering_inner) - com + v([1.0, 2.0, 0.5]).normalize() * 0.002);
    assert!((v(state.corners[0].points.steering_inner) - expected).norm() < 1e-12);
}
#[test]
fn skew_derivatives_refine_and_rest_lengths_are_exact() {
    let mut p = Project::example();
    for c in &mut p.corners {
        c.upper_front[1] += 0.035;
        c.upper_rear[2] += 0.023;
        c.lower_front[2] -= 0.018;
        c.lower_rear[1] -= 0.021;
        c.spring_damper.preload = 250.0;
    }
    let m = Motion {
        heave: 0.015,
        rack_front: 0.002,
        roll: 0.01,
        ..Motion::default()
    };
    let a = analyze_with_step(&p, &m, 0.0004).unwrap();
    let b = analyze_with_step(&p, &m, 0.0002).unwrap();
    for (a, b) in a.corners.iter().zip(&b.corners) {
        for (x, y, tol) in [
            (a.motion_ratio.value, b.motion_ratio.value, 1e-5),
            (
                a.motion_ratio_gradient_per_m.value,
                b.motion_ratio_gradient_per_m.value,
                0.002,
            ),
            (
                a.camber_gain_deg_per_m.value,
                b.camber_gain_deg_per_m.value,
                0.002,
            ),
            (
                a.toe_gain_deg_per_m.value,
                b.toe_gain_deg_per_m.value,
                0.002,
            ),
        ] {
            assert!((x.unwrap() - y.unwrap()).abs() < tol, "{x:?} {y:?}");
        }
        assert!(b.projected_front_view_ic.point_yz_m.is_some());
    }
    let c = &p.corners[0];
    let s = &b.state.corners[0];
    let distance = |a: Point, b: Point| {
        nalgebra::Vector3::from(a).metric_distance(&nalgebra::Vector3::from(b))
    };
    assert_eq!(
        s.metrics.rest_upper_front_leg_m,
        distance(c.upper_front, c.upper_ball)
    );
    assert_eq!(
        s.metrics.rest_pushrod_m,
        distance(c.pushrod_pickup, c.rocker_pushrod)
    );
    assert!(
        (s.metrics.rest_shock_length_m - s.metrics.shock_length_m - s.metrics.shock_compression_m)
            .abs()
            < 1e-12
    );
}
#[test]
fn vertical_torus_contact_is_on_ring_not_in_hole() {
    let mut c = Project::example().corners[0].clone();
    c.tire_profile = TireProfile::Torus;
    c.spindle_axis = [[1.3, 0.9, 0.2], [1.3, 0.9, 0.4]];
    let s = solve_corner(&c, 0.0, 0.0).unwrap();
    let d = nalgebra::Vector3::from(s.points.contact_point)
        - nalgebra::Vector3::from(s.points.wheel_center);
    assert!((d.x.hypot(d.y) - 0.2).abs() < 1e-12);
    assert!((d.z + 0.1).abs() < 1e-12);
    assert!(s.contact_ambiguity.is_some());
}
#[test]
fn undefined_projected_geometry_and_spring_knots_keep_reasons() {
    use dw_core::analysis::{geometric_roll_center, projected_center};
    let zero = projected_center([0.8, 0.6], [0.8, 0.3], [0.0, 0.0], [0.0, 1.0]);
    assert!(zero.point_yz_m.is_none() && zero.reason.is_some());
    let coincident = projected_center([0.8, 0.6], [0.4, 0.6], [0.0, 1.0], [0.0, 1.0]);
    assert!(coincident.direction_yz.is_none() && coincident.reason.is_some());
    assert!(geometric_roll_center([0.9, 0.0], &zero, [-0.9, 0.0], &zero)
        .reason
        .is_some());
    let mut p = Project::example();
    for c in &mut p.corners {
        c.spring_damper.spring_curve = Some(vec![[-0.1, -100.0], [0.0, 0.0], [0.1, 200.0]]);
    }
    let a = analyze(&p, &Motion::default()).unwrap();
    assert!(a.corners[0].spring_wheel_rate_n_per_m.value.is_none());
    assert!(a.corners[0]
        .spring_wheel_rate_n_per_m
        .reason
        .as_ref()
        .unwrap()
        .contains("knot"));
    assert_eq!(
        a.state.corners[0].metrics.motion_ratio,
        a.corners[0].motion_ratio.value
    );
}
#[test]
fn old_corner_schema_defaults_and_cylinder_contact_line() {
    let mut value = serde_json::to_value(Project::example().corners[0].clone()).unwrap();
    value.as_object_mut().unwrap().remove("tire_profile");
    value.as_object_mut().unwrap().remove("rack_axis");
    let mut c: Corner = serde_json::from_value(value).unwrap();
    assert_eq!(c.tire_profile, TireProfile::Disk);
    assert_eq!(c.rack_axis, [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    c.tire_profile = TireProfile::Cylinder;
    let s = solve_corner(&c, 0.0, 0.0).unwrap();
    assert!(s.contact_ambiguity.as_ref().unwrap().contains("midpoint"));
    assert_eq!(s.points.contact_point[1], s.points.wheel_center[1]);
}
#[test]
fn skew_first_derivatives_match_independent_corner_samples() {
    let mut p = Project::example();
    let c = &mut p.corners[0];
    c.upper_front[1] += 0.035;
    c.upper_rear[2] += 0.023;
    c.lower_front[2] -= 0.018;
    c.lower_rear[1] -= 0.021;
    let a = analyze(&p, &Motion::default()).unwrap();
    let c = &p.corners[0];
    let h = 0.0007;
    let lo = solve_corner(c, -h, 0.0).unwrap();
    let hi = solve_corner(c, h, 0.0).unwrap();
    let ratio = (hi.metrics.shock_compression_m - lo.metrics.shock_compression_m) / (2.0 * h);
    let camber = (hi.metrics.camber_deg - lo.metrics.camber_deg) / (2.0 * h);
    let toe = (hi.metrics.toe_deg - lo.metrics.toe_deg) / (2.0 * h);
    assert!((ratio - a.corners[0].motion_ratio.value.unwrap()).abs() < 2e-5);
    assert!((camber - a.corners[0].camber_gain_deg_per_m.value.unwrap()).abs() < 0.002);
    assert!((toe - a.corners[0].toe_gain_deg_per_m.value.unwrap()).abs() < 0.002);
}
#[test]
fn interconnect_wheel_rate_absent_without_configuration_present_when_configured() {
    let p = Project::example();
    let a = analyze(&p, &Motion::default()).unwrap();
    assert!(a.front.heave_wheel_rate_n_per_m.value.is_none());
    assert!(a.front.roll_wheel_rate_n_per_m.value.is_none());

    let p = Project::example_with_interconnect();
    let a = analyze(&p, &Motion::default()).unwrap();
    let heave = a
        .front
        .heave_wheel_rate_n_per_m
        .value
        .expect("front heave interconnect configured");
    let roll = a
        .front
        .roll_wheel_rate_n_per_m
        .value
        .expect("front roll interconnect configured");
    assert!(heave > 0.0, "{heave}");
    assert!(roll > 0.0, "{roll}");
    // example_with_interconnect() only configures front_interconnect.
    assert!(a.rear.heave_wheel_rate_n_per_m.value.is_none());
    assert!(a.rear.roll_wheel_rate_n_per_m.value.is_none());
}
#[test]
fn interconnect_wheel_rate_matches_independent_corner_samples() {
    let p = Project::example_with_interconnect();
    let a = analyze(&p, &Motion::default()).unwrap();
    let fl = &p.corners[0];
    let fr = &p.corners[1];
    assert_eq!(fl.id, CornerId::FrontLeft);
    assert_eq!(fr.id, CornerId::FrontRight);
    let h = 0.0007;
    let fl_hi = solve_corner(fl, h, 0.0).unwrap();
    let fl_lo = solve_corner(fl, -h, 0.0).unwrap();
    let fr_hi = solve_corner(fr, h, 0.0).unwrap();
    let fr_lo = solve_corner(fr, -h, 0.0).unwrap();
    // Symmetric (heave) perturbation: both corners jounce together.
    let heave_hi = (fl_hi.metrics.heave_arm_compression_m.unwrap()
        + fr_hi.metrics.heave_arm_compression_m.unwrap())
        / 2.0;
    let heave_lo = (fl_lo.metrics.heave_arm_compression_m.unwrap()
        + fr_lo.metrics.heave_arm_compression_m.unwrap())
        / 2.0;
    let heave_ratio = (heave_hi - heave_lo) / (2.0 * h);
    // Antisymmetric (roll) perturbation: left up, right down.
    let roll_hi = (fl_hi.metrics.roll_arm_compression_m.unwrap()
        - fr_lo.metrics.roll_arm_compression_m.unwrap())
        / 2.0;
    let roll_lo = (fl_lo.metrics.roll_arm_compression_m.unwrap()
        - fr_hi.metrics.roll_arm_compression_m.unwrap())
        / 2.0;
    let roll_ratio = (roll_hi - roll_lo) / (2.0 * h);
    // Both interconnect springs have zero preload, and the design pose is exactly zero
    // compression, so the preload-dependent geometric-stiffness term in wheel_rate
    // (F_s * gradient) vanishes exactly here: wheel_rate reduces exactly to k*ratio^2,
    // giving a tight independent oracle rather than a leading-order approximation.
    let expected_heave = 40_000.0 * heave_ratio * heave_ratio;
    let expected_roll = 20_000.0 * roll_ratio * roll_ratio;
    let heave_rate = a.front.heave_wheel_rate_n_per_m.value.unwrap();
    let roll_rate = a.front.roll_wheel_rate_n_per_m.value.unwrap();
    assert!(
        (heave_rate - expected_heave).abs() < 1.0,
        "{heave_rate} {expected_heave}"
    );
    assert!(
        (roll_rate - expected_roll).abs() < 1.0,
        "{roll_rate} {expected_roll}"
    );
}
