use dw_core::*;
#[test]
fn analytic_parallel_arm_arc() {
    let c = &Project::example().corners[0];
    let s = solve_corner(c, 0.03, 0.0).unwrap();
    assert!((s.points.wheel_center[2] - 0.33).abs() < 1e-8);
    assert!((s.points.wheel_center[1] - (0.4 + (0.16_f64 - 0.0009).sqrt() + 0.1)).abs() < 1e-8);
    assert!(s.metrics.camber_deg.abs() < 1e-6);
}
fn distance(a: Point, b: Point) -> f64 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}
fn check_links(c: &Corner, s: &CornerState) {
    let p = &s.points;
    for (a, b, x, y) in [
        (c.upper_front, c.upper_ball, p.upper_front, p.upper_ball),
        (c.upper_rear, c.upper_ball, p.upper_rear, p.upper_ball),
        (c.lower_front, c.lower_ball, p.lower_front, p.lower_ball),
        (c.lower_rear, c.lower_ball, p.lower_rear, p.lower_ball),
        (
            c.steering_inner,
            c.steering_outer,
            p.steering_inner,
            p.steering_outer,
        ),
        (
            c.pushrod_pickup,
            c.rocker_pushrod,
            p.pushrod_pickup,
            p.rocker_pushrod,
        ),
        (c.upper_ball, c.lower_ball, p.upper_ball, p.lower_ball),
    ] {
        assert!((distance(a, b) - distance(x, y)).abs() < 1e-8);
    }
    assert!(s.max_residual_m <= 1e-8);
}
#[test]
fn every_pickup_follows_its_owner() {
    for owner in [
        PushrodBody::UpperArm,
        PushrodBody::LowerArm,
        PushrodBody::Knuckle,
    ] {
        let mut c = Project::example().corners[0].clone();
        c.pushrod_body = owner;
        if owner == PushrodBody::UpperArm {
            c.pushrod_pickup[2] = 0.6;
        }
        let s = solve_corner(&c, 0.03, 0.0).unwrap();
        check_links(&c, &s);
        let dz = s.points.pushrod_pickup[2] - c.pushrod_pickup[2];
        assert!(
            (dz - if owner == PushrodBody::Knuckle {
                0.03
            } else {
                0.0225
            })
            .abs()
                < 1e-8
        );
        let spindle = [
            s.points.spindle_axis[1][0] - s.points.spindle_axis[0][0],
            s.points.spindle_axis[1][1] - s.points.spindle_axis[0][1],
            s.points.spindle_axis[1][2] - s.points.spindle_axis[0][2],
        ];
        assert!(distance(spindle, [0.0, 0.2, 0.0]) < 1e-8);
    }
}
#[test]
fn unreachable_and_nonfinite_requests_fail() {
    let c = &Project::example().corners[0];
    assert!(solve_corner(c, 0.5, 0.0).is_err());
    assert!(solve_corner(c, f64::NAN, 0.0).is_err());
}
#[test]
fn steering_changes_toe_and_mirrors() {
    let p = Project::example();
    let l = solve_corner(&p.corners[0], 0.02, 0.005).unwrap();
    let r = solve_corner(&p.corners[1], 0.02, -0.005).unwrap();
    assert!(l.metrics.toe_deg.abs() > 0.1);
    assert!((l.metrics.toe_deg - r.metrics.toe_deg).abs() < 1e-5);
    assert!((l.metrics.camber_deg - r.metrics.camber_deg).abs() < 1e-5);
    check_links(&p.corners[0], &l);
    check_links(&p.corners[1], &r);
}
#[test]
fn skewed_upper_axis_closes_and_branch_is_continuous() {
    let mut c = Project::example().corners[0].clone();
    c.upper_front[2] += 0.04;
    c.upper_rear[1] -= 0.02;
    let mut prior = solve_corner(&c, 0.0, 0.0).unwrap();
    for i in 1..11 {
        let s = solve_corner(&c, i as f64 * 0.003, 0.0).unwrap();
        check_links(&c, &s);
        assert!((s.metrics.rocker_angle_rad - prior.metrics.rocker_angle_rad).abs() < 0.1);
        assert!(distance(s.points.wheel_center, prior.points.wheel_center) < 0.01);
        prior = s;
    }
}
#[test]
fn reversed_spindle_endpoints_preserve_alignment() {
    let mut c = Project::example().corners[0].clone();
    let a = solve_corner(&c, 0.02, 0.005).unwrap();
    c.spindle_axis.swap(0, 1);
    let b = solve_corner(&c, 0.02, 0.005).unwrap();
    assert!((a.metrics.toe_deg - b.metrics.toe_deg).abs() < 1e-9);
    assert!((a.metrics.camber_deg - b.metrics.camber_deg).abs() < 1e-9);
}
#[test]
fn analytic_rest_motion_ratio() {
    let c = &Project::example().corners[0];
    let s = solve_corner(c, 0.0, 0.0).unwrap();
    // Pushrod differentiation gives rocker speed 5 rad/m. Shock end dy/dj=-0.75,
    // projected on the rest shock (dy,dz)=(0.25,0.15).
    assert!((s.metrics.motion_ratio.unwrap() - 0.1875 / 0.085_f64.sqrt()).abs() < 1e-5);
}
#[test]
fn unconstrained_rocker_returns_explicit_error() {
    let mut c = Project::example().corners[0].clone();
    c.pushrod_body = PushrodBody::Knuckle;
    c.pushrod_pickup = [1.3, 0.4, 0.8];
    c.validate().unwrap();
    let error = solve_corner(&c, 0.0, 0.0).unwrap_err();
    assert!(error.message.contains("underdetermined rocker"));
}
#[test]
fn signed_design_alignment() {
    let mut c = Project::example().corners[0].clone();
    c.upper_ball[0] -= 0.03;
    c.upper_ball[1] -= 0.03;
    c.spindle_axis[1][0] += 0.02;
    c.spindle_axis[1][2] -= 0.02;
    let s = solve_corner(&c, 0.0, 0.0).unwrap();
    assert!((s.metrics.caster_deg - 0.1_f64.atan().to_degrees()).abs() < 1e-8);
    assert!((s.metrics.kpi_deg - 0.1_f64.atan().to_degrees()).abs() < 1e-8);
    assert!((s.metrics.toe_deg - 0.1_f64.atan().to_degrees()).abs() < 1e-8);
    assert!(s.metrics.camber_deg > 0.0);
}
