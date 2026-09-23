use dw_core::{Project, PushrodBody};

#[test]
fn example_validates_and_tires_rest_on_ground() {
    let p = Project::example();
    p.validate().unwrap();
    for c in &p.corners {
        assert_eq!(c.wheel_center[2], c.tire_radius);
    }
}
#[test]
fn collapsed_axis_is_rejected() {
    let mut p = Project::example();
    p.corners[0].upper_front = p.corners[0].upper_rear;
    assert!(p.validate().is_err());
}
#[test]
fn collinear_wishbone_is_rejected() {
    let mut p = Project::example();
    p.corners[0].lower_ball = [1.3, 0.4, 0.3];
    assert!(p.validate().is_err());
}
#[test]
fn nonfinite_coordinates_are_rejected() {
    let mut p = Project::example();
    p.corners[0].shock_chassis[0] = f64::NAN;
    assert!(p.validate().is_err());
}
#[test]
fn invalid_radius_is_rejected() {
    let mut p = Project::example();
    p.corners[0].tire_radius = -0.1;
    assert!(p.validate().is_err());
}
#[test]
fn duplicate_corner_ids_are_rejected() {
    let mut p = Project::example();
    p.corners[0].id = p.corners[1].id;
    assert!(p.validate().is_err());
}
#[test]
fn invalid_mass_and_force_parameters_are_rejected() {
    let mut p = Project::example();
    p.chassis.sprung_mass = 0.0;
    assert!(p.validate().is_err());
    let mut p = Project::example();
    p.corners[0].spring_damper.rebound_damping = -1.0;
    assert!(p.validate().is_err());
}
#[test]
fn example_with_interconnect_validates() {
    Project::example_with_interconnect().validate().unwrap();
}
#[test]
fn interconnect_hardpoints_require_pairing() {
    let mut p = Project::example_with_interconnect();
    p.corners[0].heave_arm_anchor = None;
    assert!(p.validate().is_err());
    let mut p = Project::example_with_interconnect();
    p.corners[0].rocker_roll_arm = None;
    assert!(p.validate().is_err());
}
#[test]
fn interconnect_requires_both_corners_fully_configured() {
    let mut p = Project::example_with_interconnect();
    p.corners[1].rocker_heave_arm = None;
    p.corners[1].heave_arm_anchor = None;
    assert!(p.validate().is_err());
}
#[test]
fn corner_may_carry_interconnect_hardpoints_without_an_active_interconnect() {
    // example_with_interconnect() gives every corner the hardpoints but only configures
    // front_interconnect; rear_interconnect stays None and the project still validates.
    let p = Project::example_with_interconnect();
    assert!(p.corners[2].rocker_heave_arm.is_some());
    assert!(p.rear_interconnect.is_none());
    p.validate().unwrap();
}
#[test]
fn interconnect_spring_length_limits_are_rejected() {
    let mut p = Project::example_with_interconnect();
    p.front_interconnect.as_mut().unwrap().heave.min_length_m = Some(0.01);
    assert!(p.validate().is_err());
    let mut p = Project::example_with_interconnect();
    p.front_interconnect.as_mut().unwrap().roll.max_length_m = Some(0.5);
    assert!(p.validate().is_err());
}
#[test]
fn interconnect_spring_rate_and_damping_validated() {
    let mut p = Project::example_with_interconnect();
    p.front_interconnect.as_mut().unwrap().heave.spring_rate = -1.0;
    assert!(p.validate().is_err());
    let mut p = Project::example_with_interconnect();
    p.front_interconnect.as_mut().unwrap().roll.rebound_damping = -1.0;
    assert!(p.validate().is_err());
}
#[test]
fn degenerate_interconnect_lever_is_rejected() {
    let mut p = Project::example_with_interconnect();
    let axis = p.corners[0].rocker_axis;
    // Place the heave arm tip on the rocker axis line: degenerate lever.
    p.corners[0].rocker_heave_arm = Some(axis[0]);
    assert!(p.validate().is_err());
}
#[test]
fn nonfinite_interconnect_hardpoint_is_rejected() {
    let mut p = Project::example_with_interconnect();
    p.corners[0].roll_arm_anchor = Some([f64::NAN, 0.0, 0.0]);
    assert!(p.validate().is_err());
}
#[test]
fn interconnect_round_trips_through_json() {
    let p = Project::example_with_interconnect();
    let json = serde_json::to_string(&p).unwrap();
    let back: Project = serde_json::from_str(&json).unwrap();
    assert_eq!(back, p);
    back.validate().unwrap();
}
#[test]
fn ownership_round_trips_and_defaults_are_backward_compatible() {
    for owner in [
        PushrodBody::UpperArm,
        PushrodBody::LowerArm,
        PushrodBody::Knuckle,
    ] {
        let mut p = Project::example();
        p.corners[0].pushrod_body = owner;
        let json = serde_json::to_string(&p).unwrap();
        assert_eq!(serde_json::from_str::<Project>(&json).unwrap(), p);
    }
    let mut v = serde_json::to_value(Project::example()).unwrap();
    v.as_object_mut().unwrap().remove("chassis");
    for c in v["corners"].as_array_mut().unwrap() {
        c.as_object_mut().unwrap().remove("spring_damper");
    }
    serde_json::from_value::<Project>(v)
        .unwrap()
        .validate()
        .unwrap();
}
