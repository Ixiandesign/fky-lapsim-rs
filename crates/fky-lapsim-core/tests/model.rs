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
