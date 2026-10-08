use fky_lapsim_core::lap::LapVehicle;
use fky_lapsim_core::CornerId;

#[test]
fn the_synthetic_demo_validates_and_round_trips_through_json() {
    let car = LapVehicle::synthetic_demo().unwrap();
    car.validate().unwrap();
    let back: LapVehicle = serde_json::from_str(&serde_json::to_string(&car).unwrap()).unwrap();
    back.validate().unwrap();
    assert_eq!(back.wheels.len(), 4);
}

#[test]
fn validation_rejects_a_tire_radius_that_disagrees_with_the_suspension_envelope() {
    let mut car = LapVehicle::synthetic_demo().unwrap();
    car.wheels[0].tire.model.radius_m += 0.01;
    assert!(car.validate().is_err());
}

#[test]
fn validation_requires_exactly_one_wheel_per_corner() {
    let mut car = LapVehicle::synthetic_demo().unwrap();
    car.wheels[1].id = CornerId::FrontLeft; // duplicate; front right is now missing
    assert!(car.validate().is_err());
}

#[test]
fn validation_rejects_bad_scalars() {
    let mut car = LapVehicle::synthetic_demo().unwrap();
    car.steering_limit_m = 0.0;
    assert!(car.validate().is_err());
    let mut car = LapVehicle::synthetic_demo().unwrap();
    car.wheels[2].spin_inertia_kg_m2 = -1.0;
    assert!(car.validate().is_err());
}

#[test]
fn corner_order_in_the_suspension_does_not_matter() {
    let mut car = LapVehicle::synthetic_demo().unwrap();
    car.suspension.corners.swap(0, 3);
    car.wheels.swap(1, 2);
    car.validate().unwrap();
}
