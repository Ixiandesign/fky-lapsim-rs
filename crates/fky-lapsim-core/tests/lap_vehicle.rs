use dw_core::lap::{LapVehicle,LapState,LapControls,evaluate_vehicle};

#[test]
fn coast_and_lateral_slip_preserve_force_balance_and_frame_signs() {
    let mut car=LapVehicle::synthetic_demo().unwrap();
    car.aero.reference_area_m2=0.;
    let s=LapState::rolling(&car,10.).unwrap();
    let c=LapControls {gear:0,..Default::default()};
    let at_rest=evaluate_vehicle(&car,&s,c).unwrap();
    assert!(at_rest.derivative.u_m_s2.abs()<1e-7);
    assert!(at_rest.derivative.v_m_s2.abs()<1e-7);
    assert!(at_rest.derivative.displacement_acceleration[0].abs()<1e-4);
    assert!((at_rest.wheels.iter().map(|w|w.normal_load_n).sum::<f64>()-300.*9.81).abs()<0.02);
    let mut sliding=s; sliding.planar.v_m_s=0.5;
    let e=evaluate_vehicle(&car,&sliding,c).unwrap();
    assert!(e.derivative.v_m_s2<0.);
    let sum=e.wheels.iter().map(|w|w.normal_load_n).sum::<f64>();
    assert!((sum-300.*(9.81+e.derivative.displacement_acceleration[0])).abs()<0.02);
    for w in &e.wheels {assert!(w.fy_n<0.);}
    car.suspension.corners.swap(0,3);car.wheels.swap(1,2);
    let permuted=evaluate_vehicle(&car,&sliding,c).unwrap();
    assert!((permuted.derivative.v_m_s2-e.derivative.v_m_s2).abs()<1e-10);
}

#[test]
fn driven_wheel_spin_obeys_torque_balance_without_invented_traction() {
    let mut car=LapVehicle::synthetic_demo().unwrap();car.aero.reference_area_m2=0.;
    let s=LapState::rolling(&car,10.).unwrap();
    let c=LapControls {throttle:0.5,gear:0,..Default::default()};
    let e=evaluate_vehicle(&car,&s,c).unwrap();
    assert!(e.derivative.u_m_s2.abs()<1e-7); // At zero slip, torque initially spins wheels.
    assert!(e.derivative.wheel_acceleration_rad_s2[2]>0.);
    for i in 0..4 {
        let w=&e.wheels[i];let j=car.wheels[i].spin_inertia_kg_m2;
        assert!((j*e.derivative.wheel_acceleration_rad_s2[i]-(w.drive_torque_nm-w.brake_torque_nm-w.fx_n*w.rolling_radius_m)).abs()<1e-9);
    }
}
