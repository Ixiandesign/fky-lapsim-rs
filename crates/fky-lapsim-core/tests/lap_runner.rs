use dw_core::lap::{LapVehicle,LapState,LapControls,advance_vehicle,LapRequest,simulate_lap_controlled};

#[test]
fn four_wheel_rk4_coasts_at_constant_speed_and_cancels_without_fake_lap_time() {
    let mut car=LapVehicle::synthetic_demo().unwrap();car.aero.reference_area_m2=0.;
    let s=LapState::rolling(&car,10.).unwrap();
    let mut a=s.clone();
    for _ in 0..10 {a=advance_vehicle(&car,&a,LapControls::default(),0.005).unwrap();}
    assert!((a.planar.x_m-0.5).abs()<1e-7);
    assert!((a.planar.u_m_s-10.).abs()<1e-7);
    assert!(a.planar.y_m.abs()<1e-7);
    let track=dw_core::track::Track::circle(30.,6.,128).unwrap();
    let result=simulate_lap_controlled(&car,&track,&LapRequest::default(),&||true).unwrap();
    assert!(!result.completed);assert!(result.lap_time_s.is_none());
    assert_eq!(result.termination,"cancelled");
}

#[test]
fn numerical_work_and_driver_domain_are_bounded() {
    let mut r=LapRequest::default();r.dt_s=0.;assert!(r.validate().is_err());
    r.dt_s=0.000001;assert!(r.validate().is_err());
    r=LapRequest::default();r.lookahead_m=-1.;assert!(r.validate().is_err());
}
