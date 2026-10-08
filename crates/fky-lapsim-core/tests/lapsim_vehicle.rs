mod common;
use common::within;
use fky_lapsim_core::lap::LapVehicle;
use fky_lapsim_core::lapsim::apex::top_speed;
use fky_lapsim_core::lapsim::scenarios::corner;
use fky_lapsim_core::lapsim::vehicle::{build_tyre_table, derive_axle_tyre, QssExtras, QssVehicle};
use fky_lapsim_core::tire_configuration::ConfiguredTire;
use fky_lapsim_core::CornerId;

#[test]
fn derived_friction_matches_the_magic_formula_peak() {
    let tire = ConfiguredTire::synthetic_demo(0.2);
    let d = derive_axle_tyre(&tire, 800.0).unwrap();
    // the synthetic tyre has PDY1 = 1.4, PDX1 = 1.4 at its reference load
    assert!(
        d.axle_tyre.muy > 1.0 && d.axle_tyre.muy < 2.0,
        "muy {}",
        d.axle_tyre.muy
    );
    assert!(d.axle_tyre.mux > 1.0 && d.axle_tyre.mux < 2.0);
    assert!(d.axle_tyre.cornering_stiffness_n_per_deg > 0.0);
    assert!(d.source.starts_with("synthetic: "), "{}", d.source);
    // the fitted line reproduces the tyre at the fit loads
    let expected_static = d.axle_tyre.muy;
    within(
        d.axle_tyre.muy + d.axle_tyre.muy_sens_per_n * (d.axle_tyre.muy_norm_kg * 9.81 - 800.0),
        expected_static,
        1e-9,
    );
}

#[test]
fn derivation_rejects_loads_outside_the_tyre_domain() {
    let tire = ConfiguredTire::synthetic_demo(0.2);
    assert!(derive_axle_tyre(&tire, 100_000.0).is_err());
    assert!(derive_axle_tyre(&tire, -5.0).is_err());
}

#[test]
fn tyre_table_covers_load_and_camber_and_never_extrapolates() {
    let tire = ConfiguredTire::synthetic_demo(0.2);
    let t = build_tyre_table(&tire, 800.0, CornerId::FrontLeft).unwrap();
    let (mx, my) = t.mu(800.0, 0.0).unwrap();
    assert!(mx > 0.5 && my > 0.5);
    assert!(t.mu(1.0e6, 0.0).is_err());
    assert!(t.load_n.windows(2).all(|w| w[1] > w[0]));
}

#[test]
fn the_demo_vehicle_builds_and_is_physically_sensible() {
    let car = LapVehicle::synthetic_demo().unwrap();
    let v = QssVehicle::build(&car, &QssExtras::synthetic_demo()).unwrap();
    let b = &v.bike;
    b.validate().unwrap();
    within(b.mass_kg, car.suspension.chassis.sprung_mass, 1e-12);
    assert!(
        b.wd_front_pct > 35.0 && b.wd_front_pct < 65.0,
        "WD {}",
        b.wd_front_pct
    );
    assert!(b.wheelbase_m > 1.0 && b.wheelbase_m < 3.5);
    assert!(b.cg_height_m > 0.1 && b.cg_height_m < 0.8);
    assert_eq!(v.derived_tyres.len(), 2);
    assert!(v.tractive.top_speed_m_s() > 20.0);
    // static wheel loads add up to the vehicle weight
    let sum: f64 = v.full_car.params.static_wheel_load_n.iter().sum();
    within(sum, b.mass_kg * 9.81, 1e-3);
}

#[test]
fn thesis_exact_and_coupled_models_both_run_a_corner_on_the_demo_car() {
    let car = LapVehicle::synthetic_demo().unwrap();
    let v = QssVehicle::build(&car, &QssExtras::synthetic_demo()).unwrap();
    let thesis = v.step_model(false, true).unwrap();
    let coupled = v.step_model(true, true).unwrap();
    let top = top_speed(thesis.as_ref(), &v.tractive).unwrap();
    let a = corner(thesis.as_ref(), top, 20.0).unwrap();
    let b = corner(coupled.as_ref(), top, 20.0).unwrap();
    // lateral transfer + camber cost some grip but the two agree to within ~15 %
    assert!(b.speed_m_s <= a.speed_m_s * 1.02);
    assert!(
        b.speed_m_s > 0.85 * a.speed_m_s,
        "{} vs {}",
        b.speed_m_s,
        a.speed_m_s
    );
}

#[test]
fn an_aero_map_requires_the_matrix() {
    use fky_lapsim_core::lapsim::aeromap::{AeroMap, Sensitivity};
    let car = LapVehicle::synthetic_demo().unwrap();
    let mut extras = QssExtras::synthetic_demo();
    let grid = vec![5.0, 25.0, 45.0];
    let fill = |x: f64| vec![vec![x; 3]; 3];
    let zero = Sensitivity {
        angle_deg: vec![0.0, 1.0],
        czt_pct: vec![0.0; 2],
        ab_pct: vec![0.0; 2],
        cx_pct: vec![0.0; 2],
    };
    extras.aero_map = Some(AeroMap {
        front_rh_mm: grid.clone(),
        rear_rh_mm: grid,
        czt: fill(2.0),
        ab_front_pct: fill(45.0),
        cx: fill(0.8),
        roll: zero.clone(),
        yaw: zero,
    });
    let v = QssVehicle::build(&car, &extras).unwrap();
    assert!(v.step_model(false, true).is_err());
    assert!(v.step_model(true, true).is_ok());
}

#[test]
fn invalid_extras_are_rejected() {
    let car = LapVehicle::synthetic_demo().unwrap();
    let mut e = QssExtras::synthetic_demo();
    e.unsprung[0].tyre_vertical_stiffness_n_m = 0.0;
    assert!(QssVehicle::build(&car, &e).is_err());
    let mut e = QssExtras::synthetic_demo();
    e.steer_ratio = 0.0;
    assert!(QssVehicle::build(&car, &e).is_err());
}
