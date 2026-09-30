use fky_lapsim_core::tire::{TireInput, TireModel};

fn fixture() -> TireModel {
    // C=1,E=0: sin(atan(x)) = x/sqrt(1+x*x), an independent analytic oracle.
    serde_json::from_value(serde_json::json!({
        "reference_load_n":1000., "radius_m":0.25,
        "coefficients":{"PCX1":1.,"PDX1":1.,"PKX1":10.,
        "PCY1":1.,"PDY1":1.,"PKY1":20.,"PKY2":1.,"PKY4":2.,
        "RBX1":10.,"RCX1":1.,"RBY1":10.,"RCY1":1.,
        "QBZ1":10.,"QCZ1":1.,"QDZ1":0.2}
    }))
    .unwrap()
}
#[test]
fn pure_and_combined_forces_match_closed_form_oracles() {
    let t = fixture();
    let input = TireInput {
        normal_load_n: 1000.,
        slip_angle_rad: 0.1_f64.atan(),
        slip_ratio: 0.1,
        camber_rad: 0.,
        speed_m_s: 10.,
    };
    let s = t.evaluate(input).unwrap();
    assert!((s.pure_fx_n - 707.1067811865476).abs() < 1e-7);
    assert!((s.pure_fy_n - 894.4271909999159).abs() < 1e-7);
    assert!((s.fx_n - 500.).abs() < 1e-7);
    assert!((s.fy_n - 632.4555320336759).abs() < 1e-7);
    // QDZ1*R0 = 0.05 m. Equivalent combined slip is sqrt(.1²+.05²),
    // and cos(atan(10*equivalent)) = 1/sqrt(2.25).
    let expected_trail = 0.05 / 1.5 * 10. / (10. * 1.01_f64.sqrt() + 0.1);
    assert!((s.pneumatic_trail_m - expected_trail).abs() < 1e-12);
    assert!((s.mz_nm + expected_trail * 632.4555320336759).abs() < 1e-9);
    assert_eq!(s.longitudinal_stiffness_n, 10000.);
    assert_eq!(s.cornering_stiffness_n, 20000.);
    let pure_x = t
        .evaluate(TireInput {
            slip_angle_rad: 0.,
            ..input
        })
        .unwrap();
    assert!((pure_x.fx_n - pure_x.pure_fx_n).abs() < 1e-10);
    let pure_y = t
        .evaluate(TireInput {
            slip_ratio: 0.,
            ..input
        })
        .unwrap();
    assert!((pure_y.fy_n - pure_y.pure_fy_n).abs() < 1e-10);
}
#[test]
fn tire_is_odd_without_shifts_and_rejects_invalid_states_and_coefficients() {
    let t = fixture();
    let input = TireInput {
        normal_load_n: 1000.,
        slip_angle_rad: 0.04,
        slip_ratio: 0.02,
        camber_rad: 0.,
        speed_m_s: 10.,
    };
    let a = t.evaluate(input).unwrap();
    let b = t
        .evaluate(TireInput {
            slip_angle_rad: -0.04,
            slip_ratio: -0.02,
            ..input
        })
        .unwrap();
    assert!((a.fx_n + b.fx_n).abs() < 1e-10);
    assert!((a.fy_n + b.fy_n).abs() < 1e-10);
    assert!((a.mz_nm + b.mz_nm).abs() < 1e-10);
    let zero = t
        .evaluate(TireInput {
            normal_load_n: 0.,
            ..input
        })
        .unwrap();
    assert_eq!((zero.fx_n, zero.fy_n, zero.mz_nm), (0., 0., 0.));
    assert!(t
        .evaluate(TireInput {
            normal_load_n: -1.,
            ..input
        })
        .is_err());
    assert!(t
        .evaluate(TireInput {
            slip_ratio: f64::NAN,
            ..input
        })
        .is_err());
    let mut bad = t.clone();
    bad.coefficients.insert("a0".into(), 1.);
    assert!(bad.validate().is_err());
    bad = t.clone();
    bad.coefficients.remove("PKY4");
    assert!(bad.validate().is_err());
}
#[test]
fn load_sensitivity_and_camber_stiffness_follow_the_book_convention() {
    let mut t = fixture();
    t.coefficients.insert("PDY2".into(), -0.2);
    t.coefficients.insert("PKY6".into(), 2.);
    let input = TireInput {
        normal_load_n: 1000.,
        slip_angle_rad: 0.,
        slip_ratio: 0.,
        camber_rad: 1e-6,
        speed_m_s: 10.,
    };
    let a = t.evaluate(input).unwrap();
    assert!((a.fy_n / 1e-6 - 2000.).abs() < 0.01);
    let b = t
        .evaluate(TireInput {
            normal_load_n: 1500.,
            camber_rad: 0.,
            ..input
        })
        .unwrap();
    assert!((b.lateral_friction - 0.9).abs() < 1e-12);
}
