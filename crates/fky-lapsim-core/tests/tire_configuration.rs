use dw_core::tire_configuration::{ConfiguredTire, TireDomain, Provenance, tire_input};
use dw_core::CornerId;

#[test]
fn frame_conversion_opposes_lateral_sliding_on_both_sides() {
    let t = ConfiguredTire::synthetic_demo(0.25);
    for id in [CornerId::FrontLeft, CornerId::FrontRight] {
        let input = tire_input(id, 700., [10., 1.], 40., 0.25, 2.).unwrap();
        assert!((input.slip_angle_rad - 0.1_f64.atan()).abs() < 1e-14);
        assert_eq!(input.slip_ratio, 0.);
        let expected_gamma = if id == CornerId::FrontLeft { -2_f64 } else { 2_f64 }.to_radians();
        assert!((input.camber_rad - expected_gamma).abs() < 1e-14);
        let force = t.evaluate(input).unwrap();
        assert!(force.fy_n > 0.); // Book Fy -> body-left Fy = -book Fy.
        assert!(-force.fy_n * 1. < 0.); // Tire dissipates lateral sliding power.
    }
}

#[test]
fn declared_domain_and_provenance_are_enforced_not_silently_clamped() {
    let mut t = ConfiguredTire::synthetic_demo(0.25);
    assert!(matches!(t.provenance, Provenance::Synthetic { .. }));
    t.domain = TireDomain { normal_load_n: [100., 1000.], max_abs_slip_angle_rad: 0.2,
        max_abs_slip_ratio: 0.3, max_abs_camber_rad: 0.15, speed_m_s: [1., 40.] };
    let input = tire_input(CornerId::FrontLeft, 700., [10., 0.], 40., 0.25, 0.).unwrap();
    assert!(t.evaluate(input).is_ok());
    for bad in [dw_core::tire::TireInput { normal_load_n: 1001., ..input },
        dw_core::tire::TireInput { speed_m_s: 0., ..input },
        dw_core::tire::TireInput { slip_ratio: 0.31, ..input }] {
        assert!(t.evaluate(bad).is_err());
    }
    t.provenance = Provenance::Measured { source: "".into(), fit_notes: "".into() };
    assert!(t.validate().is_err());
    assert!(tire_input(CornerId::FrontLeft, 700., [0., 0.], 0., 0.25, 0.).is_err());
}
