mod common;
use common::within;
use fky_lapsim_core::lapsim::thesis::{p19, BikeParams};

#[test]
fn p19_geometry_and_aero_split_match_thesis_equations() {
    let p = p19();
    p.validate().unwrap();
    // Eq. 2-1, 2-2: a = (1 - WD/100) * WB, b = WB - a
    within(p.a_dist_m(), 0.51 * 1.53, 1e-12);
    within(p.b_dist_m(), 1.53 - 0.51 * 1.53, 1e-12);
    // Eq. 2-7, 2-8: CzF = CzT * AB/100, CzR = CzT * (1 - AB/100)
    within(p.cz_front(), 4.5 * 0.45, 1e-12);
    within(p.cz_rear(), 4.5 * 0.55, 1e-12);
}

#[test]
fn p19_uses_the_load_sensitivity_that_reproduces_the_thesis() {
    // Thesis Table 2-3 prints "10E-4"; only 1e-4 per N reproduces Tables 3-13/3-14 (see Task 4).
    let p = p19();
    assert_eq!(p.front.muy_sens_per_n, 1e-4);
    assert_eq!(p.rear.mux_sens_per_n, 1e-4);
}

#[test]
fn validate_rejects_nonphysical_vehicles() {
    let mut p: BikeParams = p19();
    p.mass_kg = 0.0;
    assert!(p.validate().is_err());
    let mut p = p19();
    p.wd_front_pct = 120.0;
    assert!(p.validate().is_err());
    let mut p = p19();
    p.front.muy = f64::NAN;
    assert!(p.validate().is_err());
    let mut p = p19();
    p.aero.aero_balance_front_pct = -1.0;
    assert!(p.validate().is_err());
}
