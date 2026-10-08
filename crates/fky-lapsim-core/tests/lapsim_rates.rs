mod common;
use common::within;
use fky_lapsim_core::lapsim::rates::{lbs_in_to_n_m, rates, ThesisSuspension};
use fky_lapsim_core::lapsim::thesis::p19;

#[test]
fn lbs_per_inch_conversion_eq_2_17() {
    within(lbs_in_to_n_m(350.0), 61294.4, 1e-4);
}

#[test]
fn p19_rates_match_thesis_table_5_1() {
    let r = rates(&p19(), &ThesisSuspension::p19()).unwrap();
    within(r.kw_n_m[0], 52550.0, 1e-3);
    within(r.kw_n_m[1], 23379.0, 1e-3);
    within(r.kheave_n_m[0], 33834.0, 1e-3);
    within(r.kheave_n_m[1], 18762.0, 1e-3);
    within(r.kw_arb_n_m[0], 31837.0, 1e-3);
    within(r.kw_arb_n_m[1], 47547.0, 1e-3);
    within(r.kroll_axle_nm_per_deg[0], 597.72, 1e-3);
    within(r.kroll_axle_nm_per_deg[1], 468.66, 1e-3);
    within(r.kroll_nm_per_deg, 1066.4, 1e-3);
    within(r.mechanical_balance_front_pct, 56.05, 1e-3);
    within(r.roll_gradient_deg_per_g, 0.6328, 1e-3);
}

#[test]
fn roll_axis_uses_the_thesis_weighting_eqs_2_18_and_2_19() {
    let r = rates(&p19(), &ThesisSuspension::p19()).unwrap();
    within(
        r.roll_axis_height_at_cg_m,
        0.51 * 0.0295 + 0.49 * 0.0690,
        1e-12,
    );
    within(
        r.roll_lever_arm_m,
        0.324 - r.roll_axis_height_at_cg_m,
        1e-12,
    );
}

#[test]
fn stiffer_arb_raises_roll_stiffness_but_not_heave() {
    let p = p19();
    let base = rates(&p, &ThesisSuspension::p19()).unwrap();
    let mut s = ThesisSuspension::p19();
    s.arb_rate_n_m[0] *= 2.0;
    let r = rates(&p, &s).unwrap();
    assert!(r.kroll_axle_nm_per_deg[0] > base.kroll_axle_nm_per_deg[0]);
    assert_eq!(r.kheave_n_m, base.kheave_n_m);
    assert!(r.mechanical_balance_front_pct > base.mechanical_balance_front_pct);
}

#[test]
fn rejects_nonphysical_suspensions() {
    let mut s = ThesisSuspension::p19();
    s.spring_mr[0] = 0.0;
    assert!(rates(&p19(), &s).is_err());
    let mut s = ThesisSuspension::p19();
    s.tyre_stiffness_n_m[1] = -1.0;
    assert!(rates(&p19(), &s).is_err());
}
