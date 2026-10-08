mod common;
use common::within;
use fky_lapsim_core::lapsim::forces::{forces, max_ay, StepModel, ThesisConst};
use fky_lapsim_core::lapsim::thesis::p19;
use fky_lapsim_core::lapsim::G;

#[test]
fn static_vertical_loads_follow_eqs_2_26_to_2_28() {
    let p = p19();
    let f = forces(&p, 0.0, 0.0, p.const_aero_coeffs()).unwrap();
    within(f.mass_f_n, 250.0 * G * 0.49, 1e-12);
    within(f.mass_r_n, 250.0 * G * 0.51, 1e-12);
    within(f.tot_n(), 250.0 * G, 1e-12);
    assert_eq!(f.df_f_n, 0.0);
}

#[test]
fn weight_transfer_moves_load_rearward_when_accelerating_eq_2_49() {
    let p = p19();
    let f0 = forces(&p, 0.0, 0.0, p.const_aero_coeffs()).unwrap();
    let f = forces(&p, 0.0, 10.0, p.const_aero_coeffs()).unwrap();
    let wt = 250.0 * 10.0 * 0.330 / 1.53;
    within(f.mass_r_n - f0.mass_r_n, wt, 1e-12);
    within(f0.mass_f_n - f.mass_f_n, wt, 1e-12);
}

#[test]
fn downforce_and_drag_follow_eqs_2_29_to_2_35() {
    let p = p19();
    let v = 20.0;
    let f = forces(&p, v, 0.0, p.const_aero_coeffs()).unwrap();
    within(f.df_f_n, 0.5 * 1.225 * (4.5 * 0.45) * 1.0 * v * v, 1e-12);
    within(f.df_r_n, 0.5 * 1.225 * (4.5 * 0.55) * 1.0 * v * v, 1e-12);
    within(f.drag_n, 0.5 * 1.225 * 1.75 * 1.0 * v * v, 1e-12);
    // Eq. 2-36: rolling resistance = Cr * total vertical load (including downforce)
    within(f.rolling_n, 0.03 * f.tot_n(), 1e-12);
    assert!(f.resist_n() > 0.0);
}

#[test]
fn friction_uses_load_sensitivity_eqs_2_12_to_2_15() {
    let p = p19();
    let f = forces(&p, 0.0, 0.0, p.const_aero_coeffs()).unwrap();
    let expected_muy_f = 1.45 + 1e-4 * (50.0 * G - f.tot_f_n / 2.0);
    within(f.muy_f, expected_muy_f, 1e-12);
    let expected_mux_r = 1.2 + 1e-4 * (50.0 * G - f.tot_r_n / 2.0);
    within(f.mux_r, expected_mux_r, 1e-12);
}

#[test]
fn rwd_acceleration_force_uses_only_the_rear_axle_eq_2_37() {
    let p = p19();
    let f = forces(&p, 10.0, 0.0, p.const_aero_coeffs()).unwrap();
    assert_eq!(f.tyres_acc_f_n, 0.0);
    within(f.tyres_acc_r_n, f.mux_r * f.tot_r_n, 1e-12);
    // Eqs. 2-41, 2-42: braking uses all four tyres
    within(f.tyres_dec_f_n, f.mux_f * f.tot_f_n, 1e-12);
    within(f.tyres_dec_r_n, f.mux_r * f.tot_r_n, 1e-12);
}

#[test]
fn lateral_force_is_mu_times_axle_load_eqs_2_46_to_2_48() {
    let p = p19();
    let f = forces(&p, 15.0, 0.0, p.const_aero_coeffs()).unwrap();
    within(f.fy_f_n, f.muy_f * f.tot_f_n, 1e-12);
    within(f.fy_total_n(), f.fy_f_n + f.fy_r_n, 1e-12);
}

#[test]
fn correlation_factors_scale_forces() {
    let mut p = p19();
    p.correlation.muy = 0.9;
    p.correlation.mux_accel = 1.08;
    p.correlation.mux_brake = 0.9;
    p.correlation.aero = 0.5;
    let base = p19();
    let f = forces(&p, 20.0, 0.0, p.const_aero_coeffs()).unwrap();
    let g0 = forces(&base, 20.0, 0.0, base.const_aero_coeffs()).unwrap();
    within(f.df_f_n, 0.5 * g0.df_f_n, 1e-12);
    within(f.drag_n, 0.5 * g0.drag_n, 1e-12);
    // mu is load-dependent, so compare against the formula with the scaled downforce
    within(f.fy_f_n, 0.9 * f.muy_f * f.tot_f_n, 1e-12);
    within(f.tyres_acc_r_n, 1.08 * f.mux_r * f.tot_r_n, 1e-12);
    within(f.tyres_dec_f_n, 0.9 * f.mux_f * f.tot_f_n, 1e-12);
}

#[test]
fn lift_off_is_an_error_not_a_clamp() {
    let p = p19();
    // enormous forward acceleration unloads the front axle
    assert!(forces(&p, 0.0, 200.0, p.const_aero_coeffs()).is_err());
}

#[test]
fn thesis_const_model_ignores_weight_transfer_when_disabled() {
    let off = ThesisConst {
        params: p19(),
        weight_transfer: false,
    };
    let on = ThesisConst {
        params: p19(),
        weight_transfer: true,
    };
    let a = off.instant(10.0, 8.0, 0.0).unwrap().forces;
    let b = on.instant(10.0, 8.0, 0.0).unwrap().forces;
    within(a.mass_r_n, 250.0 * G * 0.51, 1e-12);
    assert!(b.mass_r_n > a.mass_r_n);
}

#[test]
fn max_ay_is_a_fixed_point_of_lateral_force() {
    let m = ThesisConst {
        params: p19(),
        weight_transfer: false,
    };
    let v = 12.0;
    let ay = max_ay(&m, v, 0.0).unwrap();
    let f = m.instant(v, 0.0, ay).unwrap().forces;
    within(ay, f.fy_total_n() / 250.0, 1e-9);
}
