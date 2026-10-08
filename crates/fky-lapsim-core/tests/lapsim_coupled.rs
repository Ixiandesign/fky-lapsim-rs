mod common;
use common::within;
use fky_lapsim_core::lapsim::coupled::{CoupleSettings, Coupled, TyreMode, TyreTable};
use fky_lapsim_core::lapsim::forces::{StepModel, ThesisConst};
use fky_lapsim_core::lapsim::fullcar::{thesis_params, FullCar};
use fky_lapsim_core::lapsim::rates::ThesisSuspension;
use fky_lapsim_core::lapsim::thesis::p19;
use fky_lapsim_core::lapsim::G;

fn coupled() -> Coupled {
    Coupled {
        params: p19(),
        car: FullCar::from_rates(thesis_params(&p19(), &ThesisSuspension::p19()).unwrap()).unwrap(),
        aero_map: None,
        tyres: TyreMode::Thesis,
        settings: CoupleSettings::default(),
    }
}

#[test]
fn without_lateral_transfer_it_reduces_to_the_thesis_bike_model() {
    let c = coupled();
    let t = ThesisConst {
        params: p19(),
        weight_transfer: true,
    };
    for (v, ax) in [(0.0, 0.0), (10.0, 0.0), (25.0, 6.0), (20.0, -9.0)] {
        let a = c.instant(v, ax, 0.0).unwrap().forces;
        let b = t.instant(v, ax, 0.0).unwrap().forces;
        within(a.tot_f_n, b.tot_f_n, 1e-6);
        within(a.tot_r_n, b.tot_r_n, 1e-6);
        within(a.fy_total_n(), b.fy_total_n(), 1e-6);
        within(a.tyres_acc_n(), b.tyres_acc_n(), 1e-6);
        within(a.tyres_dec_n(), b.tyres_dec_n(), 1e-6);
    }
}

#[test]
fn lateral_transfer_reduces_grip_through_per_wheel_load_sensitivity() {
    let c = coupled();
    let t = ThesisConst {
        params: p19(),
        weight_transfer: true,
    };
    let (v, ay) = (15.0, 1.4 * G);
    let inst = c.instant(v, 0.0, ay).unwrap();
    let thesis = t.instant(v, 0.0, ay).unwrap().forces;
    assert!(inst.forces.fy_total_n() < thesis.fy_total_n());
    // closed form: front lateral force = sum over the two front wheels of muy(Fz_i) * Fz_i
    let fz = inst.attitude.wheel_fz_n;
    let expected: f64 = (0..2)
        .map(|i| (1.45 + 1e-4 * (50.0 * G - fz[i])) * fz[i])
        .sum();
    within(inst.forces.fy_f_n, expected, 1e-9);
    // outside (right) wheels carry more in a left turn
    assert!(fz[1] > fz[0] && fz[3] > fz[2]);
    assert!(inst.attitude.roll_deg > 0.0);
}

#[test]
fn bike_model_and_matrix_agree_on_axle_loads_after_convergence() {
    let c = coupled();
    for (v, ax, ay) in [(12.0, 0.0, 8.0), (20.0, -10.0, 6.0), (18.0, 5.0, 10.0)] {
        let i = c.instant(v, ax, ay).unwrap();
        let t = ThesisConst {
            params: p19(),
            weight_transfer: true,
        };
        let b = t.instant(v, ax, ay).unwrap().forces;
        within(i.forces.tot_f_n, b.tot_f_n, 5e-3);
        within(i.forces.tot_r_n, b.tot_r_n, 5e-3);
        assert!(i.attitude.iterations >= 1 && i.attitude.iterations < 100);
    }
}

#[test]
fn the_fixed_point_does_not_depend_on_the_initial_guess_or_relaxation() {
    let base = coupled().instant(15.0, -4.0, 9.0).unwrap();
    let mut other = coupled();
    other.settings.initial_wheel_fz_n = Some([100.0, 2000.0, 300.0, 1500.0]);
    let b = other.instant(15.0, -4.0, 9.0).unwrap();
    for i in 0..4 {
        within(b.attitude.wheel_fz_n[i], base.attitude.wheel_fz_n[i], 1e-6);
    }
    let mut slow = coupled();
    slow.settings.relax = 0.3;
    let s = slow.instant(15.0, -4.0, 9.0).unwrap();
    for i in 0..4 {
        within(s.attitude.wheel_fz_n[i], base.attitude.wheel_fz_n[i], 1e-6);
    }
    assert!(s.attitude.iterations >= base.attitude.iterations);
}

#[test]
fn non_convergence_is_an_error_not_a_stale_result() {
    let mut c = coupled();
    c.settings.max_iter = 1;
    c.settings.load_tol_n = 1e-12;
    assert!(c.instant(20.0, -4.0, 9.0).is_err());
}

#[test]
fn wheel_lift_off_is_an_error() {
    let c = coupled();
    // far beyond any grip: the inside wheels unload completely
    assert!(c.instant(5.0, 0.0, 6.0 * G).is_err());
}

#[test]
fn table_mode_uses_wheel_camber_and_never_extrapolates() {
    let table = TyreTable {
        load_n: vec![100.0, 3000.0],
        camber_deg: vec![-4.0, 4.0],
        mux: vec![vec![1.2, 1.2], vec![1.2, 1.2]],
        muy: vec![vec![1.4, 1.4], vec![1.4, 1.4]],
    };
    within(table.mu(1500.0, 0.0).unwrap().1, 1.4, 1e-12);
    assert!(table.mu(50.0, 0.0).is_err());
    assert!(table.mu(1500.0, 9.0).is_err());
    let mut c = coupled();
    c.tyres = TyreMode::Table([table.clone(), table]);
    let i = c.instant(15.0, 0.0, 8.0).unwrap();
    // flat 1.4 friction regardless of load: lateral force = 1.4 * total vertical load
    within(i.forces.fy_total_n(), 1.4 * i.forces.tot_n(), 1e-9);
}
