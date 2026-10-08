mod common;
use common::within;
use fky_lapsim_core::lapsim::aeromap::{AeroMap, Sensitivity, ThesisAeroMap};
use fky_lapsim_core::lapsim::forces::{StepModel, ThesisConst};
use fky_lapsim_core::lapsim::rates::ThesisSuspension;
use fky_lapsim_core::lapsim::thesis::p19;

fn zero_sens(max: f64) -> Sensitivity {
    Sensitivity {
        angle_deg: vec![0.0, max],
        czt_pct: vec![0.0, 0.0],
        ab_pct: vec![0.0, 0.0],
        cx_pct: vec![0.0, 0.0],
    }
}

fn flat_map(czt: f64, ab: f64, cx: f64) -> AeroMap {
    let grid = vec![5.0, 25.0, 45.0];
    let fill = |x: f64| vec![vec![x; 3]; 3];
    AeroMap {
        front_rh_mm: grid.clone(),
        rear_rh_mm: grid,
        czt: fill(czt),
        ab_front_pct: fill(ab),
        cx: fill(cx),
        roll: zero_sens(1.5),
        yaw: zero_sens(20.0),
    }
}

#[test]
fn bilinear_interpolation_is_exact_on_a_planar_field_and_flags_clamping() {
    let mut m = flat_map(0.0, 45.0, 1.75);
    for i in 0..3 {
        for j in 0..3 {
            m.czt[i][j] = 2.0 + 0.05 * m.front_rh_mm[i] + 0.03 * m.rear_rh_mm[j];
        }
    }
    let p = m.eval(15.0, 35.0, 0.0, 0.0).unwrap();
    within(p.czt, 2.0 + 0.05 * 15.0 + 0.03 * 35.0, 1e-12);
    assert!(!p.clamped);
    let edge = m.eval(100.0, 35.0, 0.0, 0.0).unwrap();
    assert!(edge.clamped);
    within(edge.czt, 2.0 + 0.05 * 45.0 + 0.03 * 35.0, 1e-12);
}

#[test]
fn roll_and_yaw_sensitivities_reproduce_thesis_tables_6_3_to_6_5() {
    let mut m = flat_map(4.61, 45.0, 1.75);
    m.roll = Sensitivity {
        angle_deg: vec![0.0, 0.8, 1.5],
        czt_pct: vec![0.0, -11.43, -13.0],
        ab_pct: vec![0.0, 0.0, 0.0],
        cx_pct: vec![0.0, -11.43, -13.0],
    };
    m.yaw = Sensitivity {
        angle_deg: vec![0.0, 6.5, 20.0],
        czt_pct: vec![0.0, -12.52, -30.0],
        ab_pct: vec![0.0, 0.0, 0.0],
        cx_pct: vec![0.0, -12.52, -30.0],
    };
    within(m.eval(17.0, 20.0, 0.0, 0.0).unwrap().czt, 4.61, 1e-12);
    within(m.eval(17.0, 20.0, 0.8, 0.0).unwrap().czt, 4.08, 5e-3);
    within(m.eval(17.0, 20.0, 0.8, 6.5).unwrap().czt, 3.57, 5e-3);
    // angles are symmetric (left/right)
    within(m.eval(17.0, 20.0, -0.8, -6.5).unwrap().czt, 3.57, 5e-3);
}

#[test]
fn validate_rejects_bad_maps() {
    let mut m = flat_map(4.5, 45.0, 1.75);
    m.front_rh_mm = vec![5.0, 5.0, 45.0];
    assert!(m.validate().is_err());
    let mut m = flat_map(4.5, 45.0, 1.75);
    m.czt.pop();
    assert!(m.validate().is_err());
    let mut m = flat_map(4.5, 120.0, 1.75);
    assert!(m.validate().is_err());
    m = flat_map(4.5, 45.0, 1.75);
    m.czt[1][1] = f64::NAN;
    assert!(m.validate().is_err());
}

fn model(map: AeroMap, wt: bool) -> ThesisAeroMap {
    ThesisAeroMap {
        params: p19(),
        suspension: ThesisSuspension::p19(),
        map,
        weight_transfer: wt,
    }
}

#[test]
fn a_flat_map_equals_the_constant_aero_model() {
    let m = model(flat_map(4.5, 45.0, 1.75), true);
    let c = ThesisConst {
        params: p19(),
        weight_transfer: true,
    };
    for (v, ax, ay) in [(5.0, 0.0, 0.0), (20.0, 5.0, 0.0), (15.0, -8.0, 12.0)] {
        let a = m.instant(v, ax, ay).unwrap().forces;
        let b = c.instant(v, ax, ay).unwrap().forces;
        within(a.tot_f_n, b.tot_f_n, 1e-9);
        within(a.tot_r_n, b.tot_r_n, 1e-9);
        within(a.drag_n, b.drag_n, 1e-9);
    }
}

#[test]
fn ride_height_reproduces_the_thesis_skidpad_state_table_7_14() {
    // CzT 4.22, front aero balance 58.6 %, 12.15 m/s: thesis reports FRH 31.7 mm, RRH 40.8 mm
    let m = model(flat_map(4.22, 58.6, 1.75), false);
    let a = m.instant(12.15, 0.0, 0.0).unwrap().attitude;
    within(a.front_rh_mm, 31.7, 5e-3);
    within(a.rear_rh_mm, 40.8, 5e-3);
}

#[test]
fn ride_height_follows_downforce_and_weight_transfer() {
    let m = model(flat_map(4.5, 45.0, 1.75), true);
    let slow = m.instant(5.0, 0.0, 0.0).unwrap().attitude;
    let fast = m.instant(30.0, 0.0, 0.0).unwrap().attitude;
    assert!(fast.front_rh_mm < slow.front_rh_mm && fast.rear_rh_mm < slow.rear_rh_mm);
    // static ride height at standstill
    let rest = m.instant(0.0, 0.0, 0.0).unwrap().attitude;
    within(rest.front_rh_mm, 35.0, 1e-9);
    within(rest.rear_rh_mm, 45.0, 1e-9);
    // braking dips the nose, accelerating squats the tail
    let braking = m.instant(20.0, -12.0, 0.0).unwrap().attitude;
    let cruise = m.instant(20.0, 0.0, 0.0).unwrap().attitude;
    assert!(braking.front_rh_mm < cruise.front_rh_mm);
    assert!(braking.rear_rh_mm > cruise.rear_rh_mm);
    let accel = m.instant(20.0, 8.0, 0.0).unwrap().attitude;
    assert!(accel.rear_rh_mm < cruise.rear_rh_mm);
}

#[test]
fn full_anti_features_remove_the_weight_transfer_ride_height_change() {
    let mut m = model(flat_map(4.5, 45.0, 1.75), true);
    m.suspension.anti_dive_front_pct = 100.0;
    m.suspension.anti_lift_rear_pct = 100.0;
    let braking = m.instant(20.0, -12.0, 0.0).unwrap().attitude;
    let cruise = m.instant(20.0, 0.0, 0.0).unwrap().attitude;
    within(braking.front_rh_mm, cruise.front_rh_mm, 1e-9);
    within(braking.rear_rh_mm, cruise.rear_rh_mm, 1e-9);
}

#[test]
fn roll_and_yaw_reduce_downforce_when_cornering() {
    let mut map = flat_map(4.5, 45.0, 1.75);
    map.roll = Sensitivity {
        angle_deg: vec![0.0, 1.5],
        czt_pct: vec![0.0, -10.0],
        ab_pct: vec![0.0, 0.0],
        cx_pct: vec![0.0, 0.0],
    };
    map.yaw = Sensitivity {
        angle_deg: vec![0.0, 20.0],
        czt_pct: vec![0.0, -20.0],
        ab_pct: vec![0.0, 0.0],
        cx_pct: vec![0.0, 0.0],
    };
    let m = model(map, false);
    let straight = m.instant(15.0, 0.0, 0.0).unwrap().attitude;
    let corner = m.instant(15.0, 0.0, 15.0).unwrap().attitude;
    assert!(corner.cz_total < straight.cz_total);
    assert!(corner.roll_deg > 0.0); // left turn rolls positive (left side up)
    assert!(corner.iterations > 0 && corner.iterations < 100);
}

#[test]
fn correlation_aero_scales_the_map_output() {
    let mut m = model(flat_map(4.5, 45.0, 1.75), false);
    let full = m.instant(20.0, 0.0, 0.0).unwrap().forces;
    m.params.correlation.aero = 0.5;
    let half = m.instant(20.0, 0.0, 0.0).unwrap().forces;
    within(half.df_f_n, 0.5 * full.df_f_n, 1e-9);
    within(half.drag_n, 0.5 * full.drag_n, 1e-9);
}
