mod common;
use common::{p19_powertrain, within};
use fky_lapsim_core::lapsim::tractive::TractiveTable;
use fky_lapsim_core::powertrain::DynoPoint;

const R: f64 = 0.199;

#[test]
fn top_speed_is_redline_in_top_gear() {
    let pt = p19_powertrain();
    let t = TractiveTable::build(&pt, R, 1.0, 0.05).unwrap();
    let expected = 12000.0 * 2.0 * std::f64::consts::PI / 60.0 * R / (1.49 * 3.6);
    within(t.top_speed_m_s(), expected, 0.01);
}

#[test]
fn force_matches_eq_2_23_and_2_25_in_a_known_gear() {
    let pt = p19_powertrain();
    let t = TractiveTable::build(&pt, R, 1.0, 0.05).unwrap();
    let v = 25.0;
    let g = t.gear_at(v);
    let ratio = pt.gear_ratios[g] * pt.final_drive;
    let rpm = v / R * ratio * 60.0 / (2.0 * std::f64::consts::PI);
    assert!((t.rpm_at(v) - rpm).abs() < 5.0);
    let force = pt.torque_at_rpm(rpm).unwrap() * ratio * pt.efficiency / R;
    within(t.force_at(v), force, 0.01);
}

#[test]
fn engine_scalar_scales_force_linearly() {
    let pt = p19_powertrain();
    let a = TractiveTable::build(&pt, R, 1.0, 0.05).unwrap();
    let b = TractiveTable::build(&pt, R, 0.8, 0.05).unwrap();
    within(b.force_at(20.0), 0.8 * a.force_at(20.0), 1e-9);
}

#[test]
fn force_is_held_constant_below_the_first_reachable_speed() {
    let pt = p19_powertrain();
    let t = TractiveTable::build(&pt, R, 1.0, 0.05).unwrap();
    assert!((t.force_at(0.0) - t.force_at(0.5)).abs() < 1e-9);
    assert!(t.force_at(0.0) > 0.0);
    assert_eq!(t.gear_at(0.0), 0);
}

#[test]
fn gear_never_decreases_with_speed_after_the_shift_filter() {
    let mut pt = p19_powertrain();
    // a dyno dip that would cause an undesired downshift (thesis Fig. 2-18)
    pt.torque_curve = vec![
        DynoPoint {
            rpm: 2000.0,
            torque_nm: 30.0,
        },
        DynoPoint {
            rpm: 5000.0,
            torque_nm: 44.0,
        },
        DynoPoint {
            rpm: 6000.0,
            torque_nm: 20.0,
        },
        DynoPoint {
            rpm: 7000.0,
            torque_nm: 44.0,
        },
        DynoPoint {
            rpm: 12000.0,
            torque_nm: 33.0,
        },
    ];
    let t = TractiveTable::build(&pt, R, 1.0, 0.05).unwrap();
    for w in t.gear.windows(2) {
        assert!(w[1] >= w[0], "gear decreased: {} -> {}", w[0], w[1]);
    }
}

#[test]
fn above_top_speed_force_is_zero_and_bad_input_is_rejected() {
    let pt = p19_powertrain();
    let t = TractiveTable::build(&pt, R, 1.0, 0.05).unwrap();
    assert_eq!(t.force_at(t.top_speed_m_s() + 1.0), 0.0);
    assert!(TractiveTable::build(&pt, -1.0, 1.0, 0.05).is_err());
    assert!(TractiveTable::build(&pt, R, 1.0, 0.0).is_err());
}
