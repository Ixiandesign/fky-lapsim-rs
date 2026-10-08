#![allow(dead_code)]
use fky_lapsim_core::powertrain::{DrivenAxle, DynoPoint, Powertrain};

/// Assert `actual` is within relative error `rel` of `expected` (absolute `rel` if expected is 0).
pub fn within(actual: f64, expected: f64, rel: f64) {
    let scale = if expected == 0.0 { 1.0 } else { expected.abs() };
    assert!(
        (actual - expected).abs() <= rel * scale,
        "actual {actual} not within {rel} (rel) of expected {expected}"
    );
}

/// P19-like powertrain: thesis Tables 2-5/2-6 ratios and efficiency; the dyno curve is invented
/// (the thesis's engine file is not published), so tests must not assert absolute acceleration times.
pub fn p19_powertrain() -> Powertrain {
    Powertrain {
        torque_curve: vec![
            DynoPoint {
                rpm: 2000.0,
                torque_nm: 30.0,
            },
            DynoPoint {
                rpm: 4500.0,
                torque_nm: 38.0,
            },
            DynoPoint {
                rpm: 7000.0,
                torque_nm: 44.0,
            },
            DynoPoint {
                rpm: 9500.0,
                torque_nm: 42.0,
            },
            DynoPoint {
                rpm: 12000.0,
                torque_nm: 33.0,
            },
        ],
        idle_rpm: 2000.0,
        redline_rpm: 12000.0,
        gear_ratios: vec![4.51, 3.23, 2.49, 2.03, 1.71, 1.49],
        final_drive: 3.6,
        efficiency: 0.9,
        driven_axle: DrivenAxle::Rear,
    }
}

/// A circle of `radius` metres as `n` equal (radius, length) rows (left turn).
pub fn circle_rows(radius: f64, n: usize) -> Vec<(f64, f64)> {
    let len = 2.0 * std::f64::consts::PI * radius / n as f64;
    vec![(radius, len); n]
}

/// Stadium: straight, left semicircle, straight, left semicircle, sampled every `step_m`.
pub fn stadium_rows(straight_m: f64, radius_m: f64, step_m: f64) -> Vec<(f64, f64)> {
    let ns = (straight_m / step_m).round() as usize;
    let arc = std::f64::consts::PI * radius_m;
    let na = (arc / step_m).round().max(3.0) as usize;
    let mut rows = vec![];
    for _ in 0..2 {
        rows.extend(std::iter::repeat((1.0e5, straight_m / ns as f64)).take(ns));
        rows.extend(std::iter::repeat((radius_m, arc / na as f64)).take(na));
    }
    rows
}
