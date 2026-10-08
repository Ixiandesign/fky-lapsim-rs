mod common;
use common::{p19_powertrain, within};
use fky_lapsim_core::lapsim::apex::{
    find_apexes, smooth_clusters, top_speed, vmax_point, vmax_profile,
};
use fky_lapsim_core::lapsim::forces::{max_ay, StepModel, ThesisConst};
use fky_lapsim_core::lapsim::scenarios::corner;
use fky_lapsim_core::lapsim::thesis::p19;
use fky_lapsim_core::lapsim::track_model::TrackModel;
use fky_lapsim_core::lapsim::tractive::TractiveTable;

fn setup() -> (ThesisConst, TractiveTable) {
    (
        ThesisConst {
            params: p19(),
            weight_transfer: false,
        },
        TractiveTable::build(&p19_powertrain(), 0.199, 1.0, 0.05).unwrap(),
    )
}

#[test]
fn apex_speed_is_self_consistent_with_the_friction_ellipse() {
    let (m, tr) = setup();
    let r = 30.0;
    let v = vmax_point(&m, &tr, r).unwrap();
    let f = m.instant(v, 0.0, v * v / r).unwrap().forces;
    let ax_needed = f.resist_n() / 250.0;
    let ax_tyre = f.tyres_acc_n() / 250.0;
    let ay_remain = max_ay(&m, v, 0.0).unwrap() * (1.0 - (ax_needed / ax_tyre).powi(2)).sqrt();
    within(v * v / r, ay_remain, 1e-6);
}

#[test]
fn drag_correction_lowers_the_apex_speed_below_the_isolated_corner_speed() {
    let (m, tr) = setup();
    let isolated = corner(&m, 100.0, 30.0).unwrap().speed_m_s;
    let corrected = vmax_point(&m, &tr, 30.0).unwrap();
    assert!(corrected < isolated);
    assert!(corrected > 0.9 * isolated);
}

#[test]
fn straights_are_capped_at_the_vehicle_top_speed() {
    let (m, tr) = setup();
    let top = top_speed(&m, &tr).unwrap();
    assert!(top > 20.0 && top <= tr.top_speed_m_s());
    assert_eq!(vmax_point(&m, &tr, 1.0e5).unwrap(), top);
    assert_eq!(vmax_point(&m, &tr, -1.0e5).unwrap(), top);
    assert!(vmax_point(&m, &tr, 5000.0).unwrap() <= top + 1e-9);
}

#[test]
fn apex_speed_increases_with_radius_and_is_symmetric_in_turn_direction() {
    let (m, tr) = setup();
    let a = vmax_point(&m, &tr, 15.0).unwrap();
    let b = vmax_point(&m, &tr, 40.0).unwrap();
    assert!(b > a);
    within(vmax_point(&m, &tr, -15.0).unwrap(), a, 1e-12);
}

#[test]
fn profile_matches_pointwise_values() {
    let (m, tr) = setup();
    let t = TrackModel::from_radius_length(&[(20.0, 1.0), (1.0e5, 1.0), (35.0, 1.0), (20.0, 1.0)])
        .unwrap();
    let p = vmax_profile(&m, &tr, &t).unwrap();
    assert_eq!(p.len(), 4);
    within(p[0], vmax_point(&m, &tr, 20.0).unwrap(), 1e-12);
    assert_eq!(p[0], p[3]);
}

#[test]
fn find_apexes_merges_noisy_minima_within_the_spacing() {
    let n = 80;
    let distance: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
    let mut v = vec![30.0; n];
    for i in 36..=44 {
        v[i] = 10.0 + (i as f64 - 40.0).abs();
    }
    v[38] = 9.5;
    v[42] = 9.6;
    let a = find_apexes(&v, &distance, 40.0, 3.0);
    assert_eq!(a.len(), 1);
    assert_eq!(a[0], 38);
    let mut w = v.clone();
    for i in 10..=14 {
        w[i] = 12.0 + (i as f64 - 12.0).abs();
    }
    assert_eq!(find_apexes(&w, &distance, 40.0, 3.0).len(), 2);
    assert_eq!(find_apexes(&w, &distance, 40.0, 30.0).len(), 1);
}

#[test]
fn smoothing_makes_each_cluster_monotone_into_its_apex() {
    let n = 80;
    let distance: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
    let mut v = vec![30.0; n];
    for i in 36..=44 {
        v[i] = 10.0 + (i as f64 - 40.0).abs();
    }
    v[38] = 9.5;
    v[42] = 9.6;
    let apexes = smooth_clusters(&mut v, &distance, 40.0, 3.0);
    let a = apexes[0];
    assert_eq!(v[a], 9.5);
    let first = 36;
    for i in first..a {
        assert!(v[i] >= v[i + 1] - 1e-12, "not monotone down at {i}");
    }
    let last = 42;
    for i in a..last {
        assert!(v[i] <= v[i + 1] + 1e-12, "not monotone up at {i}");
    }
}
