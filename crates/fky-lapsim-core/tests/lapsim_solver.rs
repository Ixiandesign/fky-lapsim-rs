mod common;
use common::{circle_rows, p19_powertrain, stadium_rows, within};
use fky_lapsim_core::lapsim::apex::vmax_point;
use fky_lapsim_core::lapsim::forces::ThesisConst;
use fky_lapsim_core::lapsim::scenarios::{accelerate, End, LongSettings, Solver};
use fky_lapsim_core::lapsim::solver::{simulate, LapSettings};
use fky_lapsim_core::lapsim::thesis::p19;
use fky_lapsim_core::lapsim::track_model::TrackModel;
use fky_lapsim_core::lapsim::tractive::TractiveTable;

fn setup() -> (ThesisConst, TractiveTable) {
    (
        ThesisConst {
            params: p19(),
            weight_transfer: true,
        },
        TractiveTable::build(&p19_powertrain(), 0.199, 1.0, 0.05).unwrap(),
    )
}

fn stadium() -> TrackModel {
    TrackModel::from_radius_length(&stadium_rows(120.0, 12.0, 0.5)).unwrap()
}

#[test]
fn constant_radius_flying_lap_is_length_over_apex_speed() {
    let (m, tr) = setup();
    let t = TrackModel::from_radius_length(&circle_rows(30.0, 360)).unwrap();
    let r = simulate(&m, &tr, &t, &LapSettings::default()).unwrap();
    let v = vmax_point(&m, &tr, 30.0).unwrap();
    within(r.lap_time_s, t.length_m / v, 2e-3);
    for s in &r.trace.speed_m_s {
        assert!(*s <= v + 1e-6);
    }
}

#[test]
fn straight_standing_start_matches_the_isolated_acceleration_scenario() {
    let (m, tr) = setup();
    let t = TrackModel::from_radius_length(&vec![(1.0e5, 0.25); 300]).unwrap(); // 75 m
    let s = LapSettings {
        flying: false,
        initial_speed_m_s: 0.0,
        apex_min_spacing_m: 2.0,
    };
    let r = simulate(&m, &tr, &t, &s).unwrap();
    let a = accelerate(
        &m,
        &tr,
        0.0,
        &LongSettings {
            solver: Solver::Distance { dx: 0.25 },
            end: End::TargetDistance(75.0),
        },
    )
    .unwrap();
    within(r.lap_time_s, *a.time_s.last().unwrap(), 0.01);
}

#[test]
fn stadium_lap_brakes_before_corners_and_never_exceeds_vmax() {
    let (m, tr) = setup();
    let t = stadium();
    let r = simulate(&m, &tr, &t, &LapSettings::default()).unwrap();
    let tr_ = &r.trace;
    assert_eq!(tr_.apex_index.len(), 2);
    for i in 0..tr_.speed_m_s.len() {
        assert!(tr_.speed_m_s[i] <= tr_.vmax_m_s[i] + 1e-6);
    }
    let apex_v = tr_
        .apex_index
        .iter()
        .map(|&i| tr_.speed_m_s[i])
        .fold(f64::MAX, f64::min);
    let max_v = tr_.speed_m_s.iter().cloned().fold(0.0, f64::max);
    assert!(max_v > 1.2 * apex_v);
    assert!(
        tr_.ax_m_s2.iter().any(|a| *a < -5.0),
        "no hard braking found"
    );
    assert!(
        tr_.ax_m_s2.iter().any(|a| *a > 2.0),
        "no hard acceleration found"
    );
    assert!(r.lap_time_s < t.length_m / apex_v);
    assert!(r.lap_time_s > t.length_m / max_v);
}

#[test]
fn reprocessing_inserts_exact_braking_points() {
    let (m, tr) = setup();
    let r = simulate(&m, &tr, &stadium(), &LapSettings::default()).unwrap();
    let n_inserted = r.trace.inserted.iter().filter(|b| **b).count();
    assert!(
        n_inserted >= 2,
        "expected one inserted braking point per corner, got {n_inserted}"
    );
    assert_eq!(r.trace.distance_m.len(), r.trace.speed_m_s.len());
    for w in r.trace.distance_m.windows(2) {
        assert!(w[1] > w[0]);
    }
}

#[test]
fn mirroring_the_track_leaves_the_lap_time_unchanged() {
    let (m, tr) = setup();
    let rows = stadium_rows(120.0, 12.0, 0.5);
    let mirrored: Vec<(f64, f64)> = rows.iter().map(|(r, l)| (-r, *l)).collect();
    let a = simulate(
        &m,
        &tr,
        &TrackModel::from_radius_length(&rows).unwrap(),
        &LapSettings::default(),
    )
    .unwrap();
    let b = simulate(
        &m,
        &tr,
        &TrackModel::from_radius_length(&mirrored).unwrap(),
        &LapSettings::default(),
    )
    .unwrap();
    within(a.lap_time_s, b.lap_time_s, 1e-9);
}

#[test]
fn more_lateral_grip_gives_a_faster_lap() {
    let (m, tr) = setup();
    let t = stadium();
    let base = simulate(&m, &tr, &t, &LapSettings::default())
        .unwrap()
        .lap_time_s;
    let mut p = p19();
    p.correlation.muy = 1.1;
    let grippier = ThesisConst {
        params: p,
        weight_transfer: true,
    };
    let faster = simulate(&grippier, &tr, &t, &LapSettings::default())
        .unwrap()
        .lap_time_s;
    assert!(faster < base);
}

#[test]
fn standing_start_is_slower_than_a_flying_lap() {
    let (m, tr) = setup();
    let t = stadium();
    let flying = simulate(&m, &tr, &t, &LapSettings::default())
        .unwrap()
        .lap_time_s;
    let standing = simulate(
        &m,
        &tr,
        &t,
        &LapSettings {
            flying: false,
            initial_speed_m_s: 0.0,
            apex_min_spacing_m: 2.0,
        },
    )
    .unwrap();
    assert!(standing.lap_time_s > flying);
    assert_eq!(standing.trace.speed_m_s[0], 0.0);
}

#[test]
fn rejects_invalid_settings() {
    let (m, tr) = setup();
    let t = TrackModel::from_radius_length(&circle_rows(30.0, 100)).unwrap();
    let bad = LapSettings {
        flying: true,
        initial_speed_m_s: -1.0,
        apex_min_spacing_m: 2.0,
    };
    assert!(simulate(&m, &tr, &t, &bad).is_err());
    let bad = LapSettings {
        flying: true,
        initial_speed_m_s: 0.0,
        apex_min_spacing_m: 0.0,
    };
    assert!(simulate(&m, &tr, &t, &bad).is_err());
}

#[test]
fn an_abrupt_straight_to_corner_step_never_asks_for_impossible_braking() {
    // radius jumps from a straight to ~9 m within one 0.1 m section (a traced skidpad does this):
    // the cap drops by a factor of three, so the braking trace must start before the step and the
    // final profile must obey the braking limit everywhere
    let (m, tr) = setup();
    let mut rows = vec![(1.0e5, 0.1); 400];
    // a slowly tightening corner: the cap falls along it, so its apex is at the far end
    rows.extend((0..300).map(|k| (9.2 - 0.2 * k as f64 / 300.0, 0.1)));
    rows.extend(vec![(1.0e5, 0.1); 100]);
    let t = TrackModel::from_radius_length(&rows).unwrap();
    let r = simulate(&m, &tr, &t, &LapSettings::default()).unwrap();
    let worst = r.trace.ax_m_s2.iter().fold(0.0_f64, |a, x| a.max(x.abs()));
    assert!(
        worst < 3.0 * 9.81,
        "profile needs {worst:.1} m/s², beyond the tyres"
    );
}
