mod common;
use common::{circle_rows, stadium_rows, within};
use fky_lapsim_core::lapsim::track_model::{TrackModel, STRAIGHT_RADIUS_M};
use fky_lapsim_core::track::Track;

#[test]
fn from_radius_length_accumulates_distance() {
    let t = TrackModel::from_radius_length(&[(10.0, 2.0), (-20.0, 3.0), (1.0e5, 5.0)]).unwrap();
    assert_eq!(t.distance_m, vec![0.0, 2.0, 5.0]);
    assert_eq!(t.length_m, 10.0);
    assert_eq!(t.section_m(0), 2.0);
    assert_eq!(t.section_m(2), 5.0); // periodic closing section
}

#[test]
fn validation_rejects_bad_tracks() {
    assert!(TrackModel::from_radius_length(&[(10.0, 1.0), (10.0, 1.0)]).is_err()); // <3 points
    assert!(TrackModel::from_radius_length(&[(10.0, 1.0), (0.0, 1.0), (5.0, 1.0)]).is_err());
    assert!(TrackModel::from_radius_length(&[(10.0, 1.0), (10.0, -1.0), (5.0, 1.0)]).is_err());
    assert!(TrackModel::from_radius_length(&[(10.0, 1.0), (f64::NAN, 1.0), (5.0, 1.0)]).is_err());
}

#[test]
fn from_track_recovers_circle_radius_and_length() {
    let circle = Track::circle(20.0, 3.0, 400).unwrap();
    let t = TrackModel::from_track(&circle, 0.5).unwrap();
    within(t.length_m, 2.0 * std::f64::consts::PI * 20.0, 5e-3);
    for r in &t.radius_m {
        within(*r, 20.0, 0.02); // positive: counter-clockwise = left
    }
}

#[test]
fn from_track_marks_straights() {
    // a polygon sampled every 5 m along its straights: collinear vertices have zero curvature
    let mut pts = vec![];
    for x in (0..=50).step_by(5) {
        pts.push([x as f64, 0.0]);
    }
    for y in (5..=10).step_by(5) {
        pts.push([50.0, y as f64]);
    }
    for x in (0..50).rev().step_by(5) {
        pts.push([x as f64, 10.0]);
    }
    let track = Track {
        centerline_m: pts,
        width_m: 3.0,
    };
    let t = TrackModel::from_track(&track, 1.0).unwrap();
    let straights = t
        .radius_m
        .iter()
        .filter(|r| r.abs() >= STRAIGHT_RADIUS_M * 0.5)
        .count();
    assert!(straights > t.len() / 2, "{straights} of {}", t.len());
}

#[test]
fn fine_mesh_refines_only_tight_corners_and_conserves_length() {
    let rows = stadium_rows(100.0, 10.0, 1.0);
    let t = TrackModel::from_radius_length(&rows).unwrap();
    let f = t.fine_mesh(35.0, 2.0, 0.1).unwrap();
    within(f.length_m, t.length_m, 1e-12);
    assert!(f.len() > t.len());
    // the middle of a straight (far from corners) keeps its coarse spacing
    let far: Vec<f64> = (0..f.len())
        .filter(|&i| f.radius_m[i].abs() >= STRAIGHT_RADIUS_M * 0.5)
        .map(|i| f.section_m(i))
        .collect();
    assert!(far.iter().any(|s| *s > 0.9));
    // all sections inside corners are at most the requested step (+ rounding)
    for i in 0..f.len() {
        if f.radius_m[i].abs() < 35.0 {
            assert!(f.section_m(i) <= 0.1 + 1e-9, "section {}", f.section_m(i));
        }
    }
}

#[test]
fn moving_average_smooths_noise_but_keeps_length() {
    let mut rows = circle_rows(20.0, 200);
    rows[100].0 = 10.0; // spike
    let t = TrackModel::from_radius_length(&rows).unwrap();
    let s = t.moving_average_curvature(5);
    within(s.length_m, t.length_m, 1e-12);
    assert!((s.radius_m[100] - 20.0).abs() < (t.radius_m[100] - 20.0).abs());
    assert_eq!(t.moving_average_curvature(4).len(), t.len()); // even windows are raised to odd
}
