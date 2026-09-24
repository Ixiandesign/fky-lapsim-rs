use dw_core::study::{motion_grid, AxisRange, GridMode, MotionGrid};

#[test]
fn combined_grid_includes_endpoints_and_fixed_rack() {
    let g = MotionGrid {
        heave: AxisRange {
            start: -0.01,
            end: 0.01,
            count: 3,
        },
        roll: AxisRange {
            start: -0.02,
            end: 0.02,
            count: 2,
        },
        rack_front: AxisRange {
            start: 0.003,
            end: 0.003,
            count: 1,
        },
        ..Default::default()
    };
    let m = motion_grid(&g).unwrap();
    assert_eq!(m.len(), 6);
    assert_eq!(
        m.iter().map(|m| (m.heave, m.roll)).collect::<Vec<_>>(),
        vec![
            (-0.01, -0.02),
            (-0.01, 0.02),
            (0., -0.02),
            (0., 0.02),
            (0.01, -0.02),
            (0.01, 0.02)
        ]
    );
    assert!(m.iter().all(|m| m.rack_front == 0.003 && m.pitch == 0.));
}

#[test]
fn linked_motion_follows_one_path_and_retains_reversed_ranges() {
    let g = MotionGrid {
        mode: GridMode::Linked,
        heave: AxisRange {
            start: 0.02,
            end: -0.02,
            count: 3,
        },
        pitch: AxisRange {
            start: -0.1,
            end: 0.1,
            count: 3,
        },
        ..Default::default()
    };
    let m = motion_grid(&g).unwrap();
    assert_eq!(m.len(), 3);
    assert_eq!((m[0].heave, m[0].pitch), (0.02, -0.1));
    assert_eq!((m[1].heave, m[1].pitch), (0., 0.));
    assert_eq!((m[2].heave, m[2].pitch), (-0.02, 0.1));
}

#[test]
fn invalid_and_excessive_grids_fail_before_allocation() {
    for range in [
        AxisRange {
            start: 0.,
            end: 1.,
            count: 1,
        },
        AxisRange {
            start: 0.,
            end: 0.,
            count: 0,
        },
        AxisRange {
            start: f64::NAN,
            end: 0.,
            count: 2,
        },
        AxisRange {
            start: -f64::MAX,
            end: f64::MAX,
            count: 3,
        },
        AxisRange {
            start: 0.,
            end: 1.,
            count: usize::MAX,
        },
    ] {
        assert!(motion_grid(&MotionGrid {
            heave: range,
            ..Default::default()
        })
        .is_err());
    }
    let mut g = MotionGrid {
        heave: AxisRange {
            start: 0.,
            end: 1.,
            count: 101,
        },
        roll: AxisRange {
            start: 0.,
            end: 1.,
            count: 101,
        },
        ..Default::default()
    };
    assert!(motion_grid(&g).is_err());
    g.mode = GridMode::Linked;
    g.roll.count = 3;
    assert!(motion_grid(&g).is_err());
}
