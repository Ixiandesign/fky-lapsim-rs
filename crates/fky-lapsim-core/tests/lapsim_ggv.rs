mod common;
use common::{p19_powertrain, within};
use fky_lapsim_core::lapsim::forces::{max_ay, StepModel, ThesisConst};
use fky_lapsim_core::lapsim::ggv::{ellipse_remaining, ggv};
use fky_lapsim_core::lapsim::thesis::p19;
use fky_lapsim_core::lapsim::tractive::TractiveTable;

#[test]
fn ellipse_matches_eq_4_9() {
    within(
        ellipse_remaining(10.0, 15.0, 9.0),
        10.0 * (1.0f64 - 0.36).sqrt(),
        1e-12,
    );
    assert_eq!(ellipse_remaining(10.0, 15.0, 0.0), 10.0);
    assert_eq!(ellipse_remaining(10.0, 15.0, 15.0), 0.0);
    assert_eq!(ellipse_remaining(10.0, 15.0, 20.0), 0.0);
    assert_eq!(ellipse_remaining(10.0, 0.0, 0.0), 0.0);
    assert_eq!(
        ellipse_remaining(10.0, 15.0, -9.0),
        ellipse_remaining(10.0, 15.0, 9.0)
    );
}

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
fn ggv_boundary_closes_on_the_lateral_limit_and_orders_accel_above_brake() {
    let (m, tr) = setup();
    let n = 41;
    let g = ggv(&m, &tr, &[10.0, 20.0], n).unwrap();
    for (k, v) in [10.0, 20.0].iter().enumerate() {
        let ay_max = max_ay(&m, *v, 0.0).unwrap();
        let (ay, ax) = (&g.ay_m_s2[k], &g.ax_m_s2[k]);
        assert_eq!(ay.len(), 2 * n);
        assert_eq!(ax.len(), 2 * n);
        within(ay[0], ay_max, 1e-9);
        within(ay[n - 1], -ay_max, 1e-9);
        let resist = m.instant(*v, 0.0, 0.0).unwrap().forces.resist_n() / 250.0;
        // at the lateral limit no longitudinal grip is left: only drag remains
        within(ax[0], -resist, 1e-6);
        // pair each upper point with the lower-half point at the same ay: upper >= lower
        for i in 0..n {
            let upper = ax[i];
            let lower = ax[2 * n - 1 - i]; // lower half runs -max -> +max; mirrored index has the same ay
            assert!((ay[i] - ay[2 * n - 1 - i]).abs() < 1e-9);
            assert!(upper >= lower - 1e-9);
        }
    }
}

#[test]
fn lateral_capability_grows_with_speed_through_downforce() {
    let (m, tr) = setup();
    let g = ggv(&m, &tr, &[5.0, 15.0, 30.0], 21).unwrap();
    let ay: Vec<f64> = g.ay_m_s2.iter().map(|a| a[0]).collect();
    assert!(ay[0] < ay[1] && ay[1] < ay[2]);
}

#[test]
fn rejects_empty_input() {
    let (m, tr) = setup();
    assert!(ggv(&m, &tr, &[], 21).is_err());
    assert!(ggv(&m, &tr, &[10.0], 1).is_err());
}
