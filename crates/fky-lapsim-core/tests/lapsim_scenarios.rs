mod common;
use common::within;
use fky_lapsim_core::lapsim::forces::ThesisConst;
use fky_lapsim_core::lapsim::scenarios::{corner, steering};
use fky_lapsim_core::lapsim::thesis::p19;

fn model(aero: f64) -> ThesisConst {
    let mut p = p19();
    p.correlation.aero = aero;
    ThesisConst {
        params: p,
        weight_transfer: false,
    }
}

#[test]
fn skidpad_reproduces_thesis_tables_3_13_and_3_14() {
    let c = corner(&model(1.0), 30.0, 9.125).unwrap();
    within(c.speed_m_s * 3.6, 43.97, 0.01);
    within(c.ay_m_s2, 16.35, 0.01);
    within(
        2.0 * std::f64::consts::PI * 9.125 / c.speed_m_s,
        4.694,
        0.01,
    );
    within(c.steer_wheel_deg, 40.2, 0.01);
}

#[test]
fn skidpad_with_half_aero_matches_table_3_15() {
    let c = corner(&model(0.5), 30.0, 9.125).unwrap();
    within(c.speed_m_s * 3.6, 42.33, 0.01);
    within(c.ay_m_s2, 15.14, 0.01);
    within(
        2.0 * std::f64::consts::PI * 9.125 / c.speed_m_s,
        4.876,
        0.01,
    );
    within(c.steer_wheel_deg, 37.3, 0.01);
}

#[test]
fn steering_matrix_satisfies_both_rows_of_eq_2_53() {
    let mut p = p19();
    p.rear.cornering_stiffness_n_per_deg = 260.0; // unequal axles
    let (delta, beta) = steering(&p, 12.0, 9.0).unwrap();
    let cf = 2.0 * p.front.cornering_stiffness_n_per_deg;
    let cr = 2.0 * p.rear.cornering_stiffness_n_per_deg;
    let (a, b) = (p.a_dist_m(), p.b_dist_m());
    within(
        cf * delta + (cr - cf) * beta,
        p.mass_kg * 12.0 * 12.0 / 9.0,
        1e-9,
    );
    within(a * cf * delta + (b * cr - a * cf) * beta, 0.0, 1e-9);
}

#[test]
fn right_turns_have_negative_angles() {
    let p = p19();
    let (dl, bl) = steering(&p, 10.0, 20.0).unwrap();
    let (dr, br) = steering(&p, 10.0, -20.0).unwrap();
    within(dr, -dl, 1e-12);
    within(br, -bl, 1e-12);
}

#[test]
fn corner_is_capped_at_top_speed_and_rejects_bad_radius() {
    let c = corner(&model(1.0), 8.0, 200.0).unwrap();
    assert!((c.speed_m_s - 8.0).abs() < 1e-9);
    assert!(corner(&model(1.0), 30.0, 0.0).is_err());
    assert!(corner(&model(1.0), 30.0, f64::NAN).is_err());
}

#[test]
fn more_downforce_means_a_higher_cornering_speed() {
    let slow = corner(&model(0.5), 60.0, 40.0).unwrap().speed_m_s;
    let fast = corner(&model(1.0), 60.0, 40.0).unwrap().speed_m_s;
    assert!(fast > slow);
}

use common::p19_powertrain;
use fky_lapsim_core::lapsim::scenarios::{accelerate, brake, End, LongSettings, Solver};
use fky_lapsim_core::lapsim::tractive::TractiveTable;
use fky_lapsim_core::powertrain::DrivenAxle;

fn last(v: &[f64]) -> f64 {
    *v.last().unwrap()
}
fn wt(on: bool) -> ThesisConst {
    ThesisConst {
        params: p19(),
        weight_transfer: on,
    }
}

#[test]
fn fs_brake_test_matches_tables_3_6_and_3_7() {
    let v0 = 55.0 / 3.6;
    let t = brake(
        &wt(false),
        v0,
        &LongSettings {
            solver: Solver::Time { dt: 0.001 },
            end: End::Natural,
        },
    )
    .unwrap();
    within(last(&t.time_s), 1.157, 0.01);
    within(last(&t.distance_m), 8.43, 0.01);
    let d = brake(
        &wt(false),
        v0,
        &LongSettings {
            solver: Solver::Distance { dx: 0.15 },
            end: End::Natural,
        },
    )
    .unwrap();
    within(last(&d.time_s), 1.154, 0.01);
    within(last(&d.distance_m), 8.55, 0.01);
}

#[test]
fn common_braking_matches_tables_3_6_3_7_and_3_8() {
    let (v0, v1) = (100.0 / 3.6, 35.0 / 3.6);
    let t = brake(
        &wt(false),
        v0,
        &LongSettings {
            solver: Solver::Time { dt: 0.001 },
            end: End::TargetSpeed(v1),
        },
    )
    .unwrap();
    within(last(&t.time_s), 1.020, 0.01);
    within(last(&t.distance_m), 18.20, 0.01);
    let d = brake(
        &wt(false),
        v0,
        &LongSettings {
            solver: Solver::Distance { dx: 0.15 },
            end: End::TargetSpeed(v1),
        },
    )
    .unwrap();
    // The thesis distance solver completes its last full 0.15 m step past the target speed, which
    // inflates its time to 1.032 s; stopping exactly at the target gives 1.017 s, within 0.3 % of
    // the time-based solver. Assert both facts.
    within(last(&d.time_s), 1.032, 0.02);
    within(last(&d.time_s), last(&t.time_s), 0.01);
    within(last(&d.distance_m), 18.30, 0.01);
    let w = brake(
        &wt(true),
        v0,
        &LongSettings {
            solver: Solver::Distance { dx: 0.15 },
            end: End::TargetSpeed(v1),
        },
    )
    .unwrap();
    within(last(&w.time_s), 1.047, 0.02);
}

fn strong_tractive() -> TractiveTable {
    TractiveTable::build(&p19_powertrain(), 0.199, 5.0, 0.05).unwrap()
}

#[test]
fn weight_transfer_helps_rwd_and_hurts_fwd_acceleration() {
    let s = LongSettings {
        solver: Solver::Distance { dx: 0.1 },
        end: End::TargetDistance(20.0),
    };
    let tr = strong_tractive();
    let rwd_off = accelerate(&wt(false), &tr, 0.0, &s).unwrap();
    let rwd_on = accelerate(&wt(true), &tr, 0.0, &s).unwrap();
    assert!(last(&rwd_on.time_s) < last(&rwd_off.time_s));
    let mut fwd = p19();
    fwd.drive = DrivenAxle::Front;
    let off = ThesisConst {
        params: fwd.clone(),
        weight_transfer: false,
    };
    let on = ThesisConst {
        params: fwd,
        weight_transfer: true,
    };
    let f_off = accelerate(&off, &tr, 0.0, &s).unwrap();
    let f_on = accelerate(&on, &tr, 0.0, &s).unwrap();
    assert!(last(&f_on.time_s) > last(&f_off.time_s));
}

#[test]
fn time_and_distance_solvers_agree_over_75_m() {
    let tr = TractiveTable::build(&p19_powertrain(), 0.199, 1.0, 0.05).unwrap();
    let run = |m: &ThesisConst, solver: Solver| {
        accelerate(
            m,
            &tr,
            0.0,
            &LongSettings {
                solver,
                end: End::TargetDistance(75.0),
            },
        )
        .unwrap()
    };
    // without weight transfer the thesis reports <0.2 % between the solvers (Tables 3-2, 3-3)
    let t = run(&wt(false), Solver::Time { dt: 0.001 });
    let d = run(&wt(false), Solver::Distance { dx: 0.25 });
    within(last(&d.time_s), last(&t.time_s), 0.005);
    within(last(&d.speed_m_s), last(&t.speed_m_s), 0.005);
    // with weight transfer the previous step's ax lags by a whole distance step at launch, so the
    // 0.25 m solver is only close; a finer distance step converges onto the time solver
    let t = run(&wt(true), Solver::Time { dt: 0.001 });
    let coarse = run(&wt(true), Solver::Distance { dx: 0.25 });
    let fine = run(&wt(true), Solver::Distance { dx: 0.02 });
    within(last(&coarse.time_s), last(&t.time_s), 0.03);
    assert!(
        (last(&fine.time_s) - last(&t.time_s)).abs()
            < (last(&coarse.time_s) - last(&t.time_s)).abs()
    );
    within(last(&fine.time_s), last(&t.time_s), 0.005);
}

#[test]
fn natural_acceleration_ends_at_the_force_balance_top_speed() {
    let tr = TractiveTable::build(&p19_powertrain(), 0.199, 1.0, 0.05).unwrap();
    let t = accelerate(
        &wt(true),
        &tr,
        0.0,
        &LongSettings {
            solver: Solver::Distance { dx: 0.5 },
            end: End::Natural,
        },
    )
    .unwrap();
    assert!(last(&t.speed_m_s) < tr.top_speed_m_s());
    assert!(last(&t.ax_m_s2) < 0.05, "final ax {}", last(&t.ax_m_s2));
    for w in t.speed_m_s.windows(2) {
        assert!(w[1] >= w[0] - 1e-9);
    }
    for w in t.time_s.windows(2) {
        assert!(w[1] > w[0]);
    }
}

#[test]
fn invalid_scenario_inputs_are_rejected() {
    let tr = strong_tractive();
    let ok = LongSettings {
        solver: Solver::Time { dt: 0.001 },
        end: End::Natural,
    };
    assert!(accelerate(&wt(false), &tr, -1.0, &ok).is_err());
    let bad = LongSettings {
        solver: Solver::Time { dt: 0.0 },
        end: End::Natural,
    };
    assert!(accelerate(&wt(false), &tr, 0.0, &bad).is_err());
    let bad = LongSettings {
        solver: Solver::Distance { dx: -0.1 },
        end: End::Natural,
    };
    assert!(brake(&wt(false), 10.0, &bad).is_err());
}

#[test]
fn a_neutral_weight_distribution_falls_back_to_the_bicycle_steady_state() {
    use fky_lapsim_core::lapsim::scenarios::bicycle_steady_state;
    let mut p = p19();
    p.wd_front_pct = 50.0; // a = b: the thesis matrix is singular
    let (d, b) = steering(&p, 12.0, 20.0).unwrap();
    let (d2, b2) = bicycle_steady_state(&p, 12.0, 20.0).unwrap();
    within(d, d2, 1e-12);
    within(b, b2, 1e-12);
    assert!(d.is_finite() && d > 0.0);
    // right turn mirrors
    let (dr, br) = steering(&p, 12.0, -20.0).unwrap();
    within(dr, -d, 1e-12);
    within(br, -b, 1e-12);
    // P19 itself is untouched (thesis-exact)
    let (dp, _) = steering(&p19(), 43.97 / 3.6, 9.125).unwrap();
    within(dp * 3.74, 40.2, 5e-3);
}

#[test]
fn bicycle_steady_state_satisfies_force_and_moment_balance() {
    use fky_lapsim_core::lapsim::scenarios::bicycle_steady_state;
    let p = p19();
    let (v, r) = (15.0, 25.0);
    let (d, be) = bicycle_steady_state(&p, v, r).unwrap();
    let k = (1.0_f64 / r).to_degrees();
    let (cf, cr) = (
        2.0 * p.front.cornering_stiffness_n_per_deg,
        2.0 * p.rear.cornering_stiffness_n_per_deg,
    );
    let (a, b) = (p.a_dist_m(), p.b_dist_m());
    let (af, ar) = (d - be - a * k, -be + b * k);
    within(cf * af + cr * ar, p.mass_kg * v * v / r, 1e-9);
    within(a * cf * af - b * cr * ar, 0.0, 1e-6);
}
