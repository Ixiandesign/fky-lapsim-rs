use fky_lapsim_core::results::result_table;
use serde_json::json;

#[test]
fn table_preserves_failed_samples_nulls_and_all_metrics_by_corner_identity() {
    let data = json!([
        {"motion":{"heave":0.},"state":{"corners":[
            {"id":"front_left","metrics":{"caster_deg":4.,"shock_force_n":125.}},
            {"id":"front_right","metrics":{"caster_deg":5.}}
        ]}},
        {"motion":{"heave":1.},"state":null,"error":"could not close, \"left\""},
        {"motion":{"heave":0.01},"state":{"corners":[
            {"id":"front_right","metrics":{"caster_deg":6.}},
            {"id":"front_left","metrics":{"caster_deg":7.,"shock_force_n":null}}
        ]}}
    ]);
    let t = result_table("sweep", &data).unwrap();
    assert_eq!(t.rows.len(), 3);
    let caster = t
        .columns
        .iter()
        .position(|c| c == "state.corners.front_left.metrics.caster_deg")
        .unwrap();
    assert_eq!(t.rows[0][caster], json!(4.));
    assert!(t.rows[1][caster].is_null());
    assert_eq!(t.rows[2][caster], json!(7.));
    assert!(t.csv().contains("\"could not close, \"\"left\"\"\""));
    assert!(t.columns.iter().any(|c| c.ends_with("shock_force_n")));
}

#[test]
fn ride_tables_use_recorded_corner_order_and_export_loads_energy_and_interconnects() {
    let r = json!({"corner_ids":["rear_right","front_left","rear_left","front_right"],
        "samples":[{"time_s":0.1,"support_reaction_n":[40.,10.,30.,20.],
            "velocity":[1.,2.,3.],"energy_balance_error_j":0.001,
            "front_interconnect":{"roll_force_n":50.}}]});
    let t = result_table("ride", &r).unwrap();
    let column = |s| t.columns.iter().position(|c| c == s).unwrap();
    assert_eq!(
        t.rows[0][column("support_reaction_n.front_left")],
        json!(10.)
    );
    assert_eq!(t.rows[0][column("velocity.2")], json!(3.));
    assert_eq!(
        t.rows[0][column("front_interconnect.roll_force_n")],
        json!(50.)
    );
    assert!(result_table("sweep", &json!({})).is_err());
    assert!(result_table("unknown", &json!([])).is_err());
}

#[test]
fn lap_tables_have_one_row_per_trace_point_and_merge_channels() {
    let run = json!({
        "trace": {"apex_index": [2], "distance_m": [0., 1., 2.], "speed_m_s": [10., 11., 12.], "radius_m": [1e5, 20., 20.]},
        "channels": {"distance_m": [0., 1., 2.], "speed_kmh": [36., 39.6, 43.2], "gear": [1, 1, 2]}
    });
    let t = result_table("lap", &run).unwrap();
    assert_eq!(t.rows.len(), 3);
    let column = |s: &str| t.columns.iter().position(|c| c == s).unwrap();
    assert_eq!(t.rows[1][column("speed_m_s")], json!(11.));
    assert_eq!(t.rows[1][column("speed_kmh")], json!(39.6));
    assert_eq!(t.rows[2][column("gear")], json!(2));
    // the channel copy of distance does not duplicate the trace column
    assert_eq!(t.columns.iter().filter(|c| *c == "distance_m").count(), 1);
    assert!(!t.columns.iter().any(|c| c == "apex_index"));
}

#[test]
fn lap_tables_reject_a_missing_trace_and_mismatched_arrays() {
    assert!(result_table("lap", &json!({"metrics": {}})).is_err());
    assert!(result_table(
        "lap",
        &json!({"trace": {"distance_m": [0., 1.], "speed_m_s": [1.]}})
    )
    .is_err());
    assert!(result_table(
        "lap",
        &json!({"trace": {"distance_m": [0., 1.]}, "channels": {"gear": [1]}})
    )
    .is_err());
}
