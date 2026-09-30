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
