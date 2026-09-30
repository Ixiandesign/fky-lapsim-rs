//! `cargo run -p fky-lapsim-core --release --example ride` emits the project, request and run.
//! Redirect stdout to a JSON file for visualization or downstream validation.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (project, request) = fky_lapsim_core::dynamics::formula_car_demo()?;
    let run = fky_lapsim_core::ride(&project, &request)?;
    eprintln!(
        "{}: {} samples, last valid time {} s, termination {:?}",
        run.model_fidelity,
        run.samples.len(),
        run.samples.last().map_or(0.0, |s| s.time_s),
        run.termination
    );
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "project": project, "request": request, "run": run
        }))?
    );
    Ok(())
}
