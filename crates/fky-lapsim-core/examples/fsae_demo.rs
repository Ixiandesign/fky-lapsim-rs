//! `cargo run -p fky-lapsim-core --release --example fsae_demo` loads the FSAE generic
//! project (examples/fsae-generic-project.json, built from a published OptimumG
//! case study - see examples/build_fsae_project.py for provenance) and exercises
//! geometry solving, a motion sweep, and a native geometry optimization directly
//! through the Rust API. Run from the repository root so the relative example
//! path resolves.

use fky_lapsim_core::optimize::{Aggregation, OptimizationRequest, Target, Variable, VariableKind};
use fky_lapsim_core::{analyze, simulate, sweep, Motion, Project};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string("examples/fsae-generic-project.json")
        .map_err(|e| format!("{e}: run this from the repository root"))?;
    let project: Project = serde_json::from_str(&text)?;
    project.validate()?;
    println!("Loaded: {}", project.name);

    let zero = Motion::default();
    let state = simulate(&project, &zero)?;
    println!("\n-- Static geometry --");
    for corner in &state.corners {
        println!(
            "  {:?}: camber={:+.2} deg  toe={:+.3} deg  motion_ratio={:?}",
            corner.id,
            corner.metrics.camber_deg,
            corner.metrics.toe_deg,
            corner.metrics.motion_ratio
        );
    }

    let analysis = analyze(&project, &zero)?;
    println!(
        "  front track={:.1} mm, rear track={:.1} mm",
        analysis.front.wheel_track_m * 1000.0,
        analysis.rear.wheel_track_m * 1000.0
    );

    // This project's exact rear pull-rod geometry cannot close beyond about
    // -0.027 m heave (see examples/build_fsae_project.py); -0.03 m is included
    // deliberately to show the sweep retaining that failure explicitly.
    println!("\n-- Motion sweep (heave, mm -> camber, deg) --");
    let motions: Vec<Motion> = [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02]
        .iter()
        .map(|&heave| Motion {
            heave,
            ..Motion::default()
        })
        .collect();
    for sample in sweep(&project, &motions) {
        match (&sample.state, &sample.error) {
            (Some(s), _) => println!(
                "  {:+.1} mm -> {:+.3} deg",
                sample.motion.heave * 1000.0,
                s.corners[0].metrics.camber_deg
            ),
            (None, Some(e)) => {
                println!("  {:+.1} mm -> FAILED: {}", sample.motion.heave * 1000.0, e)
            }
            (None, None) => unreachable!(),
        }
    }

    println!("\n-- Geometry optimization: reduce front camber gain in roll --");
    let request = OptimizationRequest {
        variables: vec![
            Variable {
                path: "/corners/0/upper_front/2".into(),
                kind: VariableKind::Continuous {
                    lower: 0.235,
                    upper: 0.275,
                },
            },
            Variable {
                path: "/corners/0/upper_rear/2".into(),
                kind: VariableKind::Continuous {
                    lower: 0.230,
                    upper: 0.270,
                },
            },
        ],
        relations: vec![
            fky_lapsim_core::optimize::Relation {
                source: "/corners/0/upper_front/2".into(),
                destination: "/corners/1/upper_front/2".into(),
                factor: 1.0,
                offset: 0.0,
            },
            fky_lapsim_core::optimize::Relation {
                source: "/corners/0/upper_rear/2".into(),
                destination: "/corners/1/upper_rear/2".into(),
                factor: 1.0,
                offset: 0.0,
            },
        ],
        targets: vec![Target {
            corner: Some(fky_lapsim_core::CornerId::FrontLeft),
            metric: "analysis.camber_gain_deg_per_m".into(),
            value: 0.0,
            values: None,
            scale: 1.0,
            weight: 1.0,
            aggregation: Aggregation::MeanSquared,
        }],
        scenarios: vec![
            Motion {
                heave: -0.015,
                ..Motion::default()
            },
            Motion::default(),
            Motion {
                heave: 0.015,
                ..Motion::default()
            },
            Motion {
                roll: -2.5_f64.to_radians(),
                ..Motion::default()
            },
            Motion {
                roll: 2.5_f64.to_radians(),
                ..Motion::default()
            },
        ],
        seed: 7,
        population_size: 10,
        generations: 6,
        max_evaluations: 100,
        max_seconds: 20.0,
        workers: 2,
        ..Default::default()
    };
    let result = fky_lapsim_core::optimize::optimize(&project, &request)?;
    println!("  status: {}", result.status);
    if let Some(baseline) = &result.baseline {
        println!("  baseline score: {:?}", baseline.evaluation.score);
    }
    if let Some(best) = &result.best_feasible {
        println!("  best feasible score: {:?}", best.evaluation.score);
    }
    println!(
        "  candidate attempts: {}, elapsed: {:.2} s",
        result.candidate_attempts, result.elapsed_seconds
    );
    Ok(())
}
