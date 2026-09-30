# fky-lapsim-core

Rigid-body kinematics, geometry analysis, ride dynamics, and optimization for
double-wishbone suspension systems, aimed at formula-style race cars. Pure
Rust, no I/O, no Python or UI dependency — the numerical engine behind
[FKY-LAPSIM](https://github.com/Ixiandesign/FKY-LAPSIM).

All bodies are rigid except the spring and damper, which are explicit force
laws. Coordinates are SI (metres), x forward / y left / z up; commanded
angles are radians. The corner solver uses quaternion tangent-space
Newton–Raphson with continuation for large travel; see
[the research notes](https://github.com/Ixiandesign/FKY-LAPSIM/blob/main/docs/research/suspension-research.md)
for the numerical method and physics assumptions behind it.

```rust
use fky_lapsim_core::{analyze, simulate, Motion, Project};

fn main() -> Result<(), fky_lapsim_core::Error> {
    let project = Project::example();
    project.validate()?;

    let motion = Motion {
        heave: 0.01,
        roll: 1.0_f64.to_radians(),
        ..Motion::default()
    };
    let state = simulate(&project, &motion)?;
    for corner in &state.corners {
        println!(
            "{:?}: camber={:.3} deg, toe={:.3} deg",
            corner.id, corner.metrics.camber_deg, corner.metrics.toe_deg
        );
    }

    let analysis = analyze(&project, &motion)?;
    println!("Front track: {:.3} m", analysis.front.wheel_track_m);
    Ok(())
}
```

## What's here

- **Geometry**: every wishbone pivot/axis, the knuckle (ball joints, spindle
  axis, steering pickup), the tire (disk/cylinder/torus contact), the
  chassis, and a configurable pushrod/rocker/shock linkage — all
  independently settable per corner (`Project`, `Corner`).
- **Kinematics**: prescribed heave/roll/pitch/steering motion, solved rigid
  linkage and metrics (`simulate`, `sweep`, `solve_corner`).
- **Analysis**: motion-ratio and camber/toe gradients, wheel rate, projected
  instant and roll centers, each with an explicit reason when undefined
  (`analyze`).
- **Dynamics**: nonlinear time-domain ride simulation with reduced or
  full retained-component-inertia fidelity, equilibrium solving, and
  road/travel-limit events (`ride`).
- **Optimization**: a bounded, parallel, resumable/checkpointable
  differential-evolution search over any registered geometry, analysis, or
  ride metric, with constraints, linked variables, and uncertainty sampling
  (`optimize`).

Full usage guide, conventions, and validation evidence:
[docs/rust-guide.md](https://github.com/Ixiandesign/FKY-LAPSIM/blob/main/docs/rust-guide.md),
[docs/model-conventions.md](https://github.com/Ixiandesign/FKY-LAPSIM/blob/main/docs/model-conventions.md),
[docs/validation.md](https://github.com/Ixiandesign/FKY-LAPSIM/blob/main/docs/validation.md).

Licensed under Apache-2.0.
