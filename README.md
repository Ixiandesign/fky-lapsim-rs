# fky-lapsim-rs

The Rust physics engine behind [FKY-LAPSIM](https://github.com/Ixiandesign/FKY-LAPSIM) — double-wishbone suspension kinematics, ride dynamics, tire/powertrain/aero modelling, full-car lap simulation, and optimization — split out as a standalone library with no Python or UI dependency.

## What's here

- **[`crates/fky-lapsim-core`](crates/fky-lapsim-core)** — the numerical engine. Pure Rust, no I/O: suspension geometry and kinematics, ride dynamics, Pacejka tire forces, the lap-simulation engine, and a bounded/parallel/resumable optimization search. See [its own README](crates/fky-lapsim-core/README.md) for a usage example and what each module covers.
- **[`crates/fky-lapsim-python`](crates/fky-lapsim-python)** — PyO3 bindings that expose `fky-lapsim-core` to Python as a native extension module. This is what [FKY-LAPSIM's Python service](https://github.com/Ixiandesign/FKY-LAPSIM) builds on; it has no use on its own without a Python host.

## Building

```bash
cargo build --workspace
cargo test --workspace
```

Coordinates are SI (metres), x forward / y left / z up; commanded angles are radians. See [FKY-LAPSIM's docs](https://github.com/Ixiandesign/FKY-LAPSIM/blob/main/docs/rust-guide.md) for the full usage guide, sign/frame conventions, and validation evidence — this repo only carries the Rust source; the research notes, benchmarks, and end-user app live in the main [FKY-LAPSIM](https://github.com/Ixiandesign/FKY-LAPSIM) repo.

## License

Apache-2.0, see [LICENSE](LICENSE).

## Python

```bash
pip install maturin
maturin develop --release   # editable install into the active venv
maturin build --release -o dist   # or build a wheel: pip install dist/*.whl
```

```python
import fky_lapsim as fl
p = fl.example_project()
state = fl.simulate(p, fl.defaults()["motion"])
lap = fl.run_lap(fl.lap_vehicle_demo(), fl.track_demo(), {})
```

Every native op is a function returning plain dicts; `fl.call(op, **fields)` reaches any op directly. `fl.OptimizationSession` gives incremental, cancellable, checkpointable optimization.
