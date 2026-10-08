# fky-lapsim-python

PyO3 bindings exposing [`fky-lapsim-core`](../fky-lapsim-core) to Python as the
`fky_lapsim._native` extension module — the native layer behind
[FKY-LAPSIM](https://github.com/Ixiandesign/FKY-LAPSIM)'s Python service. It is
`publish = false` and `crate-type = ["cdylib"]`: a Python extension module, not
a Rust library other crates depend on, so it has no meaningful standalone Rust
API beyond what is described here for maintainers.

## Shape of the bindings

Almost the entire surface is one generic, GIL-released, JSON-in/JSON-out
function:

```python
from fky_lapsim import _native
result_json = _native.call("simulate", '{"project": ..., "motion": ...}')
```

`call(op, input)` dispatches on the `op` string to a matching
`fky_lapsim_core` function, deserializing `input`'s relevant JSON keys into
that function's Rust argument types and re-serializing its result. See the
doc comment on `call` in [`src/lib.rs`](src/lib.rs) for the exhaustive,
canonical list of supported `op` values and their input/output shapes — it is
written to double as the Python docstring (`help(_native.call)`), so it is
the single source of truth for this wire contract; nothing here duplicates it.

Two operations need more than one call and get their own classes instead of
an `op` string:

- **`Session`** — a stateful, checkpointable wrapper around
  `fky_lapsim_core::optimize::OptimizationSession`, so Python can start a
  suspension-optimization search, `advance()` it incrementally, inspect
  progress, and `checkpoint()`/resume it later (including across process
  restarts).
- **`evaluate(input, cancel)`** — runs one optimizer candidate through the
  same physics evaluator the native search itself uses, for external search
  algorithms (e.g. Python `pymoo`/SciPy adapters) that want to reuse this
  crate's physics without reimplementing the search.

Both accept a **`CancellationToken`**: a flag Python can flip from another
thread to cooperatively cancel a running native call (cooperative, not
preemptive — an in-flight physics case can still finish after the flag is
set).

## Building

This crate is built as part of the workspace (`cargo build -p fky-lapsim-python`
from the repository root), but is only useful loaded from Python as an
extension module — see
[docs/python-service.md](https://github.com/Ixiandesign/FKY-LAPSIM/blob/main/docs/python-service.md)
for how `python/fky_lapsim` builds and loads it via `maturin`.

Licensed under Apache-2.0.
