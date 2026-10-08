//! Lossless scalar-column tables for persisted numerical results.
//!
//! [result_table] flattens the serialized JSON of a [crate::sweep]/[crate::ride]/
//! [crate::lapsim::simulate_lap] result, or one `optimize` result, into a single
//! [ResultTable]: one column per distinct dotted field path, one row per sample (or
//! one row total for `optimize`). Column presence can vary by sample (e.g. an
//! optional metric present on some corners but not others); missing cells are
//! `Value::Null`, never a numeric zero or an omitted column, matching this crate's
//! wider convention (see [crate::analysis::OptionalValue]) that an absent quantity is
//! never silently treated as zero.
use crate::Error;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

/// One row per sample. Null means unavailable, never a numeric zero.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultTable {
    /// Native dotted field paths (e.g. `state.corners.front_left.metrics.caster_deg`),
    /// sorted lexicographically; field names retain their native unit suffixes.
    pub columns: Vec<String>,
    /// One entry per sample, each the same length as `columns` and in the same
    /// order; a cell is `Value::Null` where that sample had no such field, and a
    /// string diagnostic (e.g. a sweep sample's `error`) is preserved as-is rather
    /// than coerced to a number.
    pub rows: Vec<Vec<Value>>,
}
impl ResultTable {
    /// RFC 4180-style CSV with doubled quotes and CRLF line endings: `columns` as the
    /// header row, then one line per row with `Value::Null` rendered as an empty
    /// field, strings quoted (embedded quotes doubled), and other JSON values
    /// rendered via their own `Display`/`to_string`.
    pub fn csv(&self) -> String {
        let quote = |s: &str| format!("\"{}\"", s.replace('"', "\"\""));
        let mut text = self
            .columns
            .iter()
            .map(|s| quote(s))
            .collect::<Vec<_>>()
            .join(",");
        text.push_str("\r\n");
        for row in &self.rows {
            text.push_str(
                &row.iter()
                    .map(|v| match v {
                        Value::Null => String::new(),
                        Value::String(s) => quote(s),
                        _ => v.to_string(),
                    })
                    .collect::<Vec<_>>()
                    .join(","),
            );
            text.push_str("\r\n");
        }
        text
    }
}

fn flatten(value: &Value, path: &str, out: &mut BTreeMap<String, Value>) {
    match value {
        Value::Object(map) => {
            for (k, v) in map {
                flatten(
                    v,
                    &if path.is_empty() {
                        k.clone()
                    } else {
                        format!("{path}.{k}")
                    },
                    out,
                );
            }
        }
        Value::Array(items) => {
            let ids: Option<Vec<&str>> = items
                .iter()
                .map(|v| v.get("id").and_then(Value::as_str))
                .collect();
            let named = ids.filter(|ids| ids.iter().collect::<BTreeSet<_>>().len() == ids.len());
            for (i, v) in items.iter().enumerate() {
                let key = named
                    .as_ref()
                    .map_or_else(|| i.to_string(), |ids| ids[i].to_string());
                flatten(v, &format!("{path}.{key}"), out);
            }
        }
        _ => {
            out.insert(path.to_owned(), value.clone());
        }
    }
}

/// Convert sweep/ride/lap samples or one optimization result into native columns.
/// `kind` selects how `result` is shaped: `"sweep"` expects `result` to already be the
/// sample array; `"ride"` expects an object with a `"samples"` array; `"lap"` expects a
/// QSS [`crate::lapsim::LapRun`] (columnar `trace`, optional `channels`; one row per trace
/// point); `"optimize"` expects `result` to be a single object, treated as one row. For
/// `"ride"`, each per-corner array field (`compression_m`, `compression_velocity_m_s`,
/// `shock_force_n`, `support_reaction_n`) is re-keyed from `result.corner_ids` before
/// flattening, so columns use the run's recorded corner identity rather than project
/// array order.
///
/// # Errors
///
/// Returns an [Error] when: `kind` is not one of `"sweep"`/`"ride"`/`"lap"`/
/// `"optimize"`, or `result`'s shape does not match the expected one for `kind` (e.g.
/// `"sweep"` given a non-array, `"ride"` missing a `"samples"` array, or `"lap"` missing
/// its `trace` or with arrays of unequal lengths);
/// there are zero samples or more than 100,000; any sample is not a JSON object; or,
/// for `"ride"`, `result.corner_ids` is present alongside a per-corner array field but
/// the two disagree: either is not exactly four entries long, or `corner_ids` does not
/// hold four distinct strings.
pub fn result_table(kind: &str, result: &Value) -> Result<ResultTable, Error> {
    let fail = |s: &str| Error { message: s.into() };
    let single;
    let records = match kind {
        "sweep" => result.as_array(),
        "ride" => result.get("samples").and_then(Value::as_array),
        "lap" => {
            let rows = lap_rows(result)?;
            return Ok(table_from(rows));
        }
        "optimize" if result.is_object() => {
            single = vec![result.clone()];
            Some(&single)
        }
        _ => None,
    }
    .ok_or_else(|| fail("unsupported result kind or malformed sample array"))?;
    if records.is_empty() || records.len() > 100_000 {
        return Err(fail("result table requires 1..100000 samples"));
    }
    let mut rows = Vec::with_capacity(records.len());
    let mut cells_list = Vec::with_capacity(records.len());
    for (i, record) in records.iter().enumerate() {
        if !record.is_object() {
            return Err(fail("result samples must be objects"));
        }
        let mut record = record.clone();
        if kind == "ride" {
            if let Some(ids) = result.get("corner_ids").and_then(Value::as_array) {
                for field in [
                    "compression_m",
                    "compression_velocity_m_s",
                    "shock_force_n",
                    "support_reaction_n",
                ] {
                    if let Some(values) = record.get(field).and_then(Value::as_array) {
                        if ids.len() != 4
                            || values.len() != 4
                            || ids.iter().any(|v| !v.is_string())
                            || ids
                                .iter()
                                .filter_map(Value::as_str)
                                .collect::<BTreeSet<_>>()
                                .len()
                                != 4
                        {
                            return Err(fail(
                                "ride corner arrays must have four distinct recorded IDs",
                            ));
                        }
                        let object = ids
                            .iter()
                            .zip(values)
                            .map(|(id, v)| (id.as_str().unwrap().to_owned(), v.clone()))
                            .collect();
                        record[field] = Value::Object(object);
                    }
                }
            }
        }
        let mut cells = BTreeMap::from([("sample".to_owned(), Value::from(i))]);
        flatten(&record, "", &mut cells);
        cells_list.push(cells);
    }
    rows.extend(cells_list);
    Ok(table_from(rows))
}

/// Column-ize per-row cell maps (missing cells become null).
fn table_from(rows: Vec<BTreeMap<String, Value>>) -> ResultTable {
    let columns: Vec<String> = rows
        .iter()
        .flat_map(|r| r.keys().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let rows = rows
        .into_iter()
        .map(|mut row| {
            columns
                .iter()
                .map(|c| row.remove(c).unwrap_or(Value::Null))
                .collect()
        })
        .collect();
    ResultTable { columns, rows }
}

/// Rows of a QSS `LapRun`: one per trace point, the trace arrays merged with the channel arrays
/// (a channel whose name the trace already has is skipped).
fn lap_rows(run: &Value) -> Result<Vec<BTreeMap<String, Value>>, Error> {
    let fail = |s: &str| Error { message: s.into() };
    let trace = run
        .get("trace")
        .and_then(Value::as_object)
        .ok_or_else(|| fail("a lap result needs a trace"))?;
    let n = trace
        .get("distance_m")
        .and_then(Value::as_array)
        .map(Vec::len)
        .ok_or_else(|| fail("lap trace has no distance_m array"))?;
    if n == 0 || n > 100_000 {
        return Err(fail("result table requires 1..100000 samples"));
    }
    let mut columns: Vec<(&String, &Vec<Value>)> = vec![];
    let channels = run.get("channels").and_then(Value::as_object);
    for source in std::iter::once(trace).chain(channels) {
        for (name, values) in source {
            let Some(values) = values.as_array() else {
                continue;
            };
            if name == "apex_index" {
                continue; // indices of the apexes, not a per-point series
            }
            if values.len() != n {
                return Err(fail("lap trace and channel arrays must have equal lengths"));
            }
            if !columns.iter().any(|(c, _)| *c == name) {
                columns.push((name, values));
            }
        }
    }
    Ok((0..n)
        .map(|i| {
            let mut row = BTreeMap::from([("sample".to_owned(), Value::from(i))]);
            for (name, values) in &columns {
                row.insert((*name).clone(), values[i].clone());
            }
            row
        })
        .collect())
}
