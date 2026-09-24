//! Lossless scalar-column tables for persisted numerical results.
use crate::Error;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

/// One row per sample. Null means unavailable, never a numeric zero.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultTable {
    /// Native field paths; field names retain their native unit suffixes.
    pub columns: Vec<String>,
    /// Cells aligned with columns, including string diagnostics.
    pub rows: Vec<Vec<Value>>,
}
impl ResultTable {
    /// RFC 4180-style CSV with doubled quotes and CRLF line endings.
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
/// Ride corner-array fields use the run's recorded IDs, independent of project order.
/// Results with no samples or an unsupported kind return an explicit error.
pub fn result_table(kind: &str, result: &Value) -> Result<ResultTable, Error> {
    let fail = |s: &str| Error { message: s.into() };
    let single;
    let records = match kind {
        "sweep" => result.as_array(),
        "ride" | "lap" => result.get("samples").and_then(Value::as_array),
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
    let mut columns = BTreeSet::new();
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
        columns.extend(cells.keys().cloned());
        rows.push(cells);
    }
    let columns: Vec<_> = columns.into_iter().collect();
    let rows = rows
        .into_iter()
        .map(|mut row| {
            columns
                .iter()
                .map(|c| row.remove(c).unwrap_or(Value::Null))
                .collect()
        })
        .collect();
    Ok(ResultTable { columns, rows })
}
