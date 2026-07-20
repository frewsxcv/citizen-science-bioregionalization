//! Port of `src/darwin_core_utils.py::_parse_meta` — parsing a Darwin Core
//! archive's `meta.xml` into the parameters needed to scan its core data file.
//!
//! Only the XML parsing lives in Rust. The scan itself (`scan_parquet`/
//! `scan_csv` + rename + cast) stays in Python: it is pure Polars plan
//! construction that must run in the pipeline's own Polars engine, and passing a
//! LazyFrame plan across the boundary is not viable (pyo3-polars serializes the
//! logical plan, and its `DSL_SCHEMA_HASH` only matches when the Rust crate is
//! built from the *exact same commit* as the installed pip polars wheel).

use std::fs;
use std::path::Path;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use quick_xml::events::Event;
use quick_xml::reader::Reader;

/// Metadata parsed from a Darwin Core archive's `meta.xml`.
/// Mirrors `src/darwin_core_utils.py::_Meta`.
struct Meta {
    core_file: String,
    has_header: bool,
    separator: String,
    columns: Vec<String>,
    quote_char: String,
    encoding: String,
    /// Insertion-ordered (term, default_value) pairs, mirroring the Python dict.
    default_fields: Vec<(String, String)>,
}

/// Reduce a Darwin Core term URI to its bare term name, mirroring Python's
/// `term_uri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]`.
fn term_from_uri(term_uri: &str) -> &str {
    let after_slash = term_uri.rsplit('/').next().unwrap_or(term_uri);
    after_slash.rsplit('#').next().unwrap_or(after_slash)
}

/// Look up an attribute (by local name) on a quick-xml start/empty element.
fn attr(e: &quick_xml::events::BytesStart, name: &str) -> Option<String> {
    for a in e.attributes().flatten() {
        if a.key.local_name().as_ref() == name.as_bytes() {
            return Some(String::from_utf8_lossy(a.value.as_ref()).into_owned());
        }
    }
    None
}

/// Parse a Darwin Core archive's `meta.xml`.
/// Mirrors `src/darwin_core_utils.py::_parse_meta`.
fn parse_meta(meta_path: &Path) -> Result<Meta, String> {
    let content = fs::read_to_string(meta_path).map_err(|e| format!("reading meta.xml: {e}"))?;
    let mut reader = Reader::from_str(&content);
    reader.config_mut().trim_text(true);

    let mut in_core = false;
    // Attributes captured from the <core> element.
    let mut separator = ",".to_string();
    let mut quote_char = "\"".to_string();
    let mut encoding = "utf-8".to_string();
    let mut has_header = false;

    let mut core_file: Option<String> = None;
    let mut fields: Vec<String> = Vec::new();
    let mut default_fields: Vec<(String, String)> = Vec::new();
    let mut id_index: Option<usize> = None;

    // When we are inside <files>, the next <location>'s text is the core file.
    let mut reading_location = false;

    let ensure_len = |fields: &mut Vec<String>, idx: usize| {
        if fields.len() <= idx {
            fields.resize(idx + 1, String::new());
        }
    };

    let handle_element = |e: &quick_xml::events::BytesStart,
                          in_core: &mut bool,
                          separator: &mut String,
                          quote_char: &mut String,
                          encoding: &mut String,
                          has_header: &mut bool,
                          fields: &mut Vec<String>,
                          default_fields: &mut Vec<(String, String)>,
                          id_index: &mut Option<usize>| {
        match e.name().local_name().as_ref() {
            b"core" => {
                *in_core = true;
                if let Some(s) = attr(e, "fieldsTerminatedBy") {
                    *separator = if s == "\\t" { "\t".to_string() } else { s };
                }
                if let Some(q) = attr(e, "fieldsEnclosedBy") {
                    *quote_char = q;
                }
                if let Some(enc) = attr(e, "encoding") {
                    *encoding = enc;
                }
                let ignore = attr(e, "ignoreHeaderLines")
                    .and_then(|v| v.parse::<i64>().ok())
                    .unwrap_or(0);
                *has_header = ignore >= 1;
            }
            b"field" if *in_core => {
                if let Some(term_uri) = attr(e, "term") {
                    let term = term_from_uri(&term_uri).to_string();
                    if let Some(idx) = attr(e, "index").and_then(|v| v.parse::<usize>().ok()) {
                        ensure_len(fields, idx);
                        fields[idx] = term;
                    } else if let Some(default_value) = attr(e, "default") {
                        default_fields.push((term, default_value));
                    }
                }
            }
            b"id" if *in_core => {
                *id_index = attr(e, "index").and_then(|v| v.parse::<usize>().ok());
            }
            _ => {}
        }
    };

    loop {
        match reader
            .read_event()
            .map_err(|e| format!("parsing meta.xml: {e}"))?
        {
            Event::Start(e) | Event::Empty(e) => {
                let local = e.name().local_name().as_ref().to_vec();
                if in_core && local == b"location" {
                    reading_location = true;
                }
                handle_element(
                    &e,
                    &mut in_core,
                    &mut separator,
                    &mut quote_char,
                    &mut encoding,
                    &mut has_header,
                    &mut fields,
                    &mut default_fields,
                    &mut id_index,
                );
            }
            Event::Text(t) if reading_location => {
                let text = t
                    .unescape()
                    .map_err(|e| format!("meta.xml location text: {e}"))?
                    .trim()
                    .to_string();
                if !text.is_empty() {
                    core_file = Some(text);
                }
            }
            Event::End(e) => {
                let local = e.name().local_name().as_ref().to_vec();
                if local == b"location" {
                    reading_location = false;
                }
                if local == b"core" {
                    break;
                }
            }
            Event::Eof => break,
            _ => {}
        }
    }

    let core_file = core_file.ok_or_else(|| "<core> missing <files>/<location>".to_string())?;

    // <id index="N"/>: if that column has no name yet, fall back to "id".
    if let Some(idx) = id_index {
        ensure_len(&mut fields, idx);
        if fields[idx].is_empty() {
            fields[idx] = "id".to_string();
        }
    }

    // Fill any empty column names with fallback names, mirroring Python.
    let columns: Vec<String> = fields
        .into_iter()
        .enumerate()
        .map(|(i, name)| {
            if name.is_empty() {
                format!("col_{i}")
            } else {
                name
            }
        })
        .collect();

    Ok(Meta {
        core_file,
        has_header,
        separator,
        columns,
        quote_char,
        encoding,
        default_fields,
    })
}

/// Parse a Darwin Core archive `meta.xml`. Returns a tuple mirroring `_Meta`'s
/// fields, which `src/darwin_core_utils.py::_parse_meta` reconstructs into the
/// `_Meta` dataclass.
#[pyfunction]
pub fn parse_darwin_core_meta(
    meta_path: String,
) -> PyResult<(
    String,
    bool,
    String,
    Vec<String>,
    String,
    String,
    Vec<(String, String)>,
)> {
    let meta = parse_meta(Path::new(&meta_path)).map_err(PyValueError::new_err)?;
    Ok((
        meta.core_file,
        meta.has_header,
        meta.separator,
        meta.columns,
        meta.quote_char,
        meta.encoding,
        meta.default_fields,
    ))
}
