# bioregion_rs

Rust core for the citizen-science bioregionalization pipeline. See
`../RUST_MIGRATION_PLAN.md` for the overall strategy. This crate is the Phase 0
scaffold: it establishes the Rust↔Python interop and the correctness harness that
every subsequent file migration will use.

## What's here

- `src/lib.rs` — a PyO3 extension module (`bioregion_rs`) exposing:
  - `darken_hex_color(hex, factor=0.5)` — pure function (plain PyO3 boundary),
    mirrors `src/colors.py`.
  - `select_geocode`/`with_geocode`/`filter_by_bounding_box` — mirrors `src/geocode.py`.
  - `build_geocode` — mirrors `src/dataframes/geocode.py`.
  - `build_taxonomy` — mirrors `src/dataframes/taxonomy.py`.
  - `build_geocode_taxa_counts` — mirrors `src/dataframes/geocode_taxa_counts.py`.
  - `build_geocode_neighbors`/`build_geocode_neighbors_no_edges` — mirrors
    `src/dataframes/geocode_neighbors.py`.
- `harness.py` — runs each Rust function and its Python counterpart on the same
  input and asserts they match. The template for migrating each file.

## Build & test

```bash
# Rust unit tests (pure logic; needs a Python on PATH for pyo3 linking):
cargo test --lib

# Build the extension into the project venv:
uv run maturin develop --manifest-path bioregion_rs/Cargo.toml

# Correctness / interop harness (Rust output vs Python output):
uv run python bioregion_rs/harness.py
```

## Version pinning (important)

Python-side `polars` and Rust-side `polars` must stay ABI-compatible; bump them
together.

| Component | Version | Notes |
|---|---|---|
| Python `polars` | 1.37.0 | the target ABI |
| Rust `polars` | 0.54.4 | matches Python 1.37 |
| `pyo3-polars` | 0.27 | pins Rust polars to 0.54 |
| `pyo3` | 0.28 | abi3-py313 |
| `h3o` | 0.8 | pure-Rust H3; produces cell ids identical to `polars-h3` |

### Known toolchain constraint

polars 0.54.4's `lazy` feature set (`polars-lazy` / `polars-stream`) does **not**
compile on the current nightly toolchain due to internal feature-unification bugs.
Phase 0 therefore uses only the **eager DataFrame API**. This is sufficient for the
robust interop path (Arrow-FFI DataFrame transfer + the parquet stage boundary).
LazyFrame plan-passing (`PyLazyFrame`) is deferred — it is the version-sensitive
path the migration plan already flags, and stages hand off via parquet in the
meantime. Re-enable `lazy` (add `"lazy"` to the `polars` features and `pyo3-polars`,
restore a `PyLazyFrame` function) once a polars/toolchain combo builds it cleanly.

## `cargo test` + PyO3 note

`pyo3/extension-module` is set only in `[tool.maturin]`, not in the crate's default
features, because it suppresses libpython linking and breaks `cargo test`'s
standalone binary. If `cargo test` can't find a Python to link, pass
`PYO3_PYTHON=$(uv run which python)`.
