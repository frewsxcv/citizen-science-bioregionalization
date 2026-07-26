# Two stages: the builder compiles the bioregion_rs extension module and
# resolves the dependency graph; the runtime image carries only the resulting
# virtualenv, so it needs neither the Rust sources nor a Rust toolchain.

# Choose Python 3.13 to match the project requirements
FROM python:3.13-slim AS builder

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /bin/uv

# maturin builds bioregion_rs from source, which needs a C toolchain and rustc.
# The crate is on Rust edition 2024, so the toolchain must be recent; `stable`
# matches what the bioregion_rs CI job uses.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# uv.lock pins the full graph; README.md is referenced by [project].readme and
# so must be present for metadata to resolve.
COPY pyproject.toml uv.lock README.md ./
COPY bioregion_rs/ ./bioregion_rs/

# The root project is virtual (see `source = { virtual = "." }` in uv.lock), so
# it is never installed itself -- only its dependencies are. --no-editable
# installs bioregion_rs as a built wheel rather than a link back to the source
# tree, which is what lets the runtime stage drop the Rust sources entirely.
RUN uv sync --frozen --no-dev --no-editable


FROM python:3.13-slim

ENV GOOGLE_APPLICATION_CREDENTIALS=/home/app_user/.config/gcloud/application_default_credentials.json

# Configure jemalloc to aggressively release memory back to the OS
# This helps reduce memory usage with Polars join operations
# See: https://github.com/pola-rs/polars/issues/25768
ENV _RJEM_MALLOC_CONF=background_thread:true,dirty_decay_ms:0,muzzy_decay_ms:0

WORKDIR /app

# A virtualenv bakes in its own absolute path, so it has to land at the same
# location it was created at in the builder stage.
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

# Copy application files
COPY notebook.py .

# Copy src directory with all modules (includes src/data/taxon_keys.json, the
# offline GBIF backbone key registry used for taxonomic scoping)
COPY src/ ./src/

# Create necessary directories
RUN mkdir -p data output tmp

EXPOSE 8080

# Create a non-root user and switch to it
RUN useradd -m app_user && \
    chown -R app_user:app_user /app && \
    mkdir -p /home/app_user/.config/marimo && \
    chown -R app_user:app_user /home/app_user/.config
USER app_user

CMD [ "marimo", "edit", "notebook.py", "--host", "0.0.0.0", "-p", "8080", "--token", "--token-password", "morusalba" ]
