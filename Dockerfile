# syntax=docker/dockerfile:1.4

# Choose Python 3.13 to match the project requirements
FROM python:3.13-slim

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:0.4.20 /uv /bin/uv
ENV UV_SYSTEM_PYTHON=1
ENV GOOGLE_APPLICATION_CREDENTIALS=/home/app_user/.config/gcloud/application_default_credentials.json

WORKDIR /app

# Copy dependency files
COPY --link pyproject.toml .

# Install the requirements using uv
RUN uv pip install -e .

# Copy application files
COPY --link notebook.py .

# Copy any additional files that might be needed
COPY --link ocean.geojson .

# Copy src directory with all modules
COPY --link src/ ./src/

# Create necessary directories
RUN mkdir -p data output tmp

EXPOSE 8080

# Create a non-root user and switch to it
RUN useradd -m app_user && \
    chown -R app_user:app_user /app && \
    mkdir -p /home/app_user/.config/marimo && \
    chown -R app_user:app_user /home/app_user/.config
USER app_user

CMD [ "marimo", "run", "notebook.py", "--host", "0.0.0.0", "-p", "8080" ]
