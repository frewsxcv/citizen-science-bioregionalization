#!/bin/sh
docker build -t citizen-science-ecoregions . && \
docker stop marimo-notebook 2>/dev/null || true && \
docker rm marimo-notebook 2>/dev/null || true && \
docker run -d -p 8080:8080 --name marimo-notebook \
  -v ~/.config/gcloud/application_default_credentials.json:/home/app_user/.config/gcloud/application_default_credentials.json:ro \
  citizen-science-ecoregions && \
echo "Container started! View logs with: docker logs -f marimo-notebook" && \
echo "Access at: http://localhost:8080"
