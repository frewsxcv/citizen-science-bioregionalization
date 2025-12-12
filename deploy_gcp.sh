#!/bin/bash

set -e

PROJECT_ID=${GCP_PROJECT_ID:-$(gcloud config get-value project)}
SERVICE_NAME="marimo-ecoregions"
REGION="us-central1"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "Building and pushing Docker image..."
gcloud builds submit --tag ${IMAGE_NAME}

echo "Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_NAME} \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --port 8080 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 3600 \
  --max-instances 10 \
  --min-instances 1 \
  --no-cpu-throttling \
  --session-affinity \
  --no-deploy-health-check

echo "Deployment complete!"
gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format='value(status.url)'
