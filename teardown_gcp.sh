#!/bin/bash

set -e

PROJECT_ID=${GCP_PROJECT_ID:-$(gcloud config get-value project)}
INSTANCE_NAME="marimo-ecoregions"
ZONE="us-central1-a"

echo "Deleting Compute Engine instance..."
if gcloud compute instances describe ${INSTANCE_NAME} --zone=${ZONE} &>/dev/null; then
  gcloud compute instances delete ${INSTANCE_NAME} --zone=${ZONE} --quiet
  echo "Instance deleted."
else
  echo "Instance ${INSTANCE_NAME} not found."
fi

echo "Deleting firewall rule..."
if gcloud compute firewall-rules describe allow-marimo &>/dev/null; then
  gcloud compute firewall-rules delete allow-marimo --quiet
  echo "Firewall rule deleted."
else
  echo "Firewall rule allow-marimo not found."
fi

echo "Teardown complete!"
