#!/bin/bash

set -e

PROJECT_ID=${GCP_PROJECT_ID:-$(gcloud config get-value project)}
INSTANCE_NAME="marimo-ecoregions"
ZONE="us-central1-a"
MACHINE_TYPE="e2-standard-2"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${INSTANCE_NAME}"

echo "Building and pushing Docker image..."
gcloud builds submit --tag ${IMAGE_NAME}

echo "Checking if instance exists..."
if gcloud compute instances describe ${INSTANCE_NAME} --zone=${ZONE} &>/dev/null; then
  echo "Instance exists. Stopping and deleting..."
  gcloud compute instances delete ${INSTANCE_NAME} --zone=${ZONE} --quiet
fi

echo "Creating firewall rule if it doesn't exist..."
if ! gcloud compute firewall-rules describe allow-marimo &>/dev/null; then
  gcloud compute firewall-rules create allow-marimo \
    --allow=tcp:8080 \
    --target-tags=marimo-server \
    --description="Allow access to Marimo notebook on port 8080"
fi

echo "Creating Compute Engine instance with container..."
gcloud compute instances create-with-container ${INSTANCE_NAME} \
  --zone=${ZONE} \
  --machine-type=${MACHINE_TYPE} \
  --tags=marimo-server \
  --container-image=${IMAGE_NAME} \
  --container-restart-policy=always \
  --boot-disk-size=20GB

echo "Deployment complete!"
echo "Getting external IP address..."
EXTERNAL_IP=$(gcloud compute instances describe ${INSTANCE_NAME} --zone=${ZONE} --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
echo ""
echo "Your Marimo notebook is available at: http://${EXTERNAL_IP}:8080"
echo ""
echo "Note: It may take 1-2 minutes for the container to start."
echo "To view logs: gcloud compute instances get-serial-port-output ${INSTANCE_NAME} --zone=${ZONE}"
