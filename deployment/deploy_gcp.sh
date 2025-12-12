#!/bin/bash

set -e

PROJECT_ID=${GCP_PROJECT_ID:-$(gcloud config get-value project)}
INSTANCE_NAME="marimo-ecoregions"
ZONE="us-central1-a"
MACHINE_TYPE="e2-standard-2"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${INSTANCE_NAME}"
DISK_NAME="${INSTANCE_NAME}-data"
DISK_SIZE="400"

echo "Building and pushing Docker image..."
gcloud builds submit deployment/Dockerfile --tag ${IMAGE_NAME}

echo "Checking if instance exists..."
if gcloud compute instances describe ${INSTANCE_NAME} --zone=${ZONE} &>/dev/null; then
  echo "Instance exists. Stopping and deleting..."
  gcloud compute instances delete ${INSTANCE_NAME} --zone=${ZONE} --quiet
fi

echo "Creating persistent disk if it doesn't exist..."
if ! gcloud compute disks describe ${DISK_NAME} --zone=${ZONE} &>/dev/null; then
  echo "Creating ${DISK_SIZE}GB persistent disk..."
  gcloud compute disks create ${DISK_NAME} \
    --zone=${ZONE} \
    --size=${DISK_SIZE}GB \
    --type=pd-standard
  echo "Disk created successfully."
else
  echo "Disk ${DISK_NAME} already exists."
fi

echo "Creating firewall rule if it doesn't exist..."
if ! gcloud compute firewall-rules describe allow-marimo &>/dev/null; then
  gcloud compute firewall-rules create allow-marimo \
    --allow=tcp:8080 \
    --target-tags=marimo-server \
    --description="Allow access to Marimo notebook on port 8080"
fi

echo "Creating Compute Engine instance with Ubuntu 22.04 LTS..."
gcloud compute instances create ${INSTANCE_NAME} \
  --zone=${ZONE} \
  --machine-type=${MACHINE_TYPE} \
  --tags=marimo-server \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=10GB \
  --disk=name=${DISK_NAME},device-name=data-disk,mode=rw,boot=no \
  --metadata-from-file=startup-script=./startup_script.sh

echo "Deployment complete!"
echo "Getting external IP address..."
EXTERNAL_IP=$(gcloud compute instances describe ${INSTANCE_NAME} --zone=${ZONE} --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
echo ""
echo "Your Marimo notebook is available at: http://${EXTERNAL_IP}:8080"
echo ""
echo "Note: It may take 1-2 minutes for the container to start."
echo "To view startup script logs: gcloud compute instances get-serial-port-output ${INSTANCE_NAME} --zone=${ZONE} | grep startup-script"
echo "To view container logs: gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --command='docker logs marimo-ecoregions'"
