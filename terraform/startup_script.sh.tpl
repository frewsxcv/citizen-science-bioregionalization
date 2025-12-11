#!/bin/bash

set -e

# Configuration (templated by Terraform)
CONTAINER_NAME="${container_name}"
PROJECT_ID="${project_id}"
IMAGE_NAME="gcr.io/$${PROJECT_ID}/$${CONTAINER_NAME}"

echo "Starting Container-Optimized OS startup script..."

# Configure Docker authentication for GCR
# Note: COS has a read-only root filesystem, so we need to use a writable HOME directory
echo "Configuring Docker authentication for GCR..."
export HOME=/home/chronos
mkdir -p $HOME/.docker
docker-credential-gcr configure-docker --registries=gcr.io

# Stop and remove the container if it exists
echo "Cleaning up old containers..."
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# Format and mount persistent disk
DEVICE_NAME="/dev/disk/by-id/google-data-disk"
MOUNT_POINT="/mnt/disks/data"

echo "Setting up persistent disk..."
if [ -b "$DEVICE_NAME" ]; then
  echo "Persistent disk found at $DEVICE_NAME"

  # Check if the disk is already formatted
  if ! blkid $DEVICE_NAME; then
    echo "Formatting disk..."
    mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard $DEVICE_NAME
  else
    echo "Disk already formatted."
  fi

  # Create mount point if it doesn't exist
  mkdir -p $MOUNT_POINT

  # Mount the disk
  echo "Mounting disk at $MOUNT_POINT..."
  mount -o discard,defaults $DEVICE_NAME $MOUNT_POINT

  # Set permissions
  chmod a+w $MOUNT_POINT

  echo "Persistent disk mounted successfully at $MOUNT_POINT"
else
  echo "Warning: Persistent disk not found at $DEVICE_NAME"
fi

# Pull the latest version of the container image
echo "Pulling container image..."
docker pull $${IMAGE_NAME}:latest

# Run the container with the data volume mounted
echo "Starting container..."
docker run \
  --name=$CONTAINER_NAME \
  --restart=always \
  --detach \
  --publish 8080:8080 \
  --volume $MOUNT_POINT:/data \
  --env DATA_DIR=/data \
  --log-driver=gcplogs \
  $${IMAGE_NAME}:latest

echo "Container $${CONTAINER_NAME} started successfully"
echo "Startup script completed successfully"
