#!/bin/bash

set -e

# Configuration
CONTAINER_NAME="marimo-ecoregions"
PROJECT_ID=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/project/project-id 2>/dev/null)
IMAGE_NAME="gcr.io/${PROJECT_ID}/${CONTAINER_NAME}"

# Wait for apt locks to be released (common on Ubuntu startup)
echo "Waiting for apt locks to be released..."
while sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 ; do
  echo "Waiting for other apt processes to finish..."
  sleep 5
done

# Update package list
echo "Updating package list..."
sudo apt-get update

# Install Docker if not already installed
if ! command -v docker &> /dev/null; then
  echo "Installing Docker..."
  sudo apt-get install -y ca-certificates curl gnupg
  sudo install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  sudo chmod a+r /etc/apt/keyrings/docker.gpg

  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

  sudo apt-get update
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

  echo "Docker installed successfully"
else
  echo "Docker already installed"
fi

# Install Ops Agent for Cloud Monitoring and Logging
echo "Installing Ops Agent..."
cd /tmp
curl -sSO https://dl.google.com/cloudagents/add-google-cloud-ops-agent-repo.sh
sudo bash add-google-cloud-ops-agent-repo.sh --also-install
rm add-google-cloud-ops-agent-repo.sh
echo "Ops Agent installation complete"

# Configure Docker authentication for GCR
echo "Configuring Docker authentication for GCR..."
sudo gcloud auth configure-docker gcr.io --quiet

# Enable incoming traffic on port 8080
echo "Configuring firewall..."
sudo iptables -A INPUT -p tcp --dport 8080 -j ACCEPT

# Stop and remove the container if it exists
echo "Cleaning up old containers..."
sudo docker stop $CONTAINER_NAME 2>/dev/null || true
sudo docker rm $CONTAINER_NAME 2>/dev/null || true

# Format and mount persistent disk
DEVICE_NAME="/dev/disk/by-id/google-data-disk"
MOUNT_POINT="/mnt/data"

echo "Setting up persistent disk..."
if [ -b "$DEVICE_NAME" ]; then
  echo "Persistent disk found at $DEVICE_NAME"

  # Check if the disk is already formatted
  if ! sudo blkid $DEVICE_NAME; then
    echo "Formatting disk..."
    sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard $DEVICE_NAME
  else
    echo "Disk already formatted."
  fi

  # Create mount point if it doesn't exist
  sudo mkdir -p $MOUNT_POINT

  # Mount the disk
  echo "Mounting disk at $MOUNT_POINT..."
  sudo mount -o discard,defaults $DEVICE_NAME $MOUNT_POINT

  # Set permissions
  sudo chmod a+w $MOUNT_POINT

  # Add to /etc/fstab for automatic mounting on reboot
  if ! grep -q $DEVICE_NAME /etc/fstab; then
    echo "Adding disk to /etc/fstab for automatic mounting..."
    echo "$DEVICE_NAME $MOUNT_POINT ext4 discard,defaults,nofail 0 2" | sudo tee -a /etc/fstab
  fi

  echo "Persistent disk mounted successfully at $MOUNT_POINT"
else
  echo "Warning: Persistent disk not found at $DEVICE_NAME"
fi

# Pull the latest version of the container image
echo "Pulling container image..."
sudo docker pull ${IMAGE_NAME}

# Run the container with the data volume mounted
echo "Starting container..."
sudo docker run \
  --name=$CONTAINER_NAME \
  --restart=always \
  --detach \
  --publish 8080:8080 \
  --volume /mnt/data:/data \
  --env DATA_DIR=/data \
  --log-driver=gcplogs \
  ${IMAGE_NAME}

echo "Container ${CONTAINER_NAME} started successfully"
echo "Startup script completed successfully"
