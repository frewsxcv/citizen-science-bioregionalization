#!/bin/bash
#
# Initial Build Script
# ====================
# Run this script before your first `terraform apply` to ensure
# a Docker image exists in Google Container Registry.
#
# The Cloud Build trigger will handle subsequent builds automatically
# when you push to the main branch.
#

set -e

# Change to repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load project ID from terraform.tfvars if it exists
if [ -f "$SCRIPT_DIR/terraform.tfvars" ]; then
  PROJECT_ID=$(grep -E '^project_id\s*=' "$SCRIPT_DIR/terraform.tfvars" | sed 's/.*=\s*"\(.*\)"/\1/' | tr -d ' ')
fi

# Fall back to gcloud config
if [ -z "$PROJECT_ID" ]; then
  PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
fi

if [ -z "$PROJECT_ID" ]; then
  echo "Error: Could not determine project ID."
  echo "Either set up terraform.tfvars or run: gcloud config set project YOUR_PROJECT_ID"
  exit 1
fi

INSTANCE_NAME="marimo-ecoregions"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${INSTANCE_NAME}"

echo "============================================"
echo "Initial Docker Image Build"
echo "============================================"
echo "Project ID:  $PROJECT_ID"
echo "Image Name:  $IMAGE_NAME"
echo "Build Context: $REPO_ROOT"
echo "============================================"
echo ""

read -p "Continue with build? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "Cancelled."
  exit 0
fi

echo ""
echo "Building and pushing Docker image..."
echo ""

cd "$REPO_ROOT"
gcloud builds submit \
  --tag "${IMAGE_NAME}:latest"

echo ""
echo "Adding 'initial' tag..."
gcloud container images add-tag \
  "${IMAGE_NAME}:latest" \
  "${IMAGE_NAME}:initial" \
  --quiet

echo ""
echo "============================================"
echo "Build complete!"
echo "============================================"
echo ""
echo "Image pushed to: ${IMAGE_NAME}:latest"
echo ""
echo "You can now run:"
echo "  cd $SCRIPT_DIR"
echo "  terraform apply"
echo ""
