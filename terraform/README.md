# Terraform Deployment for Citizen Science Bioregionalization

This directory contains Terraform configuration to deploy the Marimo ecoregions notebook to Google Cloud Platform.

## Prerequisites

1. [Terraform](https://developer.hashicorp.com/terraform/downloads) >= 1.0 installed
2. [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed and authenticated
3. A GCP project with billing enabled

## Setup

### 1. Configure Variables

Copy the example variables file and fill in your values:

```bash
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars` with your settings:

```hcl
project_id = "your-gcp-project-id"
```

### 2. Build the Initial Docker Image

Before deploying the infrastructure, you need to build and push the Docker image:

```bash
./initial_build.sh
```

This will build the container and push it to Google Container Registry.

### 3. Initialize Terraform

```bash
terraform init
```

### 4. Review the Plan

```bash
terraform plan
```

This will show you what resources will be created:
- Persistent disk for data storage
- Firewall rule for port 8080
- Compute Engine instance running the Marimo notebook

### 5. Deploy

```bash
terraform apply
```

Type `yes` when prompted to create the resources.

## Usage

After deployment, Terraform will output:

- **marimo_url**: The URL to access your Marimo notebook
- **ssh_command**: Command to SSH into the instance
- **logs_command**: Command to view container logs

Note: It may take 2-3 minutes after deployment for the container to start.

## Deploying Updates

To deploy a new version of the application:

1. Build and push the new image:
   ```bash
   ./initial_build.sh
   ```

2. Restart the instance to pick up the new image:
   ```bash
   gcloud compute instances reset marimo-ecoregions --zone=us-central1-a
   ```

Or SSH into the instance and manually pull/restart the container:
```bash
gcloud compute ssh marimo-ecoregions --zone=us-central1-a
# Then on the instance:
sudo docker pull gcr.io/YOUR_PROJECT_ID/marimo-ecoregions:latest
sudo docker stop marimo-ecoregions
sudo docker rm marimo-ecoregions
# The startup script will restart it, or you can run it manually
```

## Teardown

To destroy all resources:

```bash
terraform destroy
```

**Warning**: This will delete the persistent disk and all data stored on it.

To destroy only the compute instance (preserving the disk and data):

```bash
terraform destroy -target=google_compute_instance.marimo
```

## Troubleshooting

### View startup script logs

```bash
gcloud compute instances get-serial-port-output marimo-ecoregions --zone=us-central1-a
```

### View container logs

```bash
gcloud compute ssh marimo-ecoregions --zone=us-central1-a --command='docker logs marimo-ecoregions'
```

### SSH into the instance

```bash
gcloud compute ssh marimo-ecoregions --zone=us-central1-a
```

### Check if container is running

```bash
gcloud compute ssh marimo-ecoregions --zone=us-central1-a --command='docker ps'
```

## Files

| File | Description |
|------|-------------|
| `main.tf` | Main infrastructure configuration |
| `variables.tf` | Variable definitions |
| `outputs.tf` | Output values |
| `startup_script.sh.tpl` | Templated startup script for the VM |
| `terraform.tfvars.example` | Example variables file |
| `initial_build.sh` | Script to build and push Docker image |