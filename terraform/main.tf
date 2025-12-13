terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# Enable required APIs
resource "google_project_service" "compute" {
  service            = "compute.googleapis.com"
  disable_on_destroy = false
}





# Persistent disk for data storage
resource "google_compute_disk" "data_disk" {
  name = "${var.instance_name}-data"
  type = "pd-balanced"
  zone = var.zone
  size = var.disk_size

  labels = {
    environment = "production"
    app         = "marimo-ecoregions"
  }

  depends_on = [google_project_service.compute]
}

# Firewall rule for Marimo access
resource "google_compute_firewall" "allow_marimo" {
  name    = "allow-marimo"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["8080"]
  }

  target_tags   = ["marimo-server"]
  source_ranges = ["0.0.0.0/0"]
  description   = "Allow access to Marimo notebook on port 8080"

  depends_on = [google_project_service.compute]
}

# Compute Engine instance
resource "google_compute_instance" "marimo" {
  name         = var.instance_name
  machine_type = var.machine_type
  zone         = var.zone

  tags = ["marimo-server"]

  boot_disk {
    initialize_params {
      image = "cos-cloud/cos-stable"
      size  = 10
      type  = "pd-balanced"
    }
  }

  attached_disk {
    source      = google_compute_disk.data_disk.self_link
    device_name = "data-disk"
    mode        = "READ_WRITE"
  }

  network_interface {
    network = "default"
    access_config {
      # Ephemeral public IP
    }
  }

  metadata = {
    google-monitoring-enabled = "true"
    startup-script = templatefile("${path.module}/startup_script.sh.tpl", {
      container_name = var.instance_name
      project_id     = var.project_id
    })
  }

  service_account {
    scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]
  }

  # Allow stopping for updates
  allow_stopping_for_update = true

  depends_on = [
    google_compute_disk.data_disk,
    google_compute_firewall.allow_marimo,
    google_project_service.compute
  ]
}
