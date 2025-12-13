variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "region" {
  description = "The GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "The GCP zone"
  type        = string
  default     = "us-central1-a"
}

variable "instance_name" {
  description = "Name for the compute instance and container"
  type        = string
  default     = "marimo-ecoregions"
}

variable "machine_type" {
  description = "Machine type for the compute instance"
  type        = string
  default     = "c3-highmem-4"
}

variable "disk_size" {
  description = "Size of the persistent data disk in GB"
  type        = number
  default     = 400
}

variable "boot_disk_size" {
  description = "Size of the boot disk in GB"
  type        = number
  default     = 10
}
