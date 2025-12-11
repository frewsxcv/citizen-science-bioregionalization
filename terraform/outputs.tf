output "instance_ip" {
  description = "The external IP address of the Marimo instance"
  value       = google_compute_instance.marimo.network_interface[0].access_config[0].nat_ip
}

output "marimo_url" {
  description = "URL to access the Marimo notebook"
  value       = "http://${google_compute_instance.marimo.network_interface[0].access_config[0].nat_ip}:8080"
}

output "ssh_command" {
  description = "Command to SSH into the instance"
  value       = "gcloud compute ssh ${var.instance_name} --zone=${var.zone}"
}

output "logs_command" {
  description = "Command to view container logs"
  value       = "gcloud compute ssh ${var.instance_name} --zone=${var.zone} --command='docker logs ${var.instance_name}'"
}

output "serial_logs_command" {
  description = "Command to view startup script logs"
  value       = "gcloud compute instances get-serial-port-output ${var.instance_name} --zone=${var.zone}"
}
