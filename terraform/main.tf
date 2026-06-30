terraform {
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
}

resource "google_artifact_registry_repository" "gdpr_api" {
  location      = var.region
  repository_id = "gdpr-api"
  format        = "DOCKER"
}

resource "google_secret_manager_secret" "databricks_token" {
  secret_id = "databricks-token"
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret" "databricks_endpoint_url" {
  secret_id = "databricks-endpoint-url"
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret" "api_key" {
  secret_id = "api-key"
  replication {
    auto {}
  }
}

resource "google_cloud_run_v2_service" "gdpr_api" {
  name     = "gdpr-api"
  location = var.region

  template {
    containers {
      image = "australia-southeast1-docker.pkg.dev/gdpr-agent-api/gdpr-api/gdpr-api:latest"

      env {
        name = "DATABRICKS_TOKEN"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.databricks_token.secret_id
            version = "latest"
          }
        }
      }

      env {
        name = "DATABRICKS_ENDPOINT_URL"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.databricks_endpoint_url.secret_id
            version = "latest"
          }
        }
      }
      env {
        name = "API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.api_key.secret_id
            version = "latest"
          }
        }
      }
    }
  }
}

resource "google_secret_manager_secret_iam_member" "databricks_token_access" {
  secret_id = google_secret_manager_secret.databricks_token.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:420993726930-compute@developer.gserviceaccount.com"
}

resource "google_secret_manager_secret_iam_member" "databricks_endpoint_access" {
  secret_id = google_secret_manager_secret.databricks_endpoint_url.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:420993726930-compute@developer.gserviceaccount.com"
}

resource "google_secret_manager_secret_iam_member" "api_key" {
  secret_id = google_secret_manager_secret.api_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:420993726930-compute@developer.gserviceaccount.com"
}

resource "google_cloud_run_v2_service_iam_member" "public_access" {
  name     = google_cloud_run_v2_service.gdpr_api.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}
