
1. Create GCP project
`gcloud projects create "gdpr-agent-api"`
2. Set up billing
set up on GCP
3. Set active project 
`gcloud config set project "gdpr-agent-api"`
4. Enable APIs
`gcloud services enable run.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com`                            
5. `terraform init`
6. `terraform apply` (creates Artifact Registry, Secret Manager, Cloud Run)
7. Add secrets (token and endpoint URL without trailing newlines)
`[System.Text.Encoding]::UTF8.GetBytes("TOKEN") | Set-Content -Path "token.tmp" -Encoding Byte; gcloud secrets versions add databricks-token --data-file=token.tmp; Remove-Item token.tmp`
`[System.Text.Encoding]::UTF8.GetBytes("URL") | Set-Content -Path "token.tmp" -Encoding Byte; gcloud secrets versions add databricks-endpoint-url --data-file=token.tmp; Remove-Item token.tmp`
`[System.Text.Encoding]::UTF8.GetBytes("api-key") | Set-Content -Path "token.tmp" -Encoding Byte; gcloud secrets versions add api-key --data-file=token.tmp; Remove-Item token.tmp`
8. Configure Docker auth with GCP
`gcloud auth configure-docker australia-southeast1-docker.pkg.dev`
9. Build Docker image
`cd ..\api` (CD into api folder)
`docker build -t australia-southeast1-docker.pkg.dev/gdpr-agent-api/gdpr-api/gdpr-api:latest .`
10. Push Docker image
`docker push australia-southeast1-docker.pkg.dev/gdpr-agent-api/gdpr-api/gdpr-api:latest`
11. Redeploy Cloud Run to pick up new secrets
`gcloud run services update gdpr-api --region=australia-southeast1 --image=australia-southeast1-docker.pkg.dev/gdpr-agent-api/gdpr-api/gdpr-api:latest`