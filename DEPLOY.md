# Deployment Guide: Git & Google Cloud Run

This project contains the backend (FastAPI, PSO, Sugeno FIS, DSP) and frontend (ICU Arrhythmia Pipeline web UI). It is fully packaged for one-click deployment to **Google Cloud Run**.

---

## 1. Push to Git / GitHub

Your local git repository is already initialized, configured, and committed on branch `main`.

Run the following commands in PowerShell (replace `<YOUR_GITHUB_REPO_URL>` with your repository link):

```powershell
# 1. Add your remote GitHub repository
git remote add origin <YOUR_GITHUB_REPO_URL>

# 2. Push all code to main branch
git push -u origin main
```

*Example repository URL:* `https://github.com/mokshithreddy635/biosig-fuzzy-icu.git`

---

## 2. Deploy to Google Cloud Run (Continuous Deployment)

Using Google Cloud Run connected directly to your GitHub repository provides automatic builds, free HTTPS, zero server maintenance, and automatic scaling to zero when idle.

### Step-by-Step Instructions:

1. **Log in to Google Cloud Console**:
   - Go to [https://console.cloud.google.com/run](https://console.cloud.google.com/run).
   - Select or create a Google Cloud Project (e.g. `biosig-icu-project`).

2. **Create Service**:
   - Click **Create Service** at the top.
   - Under **Deployment platform**, choose:
     - **"Continuously deploy from a repository"** (Cloud Build).
   - Click **Set Up Cloud Build**:
     - Provider: **GitHub**.
     - Authenticate and select your repository.
     - Branch: `^main$`.
     - Build Type: **Dockerfile** (located at `/Dockerfile`, already created in the repository root).
     - Click **Save**.

3. **Configure Service Settings**:
   - **Service Name**: `biosig-fuzzy-icu` (or your preferred name).
   - **Region**: Choose a region close to you (e.g. `us-central1`, `asia-south1`, etc.).
   - **Authentication**: Select **"Allow unauthenticated invocations"** (enables public access to the web dashboard and API).
   - **Port**: Set to `8080` (default in Cloud Run and configured in the Dockerfile).
   - **Memory**: 512 MiB or 1 GiB (sufficient for the pipeline).

4. **Click "Create"**:
   - Google Cloud will automatically pull your code, build the container image, and provision a public HTTPS URL:
     `https://biosig-fuzzy-icu-<hash>-<region>.a.run.app`

---

## 3. What You Get on Google Cloud

- **Web Dashboard**: Access the interactive ICU Arrhythmia Pipeline frontend directly from the root URL `/`.
- **API Documentation**: Interactive Swagger docs available at `/docs`.
- **WebSocket Streaming**: Real-time per-hop arrhythmia classification at `/stream`.
- **Automated Tests**: Direct API health and automated test validation at `/health` and `/tests`.
