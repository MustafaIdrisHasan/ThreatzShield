# Deployment Guide

This guide covers deploying ThreatzShield to various cloud platforms.

## Prerequisites

- GitHub repository with your code
- Account on chosen platform
- Docker (for local testing)

---

## 🚂 Railway Deployment

1. **Create Account**
   - Go to [railway.app](https://railway.app)
   - Sign up with GitHub

2. **Deploy**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your ThreatzShield repository
   - Railway auto-detects Dockerfile

3. **Configure**
   - Environment variables (if needed):
     - `PORT` (Railway auto-sets this)
   - Wait for build to complete (~5-10 minutes)

4. **Access**
   - Railway provides a URL: `https://your-app.up.railway.app`
   - API available at: `https://your-app.up.railway.app/predict`

**Cost:** Free tier available, pay-as-you-go

---

## 🎨 Render Deployment

1. **Create Account**
   - Go to [render.com](https://render.com)
   - Sign up with GitHub

2. **New Web Service**
   - Click "New" → "Web Service"
   - Connect your GitHub repository

3. **Settings**
   - **Name:** `threatzshield` (or your choice)
   - **Environment:** `Docker`
   - **Region:** Choose closest
   - **Branch:** `main` (or your default)
   - **Dockerfile Path:** `./Dockerfile`

4. **Advanced Settings (Optional)**
   - Auto-deploy: ON
   - Health Check Path: `/health`

5. **Deploy**
   - Click "Create Web Service"
   - Wait for build (~5-10 minutes)

**Cost:** Free tier available, $7/month for always-on

---

## 🚀 Heroku Deployment

### Prerequisites
```bash
# Install Heroku CLI
# https://devcenter.heroku.com/articles/heroku-cli
```

### Steps
1. **Login**
   ```bash
   heroku login
   ```

2. **Create App**
   ```bash
   heroku create threatzshield-app
   ```

3. **Add Buildpack**
   ```bash
   heroku buildpacks:set heroku/python
   ```

4. **Create Procfile**
   Create `Procfile` in root:
   ```
   web: uvicorn api:app --host 0.0.0.0 --port $PORT
   ```

5. **Deploy**
   ```bash
   git push heroku main
   ```

6. **Open**
   ```bash
   heroku open
   ```

**Cost:** Free tier discontinued, starts at $7/month

---

## ☁️ AWS Deployment (EC2/ECS)

### Option 1: EC2 with Docker

1. **Launch EC2 Instance**
   - Launch EC2 instance (Ubuntu)
   - Security group: Allow port 8000

2. **SSH into Instance**
   ```bash
   ssh -i your-key.pem ubuntu@your-ec2-ip
   ```

3. **Install Docker**
   ```bash
   sudo apt-get update
   sudo apt-get install docker.io -y
   sudo usermod -aG docker ubuntu
   ```

4. **Deploy**
   ```bash
   git clone https://github.com/your-username/ThreatzShield.git
   cd ThreatzShield
   docker build -t threatzshield .
   docker run -d -p 8000:8000 threatzshield
   ```

### Option 2: ECS (Container Service)

1. **Push to ECR**
   ```bash
   aws ecr create-repository --repository-name threatzshield
   docker tag threatzshield:latest your-account.dkr.ecr.region.amazonaws.com/threatzshield:latest
   docker push your-account.dkr.ecr.region.amazonaws.com/threatzshield:latest
   ```

2. **Create ECS Task Definition**
   - Use ECS Console or CLI
   - Configure container image from ECR
   - Set port 8000

3. **Deploy Service**
   - Create ECS Service
   - Configure load balancer
   - Deploy!

---

## 🐳 Docker Build & Push (Generic)

### Build Locally
```bash
# Build
docker build -t threatzshield:latest .

# Test locally
docker run -p 8000:8000 threatzshield

# Access
curl http://localhost:8000/health
```

### Push to Registry

**Docker Hub:**
```bash
docker tag threatzshield:latest your-username/threatzshield:latest
docker login
docker push your-username/threatzshield:latest
```

**GitHub Container Registry:**
```bash
docker tag threatzshield:latest ghcr.io/your-username/threatzshield:latest
docker login ghcr.io
docker push ghcr.io/your-username/threatzshield:latest
```

---

## 🔧 Environment Variables

Set these in your deployment platform:

- `HATE_MODEL_DIR` (optional) - Path to local BERT model
- `PORT` (usually auto-set by platform)
- `API_HOST` (default: 0.0.0.0)

---

## 📝 Post-Deployment

1. **Test Health Endpoint**
   ```bash
   curl https://your-app-url.com/health
   ```

2. **Test Prediction**
   ```bash
   curl -X POST https://your-app-url.com/predict \
     -H "Content-Type: application/json" \
     -d '{"text":"Hello world"}'
   ```

3. **Update Frontend**
   - Update `API_BASE` in `frontend/index.html`:
     ```javascript
     const API_BASE = 'https://your-app-url.com';
     ```

4. **Monitor Logs**
   - Check platform logs for errors
   - Monitor response times

---

## 🐛 Troubleshooting

### Common Issues

**1. Port Already in Use**
- Check if port is already bound
- Change port in Dockerfile or platform settings

**2. Model Loading Timeout**
- First request may timeout (>30s)
- Increase timeout in platform settings

**3. Memory Issues**
- Models are large, may need 2GB+ RAM
- Upgrade instance size if needed

**4. CORS Errors**
- API has CORS enabled for all origins
- Check if frontend URL matches allowed origins

---

## 📊 Monitoring

### Health Checks
- Platform should ping `/health` endpoint
- Configure health check interval (30s recommended)

### Metrics to Monitor
- Response time (P50, P95)
- Error rate
- Memory usage
- CPU usage

---

## 🔐 Security

### Production Recommendations

1. **Rate Limiting**
   - Add rate limiting middleware
   - Prevent abuse

2. **Authentication**
   - Add API keys if needed
   - Use OAuth for user-facing endpoints

3. **HTTPS**
   - All platforms provide HTTPS
   - Ensure frontend uses HTTPS

4. **Environment Secrets**
   - Don't commit secrets
   - Use platform secret management

---

## 📚 Additional Resources

- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Railway Docs](https://docs.railway.app/)
- [Render Docs](https://render.com/docs)

---

**Need Help?** Open an issue on GitHub!


