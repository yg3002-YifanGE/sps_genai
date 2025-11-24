# Docker Deployment Guide - Assignment 5

## Quick Start

### Build the Docker Image

```bash
cd /Users/geyifan/sps_genai
docker build -t sps-genai:latest .
```

### Run the Container

```bash
docker run -d -p 8000:8000 --name sps-genai sps-genai:latest
```

Access the API at: **http://localhost:8000/docs**

---

## Assignment 5: GPT-2 Model Setup

The Docker image includes all code and dependencies for Assignment 5, but **not the trained model weights** (500MB file is too large for Docker image).

### Option 1: Train Model in Container (Quick Test)

**Recommended for demonstration:**

```bash
# Quick training (1,000 samples, ~50 minutes)
docker exec -it sps-genai python assignment5/train_gpt2_squad.py --epochs 1 --num_samples 1000 --batch_size 4

# After training, test the model
docker exec -it sps-genai python assignment5/train_gpt2_squad.py --test_only
```

### Option 2: Copy Pre-trained Model into Container

If you've already trained the model locally:

```bash
# Copy the model weights into the running container
docker cp models/gpt2_squad_finetuned/model.safetensors sps-genai:/code/models/gpt2_squad_finetuned/

# Restart the container to pick up the model
docker restart sps-genai
```

### Option 3: Use Volume Mount

Mount your local models directory:

```bash
docker run -d -p 8000:8000 \
  -v $(pwd)/models:/code/models \
  --name sps-genai \
  sps-genai:latest
```

---

## Testing the API

### Check API Status

```bash
curl http://localhost:8000/
```

### Test GPT-2 Endpoint (after training)

```bash
curl -X POST "http://localhost:8000/gpt2/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is machine learning?",
    "max_length": 150,
    "temperature": 0.7
  }'
```

### Interactive Documentation

Visit: **http://localhost:8000/docs**

---

## Available Endpoints

All previous assignments' endpoints are available:

| Endpoint | Assignment | Status |
|----------|------------|--------|
| `/generate` | 1 | ✅ Ready |
| `/embed` | 1 | ✅ Ready |
| `/similar` | 1 | ✅ Ready |
| `/predict_cifar10` | 2 | ✅ Ready (if weights available) |
| `/gan/generate` | 3 | ✅ Ready (if weights available) |
| `/ebm/generate` | 4 | ✅ Ready (if weights available) |
| `/diffusion/generate` | 4 | ✅ Ready (if weights available) |
| `/gpt2/answer` | 5 | ⚠️ Requires training |
| `/gpt2/answer/multiple` | 5 | ⚠️ Requires training |

---

## Container Management

### View Logs

```bash
docker logs -f sps-genai
```

### Access Container Shell

```bash
docker exec -it sps-genai bash
```

### Stop Container

```bash
docker stop sps-genai
```

### Remove Container

```bash
docker rm sps-genai
```

### Remove Image

```bash
docker rmi sps-genai:latest
```

---

## Troubleshooting

### Issue: GPT-2 endpoints return 500 error

**Cause:** Model not trained yet

**Solution:**
```bash
# Train the model in the container
docker exec -it sps-genai python assignment5/train_gpt2_squad.py --epochs 1 --num_samples 1000
```

### Issue: Out of memory during training

**Cause:** Container memory limit

**Solution:**
```bash
# Increase Docker memory limit (Docker Desktop settings)
# Or use smaller sample size
docker exec -it sps-genai python assignment5/train_gpt2_squad.py --epochs 1 --num_samples 500 --batch_size 2
```

### Issue: Container fails to start

**Cause:** Port 8000 already in use

**Solution:**
```bash
# Use a different port
docker run -d -p 8080:8000 --name sps-genai sps-genai:latest
# Access at http://localhost:8080
```

---

## Production Deployment Notes

For production deployment (not required for assignment):

### Option 1: Build with Pre-trained Model

Uncomment this line in Dockerfile if you want to include the model:

```dockerfile
# COPY ./models/gpt2_squad_finetuned/model.safetensors /code/models/gpt2_squad_finetuned/
```

**Pros:** Model ready immediately  
**Cons:** Large image size (~1GB)

### Option 2: Download Model on Startup

Add to startup script:

```bash
# Download from S3/cloud storage
# aws s3 cp s3://bucket/model.safetensors /code/models/gpt2_squad_finetuned/
```

### Option 3: Use External Volume

Best practice for production:

```bash
docker run -d -p 8000:8000 \
  -v /path/to/models:/code/models \
  --name sps-genai \
  sps-genai:latest
```

---

## Assignment Submission Notes

### What to Document

For the assignment, document that:

1. **Docker image builds successfully** ✅
2. **API runs in container** ✅
3. **GPT-2 model can be trained in container** ✅
4. **All endpoints accessible** ✅

### Example Documentation Text

> "The application has been containerized with Docker. All dependencies including transformers and datasets are included. The GPT-2 model can be trained within the container using:
> 
> ```bash
> docker exec -it sps-genai python assignment5/train_gpt2_squad.py --epochs 1 --num_samples 1000
> ```
> 
> Due to the model weight file size (500MB), it is not included in the Docker image by default, following best practices for container size management. See DOCKER_GUIDE.md for deployment options."

---

## Quick Reference

```bash
# Build
docker build -t sps-genai .

# Run
docker run -d -p 8000:8000 --name sps-genai sps-genai:latest

# Train GPT-2 (quick test)
docker exec -it sps-genai python assignment5/train_gpt2_squad.py --epochs 1 --num_samples 1000

# Test
curl http://localhost:8000/

# Logs
docker logs -f sps-genai

# Stop
docker stop sps-genai && docker rm sps-genai
```

---

## Summary

✅ Docker fully supports Assignment 5  
✅ Model training can be done in container  
✅ All endpoints functional  
✅ Production-ready architecture  

