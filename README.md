# SPS GenAI - Assignments 1-4

**Columbia University – APANPS5900 Applied Generative AI**  
**Author:** Yifan Ge (yg3002)  
**Repository:** https://github.com/yg3002-YifanGE/sps_genai

---

## 📋 Project Overview

This project implements multiple generative AI models and integrates them into a unified FastAPI web service. The project covers assignments 1-4 of the course, progressively building a comprehensive generative AI system.

---

## 🎯 Assignments Summary

### Assignment 1: Text Generation & Embeddings
- **Bigram text generation model**
- **SpaCy word embeddings** (en_core_web_md)
- **Word similarity search** using cosine similarity
- **FastAPI integration** with interactive documentation

**Key Endpoints:**
- `/generate` - Generate text using bigram probabilities
- `/embed` - Get word embedding vectors
- `/similar` - Find similar words based on embeddings

### Assignment 2: CNN Image Classification
- **CNN_A2 architecture** for image classification
- **CIFAR-10 dataset** (10 classes: airplane, car, bird, etc.)
- **Training and evaluation** with 64×64 RGB images
- **Model deployment** via API endpoint

**Key Features:**
- 2 Convolutional layers with MaxPooling
- 2 Fully connected layers
- Accuracy: ~70% on CIFAR-10 test set

**Key Endpoint:**
- `/predict_cifar10` - Upload image for classification

### Assignment 3: GAN for Image Generation
- **Generative Adversarial Network (GAN)** on MNIST
- **Generator:** Noise → 28×28 grayscale images
- **Discriminator:** Binary classifier (real/fake)
- **BCE loss** with Adam optimizer

**Architecture:**
- Generator: FC → ConvTranspose layers → Tanh
- Discriminator: Conv layers → FC → Sigmoid
- Latent dimension: 100

**Key Endpoint:**
- `/gan/generate` - Generate handwritten digits

### Assignment 4: Energy-Based Model & Diffusion Model
- **Energy-Based Model (EBM)** on CIFAR-10
- **Diffusion Model** with UNet on CIFAR-10
- **Fine-grained gradient control** with `torch.autograd.grad()`
- **RGB image generation** (32×32×3)

**EBM Features:**
- Langevin dynamics for sampling low-energy states
- Contrastive divergence training
- Gradient descent on **input images** (not parameters)

**Diffusion Features:**
- UNet architecture with skip connections
- Offset cosine diffusion schedule
- Reverse diffusion for generation
- Predicts noise (not images directly)

**Key Endpoints:**
- `/ebm/generate` - Generate images using EBM
- `/diffusion/generate` - Generate images using Diffusion

---

## 🏗️ Project Structure

```
sps_genai/
├── app/
│   ├── main.py              # FastAPI application with all endpoints
│   └── bigram_model.py      # Text generation model
│
├── helper_lib/
│   ├── model.py             # All models: CNN_A2, VAE, GAN, EBM, UNet
│   ├── trainer.py           # Training functions for all models
│   ├── generator.py         # Sample generation utilities
│   ├── data_loader.py       # Dataset loaders
│   └── utils.py
│
├── data/
│   ├── cifar-10-batches-py/ # CIFAR-10 dataset
│   ├── gan_G.pth            # GAN generator weights
│   ├── ebm_cifar10.pth      # EBM model weights
│   └── diffusion_cifar10.pth # Diffusion model weights
│
├── assignment2/
│   └── cnn_a2_cifar10.pt    # CNN classifier weights
│
├── train_ebm_cifar10.py     # EBM training script
├── train_diffusion_cifar10.py # Diffusion training script
├── train_gan_offline.py     # GAN training script
├── test_api.py              # API testing script
├── Dockerfile               # Docker configuration
├── pyproject.toml           # Dependencies (uv package manager)
└── requirements.txt         # pip dependencies
```

---

## 🚀 Quick Start

### 1. Install Dependencies

**Using uv (recommended):**
```bash
uv sync
uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.7.1/en_core_web_md-3.7.1-py3-none-any.whl
```

**Using pip:**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

### 2. Train Models (if needed)

```bash
# Train GAN (Assignment 3)
python train_gan_offline.py

# Train EBM (Assignment 4)
python train_ebm_cifar10.py

# Train Diffusion (Assignment 4)
python train_diffusion_cifar10.py
```

### 3. Run API Server

```bash
uvicorn app.main:app --port 8000 --reload
```

Access interactive documentation: **http://localhost:8000/docs**

### 4. Test API

```bash
python test_api.py
```

---

## 🌐 API Endpoints

| Endpoint | Method | Description | Assignment |
|----------|--------|-------------|-----------|
| `/` | GET | API status and endpoint list | - |
| `/generate` | POST | Text generation (bigram) | 1 |
| `/embed` | POST | Word embedding vector | 1 |
| `/similar` | POST | Find similar words | 1 |
| `/predict_cifar10` | POST | CIFAR-10 image classification | 2 |
| `/gan/generate` | GET | Generate MNIST digits | 3 |
| `/ebm/generate` | GET | Generate CIFAR-10 images (EBM) | 4 |
| `/diffusion/generate` | GET | Generate CIFAR-10 images (Diffusion) | 4 |

### Example Usage

**Text Generation:**
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"start_word": "The", "length": 10}'
```

**EBM Generation:**
```bash
curl "http://localhost:8000/ebm/generate?num_samples=16&steps=256"
```

**Diffusion Generation:**
```bash
curl "http://localhost:8000/diffusion/generate?num_samples=16&diffusion_steps=50"
```

---

## 🐳 Docker Deployment

### Build Image
```bash
docker build -t sps-genai:latest .
```

### Run Container
```bash
docker run -p 8000:8000 sps-genai:latest
```

Access at: **http://localhost:8000/docs**

---

## 🔑 Key Implementation Details

### Assignment 4: Fine-Grained Gradient Control

**EBM Sampling (Langevin Dynamics):**
```python
inp_imgs.requires_grad_(True)
energy = model(inp_imgs)
# Compute gradients w.r.t. INPUT (not model parameters)
grads, = torch.autograd.grad(energy, inp_imgs, grad_outputs=torch.ones_like(energy))
# Gradient descent on input to find low-energy states
inp_imgs = inp_imgs - step_size * grads
```

**Diffusion Training:**
```python
# Model predicts the NOISE, not the image
pred_noises = model(noisy_images, noise_rates ** 2)
loss = criterion(pred_noises, noises)  # Compare predicted vs true noise
```

---

## 📊 Model Specifications

### CNN_A2 (Assignment 2)
- Input: 64×64×3 RGB images
- Architecture: Conv(3→16) → Pool → Conv(16→32) → Pool → FC(100) → FC(10)
- Dataset: CIFAR-10
- Output: 10 class probabilities

### GAN (Assignment 3)
- Generator: Latent(100) → FC → ConvTranspose → 28×28×1
- Discriminator: 28×28×1 → Conv → FC → Sigmoid
- Dataset: MNIST
- Training: 10 epochs, BCE loss

### EBM (Assignment 4)
- Architecture: 4 Conv layers with Swish activation → FC → Scalar energy
- Input: 32×32×3 RGB (CIFAR-10)
- Training: Contrastive divergence, 10 epochs
- Sampling: Langevin dynamics (60 steps)

### Diffusion (Assignment 4)
- Architecture: UNet with skip connections
- Input: 32×32×3 RGB (CIFAR-10)
- Training: L1 loss on noise prediction, 5 epochs
- Sampling: Reverse diffusion (50 steps)

---

## ⚙️ Dependencies

- Python >= 3.10
- torch >= 2.1.0
- torchvision >= 0.16.0
- fastapi[standard] >= 0.116.1
- pydantic >= 2.0.0
- spacy >= 3.7.0
- numpy >= 1.24.0
- pillow >= 10.0.0
- matplotlib >= 3.7.0
- tqdm >= 4.66.0

---

## 📝 Training Details

### EBM Training
- Epochs: 10
- Learning rate: 1e-4
- Langevin steps: 60
- Regularization α: 0.1
- Time: ~1 hour (CPU)

### Diffusion Training
- Epochs: 5
- Learning rate: 1e-3
- Diffusion schedule: Offset cosine
- Time: ~20 minutes (CPU)

### GAN Training
- Epochs: 10
- Learning rate: 2e-4
- Optimizer: Adam (β1=0.5, β2=0.999)
- Time: ~15 minutes (CPU)

---

## ✅ Previous Assignment Issues - Fixed

- ✅ All dependencies now in `pyproject.toml`
- ✅ Docker configured to use port 8000 (not 80)
- ✅ `helper_lib/` and `data/` folders included in Docker
- ✅ Uvicorn starts without manual intervention
- ✅ No missing dependencies
- ✅ API launches successfully

---

## 🧪 Testing

Run the test script to verify all endpoints:
```bash
python test_api.py
```

This will:
- Test all API endpoints
- Generate sample images
- Save results to local files
- Display success/failure status

---

## 📚 References

- Course: APANPS5900 Applied Generative AI
- Modules: 1-8 (Text Generation to Diffusion Models)
- PyTorch Documentation
- FastAPI Documentation

---

## 📄 License

This project is for educational purposes as part of Columbia University coursework.

---

**Status:** ✅ All assignments (1-4) complete and deployed
