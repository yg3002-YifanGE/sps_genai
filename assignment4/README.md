# Assignment 4: Energy-Based Models & Diffusion Models

**Author:** Yifan Ge (yg3002)  
**Course:** APANPS5900 Applied Generative AI

---

## Overview

This assignment implements Energy-Based Models (EBM) and Diffusion Models for CIFAR-10 image generation.

---

## Training Scripts

### Energy-Based Model
```bash
cd /Users/geyifan/sps_genai/assignment4
python train_ebm_cifar10.py
```

### Diffusion Model
```bash
python train_diffusion_cifar10.py
```

### GAN (from Assignment 3)
```bash
python train_gan_offline.py
```

---

## Model Files

Trained models are saved to `../data/`:
- `ebm_cifar10.pth` - EBM model weights
- `diffusion_cifar10.pth` - Diffusion model weights
- `gan_G.pth` - GAN generator weights

---

## API Endpoints

Models are integrated into the main API (`../app/main.py`):
- `GET /ebm/generate` - Generate images using EBM
- `GET /diffusion/generate` - Generate images using Diffusion
- `GET /gan/generate` - Generate images using GAN

---

## Key Features

### EBM
- Langevin dynamics for sampling
- Contrastive divergence training
- CIFAR-10 RGB images (32×32×3)

### Diffusion
- UNet architecture with skip connections
- Offset cosine diffusion schedule
- Reverse diffusion for generation

### GAN
- Generator + Discriminator architecture
- MNIST digit generation
- BCE loss with Adam optimizer

---

See main `README.md` for full project documentation.

