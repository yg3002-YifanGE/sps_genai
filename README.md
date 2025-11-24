# SPS GenAI - Assignments 1-5

**Columbia University – APANPS5900 Applied Generative AI**  
**Author:** Yifan Ge (yg3002)  
**Repository:** https://github.com/yg3002-YifanGE/sps_genai

---

## 📋 Project Overview

This project implements multiple generative AI models and integrates them into a unified FastAPI web service. The project covers assignments 1-5 of the course, progressively building a comprehensive generative AI system including fine-tuned large language models.

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

### Assignment 5: Fine-tuned GPT-2 for Question Answering
- **Fine-tuned GPT-2** on SQuAD dataset (5,000 samples)
- **Custom response format**: Prefix + Answer + Suffix
- **API integration** with configurable generation parameters

**Quick Start:**
```bash
# Train model
cd assignment5
python train_gpt2_squad.py --epochs 1 --num_samples 5000 --batch_size 4

# Test model
python train_gpt2_squad.py --test_only

# Test API
python test_gpt2_api.py
```

**Key Endpoints:**
- `/gpt2/answer` - Single answer generation
- `/gpt2/answer/multiple` - Multiple diverse answers

📁 **See `assignment5/README.md` for complete documentation**

---

## 🏗️ Project Structure

```
sps_genai/
├── app/
│   ├── main.py              # FastAPI application with all endpoints
│   ├── bigram_model.py      # Text generation model
│   └── gpt2_qa.py           # Fine-tuned GPT-2 integration (Assignment 5)
│
├── helper_lib/              # Reusable ML components
│   ├── model.py             # CNN, VAE, GAN, EBM, Diffusion models
│   ├── trainer.py           # Training functions
│   ├── generator.py         # Sample generation utilities
│   ├── data_loader.py       # Dataset loaders
│   └── utils.py
│
├── assignment2/             # CNN Image Classification
│   └── cnn_a2_cifar10.pt
│
├── assignment4/             # EBM & Diffusion Models
│   ├── README.md
│   ├── train_ebm_cifar10.py
│   ├── train_diffusion_cifar10.py
│   └── train_gan_offline.py
│
├── assignment5/             # Fine-tuned GPT-2 Q&A
│   ├── README.md            # Complete documentation
│   ├── train_gpt2_squad.py  # Training script
│   └── test_gpt2_api.py     # API testing
│
├── data/                    # Trained model weights
│   ├── cifar-10-batches-py/
│   ├── gan_G.pth
│   ├── ebm_cifar10.pth
│   └── diffusion_cifar10.pth
│
├── models/
│   └── gpt2_squad_finetuned/ # Fine-tuned GPT-2 (created by training)
│
├── test_api.py              # General API testing
├── Dockerfile
├── requirements.txt
└── README.md                # This file
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
# Assignment 3: GAN
cd assignment4
python train_gan_offline.py

# Assignment 4: EBM & Diffusion
python train_ebm_cifar10.py
python train_diffusion_cifar10.py

# Assignment 5: Fine-tune GPT-2
cd ../assignment5
python train_gpt2_squad.py --epochs 1 --num_samples 5000 --batch_size 4
```

### 3. Run API Server

```bash
uvicorn app.main:app --port 8000 --reload
```

Access interactive documentation: **http://localhost:8000/docs**

### 4. Test API

```bash
# Test general endpoints
python test_api.py

# Test GPT-2 endpoints (Assignment 5)
cd assignment5
python test_gpt2_api.py
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
| `/gpt2/answer` | POST | Answer question with fine-tuned GPT-2 | 5 |
| `/gpt2/answer/multiple` | POST | Generate multiple diverse answers | 5 |

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

**GPT-2 Question Answering (Assignment 5):**
```bash
# Single answer
curl -X POST "http://localhost:8000/gpt2/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is machine learning?",
    "max_length": 150,
    "temperature": 0.7
  }'

# Multiple diverse answers
curl -X POST "http://localhost:8000/gpt2/answer/multiple" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is artificial intelligence?",
    "num_responses": 3,
    "temperature": 0.8
  }'
```

---

## 🐳 Docker Deployment

### Build Image
```bash
docker build -t sps-genai:latest .
```

### Run Container
```bash
docker run -d -p 8000:8000 --name sps-genai sps-genai:latest
```

### Train GPT-2 Model in Container (Assignment 5)
```bash
# Quick test (1,000 samples, ~50 minutes)
docker exec -it sps-genai python assignment5/train_gpt2_squad.py --epochs 1 --num_samples 1000 --batch_size 4
```

Access at: **http://localhost:8000/docs**

📖 **See `DOCKER_GUIDE.md` for complete Docker deployment guide**

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

### Fine-tuned GPT-2 (Assignment 5)
- Architecture: GPT-2 (124M parameters, 12 layers, 768 hidden size)
- Dataset: SQuAD (Stanford Question Answering Dataset)
- Training: Causal language modeling, 3 epochs
- Format: Custom prefix/suffix for structured responses
- Generation: Autoregressive with temperature sampling

---

## ⚙️ Dependencies

- Python >= 3.10
- torch >= 2.1.0
- torchvision >= 0.16.0
- fastapi[standard] >= 0.116.1
- pydantic >= 2.0.0
- spacy >= 3.7.0
- transformers >= 4.35.0 (Assignment 5)
- datasets >= 2.14.0 (Assignment 5)
- accelerate >= 0.24.0 (Assignment 5)
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

### GPT-2 Fine-tuning (Assignment 5)
- Base model: openai-community/gpt2 (124M params)
- **Dataset: SQuAD (5,000 samples from 87k total)** ⭐
- Epochs: 1
- Learning rate: 5e-5
- Batch size: 4
- Max sequence length: 512 tokens
- Time: ~4-5 hours (CPU)
- Format: Custom prefix/suffix wrapping

**Dataset Size Rationale:**
We use 5,000 samples (instead of the full 87k) because:
1. **Task-appropriate**: This is fine-tuning for format learning, not training from scratch
2. **Computationally efficient**: Full dataset would take 80+ hours on CPU
3. **Sufficient for format**: 5,000 examples adequately teach the response template
4. **Assignment-compliant**: Meets all requirements while being practical

**Training Command:**
```bash
# Recommended configuration (used for this submission)
python train_gpt2_squad_simple.py --epochs 1 --num_samples 5000 --batch_size 4

# Alternative: Quick test (1,000 samples, ~50 minutes)
python train_gpt2_squad_simple.py --epochs 1 --num_samples 1000 --batch_size 4

# Alternative: Full dataset (not recommended - 80+ hours)
python train_gpt2_squad_simple.py --epochs 1 --batch_size 4

# Test existing model
python train_gpt2_squad_simple.py --test_only
```

---

## ✅ Assignment 5: Fine-tuned LLM Features

**What's New in Assignment 5:**
- ✅ Fine-tuned GPT-2 model on SQuAD dataset
- ✅ Custom response format with prefix/suffix
- ✅ Question answering API endpoints
- ✅ Multiple response generation
- ✅ Configurable generation parameters (temperature, max_length, top_p)
- ✅ Lazy model loading for efficient API startup
- ✅ Comprehensive testing script
- ✅ Full documentation and examples

**Custom Response Format:**
```
That is a great question. [Question] [Answer] Let me know if you have any other questions.
```

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
- Modules: 1-11 (Text Generation to Reinforcement Learning)
- PyTorch Documentation
- FastAPI Documentation
- HuggingFace Transformers Documentation
- SQuAD Dataset: https://huggingface.co/datasets/rajpurkar/squad

---

## 📄 License

This project is for educational purposes as part of Columbia University coursework.

---

**Status:** ✅ All assignments (1-5) complete and deployed

---

## 🎓 Assignment 5 Specific Notes

### Fine-tuning Process
1. **Data Preparation**: SQuAD dataset is automatically downloaded from HuggingFace
2. **Format Transformation**: Q&A pairs are wrapped with custom prefix/suffix
3. **Tokenization**: GPT-2 tokenizer with padding and truncation
4. **Training**: Causal language modeling with teacher forcing
5. **Saving**: Model and tokenizer saved to `models/gpt2_squad_finetuned/`

### API Integration
- **Lazy Loading**: Model loads only when first API call is made
- **Device Detection**: Automatically uses GPU if available
- **Error Handling**: Comprehensive error messages for missing models
- **Validation**: Parameter validation for generation settings

### Testing
```bash
# Run comprehensive tests
python test_gpt2_api.py
```

Tests include:
- API health check
- Single answer generation with multiple questions
- Multiple answer generation
- Parameter validation
- Sample output generation for documentation

### Performance Tips
- **GPU**: Use CUDA-enabled GPU for faster inference (3-5x speedup)
- **Batch Size**: Adjust based on available memory
- **Temperature**: Lower (0.5-0.7) for focused answers, higher (0.8-1.0) for creativity
- **Max Length**: Balance between completeness and inference time

---
