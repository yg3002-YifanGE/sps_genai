# SPS GenAI — Assignment 3: GAN Architecture & API Integration  
*Columbia University – APANPS5900 Applied Generative AI*  
**Author:** Yifan Ge (yg3002)

---

## 🎯 Project Overview

This project implements a **Generative Adversarial Network (GAN)** using **PyTorch** to generate hand-written digits from the **MNIST** dataset, and integrates the trained model into a **FastAPI** web service.

This work extends previous assignments:
- **Assignment 1:** FastAPI + Docker + spaCy embeddings  
- **Assignment 2:** CNN_A2 model trained on CIFAR-10 dataset  
- **Assignment 3:** GAN model design, training, and API deployment  

---

## 🧩 GAN Architecture

### **Generator**
- Input: Noise vector *(batch_size, 100)*
- Fully connected → reshape to (128, 7×7)
- ConvTranspose2d(128→64, kernel=4, stride=2, pad=1) + BatchNorm2d + ReLU
- ConvTranspose2d(64→1, kernel=4, stride=2, pad=1) + **Tanh**
- Output: Image (1×28×28) in range [-1, 1]

### **Discriminator**
- Input: Image (1×28×28)
- Conv2d(1→64, kernel=4, stride=2, pad=1) + LeakyReLU(0.2)
- Conv2d(64→128, kernel=4, stride=2, pad=1) + BatchNorm2d + LeakyReLU(0.2)
- Flatten → Linear(128×7×7 → 1) + **Sigmoid**

---

## 🧠 Training Details

- **Dataset:** MNIST (grayscale 28×28)
- **Loss Function:** Binary Cross Entropy (BCE)
- **Optimizer:** Adam (lr=2e-4, betas=(0.5, 0.999))
- **Epochs:** 10  
- **Batch Size:** 128  
- **Latent Dim:** 100  

After training, the generator weights are saved to:

```
data/gan_G.pth
```

A helper training script is provided:

```bash
python train_gan_offline.py
```

This script uses:
```python
from helper_lib.model import get_model
from helper_lib.trainer import train_gan
from helper_lib.data_loader import get_data_loader
```

---

## 🚀 FastAPI Integration

The trained GAN is integrated into the existing API (see `app/main.py`).  
You can run the full API locally:

```bash
uvicorn app.main:app --reload
```

Then open:
👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Available endpoints:
| Endpoint | Description |
|-----------|--------------|
| `/generate` | Text generation using bigram model |
| `/embed` | Word embeddings (spaCy) |
| `/similar` | Find most similar words |
| `/predict_cifar10` | CIFAR-10 image classification |
| `/gan/generate?num_samples=16` | Generate hand-written digits (base64 PNG) |

Example response from `/gan/generate`:
```json
{
  "num_samples": 16,
  "image_base64_png": "iVBORw0K..."
}
```

---

## 🧱 Project Structure

```
sps_genai/
├── app/
│   ├── main.py                # FastAPI endpoints
│   ├── bigram_model.py
│
├── helper_lib/
│   ├── model.py               # GAN, VAE, CNN_A2 models
│   ├── trainer.py             # train_gan, train_vae_model, train_model
│   ├── generator.py           # generate_gan_samples()
│   ├── data_loader.py         # MNIST/CIFAR loaders
│
├── data/
│   └── gan_G.pth              # Trained Generator weights
│
├── train_gan_offline.py       # Offline GAN training script
├── main.py                    # Clean entry point (no side effects)
├── Dockerfile
└── README.md
```

---

## ⚙️ Dependencies

Recommended environment:
```bash
python >= 3.10
torch >= 2.1.0
torchvision
fastapi
uvicorn
matplotlib
pillow
numpy
spacy
```

Install all dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

---

## 📸 Sample Output

Example 4×4 generated digit grid (from `/gan/generate`):

![GAN output example](https://raw.githubusercontent.com/yg3002-YifanGE/sps_genai/main/docs/example_gan_grid.png)

---

## 🧾 Notes
- The MNIST dataset will be automatically downloaded to `/data/MNIST/`
- The `data/MNIST/` folder is ignored via `.gitignore`  
- Only `data/gan_G.pth` is required for inference
- All code tested on macOS, Python 3.10, PyTorch (CPU)

---

**✅ Status:** Complete – GAN model trained and API deployed successfully.
