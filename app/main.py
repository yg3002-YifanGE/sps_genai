from typing import List
from pathlib import Path
import io
import base64

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

# ---------------------------------
# Bigram text model (Assignment 1)
# ---------------------------------
from app.bigram_model import BigramModel

# ---------------------------------
# Embedding dependencies (Assignment 1)
# NOTE: requires spacy + en_core_web_md in Docker / env
# ---------------------------------
import spacy
import numpy as np

# We load SpaCy model once, as a module-level singleton.
# This is OK at import time because it's lightweight compared to training.
nlp = spacy.load("en_core_web_md")

# ---------------------------------
# Torch / CV / Deep Models
# ---------------------------------
import torch
import torch.nn.functional as F
from torchvision import transforms, utils as tv_utils
from PIL import Image

from helper_lib.model import get_model  # must include CNN_A2, VAE, GAN, EBM, Diffusion
from helper_lib.generator import generate_gan_samples  # we will not call directly in API, but keep import for clarity
from helper_lib.trainer import DiffusionModelWrapper, offset_cosine_diffusion_schedule, generate_ebm_samples

# ---------------------------------
# FastAPI app
# ---------------------------------
app = FastAPI(
    title="SPS GenAI API",
    version="0.4.0",
    description="Text bigram generation, embeddings, CIFAR10 classification, GAN sampling, EBM generation, and Diffusion generation."
)

# ---------------------------------
# Example corpus for Bigram model
# ---------------------------------
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]
bigram_model = BigramModel(corpus)

# ---------------------------------
# Pydantic request / response schemas
# ---------------------------------
class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

class EmbeddingRequest(BaseModel):
    word: str

class EmbeddingResponse(BaseModel):
    word: str
    dim: int
    vector: List[float]

class SimilarRequest(BaseModel):
    word: str
    top_k: int = 5

class SimilarItem(BaseModel):
    token: str
    score: float

class SimilarResponse(BaseModel):
    word: str
    top_k: int
    results: List[SimilarItem]

class GANGenerateResponse(BaseModel):
    num_samples: int
    image_base64_png: str  # base64-encoded PNG of a grid of generated digits

class EBMGenerateResponse(BaseModel):
    num_samples: int
    image_base64_png: str  # base64-encoded PNG of a grid of generated digits

class DiffusionGenerateResponse(BaseModel):
    num_samples: int
    image_base64_png: str  # base64-encoded PNG of a grid of generated images


# ---------------------------------
# Basic sanity endpoint
# ---------------------------------
@app.get("/")
def read_root():
    return {"status": "ok",
            "message": "SPS GenAI API is running",
            "endpoints": ["/generate", "/embed", "/similar", "/predict_cifar10", 
                         "/gan/generate", "/ebm/generate", "/diffusion/generate"]}


# ---------------------------------
# Text generation (Assignment 1 bigram model)
# ---------------------------------
@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    """
    Generate text from the custom bigram model.
    """
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


# ---------------------------------
# Word embedding / similarity (Assignment 1)
# ---------------------------------
@app.post("/embed", response_model=EmbeddingResponse)
def get_embedding(req: EmbeddingRequest):
    """
    Return SpaCy word embedding vector (en_core_web_md).
    NOTE for grading: we return pure Python list to avoid tensor/ndarray serialization issues.
    """
    doc = nlp(req.word)
    vec = doc.vector  # numpy array
    return EmbeddingResponse(
        word=req.word,
        dim=int(vec.shape[0]),
        vector=vec.tolist()
    )


@app.post("/similar", response_model=SimilarResponse)
def most_similar(req: SimilarRequest):
    """
    Compute cosine similarity between the query word vector and
    SpaCy vocab entries that have vectors. Return top_k results.
    """
    query_vec = nlp(req.word).vector
    # guard against zero vector
    if np.linalg.norm(query_vec) == 0:
        return SimilarResponse(word=req.word, top_k=req.top_k, results=[])

    results: List[SimilarItem] = []

    for lex in nlp.vocab:
        # We skip words that don't have vectors, are not alphabetic, or are common stop words,
        # to avoid junk like punctuation or ultra-frequent filler tokens.
        if not lex.has_vector or not lex.is_alpha or lex.is_stop:
            continue

        v = lex.vector
        denom = (np.linalg.norm(query_vec) * np.linalg.norm(v))
        if denom == 0:
            continue

        score = float(np.dot(query_vec, v) / denom)
        results.append(SimilarItem(token=lex.text, score=score))

    # sort descending by cosine score
    results.sort(key=lambda x: x.score, reverse=True)

    # slice top_k (ensure >=1)
    k = max(1, req.top_k)
    topk = results[:k]

    return SimilarResponse(
        word=req.word,
        top_k=req.top_k,
        results=topk
    )


# ---------------------------------
# CIFAR-10 classifier (Assignment 2)
# ---------------------------------

# CIFAR-10 label names in canonical order
CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

# global state for CIFAR10 classifier
_device = "cuda" if torch.cuda.is_available() else "cpu"
_cifar_model = None

# preprocessing pipeline must match your training setup (64x64 resize etc.)
_cifar_preprocess = transforms.Compose([
    transforms.Resize((64, 64)),  # your CNN_A2 expects 64x64
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),  # CIFAR10 mean
        (0.2470, 0.2435, 0.2616),  # CIFAR10 std
    ),
])

def _cifar_weights_path() -> Path:
    # expected training checkpoint for assignment2 CNN model
    # e.g. project_root/assignment2/cnn_a2_cifar10.pt
    return Path(__file__).resolve().parent.parent / "assignment2" / "cnn_a2_cifar10.pt"


@app.on_event("startup")
def _load_models_on_startup():
    """
    Startup hook:
    - Load CIFAR10 classifier weights if available.
    - (We do NOT auto-train anything here; only lightweight loading.)
    This prevents the "I had to remove app call..." complaint:
    import-time side effects are controlled.
    """
    global _cifar_model

    # load CIFAR model if weights exist
    weights = _cifar_weights_path()
    if weights.exists():
        model = get_model("CNN_A2").to(_device)
        state = torch.load(weights, map_location=_device)
        model.load_state_dict(state)
        model.eval()
        _cifar_model = model
        print(f"[startup] CIFAR-10 model loaded on '{_device}' from {weights}")
    else:
        # we don't raise here, because we still want other endpoints to work
        print(f"[startup] WARNING: CIFAR-10 weights not found at {weights}. "
              f"/predict_cifar10 will 500 until you provide them.")


@app.post("/predict_cifar10")
async def predict_cifar10(file: UploadFile = File(...)):
    """
    Classify an uploaded RGB image into CIFAR-10 classes.
    Returns:
      - pred_label
      - pred_id
      - probs: softmax probabilities over 10 classes
    """
    if _cifar_model is None:
        raise HTTPException(
            status_code=500,
            detail="CIFAR10 model not loaded. "
                   "Please provide weights at assignment2/cnn_a2_cifar10.pt."
        )

    try:
        img = Image.open(file.file).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {e}")

    x = _cifar_preprocess(img).unsqueeze(0).to(_device)  # (1,3,64,64)

    with torch.no_grad():
        logits = _cifar_model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().tolist()
        pred_id = int(torch.argmax(logits, dim=1).item())

    return {
        "pred_id": pred_id,
        "pred_label": CIFAR10_CLASSES[pred_id],
        "probs": probs
    }


# ---------------------------------
# GAN sampling (Assignment 3)
# ---------------------------------

# We will lazily load the trained GAN generator weights on demand,
# instead of at import time. This avoids heavy side effects for graders,
# and it avoids retraining.
_gan_generator_cached = None
_gan_latent_dim = 100  # matches assignment spec


def _gan_generator_weights_path() -> Path:
    # We'll assume you save the trained generator's state_dict here after training.
    # e.g. torch.save(G.state_dict(), "data/gan_G.pth")
    return Path(__file__).resolve().parent.parent / "data" / "gan_G.pth"


def _load_gan_generator():
    """
    Load (and cache) the trained GAN generator model.
    Required for /gan/generate.
    """
    global _gan_generator_cached

    if _gan_generator_cached is not None:
        return _gan_generator_cached

    # get_model("GAN") returns {"generator": G, "discriminator": D}
    models = get_model("GAN", latent_dim=_gan_latent_dim)
    G = models["generator"]

    weights_path = _gan_generator_weights_path()
    if not weights_path.exists():
        raise RuntimeError(
            f"GAN generator weights not found at {weights_path}. "
            "You must train the GAN offline (train_gan in helper_lib.trainer) "
            "and save G.state_dict() there as gan_G.pth"
        )

    state = torch.load(weights_path, map_location="cpu")
    G.load_state_dict(state)
    G.eval()  # inference mode only
    _gan_generator_cached = G
    print(f"[lazy-load] GAN generator loaded from {weights_path}")
    return _gan_generator_cached


def _sample_gan_grid_png(num_samples: int = 16) -> bytes:
    """
    1. Load generator
    2. Sample latent noise
    3. Generate fake digits
    4. Make a grid image (torchvision.utils.make_grid)
    5. Convert to PNG bytes
    """
    G = _load_gan_generator()

    with torch.no_grad():
        z = torch.randn(num_samples, _gan_latent_dim)
        fake_imgs = G(z).cpu()  # (N,1,28,28), range [-1,1] due to Tanh
        # map [-1,1] -> [0,1] for visualization
        fake_imgs = (fake_imgs + 1.0) / 2.0
        fake_imgs = fake_imgs.clamp(0.0, 1.0)

        # make a square-ish grid automatically
        grid = tv_utils.make_grid(fake_imgs, nrow=int(np.ceil(np.sqrt(num_samples))), pad_value=1.0)
        # grid: (3 or 1, H, W). For grayscale it'll be (1,H,W)

        # convert to PIL
        ndarr = (grid.mul(255).byte().permute(1, 2, 0).numpy())
        pil_img = Image.fromarray(ndarr.squeeze())  # squeeze() handles 1-channel

    # dump to PNG bytes
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


@app.get("/gan/generate", response_model=GANGenerateResponse)
def gan_generate(num_samples: int = 16):
    """
    Assignment 3 endpoint:
    Returns a base64-encoded PNG of a grid of generated MNIST-like digits.
    Steps:
    - lazily load trained generator weights from data/gan_G.pth
    - sample latent noise
    - generate fake digits
    - return as base64 so the grader / frontend can render
    """
    if num_samples <= 0:
        raise HTTPException(status_code=400, detail="num_samples must be > 0")

    try:
        png_bytes = _sample_gan_grid_png(num_samples=num_samples)
    except RuntimeError as e:
        # most common error: weights file missing
        raise HTTPException(status_code=500, detail=str(e))

    # base64 encode for transport
    b64_img = base64.b64encode(png_bytes).decode("ascii")

    return GANGenerateResponse(
        num_samples=num_samples,
        image_base64_png=b64_img
    )


# ---------------------------------
# EBM sampling (Assignment 4)
# ---------------------------------

# Global state for EBM
_ebm_model_cached = None


def _ebm_weights_path() -> Path:
    """Path to trained EBM weights for CIFAR-10"""
    return Path(__file__).resolve().parent.parent / "data" / "ebm_cifar10.pth"


def _load_ebm_model():
    """Load (and cache) the trained EBM model for CIFAR-10"""
    global _ebm_model_cached

    if _ebm_model_cached is not None:
        return _ebm_model_cached

    # CIFAR-10 uses RGB (3 channels)
    model = get_model("EBM", num_channels=3)
    weights_path = _ebm_weights_path()
    
    if not weights_path.exists():
        raise RuntimeError(
            f"EBM weights not found at {weights_path}. "
            "You must train the EBM on CIFAR-10 first: python train_ebm_cifar10.py"
        )

    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    _ebm_model_cached = model
    print(f"[lazy-load] EBM model (CIFAR-10) loaded from {weights_path}")
    return _ebm_model_cached


def _sample_ebm_grid_png(num_samples: int = 16, steps: int = 256, step_size: float = 10.0) -> bytes:
    """
    Generate samples from EBM using Langevin dynamics.
    1. Load EBM model (CIFAR-10 RGB)
    2. Start from random noise
    3. Run gradient descent on input to find low-energy states
    4. Make a grid image
    5. Convert to PNG bytes
    """
    model = _load_ebm_model()
    device = "cpu"  # use CPU for inference by default

    with torch.no_grad():
        # Start from random noise [-1, 1] for RGB images
        x = torch.rand((num_samples, 3, 32, 32), device=device) * 2 - 1

    # Run Langevin dynamics (gradient descent on input)
    for _ in range(steps):
        # Add noise
        with torch.no_grad():
            noise = torch.randn_like(x) * 0.01
            x = (x + noise).clamp(-1.0, 1.0)

        x.requires_grad_(True)

        # Compute energy and gradients
        energy = model(x)
        grads, = torch.autograd.grad(energy, x, grad_outputs=torch.ones_like(energy))

        # Gradient descent on input
        with torch.no_grad():
            grads = grads.clamp(-0.03, 0.03)
            x = (x - step_size * grads).clamp(-1.0, 1.0)

    samples = x.detach().cpu()
    
    # Rescale from [-1,1] to [0,1]
    samples = (samples + 1.0) / 2.0
    samples = samples.clamp(0.0, 1.0)

    # Make grid
    grid = tv_utils.make_grid(samples, nrow=int(np.ceil(np.sqrt(num_samples))), pad_value=1.0)
    
    # Convert to PIL (RGB image)
    ndarr = (grid.mul(255).byte().permute(1, 2, 0).numpy())
    pil_img = Image.fromarray(ndarr)

    # Dump to PNG bytes
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


@app.get("/ebm/generate", response_model=EBMGenerateResponse)
def ebm_generate(num_samples: int = 16, steps: int = 256, step_size: float = 10.0):
    """
    Assignment 4 endpoint for Energy-Based Model:
    Returns a base64-encoded PNG of a grid of generated CIFAR-10-like images.
    
    Args:
        num_samples: Number of samples to generate (default 16)
        steps: Number of Langevin dynamics steps (default 256)
        step_size: Step size for gradient descent (default 10.0)
    
    Steps:
    - lazily load trained EBM weights from data/ebm_cifar10.pth
    - run Langevin dynamics to sample low-energy states
    - return as base64 PNG
    """
    if num_samples <= 0:
        raise HTTPException(status_code=400, detail="num_samples must be > 0")

    try:
        png_bytes = _sample_ebm_grid_png(num_samples=num_samples, steps=steps, step_size=step_size)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # base64 encode for transport
    b64_img = base64.b64encode(png_bytes).decode("ascii")

    return EBMGenerateResponse(
        num_samples=num_samples,
        image_base64_png=b64_img
    )


# ---------------------------------
# Diffusion sampling (Assignment 4)
# ---------------------------------

# Global state for Diffusion
_diffusion_model_cached = None


def _diffusion_weights_path() -> Path:
    """Path to trained Diffusion model weights for CIFAR-10"""
    return Path(__file__).resolve().parent.parent / "data" / "diffusion_cifar10.pth"


def _load_diffusion_model():
    """Load (and cache) the trained Diffusion model for CIFAR-10"""
    global _diffusion_model_cached

    if _diffusion_model_cached is not None:
        return _diffusion_model_cached

    # Create UNet model for CIFAR-10 (RGB, 3 channels)
    unet = get_model("Diffusion", image_size=32, num_channels=3, embedding_dim=32)
    
    # Wrap in DiffusionModelWrapper
    diffusion_wrapper = DiffusionModelWrapper(unet, offset_cosine_diffusion_schedule)
    
    weights_path = _diffusion_weights_path()
    
    if not weights_path.exists():
        raise RuntimeError(
            f"Diffusion weights not found at {weights_path}. "
            "You must train the Diffusion model on CIFAR-10 first: python train_diffusion_cifar10.py"
        )

    checkpoint = torch.load(weights_path, map_location="cpu")
    diffusion_wrapper.network.load_state_dict(checkpoint["model_state"])
    
    # Set normalizer if available
    if "normalizer_mean" in checkpoint:
        diffusion_wrapper.set_normalizer(checkpoint["normalizer_mean"], checkpoint["normalizer_std"])
    
    diffusion_wrapper.eval()
    _diffusion_model_cached = diffusion_wrapper
    print(f"[lazy-load] Diffusion model (CIFAR-10) loaded from {weights_path}")
    return _diffusion_model_cached


def _sample_diffusion_grid_png(num_samples: int = 16, diffusion_steps: int = 50, image_size: int = 32) -> bytes:
    """
    Generate samples from Diffusion model.
    1. Load Diffusion model
    2. Start from random noise
    3. Run reverse diffusion process
    4. Make a grid image
    5. Convert to PNG bytes
    """
    model = _load_diffusion_model()

    # Generate samples
    with torch.no_grad():
        samples = model.generate(
            num_images=num_samples,
            diffusion_steps=diffusion_steps,
            image_size=image_size
        ).cpu()

    # Make grid
    grid = tv_utils.make_grid(samples, nrow=int(np.ceil(np.sqrt(num_samples))), pad_value=1.0)
    
    # Convert to PIL (RGB image)
    ndarr = (grid.mul(255).byte().permute(1, 2, 0).numpy())
    pil_img = Image.fromarray(ndarr)

    # Dump to PNG bytes
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


@app.get("/diffusion/generate", response_model=DiffusionGenerateResponse)
def diffusion_generate(num_samples: int = 16, diffusion_steps: int = 50, image_size: int = 32):
    """
    Assignment 4 endpoint for Diffusion Model:
    Returns a base64-encoded PNG of a grid of generated CIFAR-10-like images.
    
    Args:
        num_samples: Number of samples to generate (default 16)
        diffusion_steps: Number of reverse diffusion steps (default 50)
        image_size: Size of generated images (default 32)
    
    Steps:
    - lazily load trained Diffusion model weights from data/diffusion_cifar10.pth
    - run reverse diffusion process
    - return as base64 PNG
    """
    if num_samples <= 0:
        raise HTTPException(status_code=400, detail="num_samples must be > 0")
    if diffusion_steps <= 0:
        raise HTTPException(status_code=400, detail="diffusion_steps must be > 0")

    try:
        png_bytes = _sample_diffusion_grid_png(
            num_samples=num_samples, 
            diffusion_steps=diffusion_steps,
            image_size=image_size
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # base64 encode for transport
    b64_img = base64.b64encode(png_bytes).decode("ascii")

    return DiffusionGenerateResponse(
        num_samples=num_samples,
        image_base64_png=b64_img
    )
