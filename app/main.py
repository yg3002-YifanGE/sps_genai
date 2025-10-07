from typing import List
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.bigram_model import BigramModel

# --------- Embedding dependency ---------
# It is necessary to install `spacy` in the Dockerfile or locally and download `en_core_web_md`.
import spacy
import numpy as np

# Loaded only once, resident in the container.
nlp = spacy.load("en_core_web_md")

# ----- NEW: CIFAR-10 classifier dependencies -----
import torch
from torchvision import transforms
from PIL import Image

from helper_lib.model import get_model  # must include CNN_A2 in get_model

app = FastAPI(title="SPS GenAI API", version="0.2.0")

# --------- example corpus，used for Bigram ---------
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]
bigram_model = BigramModel(corpus)

# --------- Pydantic model ---------
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

# --------- Routing ---------
@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}

@app.post("/embed", response_model=EmbeddingResponse)
def get_embedding(req: EmbeddingRequest):
    """Return SpaCy word embedding vector (en_core_web_md)."""
    doc = nlp(req.word)
    vec = doc.vector  # numpy array
    return EmbeddingResponse(word=req.word, dim=int(vec.shape[0]), vector=vec.tolist())

@app.post("/similar", response_model=SimilarResponse)
def most_similar(req: SimilarRequest):
    """Return top-k most similar tokens by cosine similarity against vocab."""
    query = nlp(req.word).vector
    if np.linalg.norm(query) == 0:
        return SimilarResponse(word=req.word, top_k=req.top_k, results=[])

    results: List[SimilarItem] = []
    for lex in nlp.vocab:
        # Only use words that are vectors and letters, skip stop words to avoid meaningless high-frequency words.
        if not lex.has_vector or not lex.is_alpha or lex.is_stop:
            continue
        v = lex.vector
        denom = (np.linalg.norm(query) * np.linalg.norm(v))
        if denom == 0:
            continue
        score = float(np.dot(query, v) / denom)
        results.append(SimilarItem(token=lex.text, score=score))

    results.sort(key=lambda x: x.score, reverse=True)
    topk = results[: max(1, req.top_k)]
    return SimilarResponse(word=req.word, top_k=req.top_k, results=topk)

# =========================
# CIFAR-10 classifier
# =========================
# Class names in CIFAR-10 order
CIFAR10_CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

# Global model objects (loaded once)
_device = "cuda" if torch.cuda.is_available() else "cpu"
_cifar_model = None
_cifar_preprocess = transforms.Compose([
    transforms.Resize((64, 64)),  # must match your training input size
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])

def _weights_path() -> Path:
    # /app/main.py -> project root / assignment2/cnn_a2_cifar10.pt
    return Path(__file__).resolve().parent.parent / "assignment2" / "cnn_a2_cifar10.pt"

@app.on_event("startup")
def load_cifar10_model():
    """
    Load CNN_A2 and its weights once at service startup.
    """
    global _cifar_model
    weights = _weights_path()
    if not weights.exists():
        # Fail fast with a helpful message
        raise RuntimeError(f"Model weights not found: {weights}. "
                           f"Train and save to assignment2/cnn_a2_cifar10.pt first.")
    model = get_model("CNN_A2").to(_device)
    state = torch.load(weights, map_location=_device)
    model.load_state_dict(state)
    model.eval()
    _cifar_model = model
    print(f"[startup] CIFAR-10 model loaded on {_device} from {weights}")

@app.post("/predict_cifar10")
async def predict_cifar10(file: UploadFile = File(...)):
    """
    Classify an uploaded RGB image into one of the CIFAR-10 classes.
    Returns predicted label and softmax probabilities (list of length 10).
    """
    if _cifar_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        img = Image.open(file.file).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {e}")

    x = _cifar_preprocess(img).unsqueeze(0).to(_device)  # (1,3,64,64)

    with torch.no_grad():
        logits = _cifar_model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
        pred_id = int(torch.argmax(logits, dim=1).item())

    return {
        "pred_id": pred_id,
        "pred_label": CIFAR10_CLASSES[pred_id],
        "probs": probs
    }