from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from app.bigram_model import BigramModel

# --------- Embedding dependency ---------
# It is necessary to install `spacy` in the Dockerfile or locally and download `en_core_web_md`.
import spacy
import numpy as np

# Loaded only once, resident in the container.
nlp = spacy.load("en_core_web_md")

app = FastAPI(title="SPS GenAI API", version="0.1.0")

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
