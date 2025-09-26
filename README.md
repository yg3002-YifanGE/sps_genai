# SPS GenAI Assignment 1

This repository contains the solution for **Assignment 1: Coding Environment Setup and Simple API Implementation**  
Course: *Applied Generative AI*  
Student: *Yifan Ge (yg3002)*

---

## 📦 Features
- **Bigram Language Model** (`POST /generate`)  
  Generate text using a simple bigram probability model.
- **Word Embeddings** (`POST /embed`)  
  Return the spaCy `en_core_web_md` embedding vector for a given word.
- **Word Similarity** (`POST /similar`)  
  Find the top-k most similar words based on cosine similarity in the embedding space.
- **Containerized FastAPI Application** with Docker for reproducibility.

---

## 🚀 How to Build & Run with Docker

### 1. Build the Docker image
```bash
docker build -t sps-genai .

### 2. Run the container
docker run --name sps-genai -p 8000:80 sps-genai

# The API will now be available at:
    ## Root endpoint: http://127.0.0.1:8000/
    ## Interactive docs: http://127.0.0.1:8000/docs