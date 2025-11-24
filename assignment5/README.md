# Assignment 5: Fine-tuning GPT-2 on SQuAD

**Author:** Yifan Ge (yg3002)  
**Course:** APANPS5900 Applied Generative AI - Columbia University

---

## Overview

This assignment fine-tunes the GPT-2 model on the Stanford Question Answering Dataset (SQuAD) to generate answers in a specific format:
- **Start:** "That is a great question."
- **End:** "Let me know if you have any other questions."

---

## Quick Start

### 1. Install Dependencies
```bash
pip install "transformers>=4.35.0" "datasets>=2.14.0"
```

### 2. Train the Model
```bash
cd /Users/geyifan/sps_genai/assignment5
python train_gpt2_squad.py --epochs 1 --num_samples 5000 --batch_size 4
```
**Time:** ~4-5 hours on CPU

### 3. Test the Model
```bash
python train_gpt2_squad.py --test_only
```

### 4. Start API Server
```bash
cd /Users/geyifan/sps_genai
uvicorn app.main:app --port 8000 --reload
```

### 5. Test API Endpoints
```bash
cd /Users/geyifan/sps_genai/assignment5
python test_gpt2_api.py
```

---

## Training Configuration

### Parameters
- **Base Model:** openai-community/gpt2 (124M params)
- **Dataset:** SQuAD (5,000 samples from 87k total)
- **Epochs:** 1
- **Batch Size:** 4
- **Learning Rate:** 5e-5
- **Training Time:** ~4-5 hours (CPU)

### Why 5,000 Samples?

**Rationale:**
1. **Task-appropriate:** Fine-tuning for format learning, not training from scratch
2. **Computationally practical:** Full dataset would take 80+ hours on CPU
3. **Sufficient quality:** 5,000 examples adequately teach the response template
4. **Assignment-compliant:** Meets all requirements

**Comparison:**
- 1,000 samples: ~50 minutes (quick test)
- **5,000 samples: ~4-5 hours** ⭐ **Recommended**
- 87,599 samples: ~80 hours (not practical on CPU)

---

## Implementation Details

### Model Integration
- **File:** `../app/gpt2_qa.py`
- Lazy model loading
- Configurable generation parameters
- Single and multiple answer generation

### API Endpoints
Added to `../app/main.py`:

**POST `/gpt2/answer`** - Single answer generation
```bash
curl -X POST "http://localhost:8000/gpt2/answer" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?", "max_length": 150}'
```

**POST `/gpt2/answer/multiple`** - Multiple diverse answers
```bash
curl -X POST "http://localhost:8000/gpt2/answer/multiple" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is AI?", "num_responses": 3}'
```

---

## Files in This Directory

```
assignment5/
├── README.md                    # This file
├── train_gpt2_squad.py         # Training script (simplified, no accelerate)
├── test_gpt2_api.py            # API testing script
└── (models will be saved to ../models/gpt2_squad_finetuned/)
```

---

## Results

After training, the model will:
- ✅ Start responses with "That is a great question."
- ✅ End responses with "Let me know if you have any other questions."
- ✅ Generate coherent, relevant answers
- ✅ Work through the FastAPI endpoints

---

## Technical Notes

### No Accelerate Required
This implementation uses a simplified PyTorch training loop instead of HuggingFace Trainer, avoiding the need for the `accelerate` package.

### Device Detection
Automatically uses GPU if available, falls back to CPU.

### Progress Tracking
Real-time loss monitoring and progress bars during training.

---

## Troubleshooting

**Issue:** Out of memory
```bash
# Reduce batch size or samples
python train_gpt2_squad.py --epochs 1 --num_samples 1000 --batch_size 2
```

**Issue:** Model not found error in API
```bash
# Make sure training completed successfully
ls -la ../models/gpt2_squad_finetuned/
```

**Issue:** Slow training
- Expected on CPU: ~13 seconds per batch
- Use GPU for 10-20x speedup
- Or reduce sample size for quick testing

---

## Assignment Requirements ✅

- ✅ Fine-tuned openai-community/gpt2 model
- ✅ Used SQuAD dataset from HuggingFace
- ✅ Implemented custom response format
- ✅ Updated text generation API (Modules 3 & 7)
- ✅ Code ready for GitHub submission

---

## References

- [SQuAD Dataset](https://huggingface.co/datasets/rajpurkar/squad)
- [GPT-2 Model](https://huggingface.co/openai-community/gpt2)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)

