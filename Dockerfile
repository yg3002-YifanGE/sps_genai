# Use official Python image as the base
FROM python:3.10-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates build-essential && \
    rm -rf /var/lib/apt/lists/*

# Download and install uv (Python package manager)
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:${PATH}"

# Set the working directory
WORKDIR /code

# Copy dependency files for layer caching
COPY pyproject.toml uv.lock* /code/

# Install dependencies with uv
RUN uv sync --no-dev || uv sync

# Install the spaCy model
RUN uv pip install \
  https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.7.1/en_core_web_md-3.7.1-py3-none-any.whl

# Copy the application source code
COPY ./app /code/app
COPY ./helper_lib /code/helper_lib
COPY ./assignment5 /code/assignment5

# Copy model weights and configurations
RUN mkdir -p /code/data /code/assignment2 /code/models/gpt2_squad_finetuned
COPY ./data/*.pth /code/data/ 2>/dev/null || true
COPY ./data/*.png /code/data/ 2>/dev/null || true
COPY ./assignment2/*.pt /code/assignment2/ 2>/dev/null || true

# Copy GPT-2 model config files (small files only, not the 500MB weights)
# Users can train the model in the container if needed
COPY ./models/gpt2_squad_finetuned/*.json /code/models/gpt2_squad_finetuned/ 2>/dev/null || true
COPY ./models/gpt2_squad_finetuned/*.txt /code/models/gpt2_squad_finetuned/ 2>/dev/null || true

# Create a startup script that checks for models
RUN echo '#!/bin/bash\n\
echo "🚀 Starting SPS GenAI API..."\n\
echo ""\n\
if [ ! -f "/code/models/gpt2_squad_finetuned/model.safetensors" ]; then\n\
  echo "⚠️  GPT-2 model weights not found."\n\
  echo "   The /gpt2/answer endpoints will not work until the model is trained."\n\
  echo "   To train: docker exec -it <container> python assignment5/train_gpt2_squad.py --epochs 1 --num_samples 1000"\n\
  echo ""\n\
fi\n\
echo "📡 API starting on http://0.0.0.0:8000"\n\
echo "📖 Documentation: http://localhost:8000/docs"\n\
echo ""\n\
uvicorn app.main:app --host 0.0.0.0 --port 8000\n\
' > /code/start.sh && chmod +x /code/start.sh

# Expose port 8000 (as required by assignment feedback)
EXPOSE 8000

# Start the FastAPI app with Uvicorn on port 8000
CMD ["/code/start.sh"]
