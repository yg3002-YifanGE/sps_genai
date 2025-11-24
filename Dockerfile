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

# Copy only model weights (not full dataset)
RUN mkdir -p /code/data /code/assignment2
COPY ./data/*.pth /code/data/
COPY ./data/*.png /code/data/
COPY ./assignment2/*.pt /code/assignment2/

# Expose port 8000 (as required by assignment feedback)
EXPOSE 8000

# Start the FastAPI app with Uvicorn on port 8000
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
