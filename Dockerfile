# Use a small official Python image as the base
FROM python:3.12-slim-bookworm

# Install curl and certificates (needed to download the uv installer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Download and install the latest uv (Python package manager/runner)
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
# Make sure uv is on PATH
ENV PATH="/root/.local/bin/:${PATH}"

# Set the working directory inside the container
WORKDIR /code

# Copy dependency metadata first (for better layer caching)
# If you also have a uv.lock, you can add another COPY line for it.
COPY pyproject.toml /code/

# Install dependencies with uv
# (If you have a uv.lock and want reproducible installs, use: uv sync --frozen)
RUN uv sync

# Install the spaCy model by wheel (no pip needed inside venv)
RUN uv pip install \
  https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.7.1/en_core_web_md-3.7.1-py3-none-any.whl

# Copy the application source code
# Make sure your project has: app/main.py, app/bigram_model.py, and app/__init__.py
COPY ./app /code/app

# Start the FastAPI app with Uvicorn on port 80 inside the container
# We expose it to the host with -p 8000:80 when running `docker run`
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
