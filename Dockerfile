# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Label
LABEL maintainer="sualeh77@gmail.com"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VERSION=2.1.3 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

# Add Poetry to PATH
ENV PATH="$POETRY_HOME/bin:$PATH"

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Set working directory
WORKDIR /app

# Copy all project files first
COPY . .

# Install dependencies
RUN poetry install --only main

# Install PyTorch (CPU version)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Make entrypoint script executable (moved after COPY)
RUN chmod +x /app/entrypoint.sh

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["/app/entrypoint.sh"]