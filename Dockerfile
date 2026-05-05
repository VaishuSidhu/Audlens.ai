# --- Build Frontend ---
FROM node:20 AS frontend-build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# --- Setup Python Backend ---
FROM python:3.10-slim

# Hugging Face security requirements
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install system dependencies for audio (must be done as root)
USER root
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*
USER user

# Copy backend requirements and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY --chown=user . .

# Copy built frontend from the previous stage
COPY --from=frontend-build --chown=user /app/dist/client /app/dist/client

# Expose port 7860 (required by Hugging Face)
EXPOSE 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Command to run the backend
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
