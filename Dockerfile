# ─── Email-OpenEnv Docker Image ───────────────────────────────────────────────
FROM python:3.11-slim

# Metadata
LABEL maintainer="OpenEnv Contributors"
LABEL description="Email Open Environment – AI email processing agent"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose Gradio port
EXPOSE 7860

# Environment variable for OpenAI key (override at runtime)
ENV OPENAI_API_KEY=""

# Gradio needs this to bind to all interfaces inside Docker
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Run the Gradio app
CMD ["python", "app.py"]
