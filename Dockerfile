FROM python:3.11-slim

# Install FFmpeg, libspeex, and build tools
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libspeex1 \
    libspeex-dev \
    speex \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Expose port
EXPOSE 8080

# Run with gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "3600", "main:app"]
