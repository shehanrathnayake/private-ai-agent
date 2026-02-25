FROM python:3.11-slim

WORKDIR /app

# 1. Install torch (heavy layer, cached independently)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. Install other dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Environment variables with defaults
ENV APP_HOST=0.0.0.0
ENV APP_PORT=8000
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Run the application with hot-reload enabled
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
