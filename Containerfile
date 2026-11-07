# Containerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (layer caching)
COPY app/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app/ .

# Copy dataset
COPY data/diabetes.csv /app/data/diabetes.csv

# Train the model at build time (saves into /model — override with volume at runtime)
RUN mkdir -p /model && python train.py

# Expose Flask port
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]