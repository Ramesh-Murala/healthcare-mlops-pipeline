FROM python:3.10-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir pandas numpy scikit-learn mlflow fastapi uvicorn pydantic

# Copy app and model
COPY api/app.py .
COPY mlruns/ ./mlruns/
COPY mlflow.db .

# Expose port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]