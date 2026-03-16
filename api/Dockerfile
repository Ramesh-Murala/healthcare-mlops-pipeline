FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir pandas numpy scikit-learn fastapi uvicorn pydantic

COPY api/app.py .
COPY api/model.pkl .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]