# AI Text Classifier API

A production-style text classification service built with FastAPI and PyTorch. The project exposes a `/predict` endpoint that accepts raw text and returns a predicted label, confidence score, and full class probabilities. It is structured like a small real-world ML backend, with separated API, schema, model, service, and training layers.

## Project Overview

This project demonstrates how to package a machine learning model as an API service suitable for portfolio presentation and internship applications. It includes:

- A FastAPI inference service with validation and error handling
- A PyTorch bag-of-words neural network for sentiment-style text classification
- Shared preprocessing logic between training and inference
- A training pipeline that creates a reusable model artifact
- Request timing, request IDs, and lightweight rate limiting for production-minded API behavior
- Optional API key protection for the prediction endpoint
- API tests with `pytest`
- GitHub Actions CI for automated training and test validation
- Docker support for containerized execution

## Tech Stack

- Python
- FastAPI
- PyTorch
- scikit-learn
- pandas
- NumPy
- Uvicorn
- pytest
- Docker

## Project Structure

```text
ai-text-classifier-api/
├── app/
│   ├── main.py
│   ├── api/
│   │   └── routes.py
│   ├── core/
│   │   ├── config.py
│   │   ├── middleware.py
│   │   └── security.py
│   ├── models/
│   │   └── classifier.py
│   ├── schemas/
│   │   └── prediction.py
│   └── services/
│       ├── inference.py
│       └── preprocessing.py
├── data/
│   ├── artifacts/
│   └── sample_sentiment.csv
├── training/
│   ├── train.py
│   └── preprocess.py
├── tests/
│   └── test_api.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── requirements.txt
├── Dockerfile
├── README.md
└── .gitignore
```

## Setup Instructions

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model artifact

```bash
python -m training.train
```

This command trains a lightweight text classifier on the included sample dataset and saves the artifact to `data/artifacts/text_classifier.pt`.

### Optional environment variables

```bash
export API_KEY="your-secret-key"
export RATE_LIMIT_REQUESTS=20
export RATE_LIMIT_WINDOW_SECONDS=60
export LOG_LEVEL="INFO"
```

These settings enable simple API key protection, configurable request throttling, and application logging.

## How to Run Locally

Start the API with Uvicorn:

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

Interactive docs:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## Example API Request and Response

### Request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I loved this product, it was amazing!"}'
```

### Response

```json
{
  "label": "positive",
  "confidence": 0.7439,
  "probabilities": {
    "negative": 0.2561,
    "positive": 0.7439
  },
  "model": {
    "model_version": "0.1.0",
    "model_type": "bag-of-words-feedforward",
    "max_sequence_length": 32
  }
}
```

## Error Handling

- Empty or whitespace-only text returns a validation error
- Missing or invalid `X-API-Key` values return `401 Unauthorized` when API key protection is enabled
- Excess requests to `/predict` return `429 Too Many Requests`
- Missing model artifacts return a `503 Service Unavailable` response
- Unexpected inference failures return a `500 Internal Server Error` response

## Production Features

- `X-Request-ID` header added to every response for traceability
- `X-Process-Time-Ms` header added to every response for basic latency visibility
- `/health` endpoint exposes model load status and model metadata
- GitHub Actions workflow validates training and tests on every push and pull request

## Running Tests

```bash
pytest
```

## Docker Support

Build the container:

```bash
docker build -t ai-text-classifier-api .
```

Run the container:

```bash
docker run -p 8000:8000 ai-text-classifier-api
```

The Docker build trains the sample model during image creation so the container is ready to serve predictions immediately.

## Future Improvements

- Replace the sample bootstrap dataset with a larger, domain-specific production dataset
- Add experiment tracking and metric logging with MLflow or Weights & Biases
- Introduce model versioning and artifact storage using cloud object storage such as Amazon S3
- Extend the API to support multi-class classification tasks
- Upgrade the baseline PyTorch model to transformer-based architectures such as BERT
- Deploy the service to a cloud platform such as AWS, GCP, or Render
