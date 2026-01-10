# Voice AI Intent Classification

Intent classification system for a Voice AI assistant that helps Rwandan citizens access government services. Built for the Irembo ML Engineer take-home assignment.

## Features

- **Multilingual Support**: Handles Kinyarwanda, English, and code-switched utterances
- **High Accuracy**: 97.3% Macro F1 on held-out test set
- **Production-Ready API**: FastAPI with confidence-based fallback to human agents
- **Low Latency**: ~50ms inference time per utterance

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Installation

```bash
# Clone the repository
git clone git@github.com:chukwujekwu-code/voice-ai-intent-classifier.git
cd voice-ai-intent

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -r requirements.txt
```

### Run the API

```bash
# Start the server
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

# Or without uv
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

- **Swagger UI**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### Test the API

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Ndashaka kureba status ya application yanjye"}'

# Response
{
  "success": true,
  "prediction": {
    "intent": "check_application_status",
    "confidence": 0.955,
    "is_low_confidence": false
  },
  "input_text": "Ndashaka kureba status ya application yanjye"
}
```

## Running the Notebooks

The analysis and model training are documented in Jupyter notebooks.

```bash
# Start Jupyter
uv run jupyter notebook

# Or without uv
jupyter notebook
```

Open `notebooks/eda.ipynb` to see:
- Exploratory Data Analysis
- Model comparison (TF-IDF vs LaBSE)
- Hyperparameter tuning with GridSearchCV
- Final evaluation on test set

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check and model status |
| `/intents` | GET | List all 13 intent classes |
| `/predict` | POST | Classify a single utterance |
| `/predict/batch` | POST | Classify multiple utterances (max 100) |
| `/predict/top-k` | POST | Get top-k most likely intents |
| `/predict/with-fallback` | POST | Production endpoint with human escalation |

### Example: Production Endpoint with Fallback

```bash
curl -X POST http://localhost:8000/predict/with-fallback \
  -H "Content-Type: application/json" \
  -d '{"text": "help me please", "confidence_threshold": 0.7}'
```

Response (low confidence triggers escalation):
```json
{
  "prediction": {"intent": "speak_to_agent", "confidence": 0.45},
  "requires_review": true,
  "action": "escalate_to_human",
  "alternatives": [
    {"intent": "speak_to_agent", "probability": 0.45},
    {"intent": "complaint_or_support_ticket", "probability": 0.22}
  ],
  "message": "Low confidence (45.00%). Consider routing to human agent."
}
```

## Project Structure

```
voice-ai-intent/
├── app/                         # FastAPI application
│   ├── main.py                  # API endpoints
│   └── models.py               # Request/response models
├── src/                         # Core ML module
│   ├── config.py                # Settings and configuration
│   ├── encoder.py               # LaBSE embedding wrapper
│   └── classifier.py            # Intent classifier
├── models/                      # Trained models
│   ├── labse_logreg_classifier.joblib
│   └── labse_logreg_info.joblib
├── notebooks/
│   └── eda.ipynb                # EDA + model training
├── Datasets/
│   ├── voiceai_intent_train.csv
│   ├── voiceai_intent_val.csv
│   └── voiceai_intent_test.csv
├── requirements.txt
└── pyproject.toml
```

## Model Performance

| Model | CV F1 | Test Accuracy | Test F1 |
|-------|-------|---------------|---------|
| TF-IDF + LogReg | 0.996 | 98.55% | **98.60%** |
| LaBSE + LogReg | 0.985 | 97.10% | 97.29% |
| LaBSE + XGBoost | 0.950 | 92.75% | 91.95% |

**Deployed Model**: LaBSE + Logistic Regression (better cross-lingual generalization)

### Performance by Language

| Language | Accuracy | Samples |
|----------|----------|---------|
| Kinyarwanda (rw) | 97.4% | 39 |
| English (en) | 95.2% | 21 |
| Mixed (code-switched) | 100% | 9 |

## Intent Classes

The model classifies utterances into 13 intents:

| Intent | Description |
|--------|-------------|
| `check_application_status` | Check status of submitted application |
| `requirements_information` | Ask about required documents |
| `start_new_application` | Begin a new application |
| `payment_help` | Issues with payments |
| `fees_information` | Ask about costs/fees |
| `appointment_booking` | Schedule an appointment |
| `reset_password_login_help` | Account access issues |
| `service_eligibility` | Check if eligible for a service |
| `speak_to_agent` | Request human assistance |
| `complaint_or_support_ticket` | File a complaint |
| `update_application_details` | Modify submitted application |
| `cancel_or_reschedule_appointment` | Change appointment |
| `document_upload_help` | Help uploading documents |

## Example Utterances

| Utterance | Language | Intent |
|-----------|----------|--------|
| "Ndashaka kureba status ya application yanjye" | rw + en | check_application_status |
| "What are the requirements for passport?" | en | requirements_information |
| "Nashaka gufata rendez-vous yo gufata passport" | rw + fr | appointment_booking |
| "How much does it cost?" | en | fees_information |

## Technical Decisions

| Decision | Rationale |
|----------|-----------|
| LaBSE over TF-IDF | Better cross-lingual generalization despite 1.3% lower F1 |
| Logistic Regression | Calibrated probabilities for confidence-based routing |
| Confidence threshold 0.7 | Balance between automation and human escalation |
| Singleton pattern | Avoid reloading 500MB model on each request |

## Development

```bash
# Install dev dependencies
uv sync --all-extras

# Run with auto-reload
uv run uvicorn app.main:app --reload

```

