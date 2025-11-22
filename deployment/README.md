# Personal Finance Health Predictor - Deployment Guide

FastAPI-based REST API for ML model deployment.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Trained ML models (from notebooks)
- Virtual environment

### Installation

```bash
# Navigate to deployment directory
cd deployment

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Locally

```bash
# Start the API server
uvicorn app.main:app --reload

# API will be available at:
# http://localhost:8000

# Interactive API docs:
# http://localhost:8000/docs
```

---

## 📋 API Endpoints

### Health & Info

- `GET /` - Health check
- `GET /health` - Detailed health status
- `GET /models` - List available models

### Predictions

- `POST /predict/credit-risk` - Credit risk prediction
- `POST /predict/fraud` - Fraud detection
- `POST /predict/segment` - Customer segmentation

---

## 🧪 Testing

```bash
# Run tests
pytest tests/test_api.py -v

# Or with coverage
pytest tests/ --cov=app --cov-report=html
```

---

## 📡 Example API Calls

### Credit Risk Prediction

```bash
curl -X POST "http://localhost:8000/predict/credit-risk" \
  -H "Content-Type: application/json" \
  -d '{
    "loan_amount": 10000,
    "loan_term": 36,
    "interest_rate": 10.5,
    "employment_length": 5,
    "annual_income": 65000,
    "debt_to_income": 18.5,
    "credit_score": 700,
    "num_credit_lines": 5,
    "delinquencies": 0
  }'
```

### Fraud Detection

```bash
curl -X POST "http://localhost:8000/predict/fraud" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_amount": 150.50,
    "transaction_time": 3600,
    "features": [0.1, 0.2, ..., 0.28]
  }'
```

### Customer Segmentation

```bash
curl -X POST "http://localhost:8000/predict/segment" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "credit_amount": 5000,
    "duration": 24,
    "job": 2,
    "saving_accounts": "moderate",
    "checking_account": "moderate"
  }'
```

---

## ⚠️ Important Notes

### Model Files Required

The API requires trained model files to be present:

```
../models/
├── credit_risk/
│   └── credit_risk_xgboost.pkl
├── fraud_detection/
│   └── fraud_detection_xgboost_smote.pkl
├── clustering/
│   └── hierarchical_average.pkl
├── scaler_lending.pkl
└── scaler_fraud.pkl
```

**If models don't exist**, regenerate them by running:

1. `02_Data_Preprocessing.ipynb`
2. `03_Credit_Risk_Models.ipynb`
3. `04_Fraud_Detection_Models.ipynb`
4. `06_Customer_Segmentation.ipynb`

### Feature Preprocessing

⚠️ **Critical**: The current implementation uses simplified preprocessing. In production, you must:

1. Apply the **exact same preprocessing** used during training
2. Load the saved scalers (e.g., `StandardScaler`)
3. Apply proper encoding for categorical variables
4. Match feature names and order exactly

---

## 🚀 Deployment Options

### Option 1: Local Development

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Option 2: Production with Gunicorn

```bash
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Option 3: Docker (Create Dockerfile)

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Option 4: Cloud Platforms

- **Heroku**: `heroku create` + `git push heroku main`
- **Railway**: Connect GitHub repo
- **Render**: Connect GitHub repo
- **AWS EC2/Lambda**: Deploy with Docker or Serverless

---

## 📊 API Response Examples

### Credit Risk Response

```json
{
  "prediction": "approved",
  "probability": 0.25,
  "risk_level": "low",
  "confidence": 0.75,
  "recommendation": "Approve loan with standard terms"
}
```

### Fraud Detection Response

```json
{
  "is_fraud": false,
  "fraud_probability": 0.15,
  "risk_score": 15,
  "recommendation": "APPROVE - Low fraud risk",
  "flagged_features": []
}
```

### Customer Segmentation Response

```json
{
  "segment_id": 3,
  "segment_name": "Young Standard Borrowers",
  "characteristics": [
    "Youngest customer segment",
    "Medium-term loans",
    "Building credit history"
  ],
  "recommended_products": ["Digital banking", "Credit building products"],
  "marketing_strategy": "Growth products and financial education"
}
```

---

## 🔧 Configuration

Edit `config/config.py` to customize:

- Model paths
- Prediction thresholds
- Server settings
- CORS origins
- Logging levels

---

## 📝 License

This project is part of a portfolio demonstration.

---

## 👤 Author

**Manas Ayyalaraju**

- GitHub: [@ManasAyyalaraju](https://github.com/ManasAyyalaraju)
- Project: [personal-finance-health-predictor](https://github.com/ManasAyyalaraju/personal-finance-health-predictor)
