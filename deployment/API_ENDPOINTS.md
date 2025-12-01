# 🚀 API Endpoints

Your Personal Finance Health Predictor API is running at: **http://localhost:8000**

## 📋 Available Endpoints

### Health & Info Endpoints

| Method | Endpoint | Description |
|--------|-----------|-------------|
| `GET` | `/` | Root endpoint - Health check and API info |
| `GET` | `/health` | Detailed health status with model loading status |
| `GET` | `/models` | List all available ML models |
| `GET` | `/models/credit-risk/features` | Get expected features for credit risk model |

### Prediction Endpoints

| Method | Endpoint | Description |
|--------|-----------|-------------|
| `POST` | `/predict/credit-risk` | Predict credit risk for loan application |
| `POST` | `/predict/fraud` | Detect fraud in transaction |
| `POST` | `/predict/segment` | Assign customer to behavioral segment |

### Documentation Endpoints

| Method | Endpoint | Description |
|--------|-----------|-------------|
| `GET` | `/docs` | Interactive API documentation (Swagger UI) |
| `GET` | `/redoc` | Alternative API documentation (ReDoc) |
| `GET` | `/openapi.json` | OpenAPI schema (JSON) |

---

## 📖 Example API Calls

### 1. Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "models_loaded": {
    "credit_risk": true,
    "fraud_detection": true,
    "customer_segment": true
  }
}
```

### 2. Credit Risk Prediction

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

**Response:**
```json
{
  "prediction": "approved",
  "probability": 0.25,
  "risk_level": "low",
  "confidence": 0.75,
  "recommendation": "Approve loan with standard terms"
}
```

### 3. Fraud Detection

```bash
curl -X POST "http://localhost:8000/predict/fraud" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_amount": 150.50,
    "transaction_time": 3600,
    "features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
  }'
```

**Response:**
```json
{
  "is_fraud": false,
  "fraud_probability": 0.15,
  "risk_score": 15,
  "recommendation": "APPROVE - Low fraud risk",
  "flagged_features": []
}
```

### 4. Customer Segmentation

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

**Response:**
```json
{
  "segment_id": 3,
  "segment_name": "Young Standard Borrowers",
  "characteristics": [
    "Youngest customer segment",
    "Medium-term loans",
    "Building credit history"
  ],
  "recommended_products": [
    "Digital banking",
    "Credit building products"
  ],
  "marketing_strategy": "Growth products and financial education"
}
```

---

## 🌐 Access in Browser

1. **Interactive API Docs (Swagger UI)**: http://localhost:8000/docs
2. **Alternative Docs (ReDoc)**: http://localhost:8000/redoc
3. **Health Check**: http://localhost:8000/health
4. **Models List**: http://localhost:8000/models

---

## 🧪 Testing with Python

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Credit risk prediction
payload = {
    "loan_amount": 10000,
    "loan_term": 36,
    "interest_rate": 10.5,
    "employment_length": 5,
    "annual_income": 65000,
    "debt_to_income": 18.5,
    "credit_score": 700,
    "num_credit_lines": 5,
    "delinquencies": 0
}

response = requests.post(
    "http://localhost:8000/predict/credit-risk",
    json=payload
)
print(response.json())
```

---

## 📝 Notes

- All endpoints return JSON responses
- The API uses CORS middleware (allows all origins by default)
- Models are loaded at startup and cached in memory
- Check `/health` endpoint to verify all models are loaded
- Use `/docs` for interactive testing and full API schema

---

**API Version**: 1.0.0  
**Base URL**: http://localhost:8000

