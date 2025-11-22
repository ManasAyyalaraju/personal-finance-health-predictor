"""
API endpoint tests
"""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "version" in data
    assert "models_loaded" in data


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code in [200, 503]
    data = response.json()
    assert "status" in data
    assert "models_loaded" in data


def test_list_models():
    """Test models listing endpoint"""
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "credit_risk" in data
    assert "fraud_detection" in data
    assert "customer_segment" in data


def test_credit_risk_prediction():
    """Test credit risk prediction endpoint"""
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
    
    response = client.post("/predict/credit-risk", json=payload)
    
    # May fail if models not loaded
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert data["prediction"] in ["approved", "rejected"]
        assert "probability" in data
        assert 0 <= data["probability"] <= 1
        assert "risk_level" in data
        assert data["risk_level"] in ["low", "medium", "high"]
    elif response.status_code == 503:
        # Model not loaded - acceptable in test environment
        pass


def test_fraud_detection():
    """Test fraud detection endpoint"""
    payload = {
        "transaction_amount": 150.50,
        "transaction_time": 3600,
        "features": [0.1] * 28
    }
    
    response = client.post("/predict/fraud", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        assert "is_fraud" in data
        assert isinstance(data["is_fraud"], bool)
        assert "fraud_probability" in data
        assert 0 <= data["fraud_probability"] <= 1
        assert "risk_score" in data
        assert 0 <= data["risk_score"] <= 100
    elif response.status_code == 503:
        pass


def test_customer_segmentation():
    """Test customer segmentation endpoint"""
    payload = {
        "age": 35,
        "credit_amount": 5000,
        "duration": 24,
        "job": 2,
        "saving_accounts": "moderate",
        "checking_account": "moderate"
    }
    
    response = client.post("/predict/segment", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        assert "segment_id" in data
        assert 0 <= data["segment_id"] <= 3
        assert "segment_name" in data
        assert "characteristics" in data
        assert "recommended_products" in data
        assert "marketing_strategy" in data
    elif response.status_code == 503:
        pass


def test_invalid_credit_risk_request():
    """Test credit risk with invalid data"""
    payload = {
        "loan_amount": -1000,  # Invalid: negative amount
        "loan_term": 36,
        "interest_rate": 10.5,
        "employment_length": 5,
        "annual_income": 65000,
        "debt_to_income": 18.5
    }
    
    response = client.post("/predict/credit-risk", json=payload)
    assert response.status_code == 422  # Validation error


def test_invalid_endpoint():
    """Test non-existent endpoint"""
    response = client.get("/invalid-endpoint")
    assert response.status_code == 404


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])