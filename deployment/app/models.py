"""
Pydantic models for API request/response validation
"""

from pydantic import BaseModel, Field  # pyright: ignore[reportMissingImports]
from typing import List, Optional
from enum import Enum


class ModelType(str, Enum):
    """Available model types"""
    CREDIT_RISK = "credit_risk"
    FRAUD_DETECTION = "fraud_detection"
    CUSTOMER_SEGMENT = "customer_segment"


# ==========================================
# Credit Risk Prediction Models
# ==========================================

class CreditRiskRequest(BaseModel):
    """Request model for credit risk prediction"""
    loan_amount: float = Field(..., description="Loan amount requested", gt=0)
    loan_term: int = Field(..., description="Loan term in months", ge=12, le=60)
    interest_rate: float = Field(..., description="Interest rate (%)", ge=0, le=30)
    employment_length: int = Field(..., description="Years employed", ge=0)
    annual_income: float = Field(..., description="Annual income", gt=0)
    debt_to_income: float = Field(..., description="Debt-to-income ratio", ge=0, le=100)
    credit_score: Optional[int] = Field(None, description="Credit score", ge=300, le=850)
    num_credit_lines: int = Field(5, description="Number of credit lines", ge=0)
    delinquencies: int = Field(0, description="Number of delinquencies", ge=0)
    
    class Config:
        json_schema_extra = {
            "example": {
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
        }


class CreditRiskResponse(BaseModel):
    """Response model for credit risk prediction"""
    prediction: str = Field(..., description="'approved' or 'rejected'")
    probability: float = Field(..., description="Probability of default (0-1)")
    risk_level: str = Field(..., description="'low', 'medium', or 'high'")
    confidence: float = Field(..., description="Model confidence (0-1)")
    recommendation: str = Field(..., description="Business recommendation")


# ==========================================
# Fraud Detection Models
# ==========================================

class FraudDetectionRequest(BaseModel):
    """Request model for fraud detection"""
    transaction_amount: float = Field(..., description="Transaction amount", gt=0)
    transaction_time: int = Field(..., description="Time since first transaction (seconds)", ge=0)
    # Simplified - in reality would have V1-V28 PCA features
    features: List[float] = Field(..., description="PCA transformed features (V1-V28)", min_length=28, max_length=28)
    
    class Config:
        json_schema_extra = {
            "example": {
                "transaction_amount": 150.50,
                "transaction_time": 3600,
                "features": [0.1] * 28  # Placeholder
            }
        }


class FraudDetectionResponse(BaseModel):
    """Response model for fraud detection"""
    is_fraud: bool = Field(..., description="True if fraudulent")
    fraud_probability: float = Field(..., description="Probability of fraud (0-1)")
    risk_score: int = Field(..., description="Risk score (0-100)")
    recommendation: str = Field(..., description="Recommended action")
    flagged_features: List[str] = Field([], description="Features that triggered fraud alert")


# ==========================================
# Customer Segmentation Models
# ==========================================

class CustomerSegmentRequest(BaseModel):
    """Request model for customer segmentation"""
    age: int = Field(..., description="Customer age", ge=18, le=100)
    credit_amount: float = Field(..., description="Credit amount", gt=0)
    duration: int = Field(..., description="Loan duration (months)", ge=1)
    job: int = Field(..., description="Job category (0-3)", ge=0, le=3)
    saving_accounts: Optional[str] = Field(None, description="Savings account level")
    checking_account: Optional[str] = Field(None, description="Checking account level")
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "credit_amount": 5000,
                "duration": 24,
                "job": 2,
                "saving_accounts": "moderate",
                "checking_account": "moderate"
            }
        }


class CustomerSegmentResponse(BaseModel):
    """Response model for customer segmentation"""
    segment_id: int = Field(..., description="Cluster/segment ID")
    segment_name: str = Field(..., description="Descriptive segment name")
    characteristics: List[str] = Field(..., description="Key characteristics")
    recommended_products: List[str] = Field(..., description="Product recommendations")
    marketing_strategy: str = Field(..., description="Suggested marketing approach")


# ==========================================
# Health Check & Info Models
# ==========================================

class HealthResponse(BaseModel):
    """API health check response"""
    status: str = Field(..., description="API status")
    version: str = Field(..., description="API version")
    models_loaded: dict = Field(..., description="Loaded models status")


class ModelInfo(BaseModel):
    """Model information"""
    model_type: str
    algorithm: str
    accuracy: float
    trained_date: str
    version: str