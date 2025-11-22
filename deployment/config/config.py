"""
Configuration settings for the API
"""

import os
import logging
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    APP_NAME: str = "Personal Finance Health Predictor"
    APP_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = False
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    
    # Model Settings
    MODELS_DIR: Path = Path(__file__).parent.parent.parent / "models"
    CREDIT_RISK_MODEL: str = "credit_risk/credit_risk_xgboost.pkl"
    FRAUD_MODEL: str = "fraud_detection/fraud_detection_xgboost_smote.pkl"
    SEGMENT_MODEL: str = "clustering/hierarchical_average.pkl"
    
    # Prediction Thresholds
    CREDIT_RISK_THRESHOLD: float = 0.5
    FRAUD_THRESHOLD: float = 0.5
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # CORS
    CORS_ORIGINS: list = ["*"]  # In production, specify allowed origins
    
    # Rate Limiting (if implemented)
    RATE_LIMIT_PER_MINUTE: int = 100
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Create settings instance
settings = Settings()


# Model paths
def get_model_path(model_name: str) -> Path:
    """Get full path to model file"""
    return settings.MODELS_DIR / model_name


# Validate model paths exist
def validate_models() -> dict:
    """Check if all model files exist"""
    models = {
        "credit_risk": settings.MODELS_DIR / settings.CREDIT_RISK_MODEL,
        "fraud_detection": settings.MODELS_DIR / settings.FRAUD_MODEL,
        "customer_segment": settings.MODELS_DIR / settings.SEGMENT_MODEL,
    }
    
    status = {}
    for name, path in models.items():
        status[name] = path.exists()
        if not path.exists():
            logger.warning(f"Model file not found: {path}")
    
    return status