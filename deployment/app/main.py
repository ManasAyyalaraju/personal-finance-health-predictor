"""
Personal Finance Health Predictor - FastAPI Application
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from contextlib import asynccontextmanager

from .models import (
    CreditRiskRequest, CreditRiskResponse,
    FraudDetectionRequest, FraudDetectionResponse,
    CustomerSegmentRequest, CustomerSegmentResponse,
    HealthResponse, ModelInfo
)
from .ml_models import (
    ModelLoader,
    CreditRiskPredictor,
    FraudDetector,
    CustomerSegmenter
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize model loader
model_loader = ModelLoader()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("Starting Personal Finance Health Predictor API...")
    model_status = model_loader.load_models()
    logger.info(f"Models loaded: {model_status}")
    yield
    # Shutdown
    logger.info("Shutting down API...")


# Initialize FastAPI app
app = FastAPI(
    title="Personal Finance Health Predictor API",
    description="AI-powered financial prediction endpoints for credit risk, fraud detection, and customer segmentation",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Health & Info Endpoints
# ==========================================

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def root():
    """API health check and info"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "models_loaded": model_loader._get_model_status()
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Detailed health check"""
    model_status = model_loader._get_model_status()
    
    # Check if all models are loaded
    all_loaded = all(model_status.values())
    status_code = status.HTTP_200_OK if all_loaded else status.HTTP_503_SERVICE_UNAVAILABLE
    
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "healthy" if all_loaded else "degraded",
            "version": "1.0.0",
            "models_loaded": model_status
        }
    )


@app.get("/models", tags=["Info"])
async def list_models():
    """List available models and their info"""
    return {
        "credit_risk": {
            "endpoint": "/predict/credit-risk",
            "algorithm": "XGBoost",
            "accuracy": "~70% ROC-AUC",
            "description": "Predicts loan default risk"
        },
        "fraud_detection": {
            "endpoint": "/predict/fraud",
            "algorithm": "XGBoost + SMOTE",
            "accuracy": "97.7% ROC-AUC",
            "description": "Detects fraudulent transactions"
        },
        "customer_segment": {
            "endpoint": "/predict/segment",
            "algorithm": "Hierarchical Clustering (Average Linkage)",
            "description": "Assigns customers to behavioral segments"
        }
    }


@app.get("/models/credit-risk/features", tags=["Info"])
async def get_credit_risk_features():
    """Get the expected features for the credit risk model"""
    try:
        model = model_loader.get_model('credit_risk')
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Credit risk model not loaded"
            )
        
        if hasattr(model, 'feature_names_in_'):
            features = list(model.feature_names_in_)
            return {
                "feature_count": len(features),
                "features": features
            }
        else:
            return {
                "error": "Model does not have feature_names_in_ attribute",
                "model_type": str(type(model))
            }
    except Exception as e:
        logger.error(f"Error getting model features: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model features: {str(e)}"
        )


# ==========================================
# Prediction Endpoints
# ==========================================

@app.post("/predict/credit-risk", response_model=CreditRiskResponse, tags=["Predictions"])
async def predict_credit_risk(request: CreditRiskRequest):
    """
    Predict credit risk for loan application
    
    Returns:
    - prediction: 'approved' or 'rejected'
    - probability: Probability of default (0-1)
    - risk_level: 'low', 'medium', or 'high'
    - confidence: Model confidence
    - recommendation: Business recommendation
    """
    try:
        model = model_loader.get_model('credit_risk')
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Credit risk model not loaded"
            )
        
        predictor = CreditRiskPredictor(model, model_loader)
        result = predictor.predict(request.dict())
        
        return result
    
    except Exception as e:
        logger.error(f"Credit risk prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/fraud", response_model=FraudDetectionResponse, tags=["Predictions"])
async def predict_fraud(request: FraudDetectionRequest):
    """
    Detect fraud in transaction
    
    Returns:
    - is_fraud: Boolean fraud indicator
    - fraud_probability: Probability of fraud (0-1)
    - risk_score: Risk score (0-100)
    - recommendation: Recommended action
    - flagged_features: Features that triggered alert
    """
    try:
        model = model_loader.get_model('fraud_detection')
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Fraud detection model not loaded"
            )
        
        detector = FraudDetector(model)
        result = detector.predict(request.dict())
        
        return result
    
    except Exception as e:
        logger.error(f"Fraud detection error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection failed: {str(e)}"
        )


@app.post("/predict/segment", response_model=CustomerSegmentResponse, tags=["Predictions"])
async def predict_segment(request: CustomerSegmentRequest):
    """
    Assign customer to behavioral segment
    
    Returns:
    - segment_id: Cluster ID (0-3)
    - segment_name: Descriptive name
    - characteristics: Key characteristics
    - recommended_products: Product recommendations
    - marketing_strategy: Suggested approach
    """
    try:
        model = model_loader.get_model('customer_segment')
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Customer segmentation model not loaded"
            )
        
        segmenter = CustomerSegmenter(model)
        result = segmenter.predict(request.dict())
        
        return result
    
    except Exception as e:
        logger.error(f"Segmentation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Segmentation failed: {str(e)}"
        )


# ==========================================
# Error Handlers
# ==========================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found. See /docs for available endpoints."}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)