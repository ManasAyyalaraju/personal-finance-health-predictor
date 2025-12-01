# Deployment Strategy

This document outlines the deployment strategy used for the Personal Finance Health Predictor project.

## Overview

The project is deployed as a **FastAPI-based REST API** that serves machine learning models for real-time predictions. The deployment follows a microservices architecture with clear separation of concerns.

## Architecture

### Components

1. **FastAPI Application** (`deployment/app/main.py`)
   - REST API endpoints
   - Request/response handling
   - Error handling and logging

2. **Model Loader** (`deployment/app/ml_models.py`)
   - Singleton pattern for efficient model loading
   - Model caching to avoid reloading
   - Feature preprocessing for inference

3. **Data Models** (`deployment/app/models.py`)
   - Pydantic models for request validation
   - Response schemas
   - Type safety

4. **Configuration** (`deployment/config/config.py`)
   - Model paths
   - Prediction thresholds
   - Server settings

## Deployment

### Local Development

**Use Case**: Development and testing

```bash
cd deployment
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
uvicorn app.main:app --reload
```

**Access**:
- API: `http://localhost:8000`
- Interactive Docs: `http://localhost:8000/docs`


## API Endpoints

### Health & Info
- `GET /` - Health check
- `GET /health` - Detailed health status
- `GET /models` - List available models
- `GET /models/credit-risk/features` - Get expected features

### Predictions
- `POST /predict/credit-risk` - Credit risk prediction
- `POST /predict/fraud` - Fraud detection
- `POST /predict/segment` - Customer segmentation

## Model Loading Strategy

### Singleton Pattern
- Models loaded once at application startup
- Cached in memory for fast inference
- Avoids repeated file I/O operations

### Startup Sequence
1. Application starts
2. `ModelLoader` loads all models from disk
3. Models cached in memory
4. API ready to serve predictions

### Error Handling
- Graceful degradation if models fail to load
- Health endpoint reports model status
- Detailed error messages for debugging

## Feature Preprocessing

### Critical Requirement
The API must replicate **exact same preprocessing** used during training:
1. Feature engineering (same transformations)
2. Feature scaling (using saved scalers)
3. Categorical encoding (one-hot encoding)
4. Feature ordering (matching model expectations)

### Implementation
- `preprocess_credit_risk_features()` - Credit risk preprocessing
- `preprocess_fraud_features()` - Fraud detection preprocessing
- Loads saved scalers from `models/scaler_*.pkl`

### Unit Tests
- Located in `deployment/tests/`
- Test API endpoints
- Test model loading
- Test preprocessing functions

### Running Tests
```bash
cd deployment
pytest tests/test_api.py -v
pytest tests/ --cov=app --cov-report=html
```

## Configuration Management

### Environment Variables
- Model paths
- Prediction thresholds
- Server settings
- API keys (if needed)

### Configuration File
- `deployment/config/config.py`
- Centralized settings
- Easy to modify for different environments

## Deployment Checklist

### Pre-Deployment
- [ ] All models trained and saved
- [ ] Scalers saved alongside models
- [ ] API tests passing
- [ ] Environment variables configured
- [ ] Dependencies documented

### Deployment
- [ ] Server/container provisioned
- [ ] Code deployed
- [ ] Models accessible
- [ ] Health checks passing
- [ ] API documentation accessible

### Post-Deployment
- [ ] Monitor API performance
- [ ] Check error logs
- [ ] Verify predictions are correct
- [ ] Set up alerts
- [ ] Document any issues

## Troubleshooting

### Common Issues

1. **Models Not Loading**
   - Check model file paths
   - Verify models exist in `models/` directory
   - Check file permissions

2. **Feature Mismatch Errors**
   - Ensure preprocessing matches training
   - Verify feature order
   - Check scaler compatibility

3. **Performance Issues**
   - Increase worker processes (Gunicorn)
   - Use faster hardware
   - Consider model optimization

## Future Enhancements

1. **Model Versioning**: Track model versions
2. **A/B Testing**: Test new models in production
3. **Batch Predictions**: Process multiple requests
4. **Model Monitoring**: Track prediction drift
5. **Auto-scaling**: Scale based on load
6. **Caching**: Cache common predictions

## Documentation

- **API Documentation**: Auto-generated at `/docs` (Swagger UI)
- **Alternative Docs**: Available at `/redoc`
- **Deployment Guide**: `deployment/README.md`
- **API Examples**: `API_EXAMPLES.md`

