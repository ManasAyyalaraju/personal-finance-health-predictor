"""
Utility functions for the API
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


def validate_features(features: Dict[str, Any], required_features: List[str]) -> bool:
    """
    Validate that all required features are present
    
    Args:
        features: Dictionary of feature values
        required_features: List of required feature names
    
    Returns:
        bool: True if all required features present
    """
    missing = [f for f in required_features if f not in features]
    if missing:
        logger.warning(f"Missing features: {missing}")
        return False
    return True


def preprocess_credit_risk_features(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Preprocess credit risk features for prediction
    
    Note: In production, this should match the exact preprocessing
    used during model training (scaling, encoding, etc.)
    """
    # Create feature dictionary matching training data
    features = {
        'loan_amnt': data['loan_amount'],
        'term': data['loan_term'],
        'int_rate': data['interest_rate'],
        'emp_length': data['employment_length'],
        'annual_inc': data['annual_income'],
        'dti': data['debt_to_income'],
        'open_acc': data.get('num_credit_lines', 5),
        'delinq_2yrs': data.get('delinquencies', 0),
    }
    
    # Convert to DataFrame
    df = pd.DataFrame([features])
    
    # Add any additional preprocessing here
    # - Scaling (StandardScaler)
    # - Encoding categorical variables
    # - Feature engineering
    
    return df


def preprocess_fraud_features(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Preprocess fraud detection features
    """
    features = {
        'Amount': data['transaction_amount'],
        'Time': data['transaction_time'],
    }
    
    # Add V1-V28 features
    for i in range(28):
        features[f'V{i+1}'] = data['features'][i]
    
    df = pd.DataFrame([features])
    
    return df


def preprocess_segmentation_features(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Preprocess customer segmentation features
    """
    features = {
        'Age': data['age'],
        'Credit amount': data['credit_amount'],
        'Duration': data['duration'],
        'Job': data['job'],
    }
    
    # Add encoding for categorical features if needed
    if 'saving_accounts' in data and data['saving_accounts']:
        # Map saving account levels to numeric
        saving_map = {'little': 0, 'moderate': 1, 'quite rich': 2, 'rich': 3}
        features['Saving accounts'] = saving_map.get(data['saving_accounts'], 1)
    
    if 'checking_account' in data and data['checking_account']:
        # Map checking account levels to numeric
        checking_map = {'little': 0, 'moderate': 1, 'rich': 2}
        features['Checking account'] = checking_map.get(data['checking_account'], 1)
    
    df = pd.DataFrame([features])
    
    return df


def calculate_risk_score(probability: float) -> int:
    """
    Convert probability to risk score (0-100)
    
    Args:
        probability: Probability value (0-1)
    
    Returns:
        int: Risk score (0-100)
    """
    return int(probability * 100)


def get_risk_level(probability: float) -> str:
    """
    Determine risk level from probability
    
    Args:
        probability: Probability value (0-1)
    
    Returns:
        str: Risk level ('low', 'medium', 'high')
    """
    if probability < 0.3:
        return "low"
    elif probability < 0.6:
        return "medium"
    else:
        return "high"


def format_currency(amount: float) -> str:
    """Format amount as currency"""
    return f"${amount:,.2f}"


def format_percentage(value: float) -> str:
    """Format value as percentage"""
    return f"{value * 100:.2f}%"