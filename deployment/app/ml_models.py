"""
ML Model loading and prediction utilities
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ModelLoader:
    """Singleton class to load and cache ML models"""
    
    _instance = None
    _models = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.models_dir = os.path.join(os.path.dirname(__file__), '../../models')
        self.loaded = False
    
    def load_models(self) -> Dict[str, bool]:
        """Load all trained models"""
        if self.loaded:
            return self._get_model_status()
        
        logger.info("Loading ML models...")
        status = {}
        
        # Load Credit Risk Model
        try:
            credit_path = os.path.join(self.models_dir, 'credit_risk', 'credit_risk_xgboost.pkl')
            model = joblib.load(credit_path)
            self._models['credit_risk'] = model
            
            # Log expected features for debugging
            if hasattr(model, 'feature_names_in_'):
                expected_features = list(model.feature_names_in_)
                logger.info(f"✓ Credit Risk model loaded - expects {len(expected_features)} features")
                logger.info(f"  First 10 features: {expected_features[:10]}")
                logger.info(f"  Last 10 features: {expected_features[-10:]}")
            else:
                logger.warning("Credit Risk model doesn't have feature_names_in_ attribute")
            
            # Load scaler if available
            scaler_path = os.path.join(self.models_dir, 'scaler_lending.pkl')
            if os.path.exists(scaler_path):
                self._models['credit_risk_scaler'] = joblib.load(scaler_path)
            status['credit_risk'] = True
        except Exception as e:
            logger.error(f"✗ Failed to load Credit Risk model: {e}")
            status['credit_risk'] = False
        
        # Load Fraud Detection Model
        try:
            fraud_path = os.path.join(self.models_dir, 'fraud_detection', 'fraud_detection_xgboost_smote.pkl')
            self._models['fraud_detection'] = joblib.load(fraud_path)
            # Load scaler if available
            scaler_path = os.path.join(self.models_dir, 'scaler_fraud.pkl')
            if os.path.exists(scaler_path):
                self._models['fraud_detection_scaler'] = joblib.load(scaler_path)
            status['fraud_detection'] = True
            logger.info("✓ Fraud Detection model loaded")
        except Exception as e:
            logger.error(f"✗ Failed to load Fraud Detection model: {e}")
            status['fraud_detection'] = False
        
        # Load Customer Segmentation Model
        try:
            segment_path = os.path.join(self.models_dir, 'clustering', 'hierarchical_average.pkl')
            self._models['customer_segment'] = joblib.load(segment_path)
            # Load scaler if available
            scaler_path = os.path.join(self.models_dir, 'clustering', 'scaler.pkl')
            if os.path.exists(scaler_path):
                self._models['customer_segment_scaler'] = joblib.load(scaler_path)
            status['customer_segment'] = True
            logger.info("✓ Customer Segmentation model loaded")
        except Exception as e:
            logger.error(f"✗ Failed to load Customer Segmentation model: {e}")
            status['customer_segment'] = False
        
        self.loaded = True
        return status
    
    def get_model(self, model_type: str):
        """Get a loaded model"""
        if not self.loaded:
            self.load_models()
        return self._models.get(model_type)
    
    def _get_model_status(self) -> Dict[str, bool]:
        """Get status of loaded models"""
        return {k: v is not None for k, v in self._models.items()}


# ==========================================
# Prediction Functions
# ==========================================

def preprocess_credit_risk_features(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Preprocess credit risk features to match model expectations.
    This replicates the feature engineering from the preprocessing notebook.
    """
    # Extract basic features
    loan_amnt = float(data['loan_amount'])
    loan_term = int(data['loan_term'])  # in months
    int_rate = float(data['interest_rate'])
    emp_length = data.get('employment_length', 0) or 0  # in years
    annual_inc = float(data['annual_income'])
    dti = float(data['debt_to_income'])
    open_acc = data.get('num_credit_lines') or 5
    delinq_2yrs = data.get('delinquencies') or 0
    
    # Default values for missing features (handle None explicitly)
    funded_amnt = data.get('funded_amount')
    if funded_amnt is None:
        funded_amnt = loan_amnt
    else:
        funded_amnt = float(funded_amnt)
    
    installment = data.get('installment')
    if installment is None:
        installment = loan_amnt * (1 + int_rate/100) / loan_term
    else:
        installment = float(installment)
    
    pub_rec = data.get('public_records') or 0
    revol_bal = data.get('revolving_balance') or 0
    revol_util = data.get('revolving_utilization') or 0
    total_acc = data.get('total_accounts') or open_acc
    credit_score = data.get('credit_score') or 700
    
    # Ensure all numeric values are properly typed
    pub_rec = float(pub_rec) if pub_rec is not None else 0.0
    revol_bal = float(revol_bal) if revol_bal is not None else 0.0
    revol_util = float(revol_util) if revol_util is not None else 0.0
    total_acc = float(total_acc) if total_acc is not None else float(open_acc)
    credit_score = float(credit_score) if credit_score is not None else 700.0
    emp_length = float(emp_length) if emp_length is not None else 0.0
    
    # Feature Engineering
    # 1. Loan to income ratio
    loan_to_income_ratio = loan_amnt / annual_inc if annual_inc > 0 else 0
    loan_to_income_ratio = min(loan_to_income_ratio, 2.0)  # Cap at 200%
    
    # 2. Monthly income
    monthly_income = annual_inc / 12
    
    # 3. Monthly payment burden
    monthly_payment = float(installment)
    monthly_payment_burden = monthly_payment / monthly_income if monthly_income > 0 else 0
    
    # 4. Employment length in years (already provided, but ensure it's numeric)
    emp_length_years = float(emp_length)
    
    # 5. Loan term in months (already provided)
    loan_term_months = float(loan_term)
    
    # 6. Total payment amount (estimated)
    total_payment_amount = float(monthly_payment) * float(loan_term_months)
    
    # 7. Total interest paid (estimated)
    total_interest_paid = float(total_payment_amount) - float(loan_amnt)
    
    # 8. Credit history years (estimated from credit score and accounts)
    credit_history_years = min(max((credit_score - 300) / 50, 0), 30)
    
    # 9. Risk score (composite)
    risk_score = 0
    if dti > 20:
        risk_score += 1
    if dti > 30:
        risk_score += 2
    if loan_to_income_ratio > 0.3:
        risk_score += 1
    if loan_to_income_ratio > 0.5:
        risk_score += 2
    if credit_score < 600:
        risk_score += 2
    elif credit_score < 700:
        risk_score += 1
    
    # 10. Grade risk points (estimated from interest rate)
    # Grade mapping: A=0, B=1, C=2, D=3, E=4, F=5, G=6
    if int_rate < 8:
        grade_risk_points = 0  # A
    elif int_rate < 10:
        grade_risk_points = 1  # B
    elif int_rate < 13:
        grade_risk_points = 2  # C
    elif int_rate < 15:
        grade_risk_points = 3  # D
    elif int_rate < 18:
        grade_risk_points = 4  # E
    elif int_rate < 22:
        grade_risk_points = 5  # F
    else:
        grade_risk_points = 6  # G
    
    risk_score += grade_risk_points
    
    # 11. Term encoded (36 months = 0, 60 months = 1)
    term_encoded = 1 if loan_term_months >= 60 else 0
    
    # 12. DTI Risk Category
    if dti < 10:
        dti_risk = 'Low_Risk'
    elif dti < 20:
        dti_risk = 'Moderate_Risk'
    elif dti < 30:
        dti_risk = 'High_Risk'
    else:
        dti_risk = 'Very_High_Risk'
    
    # 13. Interest Rate Category
    if int_rate < 10:
        int_rate_cat = 'Low_Rate'
    elif int_rate < 15:
        int_rate_cat = 'Medium_Rate'
    elif int_rate < 20:
        int_rate_cat = 'High_Rate'
    else:
        int_rate_cat = 'Very_High_Rate'
    
    # 14. Employment Stability
    if emp_length_years >= 10:
        emp_stability = 'Very_Experienced'
    elif emp_length_years >= 5:
        emp_stability = 'Experienced'
    elif emp_length_years >= 2:
        emp_stability = 'Stable'
    else:
        emp_stability = 'New'
    
    # 15. Revolving Utilization Category
    if revol_util < 30:
        revol_util_cat = 'Low_Util'
    elif revol_util < 70:
        revol_util_cat = 'Medium_Util'
    elif revol_util < 90:
        revol_util_cat = 'High_Util'
    else:
        revol_util_cat = 'Maxed_Out'
    
    # 16. Grade (one-hot encoded)
    grade = chr(ord('A') + min(grade_risk_points, 6))  # A through G
    
    # 17. Home ownership (default to RENT if not provided)
    home_ownership = data.get('home_ownership') or 'RENT'
    
    # 18. Verification status (default to Verified if not provided)
    verification_status = data.get('verification_status') or 'Verified'
    
    # Build feature dictionary with all required columns
    features = {
        # Continuous features
        'loan_amnt': loan_amnt,
        'funded_amnt': funded_amnt,
        'int_rate': int_rate,
        'installment': installment,
        'annual_inc': annual_inc,
        'dti': dti,
        'delinq_2yrs': delinq_2yrs,
        'open_acc': open_acc,
        'pub_rec': pub_rec,
        'revol_bal': revol_bal,
        'revol_util': revol_util,
        'total_acc': total_acc,
        'loan_to_income_ratio': loan_to_income_ratio,
        'monthly_income': monthly_income,
        'monthly_payment_burden': monthly_payment_burden,
        'emp_length_years': emp_length_years,
        'loan_term_months': loan_term_months,
        'total_payment_amount': total_payment_amount,
        'total_interest_paid': total_interest_paid,
        'credit_history_years': credit_history_years,
        'risk_score': risk_score,
        'grade_risk_points': grade_risk_points,
        'term_encoded': term_encoded,
        
        # One-hot encoded categorical features (all must be present)
        'grade_B': 1 if grade == 'B' else 0,
        'grade_C': 1 if grade == 'C' else 0,
        'grade_D': 1 if grade == 'D' else 0,
        'grade_E': 1 if grade == 'E' else 0,
        'grade_F': 1 if grade == 'F' else 0,
        'grade_G': 1 if grade == 'G' else 0,
        # grade_A is dropped (reference category)
        
        'home_ownership_MORTGAGE': 1 if home_ownership == 'MORTGAGE' else 0,
        'home_ownership_OWN': 1 if home_ownership == 'OWN' else 0,
        'home_ownership_RENT': 1 if home_ownership == 'RENT' else 0,
        
        'verification_status_Source Verified': 1 if verification_status == 'Source Verified' else 0,
        'verification_status_Verified': 1 if verification_status == 'Verified' else 0,
        # verification_status_Not Verified is dropped (reference category)
        
        'dti_risk_category_Moderate_Risk': 1 if dti_risk == 'Moderate_Risk' else 0,
        'dti_risk_category_High_Risk': 1 if dti_risk == 'High_Risk' else 0,
        'dti_risk_category_Very_High_Risk': 1 if dti_risk == 'Very_High_Risk' else 0,
        # dti_risk_category_Low_Risk is dropped (reference category)
        
        'interest_rate_category_Medium_Rate': 1 if int_rate_cat == 'Medium_Rate' else 0,
        'interest_rate_category_High_Rate': 1 if int_rate_cat == 'High_Rate' else 0,
        'interest_rate_category_Very_High_Rate': 1 if int_rate_cat == 'Very_High_Rate' else 0,
        # interest_rate_category_Low_Rate is dropped (reference category)
        
        'emp_stability_Stable': 1 if emp_stability == 'Stable' else 0,
        'emp_stability_Experienced': 1 if emp_stability == 'Experienced' else 0,
        'emp_stability_Very_Experienced': 1 if emp_stability == 'Very_Experienced' else 0,
        # emp_stability_New is dropped (reference category)
        
        'revol_util_category_Medium_Util': 1 if revol_util_cat == 'Medium_Util' else 0,
        'revol_util_category_High_Util': 1 if revol_util_cat == 'High_Util' else 0,
        'revol_util_category_Maxed_Out': 1 if revol_util_cat == 'Maxed_Out' else 0,
        # revol_util_category_Low_Util is dropped (reference category)
        
        # Loan status columns - NOTE: These are typically removed during training (data leakage)
        # Only include if the model expects them
        'loan_status_Current': 0,
        'loan_status_Default': 0,
        'loan_status_Fully Paid': 1,  # Assume new application starts as "Fully Paid" (not yet defaulted)
        'loan_status_In Grace Period': 0,
        'loan_status_Late (16-30 days)': 0,
        'loan_status_Late (31-120 days)': 0,
    }
    
    # Create DataFrame
    df = pd.DataFrame([features])
    
    return df


class CreditRiskPredictor:
    """Credit risk prediction logic"""
    
    def __init__(self, model, model_loader: Optional[ModelLoader] = None):
        self.model = model
        self.model_loader = model_loader
        # Get expected feature names from the model
        if hasattr(model, 'feature_names_in_'):
            self.expected_features = list(model.feature_names_in_)
            logger.info(f"Model expects {len(self.expected_features)} features")
        elif hasattr(model, 'get_booster'):
            # For XGBoost models
            try:
                booster = model.get_booster()
                if hasattr(booster, 'feature_names'):
                    self.expected_features = booster.feature_names
                    logger.info(f"XGBoost model expects {len(self.expected_features)} features")
                else:
                    self.expected_features = None
            except Exception as e:
                logger.warning(f"Could not get feature names from XGBoost model: {e}")
                self.expected_features = None
        else:
            self.expected_features = None
            logger.warning("Could not determine model's expected features")
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make credit risk prediction"""
        try:
            # Preprocess features to match model expectations
            features = preprocess_credit_risk_features(data)
            
            # Get expected features from model if available
            if self.expected_features:
                logger.info(f"Model expects {len(self.expected_features)} features")
                logger.info(f"Preprocessed features: {len(features.columns)} features")
                logger.info(f"Model expected features (first 10): {list(self.expected_features)[:10]}")
                logger.info(f"Model expected features (last 10): {list(self.expected_features)[-10:]}")
                
                # Ensure all expected features are present
                missing_features = set(self.expected_features) - set(features.columns)
                if missing_features:
                    logger.warning(f"Adding {len(missing_features)} missing features with default 0")
                    logger.warning(f"Missing features: {list(missing_features)}")
                    for feat in missing_features:
                        features[feat] = 0.0
                
                # Remove any extra features not expected by the model
                extra_features = set(features.columns) - set(self.expected_features)
                if extra_features:
                    logger.info(f"Removing {len(extra_features)} extra features not in model")
                    logger.info(f"Extra features: {list(extra_features)}")
                    features = features.drop(columns=list(extra_features))
                
                # Double-check: ensure we only have expected features
                current_features = set(features.columns)
                expected_set = set(self.expected_features)
                if current_features != expected_set:
                    logger.error(f"Feature mismatch after cleanup!")
                    logger.error(f"  Features in dataframe but not expected: {current_features - expected_set}")
                    logger.error(f"  Expected features not in dataframe: {expected_set - current_features}")
                    # Force alignment: keep only expected features
                    features = features[[f for f in self.expected_features if f in features.columns]]
                    # Add any missing expected features
                    for feat in expected_set - set(features.columns):
                        features[feat] = 0.0
                
                # Reorder to match model's expected order
                features = features[self.expected_features]
                logger.info(f"Final feature count: {len(features.columns)}, matching model expectations")
                logger.info(f"Final feature names match: {list(features.columns) == list(self.expected_features)}")
            else:
                logger.warning("Model feature names not available, using all features")
            
            # Apply scaler if available (only to features the scaler was trained on)
            if self.model_loader and 'credit_risk_scaler' in self.model_loader._models:
                scaler = self.model_loader._models['credit_risk_scaler']
                # Get the features the scaler expects (it was trained on a subset)
                if hasattr(scaler, 'feature_names_in_'):
                    scaler_features = list(scaler.feature_names_in_)
                    logger.info(f"Scaler expects {len(scaler_features)} features")
                    # Check that all scaler features exist in the dataframe
                    missing_scaler_features = [f for f in scaler_features if f not in features.columns]
                    if missing_scaler_features:
                        logger.warning(f"Scaler expects features not in dataframe: {missing_scaler_features}")
                    else:
                        logger.info(f"Scaling {len(scaler_features)} features in scaler's expected order")
                        # Scale only the features the scaler expects, in the exact order it expects
                        # This is critical - the scaler was trained with features in a specific order
                        features_to_scale = features[scaler_features]
                        features_scaled = scaler.transform(features_to_scale)
                        # Update the dataframe with scaled values (maintaining original column order)
                        for i, feat in enumerate(scaler_features):
                            features[feat] = features_scaled[0, i]
                        logger.info("Features scaled successfully")
                else:
                    # Fallback: scaler doesn't have feature names, use the original list
                    continuous_features = [
                        'loan_amnt', 'funded_amnt', 'int_rate', 'installment', 'annual_inc', 'dti',
                        'open_acc', 'revol_bal', 'revol_util', 'total_acc',
                        'loan_to_income_ratio', 'monthly_income', 'monthly_payment_burden',
                        'emp_length_years', 'total_payment_amount',
                        'total_interest_paid', 'credit_history_years'
                    ]
                    # Only scale features that exist in the dataframe
                    continuous_features = [f for f in continuous_features if f in features.columns]
                    if continuous_features:
                        features[continuous_features] = scaler.transform(features[continuous_features])
            
            # Get prediction probability
            # Based on testing, XGBoost works correctly with DataFrame when column names match
            # Ensure features DataFrame has columns in the exact order expected by the model
            if self.expected_features:
                # Reorder to match model's expected order (this is critical for XGBoost)
                features = features[self.expected_features]
                logger.info(f"Features reordered to match model expectations")
                logger.info(f"Feature order matches: {list(features.columns) == list(self.expected_features)}")
            
            try:
                # Use DataFrame directly - XGBoost 2.0+ validates feature names and order
                # When column names match feature_names_in_ exactly, it works correctly
                proba = self.model.predict_proba(features)[0]
                logger.info("Prediction successful with DataFrame")
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Prediction failed: {error_msg}")
                logger.error(f"Model expects {len(self.expected_features) if self.expected_features else 'unknown'} features")
                logger.error(f"DataFrame columns ({len(features.columns)}): {list(features.columns)[:10]}...")
                logger.error(f"Expected features ({len(self.expected_features) if self.expected_features else 0}): {list(self.expected_features)[:10] if self.expected_features else 'N/A'}...")
                
                # If it's a feature name error, the model file might be corrupted or from different training
                if "feature names" in error_msg.lower() or "unseen at fit time" in error_msg.lower():
                    logger.error("XGBoost feature name validation error detected!")
                    logger.error("This suggests the model file may be from a different training run.")
                    logger.error("Attempting to use model's internal feature mapping...")
                    # Try to get the actual feature names the model was trained with
                    if hasattr(self.model, 'get_booster'):
                        try:
                            booster = self.model.get_booster()
                            # XGBoost doesn't have validate_features parameter
                            # The issue is likely that the model file doesn't match the expected features
                            # Re-raise with more context
                            raise ValueError(f"Model feature mismatch. The model was trained with different features than expected. Model expects: {list(self.expected_features) if self.expected_features else 'unknown'}. Please retrain the model or check the model file.")
                        except:
                            raise ValueError(f"Model feature mismatch. Expected features: {list(self.expected_features) if self.expected_features else 'unknown'}. Error: {error_msg}")
                else:
                    raise
            
            default_prob = proba[1]  # Probability of default
            
            # Determine prediction
            prediction = "rejected" if default_prob > 0.5 else "approved"
            
            # Risk level
            if default_prob < 0.3:
                risk_level = "low"
            elif default_prob < 0.6:
                risk_level = "medium"
            else:
                risk_level = "high"
            
            # Confidence
            confidence = max(proba)
            
            # Recommendation
            if prediction == "approved" and risk_level == "low":
                recommendation = "Approve loan with standard terms"
            elif prediction == "approved" and risk_level == "medium":
                recommendation = "Approve with higher interest rate or collateral"
            else:
                recommendation = "Reject application - high default risk"
            
            return {
                "prediction": prediction,
                "probability": float(default_prob),
                "risk_level": risk_level,
                "confidence": float(confidence),
                "recommendation": recommendation
            }
        
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise


class FraudDetector:
    """Fraud detection logic"""
    
    def __init__(self, model):
        self.model = model
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect fraud in transaction"""
        # Prepare features
        features = pd.DataFrame([{
            'Amount': data['transaction_amount'],
            'Time': data['transaction_time'],
            **{f'V{i+1}': data['features'][i] for i in range(28)}
        }])
        
        try:
            # Get prediction
            proba = self.model.predict_proba(features)[0]
            fraud_prob = proba[1]  # Probability of fraud
            is_fraud = fraud_prob > 0.5
            
            # Risk score (0-100)
            risk_score = int(fraud_prob * 100)
            
            # Recommendation
            if is_fraud:
                if fraud_prob > 0.9:
                    recommendation = "BLOCK - High fraud probability. Immediate review required."
                elif fraud_prob > 0.7:
                    recommendation = "HOLD - Suspicious activity. Manual verification needed."
                else:
                    recommendation = "REVIEW - Moderate risk. Additional authentication recommended."
            else:
                recommendation = "APPROVE - Low fraud risk. Transaction appears legitimate."
            
            # Flagged features (simplified)
            flagged = []
            if data['transaction_amount'] > 1000:
                flagged.append("high_amount")
            if fraud_prob > 0.7:
                flagged.append("suspicious_pattern")
            
            return {
                "is_fraud": is_fraud,
                "fraud_probability": float(fraud_prob),
                "risk_score": risk_score,
                "recommendation": recommendation,
                "flagged_features": flagged
            }
        
        except Exception as e:
            logger.error(f"Fraud detection error: {e}")
            raise


class CustomerSegmenter:
    """Customer segmentation logic"""
    
    # Segment definitions (from Day 7)
    SEGMENTS = {
        0: {
            "name": "Senior Low-Risk Borrowers",
            "characteristics": [
                "Oldest customer group",
                "Short-term loans",
                "Low credit amounts",
                "Lowest risk profile"
            ],
            "products": [
                "Premium savings accounts",
                "Wealth management services",
                "Senior-focused insurance products"
            ],
            "strategy": "Premium products and loyalty programs"
        },
        1: {
            "name": "Mature Medium-Term Borrowers",
            "characteristics": [
                "Mature age group",
                "Medium-term loans",
                "Average credit amounts",
                "Stable employment"
            ],
            "products": [
                "Standard loan products",
                "Auto loans",
                "Home equity lines"
            ],
            "strategy": "Relationship banking and cross-selling"
        },
        2: {
            "name": "Young High-Value Borrowers",
            "characteristics": [
                "Large loan amounts",
                "Long-term commitments",
                "Higher income bracket",
                "Growth-focused"
            ],
            "products": [
                "Premium financing",
                "Investment products",
                "Business loans"
            ],
            "strategy": "Premium services with careful risk monitoring"
        },
        3: {
            "name": "Young Standard Borrowers",
            "characteristics": [
                "Youngest customer segment",
                "Medium-term loans",
                "Building credit history",
                "Digital-first"
            ],
            "products": [
                "Digital banking",
                "Credit building products",
                "Educational resources"
            ],
            "strategy": "Growth products and financial education"
        }
    }
    
    def __init__(self, model):
        self.model = model
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Assign customer to segment"""
        # Prepare features (simplified)
        features = pd.DataFrame([{
            'Age': data['age'],
            'Credit amount': data['credit_amount'],
            'Duration': data['duration'],
            'Job': data['job'],
        }])
        
        try:
            # Note: Hierarchical clustering (AgglomerativeClustering) doesn't have a predict method
            # For new data, we need to use fit_predict which will retrain on the new data
            # This is a limitation - in production, consider using KMeans or storing cluster centers
            # For now, we'll use fit_predict but note this is not ideal for production
            
            # Alternative: If the model has labels_ attribute, we could use a different approach
            # For hierarchical clustering, we typically need the full dataset to assign new points
            # This is a simplified implementation - in production, use KMeans or store cluster centers
            
            # Get cluster assignment (this will retrain on new data - not ideal but works)
            # In production, consider loading a KMeans model instead which has a proper predict method
            if hasattr(self.model, 'predict'):
                segment_id = int(self.model.predict(features)[0])
            else:
                # Hierarchical clustering workaround - assign to nearest cluster based on simple heuristics
                # This is a simplified approach - proper implementation would need cluster centers
                age = data['age']
                credit = data['credit_amount']
                duration = data['duration']
                
                # Simple heuristic-based assignment (temporary workaround)
                if age >= 50 and credit < 5000:
                    segment_id = 0  # Senior Low-Risk
                elif age >= 40 and credit < 10000:
                    segment_id = 1  # Mature Medium-Term
                elif credit >= 15000:
                    segment_id = 2  # Young High-Value
                else:
                    segment_id = 3  # Young Standard
                
                logger.warning("Using heuristic-based segmentation (hierarchical clustering doesn't support prediction on new data)")
            
            # Get segment info
            segment_info = self.SEGMENTS.get(segment_id, self.SEGMENTS[0])
            
            return {
                "segment_id": segment_id,
                "segment_name": segment_info["name"],
                "characteristics": segment_info["characteristics"],
                "recommended_products": segment_info["products"],
                "marketing_strategy": segment_info["strategy"]
            }
        
        except Exception as e:
            logger.error(f"Segmentation error: {e}")
            raise