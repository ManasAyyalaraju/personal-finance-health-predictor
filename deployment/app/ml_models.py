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


def preprocess_fraud_features(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Preprocess fraud detection features to match model expectations.
    Creates all 44 features including engineered features.
    Model expects features in this order:
    Time, V1-V28, Amount, hour, day, amount_scaled, high_amount, hour_bucket,
    transactions_in_hour, v_feature_magnitude, time_of_day_*, amount_category_*
    """
    amount = float(data['transaction_amount'])
    time = int(data['transaction_time'])
    v_features = data['features']  # V1-V28
    
    # Ensure we have exactly 28 V features
    if len(v_features) != 28:
        raise ValueError(f"Expected 28 V features, got {len(v_features)}")
    
    # Convert time (seconds since first transaction) to hour and day
    # Assuming time is in seconds since first transaction
    total_seconds = time
    hour = (total_seconds // 3600) % 24
    day = total_seconds // 86400  # Days since first transaction
    
    # Feature Engineering (matching preprocessing notebook)
    
    # 1. Hour bucket (categorize hour into buckets)
    if 0 <= hour < 6:
        hour_bucket = 0  # Night
    elif 6 <= hour < 12:
        hour_bucket = 1  # Morning
    elif 12 <= hour < 18:
        hour_bucket = 2  # Afternoon
    else:
        hour_bucket = 3  # Evening
    
    # 2. Amount scaled (normalized amount, using typical scaling)
    # Using robust scaling: (x - median) / IQR
    # For simplicity, using log transform + standardization approximation
    amount_scaled = np.log1p(amount)  # log(1 + amount) to handle small values
    
    # 3. High amount flag
    high_amount = 1 if amount > 1000 else 0
    
    # 4. Amount categories (one-hot encoded)
    if amount < 100:
        amount_category_Small = 1
        amount_category_Medium = 0
        amount_category_Large = 0
        amount_category_Very_Large = 0
    elif amount < 1000:
        amount_category_Small = 0
        amount_category_Medium = 1
        amount_category_Large = 0
        amount_category_Very_Large = 0
    elif amount < 5000:
        amount_category_Small = 0
        amount_category_Medium = 0
        amount_category_Large = 1
        amount_category_Very_Large = 0
    else:
        amount_category_Small = 0
        amount_category_Medium = 0
        amount_category_Large = 0
        amount_category_Very_Large = 1
    
    # 5. Time of day (one-hot encoded)
    # Note: Afternoon is the reference category (not included in model)
    if 6 <= hour < 12:
        time_of_day_Morning = 1
        time_of_day_Evening = 0
        time_of_day_Night = 0
    elif 12 <= hour < 17:
        # Afternoon is reference category (all 0s)
        time_of_day_Morning = 0
        time_of_day_Evening = 0
        time_of_day_Night = 0
    elif 17 <= hour < 22:
        time_of_day_Morning = 0
        time_of_day_Evening = 1
        time_of_day_Night = 0
    else:
        time_of_day_Morning = 0
        time_of_day_Evening = 0
        time_of_day_Night = 1
    
    # 6. V feature magnitude (Euclidean norm of V features)
    v_magnitude = sum(v**2 for v in v_features) ** 0.5
    
    # 7. Transactions in hour (we can't calculate from single transaction, use default)
    # This would normally be calculated from historical data
    transactions_in_hour = 1  # Default to 1 for single transaction
    
    # Build features in the EXACT order the model expects
    # Order: Time, V1-V28, Amount, hour, day, amount_scaled, high_amount, hour_bucket,
    #        transactions_in_hour, v_feature_magnitude, time_of_day_*, amount_category_*
    
    # Create ordered list of feature values matching model's expected order
    feature_values = [
        time,  # Time
    ]
    
    # Add V1-V28 features
    for i in range(28):
        feature_values.append(float(v_features[i]))
    
    # Add remaining features in expected order
    feature_values.extend([
        amount,  # Amount
        hour,  # hour
        day,  # day
        amount_scaled,  # amount_scaled
        high_amount,  # high_amount
        hour_bucket,  # hour_bucket
        transactions_in_hour,  # transactions_in_hour
        v_magnitude,  # v_feature_magnitude
        time_of_day_Evening,  # time_of_day_Evening
        time_of_day_Morning,  # time_of_day_Morning
        time_of_day_Night,  # time_of_day_Night
        amount_category_Small,  # amount_category_Small
        amount_category_Medium,  # amount_category_Medium
        amount_category_Large,  # amount_category_Large
        amount_category_Very_Large,  # amount_category_Very_Large
    ])
    
    # Define column names in the exact order the model expects
    column_names = [
        'Time',
    ]
    # Add V1-V28
    column_names.extend([f'V{i+1}' for i in range(28)])
    # Add remaining columns
    column_names.extend([
        'Amount',
        'hour',
        'day',
        'amount_scaled',
        'high_amount',
        'hour_bucket',
        'transactions_in_hour',
        'v_feature_magnitude',
        'time_of_day_Evening',
        'time_of_day_Morning',
        'time_of_day_Night',
        'amount_category_Small',
        'amount_category_Medium',
        'amount_category_Large',
        'amount_category_Very_Large',
    ])
    
    # Create DataFrame with explicit column order
    df = pd.DataFrame([feature_values], columns=column_names)
    
    # Verify we have the correct number of features
    if len(column_names) != len(feature_values):
        raise ValueError(f"Feature count mismatch: {len(column_names)} columns but {len(feature_values)} values")
    
    if len(df.columns) != 44:
        raise ValueError(f"Expected 44 features, got {len(df.columns)}: {list(df.columns)}")
    
    logger.info(f"Created DataFrame with {len(df.columns)} features in order: {list(df.columns)}")
    
    return df


class FraudDetector:
    """Fraud detection logic"""
    
    def __init__(self, model, model_loader: Optional[ModelLoader] = None):
        self.model = model
        self.model_loader = model_loader
        # Get expected feature names from the model
        if hasattr(model, 'feature_names_in_'):
            self.expected_features = list(model.feature_names_in_)
            logger.info(f"Fraud model expects {len(self.expected_features)} features")
        elif hasattr(model, 'get_booster'):
            # For XGBoost models
            try:
                booster = model.get_booster()
                if hasattr(booster, 'feature_names') and booster.feature_names:
                    self.expected_features = list(booster.feature_names)
                    logger.info(f"XGBoost fraud model expects {len(self.expected_features)} features")
                    logger.info(f"XGBoost feature order (first 10): {list(self.expected_features)[:10]}")
                    logger.info(f"XGBoost feature order (last 10): {list(self.expected_features)[-10:]}")
                else:
                    # Try to get from feature_names_in_ if available (scikit-learn wrapper)
                    if hasattr(model, 'feature_names_in_'):
                        self.expected_features = list(model.feature_names_in_)
                        logger.info(f"XGBoost (sklearn wrapper) expects {len(self.expected_features)} features")
                    else:
                        self.expected_features = None
                        logger.warning("XGBoost model does not have feature names stored")
            except Exception as e:
                logger.warning(f"Could not get feature names from XGBoost model: {e}")
                self.expected_features = None
        else:
            self.expected_features = None
            logger.warning("Could not determine fraud model's expected features")
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect fraud in transaction"""
        try:
            # Log input data to verify it's being received correctly
            logger.info(f"Received input data:")
            logger.info(f"  transaction_amount: {data.get('transaction_amount')}")
            logger.info(f"  transaction_time: {data.get('transaction_time')}")
            logger.info(f"  features length: {len(data.get('features', []))}")
            logger.info(f"  V1: {data.get('features', [])[0] if len(data.get('features', [])) > 0 else 'N/A'}")
            logger.info(f"  V14: {data.get('features', [])[13] if len(data.get('features', [])) > 13 else 'N/A'}")
            logger.info(f"  V10: {data.get('features', [])[9] if len(data.get('features', [])) > 9 else 'N/A'}")
            
            # Preprocess features to match model expectations
            features = preprocess_fraud_features(data)
            
            logger.info(f"Preprocessed features shape: {features.shape}")
            logger.info(f"Preprocessed feature columns ({len(features.columns)}): {list(features.columns)}")
            logger.info(f"Sample preprocessed values: Amount={features['Amount'].values[0]}, V1={features['V1'].values[0]}, V14={features['V14'].values[0]}")
            
            # Get expected features from model if available
            if self.expected_features:
                logger.info(f"Fraud model expects {len(self.expected_features)} features")
                logger.info(f"Model expected features: {list(self.expected_features)}")
                
                # Ensure all expected features are present FIRST
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
                
                # Verify we have all expected features before reordering
                if set(features.columns) != set(self.expected_features):
                    missing = set(self.expected_features) - set(features.columns)
                    raise ValueError(f"Missing features before reordering: {missing}. Have: {list(features.columns)}, Need: {list(self.expected_features)}")
                
                # CRITICAL: Reorder to match model's expected order
                # Create new DataFrame with features in exact order expected by model
                logger.info(f"Reordering features to match model expectations...")
                logger.info(f"Features before reorder ({len(features.columns)}): {list(features.columns)}")
                
                # Extract values in the exact order expected by model and create new DataFrame
                # This ensures the DataFrame columns are in the exact order XGBoost expects
                feature_values_ordered = []
                for feat in self.expected_features:
                    if feat not in features.columns:
                        raise ValueError(f"Required feature '{feat}' not found in DataFrame. Available: {list(features.columns)}")
                    feature_values_ordered.append(features[feat].values[0])
                
                # Create new DataFrame with features in exact order expected by model
                features = pd.DataFrame([feature_values_ordered], columns=self.expected_features)
                
                logger.info(f"Features after reorder ({len(features.columns)}): {list(features.columns)}")
                logger.info(f"Final feature count: {len(features.columns)}, matching model expectations")
                logger.info(f"Final feature order matches: {list(features.columns) == list(self.expected_features)}")
                
                # Final verification
                if len(features.columns) != len(self.expected_features):
                    raise ValueError(f"Feature count mismatch: have {len(features.columns)}, need {len(self.expected_features)}")
                if list(features.columns) != list(self.expected_features):
                    raise ValueError(f"Feature order mismatch. Have: {list(features.columns)}, Need: {list(self.expected_features)}")
            else:
                logger.warning("Model feature names not available, using all features")
            
            # Apply scaler if available
            # NOTE: Apply scaler BEFORE final reordering to ensure scaled values are preserved
            if self.model_loader and 'fraud_detection_scaler' in self.model_loader._models:
                scaler = self.model_loader._models['fraud_detection_scaler']
                # Get the features the scaler expects
                if hasattr(scaler, 'feature_names_in_'):
                    scaler_features = list(scaler.feature_names_in_)
                    logger.info(f"Fraud scaler expects {len(scaler_features)} features")
                    logger.info(f"Scaler feature order (first 5): {scaler_features[:5]}")
                    # Check that all scaler features exist in the dataframe
                    missing_scaler_features = [f for f in scaler_features if f not in features.columns]
                    if missing_scaler_features:
                        logger.warning(f"Scaler expects features not in dataframe: {missing_scaler_features}")
                    else:
                        logger.info(f"Scaling {len(scaler_features)} features in scaler's expected order")
                        # Scale only the features the scaler expects, in the exact order it expects
                        features_to_scale = features[scaler_features]
                        logger.info(f"Sample values before scaling (first 5): {features_to_scale.iloc[0, :5].values}")
                        features_scaled = scaler.transform(features_to_scale)
                        logger.info(f"Sample values after scaling (first 5): {features_scaled[0, :5]}")
                        # Update the dataframe with scaled values
                        for i, feat in enumerate(scaler_features):
                            features[feat] = features_scaled[0, i]
                        logger.info("Features scaled successfully")
                else:
                    logger.warning("Fraud scaler does not have feature_names_in_ attribute, skipping scaling")
            else:
                logger.info("No fraud detection scaler found, skipping scaling step")
            
            # Final reorder to match model's expected order (critical for XGBoost)
            if self.expected_features:
                # CRITICAL: XGBoost requires features in the exact order it was trained with
                # Extract values in the exact order from model's feature_names
                feature_values_ordered = []
                for feat in self.expected_features:
                    if feat not in features.columns:
                        raise ValueError(f"Model expects feature '{feat}' but it's not in DataFrame. Available: {list(features.columns)}")
                    feature_values_ordered.append(features[feat].values[0])
                
                # Create DataFrame with features in EXACT order expected by model
                # XGBoost uses column names to match features, so we need a DataFrame, not numpy array
                features_final = pd.DataFrame([feature_values_ordered], columns=self.expected_features)
                
                logger.info(f"Final DataFrame shape: {features_final.shape}")
                logger.info(f"Final DataFrame columns ({len(features_final.columns)}): {list(features_final.columns)}")
                logger.info(f"Feature order matches model: {list(features_final.columns) == list(self.expected_features)}")
            else:
                features_final = features
            
            # Final verification before prediction
            if self.expected_features:
                if len(features_final.columns) != len(self.expected_features):
                    raise ValueError(
                        f"Feature count mismatch before prediction: "
                        f"DataFrame has {len(features_final.columns)} features, "
                        f"model expects {len(self.expected_features)}. "
                        f"Expected features: {list(self.expected_features)}"
                    )
                if list(features_final.columns) != list(self.expected_features):
                    raise ValueError(
                        f"Feature order mismatch before prediction. "
                        f"Have: {list(features_final.columns)}, "
                        f"Need: {list(self.expected_features)}"
                    )
                logger.info(f"✅ Verified: DataFrame has {len(features_final.columns)} features in correct order")
            
            # Get prediction
            try:
                # Log sample feature values before prediction to debug
                if self.expected_features:
                    logger.info(f"Sample feature values before prediction:")
                    logger.info(f"  Time: {features_final['Time'].values[0]}")
                    logger.info(f"  Amount: {features_final['Amount'].values[0]}")
                    logger.info(f"  V1: {features_final['V1'].values[0]}")
                    logger.info(f"  V14: {features_final['V14'].values[0]}")
                    logger.info(f"  V10: {features_final['V10'].values[0]}")
                    logger.info(f"  high_amount: {features_final['high_amount'].values[0]}")
                
                # Pass DataFrame to XGBoost - it will match features by column names
                # The DataFrame columns are in the exact order the model expects
                proba = self.model.predict_proba(features_final)[0]
                logger.info(f"Fraud prediction successful: proba={proba}, fraud_prob={proba[1]:.6f}")
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Fraud prediction failed: {error_msg}")
                logger.error(f"Model expects {len(self.expected_features) if self.expected_features else 'unknown'} features")
                logger.error(f"DataFrame shape: {features_final.shape if 'features_final' in locals() else 'N/A'}")
                if self.expected_features:
                    logger.error(f"Expected feature order: {list(self.expected_features)}")
                    logger.error(f"First 10 expected: {list(self.expected_features)[:10]}")
                    logger.error(f"Last 10 expected: {list(self.expected_features)[-10:]}")
                if 'features_final' in locals():
                    logger.error(f"DataFrame columns ({len(features_final.columns)}): {list(features_final.columns)}")
                raise
            
            fraud_prob = proba[1]  # Probability of fraud from model
            
            # Calculate flagged features BEFORE applying adjustments
            flagged = []
            has_high_amount = data['transaction_amount'] > 1000
            has_anomalous_v14 = len(data['features']) > 13 and abs(data['features'][13]) > 2
            has_anomalous_v10 = len(data['features']) > 9 and abs(data['features'][9]) > 2
            
            if has_high_amount:
                flagged.append("high_amount")
            if has_anomalous_v14:
                flagged.append("anomalous_v14")
            if has_anomalous_v10:
                flagged.append("anomalous_v10")
            
            # Apply rule-based adjustments when multiple suspicious indicators are present
            # This is a common pattern in fraud detection: combine model predictions with rule-based heuristics
            original_fraud_prob = fraud_prob
            flag_count = len(flagged)
            
            # Check for extreme feature values that might indicate fraud
            # If multiple V features have extreme values (|value| > 3), increase suspicion
            extreme_v_count = sum(1 for v in data['features'] if abs(v) > 3)
            if extreme_v_count >= 5:  # 5 or more extreme V features
                fraud_prob = max(fraud_prob, 0.5)
                if "multiple_extreme_features" not in flagged:
                    flagged.append("multiple_extreme_features")
                logger.info(f"Rule-based adjustment: {extreme_v_count} extreme V features detected. "
                          f"Original prob: {original_fraud_prob:.6f}, Adjusted prob: {fraud_prob:.6f}")
            
            # If multiple critical flags are present, increase fraud probability
            # Priority: 3+ flags > high_amount + anomalous > 2 flags
            if flag_count >= 3:
                # Three or more flags: very suspicious - highest priority
                fraud_prob = max(fraud_prob, 0.7)
                logger.info(f"Rule-based adjustment: 3+ flags detected (high_amount, anomalous_v14, anomalous_v10). "
                          f"Original prob: {original_fraud_prob:.6f}, Adjusted prob: {fraud_prob:.6f}")
            elif flag_count >= 2:
                # High amount + anomalous V14/V10 is a strong indicator of fraud
                if has_high_amount and (has_anomalous_v14 or has_anomalous_v10):
                    # Increase probability significantly when high amount combines with anomalous features
                    fraud_prob = max(fraud_prob, 0.6)  # At least 60% if model says lower
                    logger.info(f"Rule-based adjustment: high_amount + anomalous features detected. "
                              f"Original prob: {original_fraud_prob:.6f}, Adjusted prob: {fraud_prob:.6f}")
                else:
                    # Two flags (but not high_amount + anomalous): moderate increase
                    fraud_prob = max(fraud_prob, min(0.4, original_fraud_prob * 2))
                    logger.info(f"Rule-based adjustment: 2 flags detected. "
                              f"Original prob: {original_fraud_prob:.6f}, Adjusted prob: {fraud_prob:.6f}")
            
            # Additional check: Very high transaction amounts should always be suspicious
            if data['transaction_amount'] > 10000:
                fraud_prob = max(fraud_prob, 0.65)
                logger.info(f"Rule-based adjustment: Very high transaction amount ({data['transaction_amount']:.2f}) detected. "
                          f"Adjusted prob: {fraud_prob:.6f}")
            
            # Ensure fraud_prob stays in valid range [0, 1]
            fraud_prob = min(1.0, max(0.0, fraud_prob))
            
            # Log final adjusted probability for debugging
            if abs(fraud_prob - original_fraud_prob) > 0.01:  # Only log if significant change
                logger.info(f"Final fraud probability after all adjustments: {original_fraud_prob:.6f} -> {fraud_prob:.6f}")
            
            # Determine fraud status with adjusted probability
            # Use lower threshold (0.3) when flags are present, standard (0.5) otherwise
            fraud_threshold = 0.3 if flag_count >= 2 else 0.5
            is_fraud = fraud_prob > fraud_threshold
            
            logger.info(f"Fraud detection result: is_fraud={is_fraud}, fraud_prob={fraud_prob:.6f}, threshold={fraud_threshold:.2f}, flag_count={flag_count}")
            
            # Risk score (0-100)
            risk_score = int(fraud_prob * 100)
            
            # Add suspicious_pattern flag if probability is high
            if fraud_prob > 0.7:
                flagged.append("suspicious_pattern")
            
            # Recommendation based on adjusted probability
            if is_fraud:
                if fraud_prob > 0.9:
                    recommendation = "BLOCK - High fraud probability. Immediate review required."
                elif fraud_prob > 0.7:
                    recommendation = "HOLD - Suspicious activity. Manual verification needed."
                else:
                    recommendation = "REVIEW - Moderate risk. Additional authentication recommended."
            else:
                # Even if not flagged as fraud, high risk should trigger review
                if fraud_prob > 0.3 or flag_count >= 2:
                    recommendation = "REVIEW - Multiple risk indicators detected. Additional verification recommended."
                else:
                    recommendation = "APPROVE - Low fraud risk. Transaction appears legitimate."
            
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