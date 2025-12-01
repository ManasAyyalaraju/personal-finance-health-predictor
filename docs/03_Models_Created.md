# Models Created

This document describes all machine learning models created in this project.

## Credit Risk Models

### Purpose
Predict the likelihood of loan default to help lenders make informed approval decisions.

### Models Trained

1. **Logistic Regression**
   - Algorithm: Linear classification with balanced class weights
   - Test ROC-AUC: 0.7037
   - Test Accuracy: 0.6302
   - Test Precision: 0.2827
   - Test Recall: 0.6775
   - **Status**: Best model (selected for deployment)

2. **Random Forest**
   - Algorithm: Ensemble of decision trees
   - Test ROC-AUC: 0.6970
   - Test Accuracy: 0.7059
   - Test Precision: 0.3137
   - Test Recall: 0.5254

3. **XGBoost**
   - Algorithm: Gradient boosting framework
   - Test ROC-AUC: 0.7021
   - Test Accuracy: 0.6502
   - Test Precision: 0.2900
   - Test Recall: 0.6436
   - **Status**: Used in deployment (alternative to best model)

4. **LightGBM**
   - Algorithm: Gradient boosting with leaf-wise tree growth
   - Test ROC-AUC: 0.7033
   - Test Accuracy: 0.6443
   - Test Precision: 0.2893
   - Test Recall: 0.6621

### Key Features
- Loan amount, term, interest rate
- Employment length, annual income
- Debt-to-income ratio, credit score
- Number of credit lines, delinquencies
- Feature-engineered variables (loan-to-income ratio, payment burden, etc.)

### Model Selection Criteria
Best model selected based on **Test ROC-AUC** score, with consideration for generalization (low overfitting).


## Fraud Detection Models

### Purpose
Detect fraudulent credit card transactions in real-time.

### Models Trained

1. **Isolation Forest**
   - Algorithm: Unsupervised anomaly detection
   - Test ROC-AUC: 0.9353
   - Test Accuracy: 0.9966
   - Test Precision: 0.1597
   - Test Recall: 0.2347
   - **Status**: Baseline model

2. **XGBoost + SMOTE**
   - Algorithm: Gradient boosting with SMOTE oversampling
   - Test ROC-AUC: 0.9799
   - Test Accuracy: 0.9984
   - Test Precision: 0.5152
   - Test Recall: 0.8673
   - **Status**: Best model (selected for deployment)

### Key Features
- Transaction amount and time
- 28 anonymized features (V1-V28) from PCA transformation
- Engineered features:
  - Hour of day, day of week
  - Amount categories (Small, Medium, Large, Very Large)
  - Time of day (Morning, Afternoon, Evening, Night)
  - High amount flag
  - V feature magnitude

### Class Imbalance Handling
- Original dataset: ~0.17% fraud cases
- SMOTE (Synthetic Minority Oversampling Technique) used to balance training data
- Critical for achieving high recall (catching fraud)

### Model Selection Criteria
Best model selected based on **Test ROC-AUC** and **Recall** (ability to catch fraud cases).


## Customer Segmentation Models

### Purpose
Group customers into behavioral segments for targeted marketing and product recommendations.

### Models Trained

1. **K-Means Clustering**
   - Tested with k = 2, 4, 5, 6 clusters
   - Algorithm: Partition-based clustering
   - **Status**: Alternative model

2. **Hierarchical Clustering**
   - Linkage methods tested:
     - Average Linkage
     - Complete Linkage
     - Ward Linkage
   - **Status**: Best model (Average Linkage selected for deployment)

3. **DBSCAN**
   - Algorithm: Density-based clustering
   - Automatically determines number of clusters
   - **Status**: Alternative model

### Optimal Number of Clusters
Selected **4 clusters** based on:
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Score
- Business interpretability

### Customer Segments Identified

1. **Segment 0: Senior Low-Risk Borrowers**
   - Oldest customer group
   - Short-term loans
   - Low credit amounts
   - Lowest risk profile

2. **Segment 1: Mature Medium-Term Borrowers**
   - Mature age group
   - Medium-term loans
   - Average credit amounts
   - Stable employment

3. **Segment 2: Young High-Value Borrowers**
   - Large loan amounts
   - Long-term commitments
   - Higher income bracket
   - Growth-focused

4. **Segment 3: Young Standard Borrowers**
   - Youngest customer segment
   - Medium-term loans
   - Building credit history
   - Digital-first

### Key Features
- Age
- Credit amount
- Loan duration
- Job category
- Savings/checking account status

### Model Selection Criteria
Best model selected based on clustering quality metrics and business interpretability.


## Model Performance Summary

| Use Case | Best Model | Test ROC-AUC | Test Accuracy | Key Metric |
|----------|-----------|--------------|---------------|------------|
| Credit Risk | Logistic Regression | 0.7037 | 0.6302 | ROC-AUC |
| Fraud Detection | XGBoost + SMOTE | 0.9799 | 0.9984 | Recall (0.8673) |
| Customer Segmentation | Hierarchical (Average) | N/A | N/A | Silhouette Score |

---

## Model Usage in Deployment

The deployment API (`deployment/app/`) uses:
- **Credit Risk**: XGBoost model (from `credit_risk_xgboost.pkl`)
- **Fraud Detection**: XGBoost + SMOTE model (from `fraud_detection_xgboost_smote.pkl`)
- **Customer Segmentation**: Hierarchical Average model (from `hierarchical_average.pkl`)

**Note**: The deployment uses XGBoost for credit risk instead of the best Logistic Regression model for consistency and better feature handling in the API.

