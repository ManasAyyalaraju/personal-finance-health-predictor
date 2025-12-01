# Notebook Summaries

This document provides a brief overview of each notebook in the project.

## 01_Data_Exploration.ipynb

**Purpose**: Initial exploration and understanding of the datasets.

**Key Activities**:
- Loads three main datasets: Lending Club, Credit Card Fraud, and German Credit
- Performs exploratory data analysis (EDA)
- Analyzes data distributions, missing values, and correlations
- Creates visualizations to understand data characteristics
- Identifies data quality issues and patterns

**Outputs**: 
- Data quality reports
- Distribution plots
- Correlation matrices
- Summary statistics

---

## 02_Data_Preprocessing.ipynb

**Purpose**: Clean, transform, and prepare data for model training.

**Key Activities**:
- Handles missing values across all datasets
- Performs feature engineering (creating new features from existing ones)
- Encodes categorical variables (one-hot encoding, label encoding)
- Scales numerical features using StandardScaler
- Splits data into training and testing sets
- Handles class imbalance using SMOTE for fraud detection data
- Saves processed datasets to `data/processed/`

**Outputs**:
- Processed CSV and PKL files for each dataset
- Training and testing splits
- Feature-engineered datasets ready for modeling

---

## 03_Credit_Risk_Models.ipynb

**Purpose**: Train and evaluate models for credit risk prediction.

**Key Activities**:
- Trains four classification models:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - LightGBM
- Evaluates models using accuracy, ROC-AUC, precision, recall, F1-score
- Compares model performance and identifies best model
- Analyzes feature importance
- Creates visualizations (ROC curves, confusion matrices, feature importance plots)
- Saves trained models to `models/credit_risk/`

**Best Model**: Logistic Regression (Test ROC-AUC: 0.7037)

**Outputs**:
- Trained model files (.pkl)
- Performance comparison visualizations
- Feature importance analysis

---

## 04_Fraud_Detection_Models.ipynb

**Purpose**: Train and evaluate models for fraud detection.

**Key Activities**:
- Handles severe class imbalance (fraud cases are rare)
- Trains two models:
  - Isolation Forest (unsupervised anomaly detection)
  - XGBoost with SMOTE (oversampling to balance classes)
- Evaluates models focusing on recall (catching fraud) and precision (minimizing false alarms)
- Creates precision-recall curves and confusion matrices
- Analyzes which features are most indicative of fraud
- Saves best model to `models/fraud_detection/`

**Best Model**: XGBoost + SMOTE (Test ROC-AUC: 0.9799, Recall: 0.8673)

**Outputs**:
- Trained fraud detection models
- Performance metrics and visualizations
- Feature importance for fraud indicators

---

## 05_Model_Evaluation.ipynb

**Purpose**: Comprehensive evaluation and comparison of all trained models.

**Key Activities**:
- Loads all saved models (credit risk and fraud detection)
- Performs cross-model comparison
- Creates unified visualizations comparing all models
- Analyzes trade-offs between different metrics
- Provides summary of model performance across use cases
- Documents key findings and recommendations

**Outputs**:
- Cross-model comparison charts
- Performance summary tables
- Evaluation reports

---

## 06_Customer_Segmentation.ipynb

**Purpose**: Perform customer segmentation using clustering algorithms.

**Key Activities**:
- Prepares German Credit dataset for clustering
- Tests multiple clustering algorithms:
  - K-Means (with different k values: 2, 4, 5, 6)
  - Hierarchical Clustering (Average, Complete, Ward linkage)
  - DBSCAN (density-based clustering)
- Evaluates clusters using silhouette score, Davies-Bouldin index, Calinski-Harabasz score
- Determines optimal number of clusters
- Creates cluster profiles and personas
- Visualizes clusters using PCA and dendrograms
- Saves best clustering model to `models/clustering/`

**Best Model**: Hierarchical Clustering with Average Linkage (4 segments)

**Outputs**:
- Clustering models
- Cluster profiles and characteristics
- Visualization plots (dendrograms, cluster scatter plots)
- Customer segment personas

---

## Notebook Execution Order

1. **01_Data_Exploration.ipynb** - Understand the data
2. **02_Data_Preprocessing.ipynb** - Prepare data for modeling
3. **03_Credit_Risk_Models.ipynb** - Train credit risk models
4. **04_Fraud_Detection_Models.ipynb** - Train fraud detection models
5. **05_Model_Evaluation.ipynb** - Compare all models
6. **06_Customer_Segmentation.ipynb** - Perform customer segmentation

**Note**: Notebooks 3, 4, and 6 can be run in parallel as they work on different datasets/models.

