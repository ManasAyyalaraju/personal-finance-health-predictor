# personal-finance-health-predictor
ML-powered credit risk prediction and savings optimization system using Python and scikit-learn

## About

This project implements machine learning models for credit risk assessment and fraud detection using multiple datasets. The project includes data preprocessing, model training, evaluation, and customer segmentation capabilities.

## Project Structure

```
personal-finance-health-predictor/
├── data/
│   ├── raw/              # Original datasets
│   └── processed/        # Preprocessed data (ignored by git)
├── models/                # Trained models (ignored by git)
├── notebooks/            # Jupyter notebooks for analysis
│   ├── 01_Data_Exploration.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   ├── 03_Credit_Risk_Models.ipynb
│   ├── 04_Fraud_Detection_Models.ipynb
│   ├── 05_Model_Evaluation.ipynb
│   └── 06_Customer_Segmentation.ipynb
└── results/              # Analysis results and visualizations
```

## Regenerating Processed Data and Models

The following files are intentionally ignored by git because they are large (some exceed GitHub's 100MB limit) and can be regenerated from the notebooks:

### Processed Data Files (`data/processed/`)

**Credit Risk Data:**
- `lending_X_train.csv`, `lending_X_train.pkl`
- `lending_X_test.csv`, `lending_X_test.pkl`
- `lending_y_train.csv`, `lending_y_train.pkl`
- `lending_y_test.csv`, `lending_y_test.pkl`
- `lending_club_step1.csv` through `lending_club_step4.csv`

**Fraud Detection Data:**
- `fraud_X_train.csv`, `fraud_X_train.pkl`
- `fraud_X_train_balanced.csv`, `fraud_X_train_balanced.pkl`
- `fraud_X_test.csv`, `fraud_X_test.pkl`
- `fraud_y_train.csv`, `fraud_y_train.pkl`
- `fraud_y_train_balanced.csv`, `fraud_y_train_balanced.pkl`
- `fraud_y_test.csv`, `fraud_y_test.pkl`
- `fraud_step2.csv`, `fraud_data_step3.csv`, `fraud_data_step4.csv`

**German Credit Data:**
- `german_credit_step1.csv` through `german_credit_step4.csv`
- `german_credit_final.csv`
- `german_credit_with_clusters.csv`

**To regenerate:**
1. Run `notebooks/02_Data_Preprocessing.ipynb` - This notebook processes all raw datasets and saves the processed files to `data/processed/`

### Model Files (`models/`)

**Credit Risk Models:**
- `credit_risk_best_model.pkl`
- `credit_risk_logistic_regression.pkl`
- `credit_risk_random_forest.pkl`
- `credit_risk_xgboost.pkl`
- `credit_risk_lightgbm.pkl`
- `scaler_lending.pkl`

**Fraud Detection Models:**
- `fraud_detection_best_model.pkl`
- `fraud_detection_isolation_forest.pkl`
- `fraud_detection_xgboost_smote.pkl`
- `scaler_fraud.pkl`

**Clustering Models:**
- `clustering/dbscan_best.pkl`
- `clustering/hierarchical_average.pkl`
- `clustering/hierarchical_complete.pkl`
- `clustering/hierarchical_ward.pkl`
- `clustering/kmeans_k2.pkl` through `kmeans_k6.pkl`
- `clustering/pca_2d.pkl`
- `clustering/scaler.pkl`

**Additional Scalers:**
- `scaler_german.pkl`

**To regenerate:**
1. **Credit Risk Models:** Run `notebooks/03_Credit_Risk_Models.ipynb`
2. **Fraud Detection Models:** Run `notebooks/04_Fraud_Detection_Models.ipynb`
3. **Clustering Models:** Run `notebooks/06_Customer_Segmentation.ipynb`

### Complete Workflow

To regenerate all processed data and models from scratch:

1. **Data Preprocessing:**
   ```bash
   jupyter notebook notebooks/02_Data_Preprocessing.ipynb
   ```
   - Processes raw data from `data/raw/`
   - Creates all processed CSV and PKL files in `data/processed/`

2. **Train Credit Risk Models:**
   ```bash
   jupyter notebook notebooks/03_Credit_Risk_Models.ipynb
   ```
   - Trains and saves all credit risk models
   - Saves scalers and best model

3. **Train Fraud Detection Models:**
   ```bash
   jupyter notebook notebooks/04_Fraud_Detection_Models.ipynb
   ```
   - Trains and saves fraud detection models
   - Uses SMOTE for balanced training data

4. **Customer Segmentation:**
   ```bash
   jupyter notebook notebooks/06_Customer_Segmentation.ipynb
   ```
   - Performs clustering analysis
   - Saves clustering models and PCA components

## Why These Files Are Ignored

- **Size:** Many files exceed 50MB, and 6 files exceed GitHub's 100MB hard limit
- **Regenerability:** All files can be recreated by running the notebooks
- **Best Practice:** Processed data and trained models are typically excluded from version control
- **Binary Files:** PKL files are binary and don't diff well in git

## Requirements

Install required packages:
```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn joblib imbalanced-learn
```

## Usage

1. Ensure raw data is in `data/raw/`
2. Run notebooks in order (01 → 06)
3. Models and processed data will be generated automatically

## License

[Add your license here]