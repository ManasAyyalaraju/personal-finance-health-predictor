# API Examples for Credit Risk Prediction

Copy and paste these JSON examples into the FastAPI docs at `http://localhost:8000/docs`

## Example 1: Low Risk - Excellent Credit (Should be APPROVED)

```json
{
  "loan_amount": 15000,
  "loan_term": 36,
  "interest_rate": 8.5,
  "employment_length": 10,
  "annual_income": 80000,
  "debt_to_income": 8,
  "credit_score": 780,
  "num_credit_lines": 8,
  "delinquencies": 0,
  "public_records": 0,
  "revolving_balance": 5000,
  "revolving_utilization": 15,
  "total_accounts": 12,
  "home_ownership": "MORTGAGE",
  "verification_status": "Verified"
}
```

## Example 2: Medium Risk - Good Credit

```json
{
  "loan_amount": 20000,
  "loan_term": 60,
  "interest_rate": 12.0,
  "employment_length": 5,
  "annual_income": 60000,
  "debt_to_income": 18,
  "credit_score": 680,
  "num_credit_lines": 6,
  "delinquencies": 1,
  "public_records": 0,
  "revolving_balance": 8000,
  "revolving_utilization": 45,
  "total_accounts": 10,
  "home_ownership": "RENT",
  "verification_status": "Verified"
}
```

## Example 3: High Risk - Poor Credit (Should be REJECTED)

```json
{
  "loan_amount": 25000,
  "loan_term": 60,
  "interest_rate": 18.5,
  "employment_length": 2,
  "annual_income": 40000,
  "debt_to_income": 35,
  "credit_score": 580,
  "num_credit_lines": 3,
  "delinquencies": 3,
  "public_records": 1,
  "revolving_balance": 12000,
  "revolving_utilization": 85,
  "total_accounts": 5,
  "home_ownership": "RENT",
  "verification_status": "Not Verified"
}
```

## Example 4: Minimal Required Fields Only

```json
{
  "loan_amount": 10000,
  "loan_term": 36,
  "interest_rate": 12.5,
  "employment_length": 5,
  "annual_income": 50000,
  "debt_to_income": 15
}
```

## Example 5: Simple Example - Common Fields

```json
{
  "loan_amount": 15000,
  "loan_term": 48,
  "interest_rate": 10.5,
  "employment_length": 7,
  "annual_income": 65000,
  "debt_to_income": 18.5,
  "credit_score": 720,
  "num_credit_lines": 5,
  "delinquencies": 0
}
```

## Field Descriptions

### Required Fields:

- `loan_amount` (float): Loan amount requested (must be > 0)
- `loan_term` (int): Loan term in months (12-60)
- `interest_rate` (float): Interest rate percentage (0-30)
- `employment_length` (int): Years employed (>= 0)
- `annual_income` (float): Annual income (must be > 0)
- `debt_to_income` (float): Debt-to-income ratio (0-100)

### Optional Fields (with defaults):

- `credit_score` (int): Credit score (300-850, default: 700)
- `num_credit_lines` (int): Number of credit lines (default: 5)
- `delinquencies` (int): Number of delinquencies (default: 0)
- `public_records` (int): Public records count (default: 0)
- `revolving_balance` (float): Revolving balance (default: 0)
- `revolving_utilization` (float): Revolving utilization % (default: 0)
- `total_accounts` (int): Total accounts (default: same as num_credit_lines)
- `home_ownership` (string): "MORTGAGE", "OWN", or "RENT" (default: "RENT")
- `verification_status` (string): "Verified", "Source Verified", or "Not Verified" (default: "Verified")

## How to Use in FastAPI Docs

1. Go to `http://localhost:8000/docs`
2. Find the `/predict/credit-risk` endpoint
3. Click "Try it out"
4. Paste one of the JSON examples above into the request body
5. Click "Execute"
6. View the response with prediction, probability, risk level, and recommendation

---

# API Examples for Fraud Detection

Copy and paste these JSON examples into the FastAPI docs at `http://localhost:8000/docs`

## Example 1: Normal Transaction - Low Risk

```json
{
  "transaction_amount": 50.25,
  "transaction_time": 3600,
  "features": [0.1, -0.2, 0.15, -0.1, 0.05, 0.2, -0.15, 0.1, -0.05, 0.1, 0.2, -0.1, 0.15, -0.2, 0.1, 0.05, -0.15, 0.2, -0.1, 0.15, 0.1, -0.2, 0.05, 0.15, -0.1, 0.2, 0.1, -0.15]
}
```

## Example 2: Suspicious Transaction - Medium Risk

```json
{
  "transaction_amount": 500.75,
  "transaction_time": 7200,
  "features": [1.5, -2.0, 1.8, -1.2, 1.0, 2.1, -1.5, 1.2, -0.8, 1.5, 2.0, -1.2, 1.8, -2.5, 1.5, 1.0, -1.8, 2.2, -1.5, 1.8, 1.5, -2.0, 1.0, 1.8, -1.2, 2.1, 1.5, -1.8]
}
```

## Example 3: High-Risk Transaction - Likely Fraud

```json
{
  "transaction_amount": 2500.00,
  "transaction_time": 10800,
  "features": [3.5, -4.0, 3.8, -3.2, 3.0, 4.1, -3.5, 3.2, -2.8, 3.5, 4.0, -3.2, 3.8, -4.5, 3.5, 3.0, -3.8, 4.2, -3.5, 3.8, 3.5, -4.0, 3.0, 3.8, -3.2, 4.1, 3.5, -3.8]
}
```

## Example 4: Small Normal Transaction

```json
{
  "transaction_amount": 25.50,
  "transaction_time": 1800,
  "features": [0.05, -0.1, 0.08, -0.05, 0.02, 0.1, -0.08, 0.05, -0.02, 0.05, 0.1, -0.05, 0.08, -0.1, 0.05, 0.02, -0.08, 0.1, -0.05, 0.08, 0.05, -0.1, 0.02, 0.08, -0.05, 0.1, 0.05, -0.08]
}
```

## Example 5: Large Normal Transaction

```json
{
  "transaction_amount": 150.00,
  "transaction_time": 5400,
  "features": [0.2, -0.3, 0.25, -0.15, 0.1, 0.3, -0.25, 0.2, -0.1, 0.2, 0.3, -0.15, 0.25, -0.3, 0.2, 0.1, -0.25, 0.3, -0.2, 0.25, 0.2, -0.3, 0.1, 0.25, -0.15, 0.3, 0.2, -0.25]
}
```

## Field Descriptions

### Required Fields:

- `transaction_amount` (float): Transaction amount in currency units (must be > 0)
- `transaction_time` (int): Time since first transaction in seconds (>= 0)
- `features` (array of 28 floats): PCA-transformed features (V1-V28) representing anonymized transaction characteristics

### About V1-V28 Features:

The V1-V28 features are PCA (Principal Component Analysis) transformed features that represent anonymized transaction characteristics. These are typically:
- Normalized values (often between -5 and 5)
- Values closer to 0 indicate more normal patterns
- Extreme values (|value| > 2-3) may indicate anomalies
- V14 and V10 are particularly important for fraud detection

### Response Fields:

- `is_fraud` (boolean): True if transaction is flagged as fraudulent
- `fraud_probability` (float): Probability of fraud (0-1)
- `risk_score` (int): Risk score from 0-100
- `recommendation` (string): Recommended action (APPROVE, REVIEW, HOLD, or BLOCK)
- `flagged_features` (array): List of features that triggered alerts

## How to Use in FastAPI Docs

1. Go to `http://localhost:8000/docs`
2. Find the `/predict/fraud` endpoint
3. Click "Try it out"
4. Paste one of the JSON examples above into the request body
5. Click "Execute"
6. View the response with fraud detection results

---

# API Examples for Customer Segmentation

Copy and paste these JSON examples into the FastAPI docs at `http://localhost:8000/docs`

## Example 1: Young Professional - Moderate Savings

```json
{
  "age": 28,
  "credit_amount": 5000,
  "duration": 24,
  "job": 2,
  "saving_accounts": "moderate",
  "checking_account": "moderate"
}
```

## Example 2: Established Customer - Rich Accounts

```json
{
  "age": 45,
  "credit_amount": 15000,
  "duration": 36,
  "job": 3,
  "saving_accounts": "rich",
  "checking_account": "rich"
}
```

## Example 3: Young Customer - Limited Savings

```json
{
  "age": 22,
  "credit_amount": 3000,
  "duration": 12,
  "job": 1,
  "saving_accounts": "little",
  "checking_account": "little"
}
```

## Example 4: Minimal Required Fields Only

```json
{
  "age": 35,
  "credit_amount": 8000,
  "duration": 18,
  "job": 2
}
```

## Example 5: High-Value Customer - Long Duration

```json
{
  "age": 52,
  "credit_amount": 25000,
  "duration": 48,
  "job": 3,
  "saving_accounts": "quite rich",
  "checking_account": "rich"
}
```

## Field Descriptions

### Required Fields:

- `age` (int): Customer age (18-100)
- `credit_amount` (float): Credit amount requested (must be > 0)
- `duration` (int): Loan duration in months (>= 1)
- `job` (int): Job category (0-3)
  - 0: Unskilled and non-resident
  - 1: Unskilled and resident
  - 2: Skilled
  - 3: Highly skilled

### Optional Fields:

- `saving_accounts` (string): Savings account level
  - "little": Low savings
  - "moderate": Moderate savings
  - "quite rich": High savings
  - "rich": Very high savings
- `checking_account` (string): Checking account level
  - "little": Low balance
  - "moderate": Moderate balance
  - "rich": High balance

### Response Fields:

- `segment_id` (int): Cluster/segment ID (0-3)
- `segment_name` (string): Descriptive segment name
- `characteristics` (array): Key characteristics of the segment
- `recommended_products` (array): Product recommendations for this segment
- `marketing_strategy` (string): Suggested marketing approach

## How to Use in FastAPI Docs

1. Go to `http://localhost:8000/docs`
2. Find the `/predict/segment` endpoint
3. Click "Try it out"
4. Paste one of the JSON examples above into the request body
5. Click "Execute"
6. View the response with customer segment assignment, characteristics, and recommendations