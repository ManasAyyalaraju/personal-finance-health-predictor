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
