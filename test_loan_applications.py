"""Test the credit risk API with different loan application scenarios"""
import requests
import json
from typing import Dict, Any

API_URL = "http://localhost:8000/predict/credit-risk"

def test_application(name: str, application: Dict[str, Any]):
    """Test a loan application and print results"""
    print("=" * 80)
    print(f"Test Case: {name}")
    print("-" * 80)
    print(f"Application Details:")
    for key, value in application.items():
        print(f"  {key}: {value}")
    print()
    
    try:
        response = requests.post(API_URL, json=application)
        response.raise_for_status()
        result = response.json()
        
        print("Prediction Result:")
        print(f"  Decision: {result['prediction'].upper()}")
        print(f"  Default Probability: {result['probability']:.2%}")
        print(f"  Risk Level: {result['risk_level'].upper()}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Recommendation: {result['recommendation']}")
        print()
        
        return result
    except Exception as e:
        print(f"  ERROR: {e}")
        print()
        return None

# Test Cases
test_cases = [
    {
        "name": "1. Low Risk - Excellent Credit",
        "application": {
            "loan_amount": 15000,
            "loan_term": 36,
            "interest_rate": 8.5,
            "employment_length": 10,
            "annual_income": 80000,
            "debt_to_income": 8,
            "num_credit_lines": 8,
            "delinquencies": 0,
            "public_records": 0,
            "revolving_balance": 5000,
            "revolving_utilization": 15,
            "total_accounts": 12,
            "credit_score": 780,
            "home_ownership": "MORTGAGE",
            "verification_status": "Verified"
        }
    },
    {
        "name": "2. Medium Risk - Good Credit",
        "application": {
            "loan_amount": 20000,
            "loan_term": 60,
            "interest_rate": 12.0,
            "employment_length": 5,
            "annual_income": 60000,
            "debt_to_income": 18,
            "num_credit_lines": 6,
            "delinquencies": 1,
            "public_records": 0,
            "revolving_balance": 8000,
            "revolving_utilization": 45,
            "total_accounts": 10,
            "credit_score": 680,
            "home_ownership": "RENT",
            "verification_status": "Verified"
        }
    },
    {
        "name": "3. High Risk - Poor Credit",
        "application": {
            "loan_amount": 25000,
            "loan_term": 60,
            "interest_rate": 18.5,
            "employment_length": 2,
            "annual_income": 40000,
            "debt_to_income": 35,
            "num_credit_lines": 3,
            "delinquencies": 3,
            "public_records": 1,
            "revolving_balance": 12000,
            "revolving_utilization": 85,
            "total_accounts": 5,
            "credit_score": 580,
            "home_ownership": "RENT",
            "verification_status": "Not Verified"
        }
    },
    {
        "name": "4. Borderline - Moderate Risk",
        "application": {
            "loan_amount": 18000,
            "loan_term": 48,
            "interest_rate": 14.0,
            "employment_length": 4,
            "annual_income": 55000,
            "debt_to_income": 25,
            "num_credit_lines": 5,
            "delinquencies": 0,
            "public_records": 0,
            "revolving_balance": 6000,
            "revolving_utilization": 55,
            "total_accounts": 8,
            "credit_score": 650,
            "home_ownership": "OWN",
            "verification_status": "Source Verified"
        }
    },
    {
        "name": "5. Small Loan - Low Amount",
        "application": {
            "loan_amount": 5000,
            "loan_term": 36,
            "interest_rate": 10.0,
            "employment_length": 7,
            "annual_income": 50000,
            "debt_to_income": 12,
            "num_credit_lines": 4,
            "delinquencies": 0,
            "public_records": 0,
            "revolving_balance": 2000,
            "revolving_utilization": 25,
            "total_accounts": 6,
            "credit_score": 720,
            "home_ownership": "MORTGAGE",
            "verification_status": "Verified"
        }
    },
    {
        "name": "6. Large Loan - High Income",
        "application": {
            "loan_amount": 35000,
            "loan_term": 60,
            "interest_rate": 9.5,
            "employment_length": 15,
            "annual_income": 120000,
            "debt_to_income": 15,
            "num_credit_lines": 10,
            "delinquencies": 0,
            "public_records": 0,
            "revolving_balance": 15000,
            "revolving_utilization": 30,
            "total_accounts": 15,
            "credit_score": 750,
            "home_ownership": "MORTGAGE",
            "verification_status": "Verified"
        }
    },
    {
        "name": "7. Young Professional - Building Credit",
        "application": {
            "loan_amount": 12000,
            "loan_term": 36,
            "interest_rate": 13.5,
            "employment_length": 1,
            "annual_income": 45000,
            "debt_to_income": 20,
            "num_credit_lines": 3,
            "delinquencies": 0,
            "public_records": 0,
            "revolving_balance": 3000,
            "revolving_utilization": 40,
            "total_accounts": 4,
            "credit_score": 670,
            "home_ownership": "RENT",
            "verification_status": "Source Verified"
        }
    },
    {
        "name": "8. High DTI - Risky",
        "application": {
            "loan_amount": 22000,
            "loan_term": 60,
            "interest_rate": 16.0,
            "employment_length": 3,
            "annual_income": 48000,
            "debt_to_income": 42,
            "num_credit_lines": 7,
            "delinquencies": 2,
            "public_records": 0,
            "revolving_balance": 10000,
            "revolving_utilization": 75,
            "total_accounts": 9,
            "credit_score": 620,
            "home_ownership": "RENT",
            "verification_status": "Not Verified"
        }
    },
    {
        "name": "9. Short Term - Low Risk",
        "application": {
            "loan_amount": 10000,
            "loan_term": 24,
            "interest_rate": 7.5,
            "employment_length": 8,
            "annual_income": 70000,
            "debt_to_income": 10,
            "num_credit_lines": 7,
            "delinquencies": 0,
            "public_records": 0,
            "revolving_balance": 4000,
            "revolving_utilization": 20,
            "total_accounts": 11,
            "credit_score": 760,
            "home_ownership": "MORTGAGE",
            "verification_status": "Verified"
        }
    },
    {
        "name": "10. Very High Risk - Multiple Red Flags",
        "application": {
            "loan_amount": 30000,
            "loan_term": 60,
            "interest_rate": 22.0,
            "employment_length": 1,
            "annual_income": 35000,
            "debt_to_income": 48,
            "num_credit_lines": 2,
            "delinquencies": 5,
            "public_records": 2,
            "revolving_balance": 15000,
            "revolving_utilization": 95,
            "total_accounts": 3,
            "credit_score": 520,
            "home_ownership": "RENT",
            "verification_status": "Not Verified"
        }
    }
]

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("CREDIT RISK PREDICTION API - TEST SUITE")
    print("=" * 80)
    print()
    
    results = []
    for test_case in test_cases:
        result = test_application(test_case["name"], test_case["application"])
        if result:
            results.append({
                "name": test_case["name"],
                "prediction": result["prediction"],
                "probability": result["probability"],
                "risk_level": result["risk_level"]
            })
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    approved = sum(1 for r in results if r["prediction"] == "approved")
    rejected = sum(1 for r in results if r["prediction"] == "rejected")
    print(f"Total Applications Tested: {len(results)}")
    print(f"Approved: {approved}")
    print(f"Rejected: {rejected}")
    print()
    print("Breakdown by Risk Level:")
    low_risk = sum(1 for r in results if r["risk_level"] == "low")
    medium_risk = sum(1 for r in results if r["risk_level"] == "medium")
    high_risk = sum(1 for r in results if r["risk_level"] == "high")
    print(f"  Low Risk: {low_risk}")
    print(f"  Medium Risk: {medium_risk}")
    print(f"  High Risk: {high_risk}")
    print()
    print("=" * 80)

