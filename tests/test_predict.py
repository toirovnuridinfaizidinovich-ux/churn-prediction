import pandas as pd

from predict import predict_batch


def test_predict_batch_returns_valid_result():
    df = pd.DataFrame([
        {
            "SeniorCitizen": 0,
            "tenure": 12,
            "MonthlyCharges": 50.5,
            "TotalCharges": 600.0,
            "gender": "Male",
            "Partner": "Yes",
            "Dependents": "No",
            "PhoneService": "Yes",
            "MultipleLines": "No phone service",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
        }
    ])

    result = predict_batch(df)

    assert len(result) == 1
    assert "churn_probability" in result.columns
    assert "churn_label" in result.columns

    probability = result["churn_probability"].iloc[0]
    label = result["churn_label"].iloc[0]

    assert 0 <= probability <= 1
    assert label in [0, 1]