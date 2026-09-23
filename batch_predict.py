# batch_predict.py
import pandas as pd
from predict import predict_batch

df = pd.read_csv("data/incoming.csv")  # новые данные
df_pred = predict_batch(df)

def get_risk(probability):
    if probability >= 0.70:
        return "high"
    elif probability >= 0.40:
        return "medium"
    return "low"

df_pred["risk_segment"] = df_pred["churn_probability"].apply(get_risk)

df_pred = df_pred.sort_values(
    by="churn_probability",
    ascending=False
)

df_pred.to_csv("data/incoming_with_predictions.csv", index=False)
print("Predictions saved to data/incoming_with_predictions.csv")