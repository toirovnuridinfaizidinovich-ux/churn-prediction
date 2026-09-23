# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict_churn

app = FastAPI(title="Churn Prediction API")

class ChurnRequest(BaseModel):
    SeniorCitizen: int
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    gender: str
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

class ChurnResponse(BaseModel):
    churn_probability: float
    churn_label: int

@app.post("/predict")
def predict(req: ChurnRequest) -> ChurnResponse:
    features = req.dict()
    res = predict_churn(features)
    return ChurnResponse(**res)

@app.get("/health")
def health():
    return {"status": "ok"}