from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
import joblib
import pandas as pd

# ---- Global storage for loaded models ----
models = {}

# ---- App lifespan: load model once on startup ----
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        models["pipeline"] = joblib.load("best_pipeline.joblib")
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
    yield
    print("App shutting down...")

# ---- FastAPI app ----
app = FastAPI(title="Drug Side Effect Severity Predictor", lifespan=lifespan)

# ---- Request schema ----
class DrugRequest(BaseModel):
    dosage_mg: float
    drug_class: str
    indications: str
    side_effects: str
    contraindications: str
    warnings: str
    approval_status: str
    expiry_date: str  # YYYY-MM-DD

# ---- Prediction endpoint ----
@app.post("/predict")
async def predict_side_effect(data: DrugRequest):
    pipeline = models.get("pipeline")
    if not pipeline:
        return JSONResponse(
            status_code=500,
            content={"status_code": 500, "message": "Internal Server Error! Model not loaded."}
        )

    # Compute days_until_expiry
    expiry_dt = pd.to_datetime(data.expiry_date)
    days_until_expiry = (expiry_dt - pd.Timestamp("today")).days

    # Prepare input data as DataFrame
    user_data = {
        "dosage_mg": data.dosage_mg,
        "drug_class": data.drug_class,
        "indications": data.indications,
        "side_effects": data.side_effects,
        "contraindications": data.contraindications,
        "warnings": data.warnings,
        "approval_status": data.approval_status,
        "days_until_expiry": days_until_expiry,
        "expiry_date": days_until_expiry,  # keep consistent with training
    }

    new_df = pd.DataFrame([user_data])

    try:
        # Get numeric prediction
        pred_numeric = pipeline.predict(new_df)[0]

        # Map numeric prediction to human-readable severity
        severity_map = {0: "Mild", 1: "Moderate", 2: "Severe"}
        pred_label = severity_map.get(pred_numeric, "Unknown")

        return {"status_code": 200, "prediction_class": pred_label}

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={
                "status_code": 400,
                "message": f"Inference failed: {e}",
                "hint": "Ensure your input data matches the columns/features used during training."
            }
        )

# ---- Run locally ----
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
