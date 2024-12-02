# 50/50 prediction, successfully ran
from fastapi import FastAPI, UploadFile, File
import random

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Dummy prediction - 50/50 chance
    prediction = random.choice([True, False])
    confidence = random.uniform(0.5, 1.0)  # Random confidence score

    return {
        "filename": file.filename,
        "contains_piano": prediction,
        "confidence": round(confidence, 3)
    }

# Add a test endpoint
@app.get("/")
async def root():
    return {"message": "Piano detector API is running"}
