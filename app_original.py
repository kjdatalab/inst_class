# app/main.py
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile):
    # For now, return dummy prediction
    return {
        "filename": file.filename,
        "prediction": "piano",
        "confidence": 0.95
    }

@app.get("/health")
async def health_check():
    return {"status": "ok"}
