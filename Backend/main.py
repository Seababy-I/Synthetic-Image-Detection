from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from model import load_model, predict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # in production, restrict to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup — not on every request
model = load_model()

@app.get("/")
async def root():
    return {"status": "ok", "message": "AI Detection API is running"}

@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    label, confidence, scores = predict(model, image)

    return {
        "prediction": label,
        "confidence": confidence,
        "scores": scores
    }