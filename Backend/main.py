from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
from model import load_model, predict

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for now allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load model once at startup — not on every request
model = load_model()

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

# Serve the frontend (from the bottom so it doesn't block API routes)
app.mount("/", StaticFiles(directory="static", html=True), name="static")