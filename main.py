import matplotlib
matplotlib.use('Agg')  # Backend non-interactif, avant tout autre import

import base64
import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle une seule fois au démarrage."""
    global predictor
    from inference import PneumoniaPredictor
    predictor = PneumoniaPredictor()
    yield


app = FastAPI(
    title="Pneumonia Detection API",
    description="Pneumonia detection with Grad-CAM explainability",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "Pneumonia Detection API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


def image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    original_resized = image.resize(
        (predictor.args.size, predictor.args.size),
        Image.BILINEAR,
    )
    original_base64 = image_to_base64(original_resized)

    result = predictor.predict(image)

    cam_array = (result["cam"] * 255).astype("uint8")
    cam_image = Image.fromarray(cam_array)
    cam_base64 = image_to_base64(cam_image)

    return {
        "filename": file.filename,
        "prediction": result["prediction"],
        "probability": result["probability"],
        "original_image": original_base64,
        "gradcam": cam_base64,
    }
