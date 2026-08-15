import base64
import io

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from inference import PneumoniaPredictor


app = FastAPI(
    title="Pneumonia Detection API",
    description="Pneumonia detection with Grad-CAM explainability",
    version="1.0.0",
)

# Allow Next.js frontend to communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Update for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once when the application starts
predictor = PneumoniaPredictor()


@app.get("/")
def root():
    return {
        "message": "Pneumonia Detection API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


def image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """Convert a PIL Image to a base64-encoded string."""
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    # Read uploaded file
    contents = await file.read()

    image = Image.open(
        io.BytesIO(contents)
    ).convert("RGB")

    # --- Original image (resized to model input size) ---
    original_resized = image.resize(
        (predictor.args.size, predictor.args.size),
        Image.BILINEAR,
    )
    original_base64 = image_to_base64(original_resized)

    # --- Prediction + Grad-CAM ---
    result = predictor.predict(image)

    # Convert Grad-CAM overlay (float32 [0,1] numpy array) to PNG
    cam_array = (result["cam"] * 255).astype("uint8")
    cam_image = Image.fromarray(cam_array)
    cam_base64 = image_to_base64(cam_image)

    return {
        "filename": file.filename,
        "prediction": result["prediction"],      # "PNEUMONIA" | "NORMAL"
        "probability": result["probability"],    # float [0, 1]
        "original_image": original_base64,       # PNG – raw X-ray (resized)
        "gradcam": cam_base64,                   # PNG – Grad-CAM overlay
    }
