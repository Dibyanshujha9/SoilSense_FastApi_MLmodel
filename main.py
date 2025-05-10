# http://localhost:8000/docs#/default/analyze_soil_image_analyze__post


# filename: main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

app = FastAPI()

# ✅ Allow CORS for MERN stack frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend domain like "http://localhost:3000"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Fertilizer recommendation map
fertilizer_map = {
    'Nitrogen Deficient': 'Urea',
    'Phosphorus Deficient': 'Superphosphate',
    'Potassium Deficient': 'Potassium Chloride',
    'Balanced': 'NPK Blend'
}

# ✅ Preprocess uploaded image
def preprocess_image(contents: bytes):
    image = Image.open(BytesIO(contents)).convert("RGB")
    image = image.resize((224, 224))
    image_np = np.array(image)
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    return image_cv2

# ✅ Analyze the image for nutrient deficiency
def analyze_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    avg_hue = np.mean(hsv[:, :, 0])
    avg_sat = np.mean(hsv[:, :, 1])
    avg_val = np.mean(hsv[:, :, 2])

    if avg_sat < 50 and avg_val < 50:
        return 'Nitrogen Deficient'
    elif avg_hue > 50 and avg_val < 50:
        return 'Phosphorus Deficient'
    elif avg_hue < 50 and avg_sat > 50:
        return 'Potassium Deficient'
    else:
        return 'Balanced'

# ✅ API Endpoint: POST /analyze
@app.post("/analyze/")
async def analyze_soil_image(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Only JPG/PNG images are supported.")

    contents = await file.read()

    try:
        image = preprocess_image(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")

    deficiency = analyze_image(image)
    recommendation = fertilizer_map.get(deficiency, 'NPK Blend')

    return {
        "deficiency": deficiency,
        "recommendation": recommendation
    }

# ✅ Entry point for local testing
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
