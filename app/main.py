from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from app.schemas import PredictionResponse
from app import model

app = FastAPI(
    title="BCRS Vision Pipeline",
    description="YOLOv8-based beverage container detection for Singapore's Beverage Container Return Scheme",
    version="1.0.0"
)

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "BCRS Vision Pipeline is running"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # This validates if the file is an image or not
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # This reads image bytes
    image_bytes = await file.read()

    # This runs prediction
    result = model.predict(image_bytes, file.filename)

    return result

@app.post("/predict/visualize")
async def predict_visualize(file: UploadFile = File(...)):
    # This validates if the files is an image or not
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    # This reads image bytes
    image_bytes = await file.read()
    # This runs prediction with visualization
    result_data, annotated_image = model.predict_with_visualization(image_bytes, file.filename)
    # This returns the annotated image
    return Response(content=annotated_image, media_type="image/jpeg")