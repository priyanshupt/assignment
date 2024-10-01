from fastapi import FastAPI, File, UploadFile
from keras.models import load_model
from PIL import Image
import numpy as np

app = FastAPI()
model = load_model("weights/best_model.h5")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("L")
    image = np.array(image.resize((256, 256)))
    image = image / 255.0
    image = np.expand_dims(image, axis=[0, -1])
    
    prediction = model.predict(image)
    return {"prediction": prediction.tolist()}
