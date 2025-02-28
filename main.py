from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import io
import cv2
import logging

# Initialize FastAPI App
app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging Setup
logging.basicConfig(level=logging.INFO)

# Load the Trained Model
try:
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    logging.info("✅ Model loaded successfully!")
except Exception as e:
    logging.error(f"❌ Error loading model: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

# Class Labels
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Confidence Threshold
CONFIDENCE_THRESHOLD = 0.5

# Image Preprocessing Function
def preprocess_image(file: UploadFile):
    try:
        file.file.seek(0)  # Reset file pointer
        contents = file.file.read()
        if not contents:
            raise ValueError("Empty file received.")

        # Save the image for debugging
        with open("debug_uploaded_image.jpg", "wb") as f:
            f.write(contents)

        # Convert image to numpy array
        img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image format. OpenCV failed to decode.")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = cv2.resize(img, (128, 128))  # Resize to match model input size

        # Convert image to array
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize to match training

        logging.info(f"🖼️ Image preprocessed, shape: {img_array.shape}")
        return img_array
    except Exception as e:
        logging.error(f"❌ Error preprocessing image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to preprocess image: {str(e)}")

# Prediction Route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        file.file.seek(0)  # Ensure correct file reading
        img_array = preprocess_image(file)

        # Make prediction
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions)
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(predictions))

        logging.info(f"📊 Raw Predictions: {predictions}")
        logging.info(f"✅ Prediction: {predicted_class}, Confidence: {confidence:.2f}")

        # Confidence Check
        if confidence < CONFIDENCE_THRESHOLD:
            logging.warning(f"⚠️ Low confidence prediction: {confidence:.2f}")
            return {
                "prediction": "Uncertain",
                "confidence": confidence,
                "all_predictions": predictions.tolist()
            }

        return {
            "prediction": predicted_class,
            "confidence": confidence,
            "all_predictions": predictions.tolist()
        }

    except Exception as e:
        logging.error(f"❌ Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Home Route
@app.get("/")
async def root():
    return {"message": "🌿 Welcome to the Plant Disease Prediction API"}
