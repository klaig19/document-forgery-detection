import joblib
import cv2 # type: ignore
import numpy as np
import os

# Load the model once (ensure this path is correct)
model = joblib.load("my_model.pkl")  # Update this with your actual model name

# Function to process image and predict forgery
def detect_forgery(filepath):
    # 1. Load the image (for now, assume it’s an image)
    image = cv2.imread(filepath)

    # 2. Preprocess the image (resize and flatten it)
    resized = cv2.resize(image, (224, 224))  # Resize image to 224x224
    flattened = resized.flatten().reshape(1, -1)  # Flatten and reshape for prediction

    # 3. Run the model prediction
    prediction = model.predict(flattened)[0]

    # 4. Determine the forgery status
    label = "forged" if prediction == 1 else "authentic"  # Assuming 1 is forged, 0 is authentic

    return {
        "status": label,
        "image_path": filepath  # Returning image path for UI display
    }
