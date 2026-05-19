import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import argparse
from forgery_detector_enhanced import EnhancedForgeryDetector
import os

# Define the necessary transforms for image preprocessing (must match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def get_forgery_overlay(image_path, overlay_path):
    detector = EnhancedForgeryDetector()
    result = detector.detect_forgery_areas(image_path, overlay_path, threshold=0.8)  # Use higher threshold
    if result['prediction'].lower() == 'fake':
        return overlay_path
    return None

def get_forgery_analysis_images(image_path, save_dir):
    """Generate individual analysis images for popup modal"""
    detector = EnhancedForgeryDetector()
    return detector.generate_individual_images(image_path, save_dir)

# Load the trained model
def load_model(model_path):
    model = models.resnet18(weights=None)  # Load the ResNet18 architecture
    model.fc = nn.Linear(model.fc.in_features, 2)  # Adjust final layer for binary classification (fake/real)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Load weights
    model.eval()  # Set model to evaluation mode
    return model

# Predict if an image is "Fake" or "Real"
def predict_document(image_path, model):
    image = Image.open(image_path).convert('RGB')  # Open and convert image to RGB
    image = transform(image).unsqueeze(0)  # Apply transforms and add batch dimension
    with torch.no_grad():  # Disable gradient tracking during prediction
        outputs = model(image)  # Forward pass through the model
        probs = torch.softmax(outputs, dim=1)  # Apply softmax to get probabilities
        confidence, predicted_class = torch.max(probs, 1)  # Get predicted class and its confidence
        class_names = ['fake', 'real']  # Class labels
        prediction = class_names[predicted_class.item()]
        return prediction, confidence.item()  # Return prediction and confidence

# CLI to run prediction from the command line
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to image')  # Image path argument
    parser.add_argument('--model-path', default='resnet18_model.pth', help='Path to trained model')  # Model path argument
    args = parser.parse_args()

    model = load_model(args.model_path)  # Load the model
    prediction, confidence = predict_document(args.image, model)  # Make prediction
    print(f"Prediction: {prediction} (Confidence: {confidence:.4f})")  # Print result
