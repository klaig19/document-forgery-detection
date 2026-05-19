import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

class GradCAM:
    def __init__(self, model_path='resnet18_model.pth'):
        # Load the model
        self.model = models.resnet18(weights=None)
        self.model.fc = torch.nn.Linear(512, 2)  # binary classification
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
        
        # Target layer (last convolutional layer)
        self.target_layer = self.model.layer4[-1]
        
        # Hook setup
        self.activations = []
        self.gradients = []
        
        # Register hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_backward_hook(self._backward_hook)
        
        # Transform for preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _forward_hook(self, module, input, output):
        self.activations.append(output)
    
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0])
    
    def generate_cam(self, image_path, save_path=None):
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        original_img = img.copy()
        input_tensor = self.transform(img).unsqueeze(0)
        
        # Forward pass
        output = self.model(input_tensor)
        pred_class = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0, pred_class].item()
        
        # Backward pass for Grad-CAM
        self.model.zero_grad()
        class_score = output[0, pred_class]
        class_score.backward()
        
        # Grad-CAM calculation
        act = self.activations[0].squeeze(0)       # [C, H, W]
        grad = self.gradients[0].squeeze(0)        # [C, H, W]
        
        weights = grad.mean(dim=(1, 2))            # Global average pooling
        cam = (weights[:, None, None] * act).sum(dim=0)  # Weighted sum
        cam = F.relu(cam)                          # Only positive influence
        
        # Normalize and resize
        cam -= cam.min()
        cam /= cam.max()
        cam = cv2.resize(cam.detach().numpy(), (224, 224))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        
        # Create overlay
        overlay_img = np.array(original_img.resize((224, 224)))
        overlay = cv2.addWeighted(overlay_img, 0.6, heatmap, 0.4, 0)
        
        # Convert BGR to RGB for matplotlib
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        
        # Display results
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(original_img)
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title("Grad-CAM Heatmap")
        plt.imshow(heatmap_rgb)
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title(f"Overlay (Prediction: {'Fake' if pred_class == 0 else 'Real'}, Confidence: {confidence:.3f})")
        plt.imshow(overlay_rgb)
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Grad-CAM visualization saved to {save_path}")
        
        plt.show()
        
        # Clear hooks for next use
        self.activations.clear()
        self.gradients.clear()
        
        return {
            'prediction': 'Fake' if pred_class == 0 else 'Real',
            'confidence': confidence,
            'heatmap': heatmap_rgb,
            'overlay': overlay_rgb
        }

def main():
    # Create output directory
    os.makedirs('gradcam_results', exist_ok=True)
    
    # Initialize Grad-CAM
    gradcam = GradCAM()
    
    # Test with some images from your test set
    test_images = [
        r'C:\Users\Kiran\Desktop\G\SEM 4\Mini Project\projectfinal\test\fake\CMB1.png',
        r'C:\Users\Kiran\Desktop\G\SEM 4\Mini Project\projectfinal\test\real\Aarushi_transcript_page-0001_aug2_crop_2_10.jpg'
    ]
    
    for i, img_path in enumerate(test_images):
        if os.path.exists(img_path):
            print(f"\nProcessing image {i+1}: {os.path.basename(img_path)}")
            save_path = f'gradcam_results/gradcam_result_{i+1}.png'
            result = gradcam.generate_cam(img_path, save_path)
            print(f"Prediction: {result['prediction']}, Confidence: {result['confidence']:.3f}")
        else:
            print(f"Image not found: {img_path}")

if __name__ == "__main__":
    main() 