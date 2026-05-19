import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from skimage import measure, morphology
from scipy import ndimage

class EnhancedForgeryDetector:
    def __init__(self, model_path='resnet18_model.pth'):
        # Load the model
        self.model = models.resnet18(weights=None)
        self.model.fc = torch.nn.Linear(512, 2)
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
    
    def detect_forgery_areas(self, image_path, save_path=None, threshold=0.7):
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        original_img = img.copy()
        input_tensor = self.transform(img).unsqueeze(0)
        
        # Forward pass
        output = self.model(input_tensor)
        pred_class = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0, pred_class].item()
        
        # Check if document is fake
        is_fake = pred_class == 0
        
        # Add confidence threshold for fake detection
        if is_fake and confidence < 0.85:  # Require 85% confidence for fake
            # If confidence is too low, treat as real
            is_fake = False
            confidence = 1.0 - confidence  # Flip confidence for real class
        
        if is_fake:
            # Detailed analysis for fake documents
            return self._analyze_fake_document(original_img, input_tensor, save_path, threshold)
        else:
            # Simple analysis for real documents
            return self._analyze_real_document(original_img, confidence, save_path)
    
    def _analyze_fake_document(self, original_img, input_tensor, save_path, threshold):
        """Detailed analysis for fake documents showing exact forgery areas"""
        # Backward pass for Grad-CAM
        self.model.zero_grad()
        output = self.model(input_tensor)
        pred_class = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0, pred_class].item()
        class_score = output[0, 0]  # Fake class
        class_score.backward()
        
        # Grad-CAM calculation
        act = self.activations[0].squeeze(0)
        grad = self.gradients[0].squeeze(0)
        
        weights = grad.mean(dim=(1, 2))
        cam = (weights[:, None, None] * act).sum(dim=0)
        cam = F.relu(cam)
        
        # Normalize and resize
        cam -= cam.min()
        cam /= cam.max()
        cam = cv2.resize(cam.detach().numpy(), (224, 224))
        
        # Fix invalid values in CAM before creating heatmap
        cam_clean = cam.copy()
        cam_clean = np.nan_to_num(cam_clean, nan=0.0, posinf=1.0, neginf=0.0)
        cam_clean = np.clip(cam_clean, 0, 1)
        
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_clean), cv2.COLORMAP_JET)
        
        # Enhanced forgery area detection
        forgery_mask = self._extract_forgery_areas(cam, threshold)
        
        # Create visualizations
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay_img = np.array(original_img.resize((224, 224)))
        
        # Create detailed overlay with forgery areas highlighted
        detailed_overlay = self._create_detailed_overlay(overlay_img, forgery_mask, cam)
        
        # Convert BGR to RGB for matplotlib
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Display results - 5 panels for fake documents (removed graph)
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Original image
        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title("Original Document", fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Grad-CAM heatmap
        axes[0, 1].imshow(heatmap_rgb)
        axes[0, 1].set_title("Forgery Detection Heatmap", fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Detailed overlay
        axes[1, 0].imshow(detailed_overlay)
        axes[1, 0].set_title("FORGERY DETECTED - Exact Areas Highlighted", fontsize=14, fontweight='bold', color='red')
        axes[1, 0].axis('off')
        
        # Analysis results
        forgery_areas = self._count_forgery_areas(forgery_mask)
        prediction_text = f"🚨 DOCUMENT IS FAKE 🚨\n\nConfidence: {confidence:.3f}\nForgery Areas: {forgery_areas}\n\nRED: High-confidence forgery\nYELLOW: Suspicious areas"
        
        axes[1, 1].text(0.1, 0.5, prediction_text, fontsize=14, fontweight='bold',
                       transform=axes[1, 1].transAxes, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))
        axes[1, 1].set_title("Forgery Analysis Results", fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"🚨 FORGERY DETECTED! Detailed analysis saved to {save_path}")
        
        # Don't show the plot - just save it
        plt.close()  # Close the figure instead of showing it
        
        # Clear hooks for next use
        self.activations.clear()
        self.gradients.clear()
        
        return {
            'prediction': 'Fake',
            'confidence': confidence,
            'forgery_mask': forgery_mask,
            'forgery_areas': forgery_areas,
            'heatmap': heatmap_rgb,
            'detailed_overlay': detailed_overlay
        }
    
    def _analyze_real_document(self, original_img, confidence, save_path):
        """Simple analysis for real documents"""
        # Display results - 2 panels for real documents
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        axes[0].imshow(original_img)
        axes[0].set_title("Original Document", fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Simple verification result
        verification_text = f"✅ DOCUMENT IS AUTHENTIC ✅\n\nConfidence: {confidence:.3f}\n\nNo forgery detected.\nDocument appears to be genuine."
        
        axes[1].text(0.1, 0.5, verification_text, fontsize=16, fontweight='bold',
                    transform=axes[1].transAxes, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
        axes[1].set_title("Authentication Result", fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ AUTHENTIC DOCUMENT! Simple verification saved to {save_path}")
        
        # Don't show the plot - just save it
        plt.close()  # Close the figure instead of showing it
        
        return {
            'prediction': 'Real',
            'confidence': confidence,
            'forgery_mask': None,
            'forgery_areas': 0,
            'heatmap': None,
            'detailed_overlay': None
        }
    
    def _extract_forgery_areas(self, cam, threshold):
        """Extract specific forgery areas using improved thresholding and validation"""
        # Apply higher threshold to reduce false positives
        high_confidence_threshold = max(threshold, 0.8)  # Minimum 0.8 for high confidence
        binary_mask = cam > high_confidence_threshold
        
        # Check if we have enough high-confidence areas
        if np.sum(binary_mask) < 100:  # Less than 100 pixels
            # Try with slightly lower threshold but still high
            medium_threshold = max(threshold, 0.75)
            binary_mask = cam > medium_threshold
            
            # If still too few areas, don't show any
            if np.sum(binary_mask) < 50:
                return np.zeros_like(cam, dtype=np.uint8)
        
        # Morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        binary_mask = morphology.binary_opening(binary_mask, kernel)
        binary_mask = morphology.binary_closing(binary_mask, kernel)
        
        # Remove small noise - increase minimum size
        binary_mask = morphology.remove_small_objects(binary_mask, min_size=100)
        
        # Fill holes
        binary_mask = ndimage.binary_fill_holes(binary_mask)
        
        # Additional validation: check if areas are too concentrated in one region
        if self._is_concentrated_in_corner(binary_mask):
            # If too concentrated, apply stricter threshold
            strict_threshold = max(threshold, 0.85)
            binary_mask = cam > strict_threshold
            binary_mask = morphology.remove_small_objects(binary_mask, min_size=150)
            binary_mask = ndimage.binary_fill_holes(binary_mask)
        
        return binary_mask.astype(np.uint8)
    
    def _is_concentrated_in_corner(self, binary_mask):
        """Check if detected areas are too concentrated in corners"""
        height, width = binary_mask.shape
        
        # Define corner regions (top-left, top-right, bottom-left, bottom-right)
        corner_size = min(height, width) // 4
        
        # Top-right corner
        top_right = binary_mask[:corner_size, -corner_size:]
        top_right_ratio = np.sum(top_right) / (corner_size * corner_size)
        
        # Top-left corner
        top_left = binary_mask[:corner_size, :corner_size]
        top_left_ratio = np.sum(top_left) / (corner_size * corner_size)
        
        # Bottom-right corner
        bottom_right = binary_mask[-corner_size:, -corner_size:]
        bottom_right_ratio = np.sum(bottom_right) / (corner_size * corner_size)
        
        # Bottom-left corner
        bottom_left = binary_mask[-corner_size:, :corner_size]
        bottom_left_ratio = np.sum(bottom_left) / (corner_size * corner_size)
        
        # Check if any corner has more than 60% of the total detected area
        total_detected = np.sum(binary_mask)
        if total_detected == 0:
            return False
            
        corner_ratios = [top_right_ratio, top_left_ratio, bottom_right_ratio, bottom_left_ratio]
        max_corner_ratio = max(corner_ratios)
        
        # If more than 60% of detected area is in one corner, it's too concentrated
        return max_corner_ratio > 0.6
    
    def _create_detailed_overlay(self, original_img, forgery_mask, cam):
        """Create detailed overlay highlighting exact forgery areas with improved validation"""
        overlay = original_img.copy()
        
        # Create red mask for forgery areas (high confidence)
        red_mask = np.zeros_like(original_img)
        red_mask[forgery_mask == 1] = [255, 0, 0]  # Red color for forgery areas
        
        # Create yellow mask for suspicious areas (higher threshold to reduce false positives)
        yellow_mask = np.zeros_like(original_img)
        # Use higher threshold for yellow areas: 0.65-0.8 instead of 0.5-0.7
        suspicious_areas = (cam > 0.65) & (cam <= 0.8) & (forgery_mask == 0)
        
        # Additional validation for yellow areas: must be connected and not too small
        if np.any(suspicious_areas):
            # Label connected components
            labeled_suspicious, num_features = measure.label(suspicious_areas, return_num=True)
            
            # Only keep components with sufficient size
            for i in range(1, num_features + 1):
                component = (labeled_suspicious == i)
                if np.sum(component) < 80:  # Minimum 80 pixels for suspicious area
                    suspicious_areas[component] = False
        
        yellow_mask[suspicious_areas] = [255, 255, 0]  # Yellow for suspicious areas
        
        # Combine masks
        combined_mask = red_mask + yellow_mask
        
        # Only create overlay if we have significant areas detected
        total_detected_pixels = np.sum(red_mask) + np.sum(yellow_mask)
        if total_detected_pixels < 200:  # Less than 200 pixels total
            # Return original image without overlay
            return overlay
        
        # Create overlay with transparency
        overlay = cv2.addWeighted(overlay, 0.7, combined_mask, 0.3, 0)
        
        # Add text annotations
        overlay_pil = Image.fromarray(overlay)
        draw = ImageDraw.Draw(overlay_pil)
        
        # Try to use a default font, fallback to basic if not available
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Add labels for different areas
        if np.any(forgery_mask):
            draw.text((10, 10), "RED: High-confidence forgery areas", fill=(255, 255, 255), font=font)
        if np.any(suspicious_areas):
            draw.text((10, 30), "YELLOW: Suspicious areas", fill=(0, 0, 0), font=font)
        
        return np.array(overlay_pil)
    
    def _count_forgery_areas(self, forgery_mask):
        """Count the number of distinct forgery areas"""
        labeled_mask, num_features = measure.label(forgery_mask, return_num=True)
        return num_features

    def generate_individual_images(self, image_path, save_dir):
        """Generate individual images for popup modal with improved validation"""
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        original_img = img.copy()
        input_tensor = self.transform(img).unsqueeze(0)
        
        # Forward pass
        output = self.model(input_tensor)
        pred_class = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0, pred_class].item()
        
        # Add confidence threshold for fake detection
        if pred_class == 0 and confidence < 0.85:  # Require 85% confidence for fake
            # If confidence is too low, treat as real
            pred_class = 1
            confidence = 1.0 - confidence
        
        if pred_class == 0:  # Fake
            # Backward pass for Grad-CAM
            self.model.zero_grad()
            class_score = output[0, 0]  # Fake class
            class_score.backward()
            
            # Grad-CAM calculation
            act = self.activations[0].squeeze(0)
            grad = self.gradients[0].squeeze(0)
            
            weights = grad.mean(dim=(1, 2))
            cam = (weights[:, None, None] * act).sum(dim=0)
            cam = F.relu(cam)
            
            # Normalize and resize
            cam -= cam.min()
            cam /= cam.max()
            cam = cv2.resize(cam.detach().numpy(), (224, 224))
            
            # Enhanced forgery area detection with improved threshold
            forgery_mask = self._extract_forgery_areas(cam, 0.8)  # Use higher threshold
            
            # Create visualizations
            # Fix invalid values in CAM before creating heatmap
            cam_clean = cam.copy()
            cam_clean = np.nan_to_num(cam_clean, nan=0.0, posinf=1.0, neginf=0.0)
            cam_clean = np.clip(cam_clean, 0, 1)
            
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_clean), cv2.COLORMAP_JET)
            overlay_img = np.array(original_img.resize((224, 224)))
            
            # Create detailed overlay with forgery areas highlighted
            detailed_overlay = self._create_detailed_overlay(overlay_img, forgery_mask, cam)
            
            # Convert BGR to RGB
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Save individual images
            os.makedirs(save_dir, exist_ok=True)
            
            # Get original image size
            original_size = original_img.size
            
            # Save original
            original_img.save(os.path.join(save_dir, 'original.png'))
            
            # Save heatmap - resize to original size
            heatmap_img = Image.fromarray(heatmap_rgb)
            heatmap_img = heatmap_img.resize(original_size, Image.Resampling.LANCZOS)
            heatmap_img.save(os.path.join(save_dir, 'heatmap.png'))
            
            # Save detailed overlay - resize to original size
            overlay_img = Image.fromarray(detailed_overlay)
            overlay_img = overlay_img.resize(original_size, Image.Resampling.LANCZOS)
            overlay_img.save(os.path.join(save_dir, 'overlay.png'))
            
            # Clear hooks for next use
            self.activations.clear()
            self.gradients.clear()
            
            return {
                'prediction': 'Fake',
                'confidence': confidence,
                'forgery_areas': self._count_forgery_areas(forgery_mask),
                'images': {
                    'original': os.path.join(save_dir, 'original.png'),
                    'heatmap': os.path.join(save_dir, 'heatmap.png'),
                    'overlay': os.path.join(save_dir, 'overlay.png')
                }
            }
        else:
            # For real documents, just save original
            os.makedirs(save_dir, exist_ok=True)
            original_img.save(os.path.join(save_dir, 'original.png'))
            
            return {
                'prediction': 'Real',
                'confidence': confidence,
                'forgery_areas': 0,
                'images': {
                    'original': os.path.join(save_dir, 'original.png')
                }
            }

def main():
    # Create output directory
    os.makedirs('enhanced_forgery_results', exist_ok=True)
    
    # Initialize enhanced detector
    detector = EnhancedForgeryDetector()
    
    # Test with images from your test set
    test_images = [
        r'C:\Users\Kiran\Desktop\G\SEM 4\Mini Project\projectfinal\test\fake\CMB1.png',
        r'C:\Users\Kiran\Desktop\G\SEM 4\Mini Project\projectfinal\test\real\Aarushi_transcript_page-0001_aug2_crop_2_10.jpg'
    ]
    
    for i, img_path in enumerate(test_images):
        if os.path.exists(img_path):
            print(f"\n{'='*60}")
            print(f"Processing image {i+1}: {os.path.basename(img_path)}")
            print(f"{'='*60}")
            
            save_path = f'enhanced_forgery_results/enhanced_analysis_{i+1}.png'
            result = detector.detect_forgery_areas(img_path, save_path)
            
            if result['prediction'] == 'Fake':
                print(f"🚨 FORGERY DETECTED! 🚨")
                print(f"Confidence: {result['confidence']:.3f}")
                print(f"Number of forgery areas: {result['forgery_areas']}")
                print("📍 RED areas: High-confidence forgery detection")
                print("⚠️  YELLOW areas: Suspicious regions")
                print("🔍 Check the detailed analysis for exact forgery locations!")
            else:
                print(f"✅ DOCUMENT IS AUTHENTIC ✅")
                print(f"Confidence: {result['confidence']:.3f}")
                print("No forgery detected - document appears genuine.")
        else:
            print(f"Image not found: {img_path}")

if __name__ == "__main__":
    main() 