#!/usr/bin/env python3
"""
Test script to verify the fixes for systematic bias in forgery detection
"""

import os
from forgery_detector_enhanced import EnhancedForgeryDetector
import matplotlib.pyplot as plt
import numpy as np

def test_forgery_detection():
    """Test the improved forgery detection with multiple images"""
    
    # Initialize the enhanced detector
    detector = EnhancedForgeryDetector()
    
    # Test images - mix of real and fake
    test_images = [
        'test/real/Aarushi_transcript_page-0001_aug2_crop_2_10.jpg',
        'test/fake/CMB1.png',
        'test/real/Anirudh (2)_page-0001_aug3_crop_2_82.jpg',
        'test/fake/CMB10.png'
    ]
    
    print("Testing improved forgery detection...")
    print("=" * 60)
    
    for i, img_path in enumerate(test_images):
        if os.path.exists(img_path):
            print(f"\nTest {i+1}: {os.path.basename(img_path)}")
            print("-" * 40)
            
            # Test detection
            result = detector.detect_forgery_areas(img_path, save_path=None, threshold=0.8)
            
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.3f}")
            
            if result['prediction'] == 'Fake':
                print(f"Forgery areas detected: {result['forgery_areas']}")
                
                # Check if forgery mask has any areas
                if result['forgery_mask'] is not None:
                    total_pixels = np.sum(result['forgery_mask'])
                    print(f"Total forgery pixels: {total_pixels}")
                    
                    if total_pixels > 0:
                        # Check distribution across image
                        height, width = result['forgery_mask'].shape
                        quarter_h, quarter_w = height // 4, width // 4
                        
                        # Check each quadrant
                        quadrants = [
                            ("Top-Left", result['forgery_mask'][:quarter_h, :quarter_w]),
                            ("Top-Right", result['forgery_mask'][:quarter_h, -quarter_w:]),
                            ("Bottom-Left", result['forgery_mask'][-quarter_h:, :quarter_w]),
                            ("Bottom-Right", result['forgery_mask'][-quarter_h:, -quarter_w:])
                        ]
                        
                        print("Distribution across quadrants:")
                        for name, quadrant in quadrants:
                            pixels = np.sum(quadrant)
                            percentage = (pixels / total_pixels * 100) if total_pixels > 0 else 0
                            print(f"  {name}: {pixels} pixels ({percentage:.1f}%)")
                    else:
                        print("No forgery areas detected (good - no false positives)")
                else:
                    print("No forgery mask generated")
            else:
                print("Document classified as real")
            
            print()
        else:
            print(f"Image not found: {img_path}")
    
    print("=" * 60)
    print("Test completed!")

if __name__ == "__main__":
    test_forgery_detection() 