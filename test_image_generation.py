import os
from forgery_detector_enhanced import EnhancedForgeryDetector

# Test the image generation
detector = EnhancedForgeryDetector()

# Test with a fake image
test_image = r'C:\Users\Kiran\Desktop\G\SEM 4\Mini Project\projectfinal\test\fake\CMB1.png'
test_dir = r'C:\Users\Kiran\Desktop\G\SEM 4\Mini Project\projectfinal\static\uploads\test_analysis'

print("Testing image generation...")
print(f"Test image: {test_image}")
print(f"Test directory: {test_dir}")

if os.path.exists(test_image):
    print("Test image exists!")
    result = detector.generate_individual_images(test_image, test_dir)
    print("Result:", result)
    
    if result and 'images' in result:
        print("\nGenerated images:")
        for key, path in result['images'].items():
            print(f"{key}: {path}")
            if os.path.exists(path):
                print(f"  ✓ File exists: {path}")
            else:
                print(f"  ✗ File missing: {path}")
else:
    print("Test image not found!") 