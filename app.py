# app.py
from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
from predict import load_model, predict_document, get_forgery_overlay, get_forgery_analysis_images
import time

UPLOAD_FOLDER = 'static/uploads'  # Folder where uploaded images will be stored
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model once at the start
model = load_model('resnet18_model.pth')  # Ensure this path is correct

# Check if file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)  # Save the file to server

        # Get prediction
        prediction, confidence = predict_document(file_path, model)

        # Generate Grad-CAM overlay if fake
        overlay_path = None
        analysis_images = None
        if prediction.lower() == 'fake':
            # Create unique filename with timestamp
            timestamp = int(time.time())
            name, ext = os.path.splitext(filename)
            overlay_filename = f'overlay_{name}_{timestamp}{ext}'
            overlay_path = os.path.join(app.config['UPLOAD_FOLDER'], overlay_filename)
            overlay_result = get_forgery_overlay(file_path, overlay_path)
            if overlay_result:
                overlay_path = overlay_result
            else:
                overlay_path = None
            
            # Generate individual analysis images for popup
            analysis_dir = os.path.join(app.config['UPLOAD_FOLDER'], f'analysis_{timestamp}')
            analysis_result = get_forgery_analysis_images(file_path, analysis_dir)
            if analysis_result and analysis_result['prediction'].lower() == 'fake':
                # Get the relative paths for web access
                analysis_images = {
                    'original': f'static/uploads/{os.path.basename(analysis_dir)}/original.png',
                    'heatmap': f'static/uploads/{os.path.basename(analysis_dir)}/heatmap.png',
                    'overlay': f'static/uploads/{os.path.basename(analysis_dir)}/overlay.png',
                    'forgery_areas': analysis_result['forgery_areas']
                }

        # Return the result as a JSON response
        return jsonify({
            'prediction': prediction,
            'confidence': confidence,
            'image': f'static/uploads/{filename}',
            'overlay': f'static/uploads/{os.path.basename(overlay_path)}' if overlay_path else None,
            'analysis_images': analysis_images
        })

    return jsonify({'error': 'Invalid file type'})

if __name__ == "__main__":
    app.run(debug=True)
