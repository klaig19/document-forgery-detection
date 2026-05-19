# Document Forgery Detection System

A deep learning powered web application that detects forged documents using computer vision. Upload an image of any document and the system tells you whether it is authentic or tampered, with visual heatmaps pinpointing the exact suspicious regions.

---

## What This Project Does

Fake documents are everywhere. Forged transcripts, manipulated certificates, altered identity proofs — detecting them manually is slow, inconsistent, and unreliable. This project automates that process using a fine tuned ResNet18 model trained on a labeled dataset of real and fake document images.

When a document is uploaded, the system runs it through the model and produces one of two outcomes:

**Authentic** — the document passes verification with a confidence score and a clean result screen.

**Forged** — the model flags it as fake and generates a Grad CAM heatmap that visually highlights the manipulated regions. Red zones indicate high confidence forgery areas. Yellow zones mark regions that look suspicious but not conclusively altered.

The whole thing runs in a browser via a Flask web interface. No command line knowledge required to use it.

---

## How It Works

The backbone of this system is ResNet18, a convolutional neural network originally trained on ImageNet. The final fully connected layer was replaced to output two classes (real vs fake) and the model was retrained on a custom document dataset using weighted cross entropy loss to handle class imbalance.

For explainability, Gradient weighted Class Activation Mapping (Grad CAM) is used. When a document is classified as fake, the system performs a backward pass through the last convolutional layer and computes a weighted activation map. This map is overlaid on the original image to show which parts of the document influenced the decision most strongly.

The enhanced forgery detector adds additional post processing on top of raw Grad CAM output including morphological cleanup, minimum area filtering, corner concentration checks, and a minimum 85% confidence threshold before flagging a document as forged. This significantly reduces false positives.

---

## Project Structure

```
cv practical project/
    app.py                         Flask web server and API routes
    predict.py                     Model loading and inference logic
    forgery_detector_enhanced.py   Enhanced Grad CAM with forgery localization
    gradcam.py                     Standalone Grad CAM implementation
    model.py                       ResNet18 binary classifier definition
    train.py                       Training script with weighted sampling
    evaluate.py                    Evaluation script with classification report
    utils.py                       Utility functions
    test_fixes.py                  Regression and fix verification tests
    test_image_generation.py       Image generation test utilities
    templates/
        index.html                 Frontend UI with modal popup for analysis
    static/
        script.js                  Frontend logic and AJAX calls
        style.css                  Application styling
        detective_image.png        UI branding asset
        uploads/                   Stores uploaded and generated images
    enhanced_forgery_results/      Sample enhanced analysis outputs
    gradcam_results/               Sample Grad CAM visualization outputs
```

---

## Tech Stack

**Backend** — Python, Flask, PyTorch, torchvision, OpenCV, scikit image, scipy, Pillow, NumPy, Matplotlib

**Frontend** — HTML, CSS, JavaScript (vanilla, with AJAX for async prediction)

**Model** — ResNet18 pretrained on ImageNet, fine tuned for binary document classification

**Explainability** — Gradient weighted Class Activation Mapping (Grad CAM)

---

## Getting Started

**1. Clone the repository**

```bash
git clone https://github.com/your-username/cv-practical-project.git
cd cv-practical-project
```

**2. Install dependencies**

```bash
pip install torch torchvision flask pillow opencv-python matplotlib scikit-image scipy numpy
```

**3. Add the trained model**

Place your trained `resnet18_model.pth` file in the root of the project directory. If you want to train from scratch, see the Training section below.

**4. Run the application**

```bash
python app.py
```

Open your browser and go to `http://localhost:5000`. Upload a document image and see the result instantly.

---

## Training the Model

If you want to retrain on your own dataset, organize your data like this:

```
dataset/
    real/
        image1.jpg
        image2.png
        ...
    fake/
        image1.jpg
        image2.png
        ...
```

Then update the dataset path in `train.py` and run:

```bash
python train.py
```

Training runs for 15 epochs by default with Adam optimizer and a learning rate of 0.001. The best model (by validation accuracy) is automatically saved as `resnet18_model.pth`. WeightedRandomSampler ensures balanced training even when class sizes differ significantly.

---

## Evaluating the Model

To generate a classification report and confusion matrix on your test set:

```bash
python evaluate.py
```

Point the `val_dataset` path in `evaluate.py` to your test folder first. The output includes per class precision, recall, and F1 score.

---

## Running Grad CAM Standalone

If you want to generate Grad CAM visualizations outside of the web app:

```bash
python gradcam.py
```

Update the `test_images` list in `gradcam.py` with paths to your test images. Results are saved to the `gradcam_results/` folder.

---

## API Reference

The Flask app exposes two endpoints:

**GET /**
Serves the main web interface.

**POST /predict**
Accepts a multipart form upload with a `file` field (PNG, JPG, or JPEG). Returns a JSON response:

```json
{
  "prediction": "fake",
  "confidence": 0.9341,
  "image": "static/uploads/sample.jpg",
  "overlay": "static/uploads/overlay_sample_1234567890.jpg",
  "analysis_images": {
    "original": "static/uploads/analysis_1234567890/original.png",
    "heatmap": "static/uploads/analysis_1234567890/heatmap.png",
    "overlay": "static/uploads/analysis_1234567890/overlay.png",
    "forgery_areas": 3
  }
}
```

If the document is classified as real, `overlay` and `analysis_images` will be `null`.

---

## Sample Results

The `enhanced_forgery_results/` and `gradcam_results/` folders contain sample outputs showing the model in action on test documents. The analysis panels display the original document, the raw Grad CAM heatmap, the annotated forgery overlay, and a summary of detected regions.

---

## Key Design Decisions

**Why ResNet18?** It strikes a good balance between accuracy and inference speed for a CPU friendly web app. Heavier architectures like ResNet50 would improve accuracy marginally but would be noticeably slower on consumer hardware.

**Why Grad CAM instead of just confidence scores?** A confidence score alone tells you what the model thinks but not why. Grad CAM gives visual evidence, which makes the prediction interpretable and far more useful in any real world verification scenario.

**Why 85% confidence threshold for fake flagging?** Documents can look visually similar to tampered ones due to compression artifacts or poor scan quality. A high confidence threshold keeps false positives down and means that when the system does flag something as fake, it is almost certainly correct.

---

## Limitations

The model's accuracy depends heavily on the quality and diversity of the training data. It performs best on the document types it was trained on (transcripts and certificates). Performance on unseen document formats may vary. The system is designed as a decision support tool rather than a replacement for human verification in high stakes scenarios.

---


