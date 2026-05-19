document.getElementById("uploadForm").addEventListener("submit", function (e) {
    e.preventDefault();
    console.log("Form submitted");

    const fileInput = document.getElementById("imageInput");
    if (!fileInput.files[0]) {
        console.log("No file selected");
        return;
    }
    
    console.log("File selected:", fileInput.files[0].name);
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    console.log("Sending request to /predict");
    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(res => {
        console.log("Response received:", res.status);
        return res.json();
    })
    .then(data => {
        console.log("Data received:", data);
        if (data.prediction) {
            document.getElementById("result").innerText = "Prediction: " + data.prediction.toUpperCase();
            document.getElementById("confidence").innerText = "Confidence: " + (data.confidence * 100).toFixed(2) + "%";
            document.getElementById("preview").src = data.image;

            document.getElementById("result").style.display = "block";
            document.getElementById("confidence").style.display = "block";
            document.getElementById("preview").style.display = "block";

            // Show inline analysis for fake documents
            if (data.prediction.toLowerCase() === 'fake' && data.analysis_images) {
                console.log("Showing fake analysis");
                showInlineAnalysis(data);
            } else {
                console.log("Showing real document result");
                // Hide analysis section for real documents
                document.getElementById("analysisSection").style.display = "none";
            }
        } else {
            console.log("Error in response:", data.error);
            document.getElementById("result").innerText = "Error: " + data.error;
        }
    })
    .catch(error => {
        console.log("Fetch error:", error);
        document.getElementById("result").innerText = "Error: " + error;
    });
});

function showInlineAnalysis(data) {
    const analysisSection = document.getElementById("analysisSection");
    const analysisImages = data.analysis_images;
    
    // Update analysis content
    document.getElementById("inlineConfidence").innerText = `Confidence: ${(data.confidence * 100).toFixed(2)}%`;
    document.getElementById("inlineForgeryAreas").innerText = `Forgery Areas Detected: ${analysisImages.forgery_areas}`;
    
    // Set image sources
    document.getElementById("inlineOriginal").src = analysisImages.original;
    document.getElementById("inlineHeatmap").src = analysisImages.heatmap;
    document.getElementById("inlineOverlay").src = analysisImages.overlay;
    
    // Set initial view to overlay
    document.getElementById("inlineSingleImageView").src = analysisImages.overlay;
    document.getElementById("inlineSingleViewTitle").innerText = "Detailed Overlay (Red/Yellow Areas)";
    
    // Show analysis section
    analysisSection.style.display = "block";
    
    // Setup view selector
    setupInlineViewSelector(analysisImages);
}

function setupInlineViewSelector(analysisImages) {
    const viewSelector = document.getElementById("inlineViewSelector");
    const singleView = document.getElementById("inlineSingleView");
    const allViews = document.getElementById("inlineAllViews");
    const singleImageView = document.getElementById("inlineSingleImageView");
    const singleViewTitle = document.getElementById("inlineSingleViewTitle");
    
    viewSelector.addEventListener("change", function() {
        const selectedView = this.value;
        
        if (selectedView === "all") {
            singleView.style.display = "none";
            allViews.style.display = "grid";
        } else {
            singleView.style.display = "block";
            allViews.style.display = "none";
            
            switch(selectedView) {
                case "overlay":
                    singleImageView.src = analysisImages.overlay;
                    singleViewTitle.innerText = "Detailed Overlay (Red/Yellow Areas)";
                    break;
                case "heatmap":
                    singleImageView.src = analysisImages.heatmap;
                    singleViewTitle.innerText = "Forgery Detection Heatmap";
                    break;
            }
        }
    });
}

// Remove close functionality - modal is not closable
// No event listeners for closing the modal
