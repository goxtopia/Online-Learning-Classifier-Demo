# Online Learnable Classifier Demo

This project demonstrates an online learnable classifier system that refines YOLO detections with user feedback.

## Features

1.  **Object Detection**: Uses YOLOv8 (nano) to detect persons in uploaded images.
2.  **Secondary Classifier**: A learnable classifier (ResNet18 features + MLPClassifier) assigns a probability of "Human" vs "False Positive".
3.  **Online Learning**: Users can click on a detection box to mark it as "True Positive" (Human) or "False Positive". The model updates instantly.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Server**:
    ```bash
    python app/main.py
    ```
    or
    ```bash
    uvicorn app.main:app --reload
    ```

3.  **Access the Demo**:
    Open [http://localhost:8000/static/index.html](http://localhost:8000/static/index.html) in your browser.

## Usage

1.  Upload an image.
2.  View detections. Green boxes indicate high confidence from the secondary classifier. Red boxes indicate low confidence.
3.  Click on a box.
4.  Select "Confirm Human" or "Mark as False Positive".
5.  Upload a similar image (or the same one) to see how the classifier predictions change.

## Architecture

-   **Backend**: FastAPI
-   **ML**: PyTorch (ResNet18 feature extractor), PyTroch (MLPClassifier), Ultralytics (YOLOv8)
-   **Frontend**: HTML5 Canvas + JavaScript
