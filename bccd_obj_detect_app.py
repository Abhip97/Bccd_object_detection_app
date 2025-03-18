import streamlit as st
import numpy as np
import onnxruntime as ort
import cv2
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# Load ONNX model
onnx_model = ort.InferenceSession("best_yolov10.onnx")

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = np.array(image)  # Convert PIL image to NumPy array
    image = cv2.resize(image, (640, 640))  # Resize to match model input
    image = image / 255.0  # Normalize pixel values
    image = np.transpose(image, (2, 0, 1))  # Change format to (C, H, W)
    image = np.expand_dims(image, axis=0).astype(np.float32)  # Add batch dimension
    return image

# Function to perform inference
def predict(image):
    input_tensor = preprocess_image(image)
    outputs = onnx_model.run(None, {"images": input_tensor})
    
    boxes, scores, class_ids = outputs[0], outputs[1], outputs[2]
    return boxes, scores, class_ids

# Function to display bounding boxes
def draw_bounding_boxes(image, boxes, scores, class_ids, threshold=0.5):
    image = np.array(image)
    for box, score, class_id in zip(boxes, scores, class_ids):
        if score > threshold:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"Class {class_id}: {score:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return Image.fromarray(image)

# Streamlit UI
st.title("🔬 Blood Sample Object Detection")
st.write("Upload a blood sample image to detect objects.")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect Objects"):
        boxes, scores, class_ids = predict(image)
        result_image = draw_bounding_boxes(image, boxes, scores, class_ids)
        st.image(result_image, caption="Detection Results", use_column_width=True)

        # Precision & Recall table (dummy example)
        df = pd.DataFrame({
            "Class": ["RBC", "WBC", "Platelets"],
            "Precision": [0.92, 0.85, 0.78],
            "Recall": [0.89, 0.80, 0.75],
        })
        st.write("### Precision & Recall Table")
        st.table(df)
