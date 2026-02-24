import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Microplastic Detector", layout="wide")

st.title("🔬 Microplastic Detection App")
st.write("Upload an image to detect microplastic particles.")

# Load model once
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Sidebar settings
st.sidebar.header("⚙️ Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    # Resize image to reasonable size (keeps aspect ratio)
    max_size = 600
    image.thumbnail((max_size, max_size))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(image, width=400)

    if st.button("🚀 Detect Microplastics"):

        with st.spinner("Detecting..."):
            results = model(image, conf=confidence)
            result_img = results[0].plot()

        with col2:
            st.subheader("🧠 Detection Result")
            st.image(result_img, width=400)

        num_detections = len(results[0].boxes)
        st.success(f"Detected {num_detections} microplastic particles.")