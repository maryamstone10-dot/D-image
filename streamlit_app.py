import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model  # <-- updated import
import io

st.set_page_config(page_title="Teachable Machine Image Classifier", page_icon="🧠")
st.title("🧠 Teachable Machine Image Classifier")

MODEL_PATH = "keras_model.h5"     # make sure this file exists in repo root
LABELS_PATH = "labels.txt"        # make sure this file exists in repo root

@st.cache_resource
def load_tm_model(path: str):
    # compile=False for inference-only loading of .h5 in Keras 3 / TF 2.20
    return load_model(path, compile=False)

@st.cache_data
def load_labels(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]

# Load model and labels with graceful errors
try:
    model = load_tm_model(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load model `{MODEL_PATH}`. Ensure it’s in the repo. Details:\n{e}")
    st.stop()

try:
    class_names = load_labels(LABELS_PATH)
except Exception as e:
    st.error(f"Could not load labels `{LABELS_PATH}`. Ensure it’s in the repo. Details:\n{e}")
    st.stop()

st.markdown("Upload an image (or take a photo). Model expects **224×224 RGB**, normalized to **[-1, 1]**.")

uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
camera_image = st.camera_input("Or take a photo")

image_file = uploaded or camera_image

if image_file:
    # Read into PIL.Image
    if hasattr(image_file, "getvalue"):  # UploadedFile or camera snapshot
        image = Image.open(io.BytesIO(image_file.getvalue())).convert("RGB")
    else:  # Fallback if it's a pathlike (rare in Streamlit Cloud)
        image = Image.open(image_file).convert("RGB")

    st.image(image, caption="Input image", use_container_width=True)

    # Resize/crop to 224x224 and normalize to [-1, 1] (as in your original script)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image).astype(np.float32)
    normalized = (image_array / 127.5) - 1.0
    data = np.expand_dims(normalized, axis=0)  # shape (1, 224, 224, 3)

    with st.spinner("Predicting..."):
        preds = model.predict(data)
        idx = int(np.argmax(preds[0]))
        score = float(preds[0][idx])
        label = class_names[idx] if idx < len(class_names) else f"Class {idx}"

    st.success(f"Prediction: **{label}**")
    st.write(f"Confidence: **{score:.4f}**")

st.caption("Make sure `keras_model.h5` and `labels.txt` are in the same folder as this app.")

