import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model once (cache to avoid reloading on every run)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# Set up Streamlit app
st.title("♻️ Scan4Clean - Trash or Not?")
st.write("Upload an image and our AI will tell you if it’s trash or not.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_size = (224, 224)  # change to your training size
    img = image.resize(img_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]  # adjust indexing for your model
    label = "Trash" if prediction > 0.5 else "Not Trash"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.markdown(f"**Prediction:** {label}")
    st.markdown(f"**Confidence:** {confidence:.2%}")
