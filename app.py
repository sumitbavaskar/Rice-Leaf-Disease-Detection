import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
model = tf.keras.models.load_model("rice_leaf_disease_model.h5")
class_names = ["Leaf smut", "Brown spot", "Bacterial leaf blight"]

st.title("🌾 Rice Leaf Disease Detection")

uploaded_file = st.file_uploader("Upload a rice leaf image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((224,224))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_array = np.expand_dims(np.array(image)/255.0, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    st.success(f"Prediction: {class_names[class_idx]} 🌱")
