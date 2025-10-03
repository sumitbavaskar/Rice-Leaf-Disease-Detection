import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
import numpy as np

# --------------------------------------------------
# Load model with caching (faster reloads)
# --------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("rice_leaf_disease_model.h5")

model = load_model()
class_names = ["Leaf smut", "Brown spot", "Bacterial leaf blight"]

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="Rice Leaf Disease Detection", page_icon="🌾", layout="centered")

st.title("🌾 Rice Leaf Disease Detection")
st.markdown("Upload a rice leaf image and the model will predict the type of disease.")

# --------------------------------------------------
# File uploader
# --------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload a rice leaf image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    try:
        # Load and preprocess image
        image = Image.open(uploaded_file).convert("RGB").resize((224,224))
        st.image(image, caption="📸 Uploaded Image", use_column_width=True)

        img_array = np.expand_dims(np.array(image) / 255.0, axis=0)

        # Prediction
        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        st.success(f"✅ Prediction: **{class_names[class_idx]}** 🌱")
        st.info(f"Confidence: {confidence:.2f}%")

        # Show top-3 probabilities
        st.subheader("🔎 Detailed Prediction Probabilities:")
        for i in np.argsort(prediction[0])[::-1]:
            st.write(f"- {class_names[i]}: {prediction[0][i]*100:.2f}%")

    except UnidentifiedImageError:
        st.error("⚠️ Invalid file type. Please upload a valid JPG/PNG image.")
    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")

else:
    st.warning("Please upload a rice leaf image to continue.")
