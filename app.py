
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Brain Tumor MRI Classification",
    page_icon="🧠",
    layout="centered"
)

# Model stored in the same GitHub repository as app.py
MODEL_PATH = "efficientnetb0_best.h5"

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]

@st.cache_resource
def load_brain_tumor_model():
    return load_model(MODEL_PATH)

model = load_brain_tumor_model()

st.title("🧠 Brain Tumor MRI Image Classification")

st.write(
    "Upload a brain MRI image to obtain an AI-assisted prediction "
    "of the tumor category and its confidence score."
)

uploaded_file = st.file_uploader(
    "Upload a Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded MRI Image",
        use_container_width=True
    )

    processed_image = image.resize((224, 224))

    image_array = np.array(
        processed_image
    ).astype("float32")

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    predictions = model.predict(
        image_array,
        verbose=0
    )

    predicted_index = np.argmax(predictions[0])

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = (
        float(predictions[0][predicted_index])
        * 100
    )

    st.subheader("Prediction Result")

    st.success(
        f"Predicted Tumor Category: {predicted_class}"
    )

    st.info(
        f"Prediction Confidence: {confidence:.2f}%"
    )

    st.subheader("Class Probabilities")

    for class_name, probability in zip(
        CLASS_NAMES,
        predictions[0]
    ):
        st.write(
            f"{class_name}: {probability * 100:.2f}%"
        )

st.caption(
    "This application is intended for AI-assisted research "
    "and educational purposes and is not a medical diagnosis."
)
