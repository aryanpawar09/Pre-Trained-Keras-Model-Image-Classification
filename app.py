import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("resnet50_model.keras")

model = load_model()

st.title("Image Classification with ResNet50")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make prediction
    preds = model.predict(img_array)
    predictions = decode_predictions(preds, top=3)[0]

    st.subheader("Predictions:")
    for label, name, confidence in predictions:
        st.write(f"**{name}:** {confidence*100:.2f}%")
