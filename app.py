import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = load_model(r'C:\Users\User\Desktop\Reena\Potato_Disease_Identifier\models\model_v1.h5')

# Define class names (edit as per your training)
class_names = ["Early Blight", "Late Blight", "Healthy"]

# Set title
st.title("🥔 Potato Disease Detector")
st.write("Upload a potato leaf image to detect the disease.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image = image.resize((256, 256))  # Match model input size
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    prediction = model.predict(image_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"Prediction: **{predicted_class}**")
