import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image as tf_image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO

st.title('Dynamic Image Classifier using TensorFlow')

# Load the pre-trained model
model = keras.applications.MobileNetV2(weights='imagenet')

# Define a function to preprocess the image


def preprocess_image(image_data):
    img = Image.open(BytesIO(image_data))
    img = img.resize((224, 224))
    x = tf_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    return x

# Define a function to make predictions


def make_prediction(image_data):
    preprocessed_image = preprocess_image(image_data)
    predictions = model.predict(preprocessed_image)
    decoded_predictions = keras.applications.mobilenet_v2.decode_predictions(
        predictions)
    return decoded_predictions[0]


# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Make predictions
    predictions = make_prediction(uploaded_file.getvalue())

    # Display the predictions
    st.write('Predictions:')
    for i, prediction in enumerate(predictions):
        st.write(f'{i+1}. {prediction[1]} ({prediction[2]*100:.2f}%)')
