import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions

#Usando o modelo já treinado do IncepctionV3 para classificar imagens (poderia ser outro modelo).
model = InceptionV3(weights='imagenet')

def classify_image(image):
    image = image.convert("RGB")
    target_size = (299, 299) #Redimensiona imagem
    image = image.resize(target_size)

    # Preprocess the image
    img_array = kimage.img_to_array(image)
    img_array = preprocess_input(img_array)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
  
    decoded_predictions = decode_predictions(predictions)[0]
    return decoded_predictions
  
def main():
    st.title("Image Classification with InceptionV3")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        predictions = classify_image(image)

        st.subheader("Classification Results:")
        for i, (imagenet_id, label, score) in enumerate(predictions):
            st.write(f"{i + 1}: {label} ({score:.2f})")

if __name__ == "__main__":
    main()
