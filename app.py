
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- Configuration ---
MODEL_PATH = 'model_modified_head.keras'
IMAGE_SIZE = (224, 224)

# Define class names here. Replace with actual coral categories!
CLASS_NAMES = ['coral_type_1', 'coral_type_2', 'coral_type_3', 'coral_type_4', 'coral_type_5', 'coral_type_6']

# --- Load the model ---
@st.cache_resource  # Cache the model loading to avoid reloading on every rerun
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()

# --- Preprocess image function ---
def preprocess_image(image):
    try:
        image = image.resize(IMAGE_SIZE)  # Resize to model input size
        image = np.array(image)           # Convert to numpy array
        image = image / 255.0             # Normalize pixel values
        image = np.expand_dims(image, axis=0) # Add batch dimension
        return image
    except Exception as e:
        # Log the error for debugging purposes
        print(f"DEBUG: Error during image preprocessing: {e}")
        return None # Indicate failure by returning None

# --- Streamlit App Layout ---
st.title('Coral Image Classifier')
st.write('Upload an image of coral and the model will predict its category.')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Preprocess the image
        processed_image = preprocess_image(image)

        if processed_image is None:
            st.error("Image preprocessing failed. Please check the image file.")
            st.stop() # Stop execution if preprocessing failed

        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = predictions[0][predicted_class_index] * 100

        st.success(f"Prediction: **{predicted_class_name}**")
        st.write(f"Confidence: **{confidence:.2f}%**")

        st.subheader("All Class Probabilities:")
        for i, (class_name, prob) in enumerate(zip(CLASS_NAMES, predictions[0])):
            st.write(f"- {class_name}: {prob * 100:.2f}%")

    except Exception as e:
        st.error(f"Error processing image: {e}")

else:
    st.info("Please upload an image to get a prediction.")

# Optional: Display model summary
# if st.checkbox('Show Model Summary'):
#     model.summary(print_fn=st.text)
