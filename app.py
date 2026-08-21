import os
import joblib
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout, Input

st.set_page_config(
    page_title="Caribbean Brain Coral ID",
    page_icon="🌊",
    layout="wide",
)

# find current directory and saves full path of trained model weights
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_FILENAME = "brain_coral_weights.pkl"
FULL_WEIGHTS_PATH = os.path.join(CURRENT_DIR, WEIGHTS_FILENAME)

@st.cache_resource
def load_coral_model():
    if not os.path.exists(FULL_WEIGHTS_PATH):
        st.error(f"❌ Model file missing. Please ensure '{WEIGHTS_FILENAME}' is saved into: {CURRENT_DIR}")
        st.stop()
        
    print("🏗️ Assembling model architecture...")
    
    # 1. Rebuild model architecture
    base_model = VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))
    
    model = Sequential([
        Input(shape=(224, 224, 3)),
        base_model,
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(6, activation='softmax')
    ])
    
    print("Adding weights into model layer nodes...")
    # 2. Load weights into layer nodes 
    raw_weights = joblib.load(FULL_WEIGHTS_PATH)
    model.set_weights(raw_weights)
    
    return model

# Securely initialize the functional model
model = load_coral_model()


# --- Configuration ---
IMAGE_SIZE = (224, 224)

# Define class names with their actual coral species
CLASS_NAMES = ['Colpophyllia natans', 'Pseudodiploria clivosa', 'Diploria labyrinthiformis', 'Pseudodiploria strigosa', 'Manicina areolata', 'Meandrina meandrites']



# --- Define function to preprocess images ---
def preprocess_image(image):
    try:
        image = image.resize(IMAGE_SIZE)  # Resize to model input size
        image = np.array(image)           # Convert to numpy array
        image = image / 255.0             # Normalize pixel values
        image = np.expand_dims(image, axis=0) # Add batch dimension
        return image
    except Exception as e:
        # Log error for debugging purposes
        print(f"DEBUG: Error during image preprocessing: {e}")
        return None # Indicate failure by returning None

# --- App Layout ---
st.title('The Caribbean Brain Coral Classifier')
st.write('Upload a Caribbean brain coral photo for Identification')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_container_width=True)
        st.write("")
        st.write("Identifying...")

        # Preprocess the image
        processed_image = preprocess_image(image)

        if processed_image is None:
            st.error("Image preprocessing failed. Please check the image file.")
            st.stop() # Stop execution if preprocessing failed

        # Make classification / species identification
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = predictions[0][predicted_class_index] * 100

        st.success(f"Species ID: **{predicted_class_name}**")
        st.write(f"Confidence: **{confidence:.2f}%**")

        st.subheader("All Caribbean Brain Coral Probabilities:")
        for i, (class_name, prob) in enumerate(zip(CLASS_NAMES, predictions[0])):
            st.write(f"- {class_name}: {prob * 100:.2f}%")

    except Exception as e:
        st.error(f"Error processing image: {e}")

else:
    st.info("Please upload an image to get an ID.")


# ==========================================
#         FOOTER PORTFOLIO ANCHOR
# ==========================================
st.write("---")
st.caption(
    """
    Designed and engineered by **Dagoberto Venera-Ponton, PhD**.  
    Open-source code available on [GitHub](https://github.com/dagovenera/Caribbean-Brain-Coral-Classifier.git).
    """
)