# Caribbean-Brain-Coral-Classifier
A fully deployed Convolutional Neural Network that identifies Caribbean species of brain coral from images

# 🖼️ Caribbean Brain Coral Deep Learning Classifier

[![Streamlit App](https://streamlit.io)](https://caribbean-brain-coral-classifier.streamlit.app/)
[![Python](https://shields.io)](https://python.org)
[![TensorFlow](https://shields.io)](https://tensorflow.org)

An end-to-end computer vision and deep learning architecture designed to automate the identification and phenotypic differentiation of closely related/similar Caribbean brain coral species from field imagery. The application utilizes Transfer Learning on top-tier deep convolutional neural networks to deliver real-time, cloud-hosted species classification.

---

## 🏗️ System Architecture & Engineering Strategy

To maximize cross-platform execution stability and bypass standard framework-level deserialization blocks (e.g., Keras 2 vs. Keras 3 versioning mismatches), this production pipeline implements a decoupled architectural deployment strategy:



1. **Transfer Learning Framework:** Utilizes a state-of-the-art **VGG16** backbone trained on millions of images. The deep convolutional feature extractor base is frozen, and a custom classification head (consisting of dense layers, batch normalization vectors, and dropout regularizations, to specialize in brain coral structures) is appended.
2. **Decoupled Serialization Integration (`train_pipeline`):** Instead of saving unstable, monolithic `.keras` or `.h5` files, the pipeline extracts the raw mathematical weight matrices. These numbers are serialized into a lightweight, platform-independent `brain_coral_weights.pkl` artifact via `joblib`.
3. **Dynamic Reassembly & Inference Engine (`app.py`):** On application initialization, the Streamlit server rebuilds a fresh, native structural frame of the convolutional network, then injects the raw numeric weights into the neural nodes, ensuring 100% environment-independent execution stability.

---

## 🛠️ Technical Toolkit

- **Languages:** Python (NumPy, Pandas, PIL)
- **Deep Learning Frameworks:** TensorFlow 2.x, Keras 3.x, VGG16 Pre-trained Backbone
- **Data Engineering & Deployment:** Joblib (Numerical Matrix Serialization), Streamlit Cloud
- **Computer Vision Pipelines:** Image resizing parameters ($224 \times 224 \times 3$), pixel normalization vectors ($1/255.0$), dynamic training data augmentation (rotating, shifting, flipping, zooming)

---

## 📋 Evaluated Coral Species Categories

The neural network computes probability distribution vectors across 6 target Caribbean reef-building corals:
*   `C_natans` (Colpophyllia natans)
*   `P_clivosa` (Pseudodiploria clivosa)
*   `D_labyrhintiformis` (Diploria labyrinthiformis)
*   `P_strigosa` (Pseudodiploria strigosa)
*   `M_areolata` (Manicina areolata)
*   `M_meandrites` (Meandrina meandrites)

---

## 🚀 Local Installation & Execution

### 1. Clone the Repository Infrastructure
```bash
git clone https://github.com
cd Caribbean-Brain-Coral-Classifier
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Calibrate Local Weights Layer
Ensure your trained `brain_coral_weights.pkl` matrix file is placed in the root directory of the cloned project folder layout.

### 3. Launch the Application Dashboard
```bash
streamlit run app.py
```

---

## 👨‍💻 Developer Profile

**Dagoberto E. Venera-Pontón, PhD**  
*Computational Biology, Advanced Statistical Modelling, & Applied Machine Learning Specialist*  
- **LinkedIn:** [://linkedin.com](https://linkedin.com/in/dagoberto-venera-ponton-phd)
- **GitHub Portfolio:** [://github.com](https://github.com/dagovenera)