import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image # type: ignore
import time

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="🧠 Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("my_model1.keras")
    return model

model = load_model()

# ------------------ CLASS LABELS ------------------
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# ------------------ CUSTOM CSS ------------------
st.markdown("""
    <style>
        body {
            background-color: #f2f7fb;+
        }
        .main {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0px 2px 12px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            color: #0077b6;
        }
        .stButton>button {
            background-color: #90e0ef;
            color: #023047;
            border-radius: 10px;
            height: 3em;
            width: 100%;
            font-size: 16px;
            font-weight: 600;
            border: none;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        }
        .stButton>button:hover {
            background-color: #caf0f8;
            color: #03045e;
        }
        .uploadedImage {
            border: 3px solid #90e0ef;
            border-radius: 10px;
            padding: 5px;
        }
        .sidebar .sidebar-content {
            background-color: #e3f2fd;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📤 Upload & Predict", "📚 About"])
st.sidebar.markdown("---")
st.sidebar.markdown("👩‍⚕️ **Developed By:** Your Name")
st.sidebar.markdown("📅 **Version:** 2.0")

# ------------------ HOME PAGE ------------------
if page == "🏠 Home":
    st.title("🧠 Brain Tumor Classification App")
    st.markdown("""
        ### Welcome!
        This app uses a **Deep Learning CNN model** to classify MRI images into the following categories:
        - 🧩 **Glioma**
        - 🧠 **Meningioma**
        - ✅ **No Tumor**
        - ⚙️ **Pituitary**

        ---
        Upload an MRI scan in the **Upload & Predict** section to see real-time predictions.
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://i.imgur.com/oT8sB6P.png", caption="Normal Brain MRI", use_column_width=True)
    with col2:
        st.image("https://i.imgur.com/95EwZHT.png", caption="Tumor MRI Example", use_column_width=True)

# ------------------ UPLOAD & PREDICT PAGE ------------------
elif page == "📤 Upload & Predict":
    st.title("📤 Upload MRI Image")
    st.markdown("Upload a brain MRI image to detect if it contains any tumor.")

    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption='🧠 Uploaded MRI Image', use_column_width=True)

        # Preprocess the image
        img = img.resize((224, 224))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        if st.button("🚀 Predict Tumor Type"):
            with st.spinner("Analyzing MRI image..."):
                time.sleep(2)
                predictions = model.predict(img_array)
                score = tf.nn.softmax(predictions[0])
                predicted_class = CLASS_NAMES[np.argmax(score)]
                confidence = 100 * np.max(score)

                st.success(f"🎯 **Prediction:** {predicted_class}")
                st.info(f"**Confidence:** {confidence:.2f}%")

                if predicted_class == "No Tumor":
                    st.balloons()
                else:
                    st.warning(f"⚠️ Detected possible **{predicted_class} tumor**. Please consult a specialist.")

    else:
        st.warning("⚠️ Please upload an image first to make a prediction.")

# ------------------ ABOUT PAGE ------------------
elif page == "📚 About":
    st.title("📚 About This Project")
    st.markdown("""
        ### 🎯 Objective
        This project aims to detect and classify **brain tumors** from MRI scans using deep learning.

        ### 🧩 Model Information
        - Model: **Convolutional Neural Network (CNN)**
        - Framework: **TensorFlow / Keras**
        - Classes: **Glioma, Meningioma, No Tumor, Pituitary**
        - Accuracy: ~98% on test dataset

        ### ⚙️ Tech Stack
        - Python 🐍
        - TensorFlow / Keras
        - Streamlit
        - OpenCV, NumPy, PIL

        ---
        **Disclaimer:**  
        This tool is for educational and research purposes only.  
        Please consult a medical professional for an actual diagnosis.
    """)
    st.success("Developed with ❤️ using Streamlit and TensorFlow")
