import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import Sequential

# ---------------- Constants ----------------
SEQUENCE_LENGTH = 30
IMAGE_SIZE = (224, 224)
MODEL_PATH = os.path.join("models", "seizure_lstm_model.h5")

# ---------------- Load Model ----------------
@st.cache_resource
def load_seizure_model():
    def custom_lstm_layer(**kwargs):
        kwargs.pop('time_major', None)
        return LSTM(**kwargs)

    try:
        model = load_model(
            MODEL_PATH,
            custom_objects={'LSTM': custom_lstm_layer}
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        dummy_input = np.zeros((1, SEQUENCE_LENGTH, 1280), dtype=np.float32)
        model.predict(dummy_input, verbose=0)
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

@st.cache_resource
def load_feature_extractor():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    model = Sequential([base_model, GlobalAveragePooling2D()])
    dummy_img = np.zeros((1, 224, 224, 3), dtype=np.float32)
    model.predict(dummy_img, verbose=0)
    return model

# ---------------- Video Processing ----------------
def extract_limited_frames(video_path, max_frames=90, step=3):
    cap = cv2.VideoCapture(video_path)
    frames, frame_count = [], 0

    while True:
        ret, frame = cap.read()
        if not ret or len(frames) >= max_frames:
            break
        if frame_count % step == 0:
            frame = cv2.resize(frame, IMAGE_SIZE)
            frame = tf.keras.applications.mobilenet_v2.preprocess_input(frame)
            frames.append(frame)
        frame_count += 1

    cap.release()
    return frames

def create_sequences(frames, max_sequences=3):
    sequences = [frames[i:i + SEQUENCE_LENGTH] for i in range(0, len(frames) - SEQUENCE_LENGTH + 1, SEQUENCE_LENGTH)]
    return sequences[:max_sequences]

def extract_features(sequences, feature_extractor):
    if not sequences:
        return np.array([])

    sequences = np.array(sequences, dtype=np.float32)
    batch_size, seq_len, h, w, c = sequences.shape
    features = []

    for seq in sequences:
        flat = seq.reshape(-1, h, w, c)
        feat = feature_extractor.predict(flat, verbose=0)
        features.append(feat.reshape(seq_len, -1))

    return np.array(features, dtype=np.float32)

def predict_seizure(video_path, model, feature_extractor):
    try:
        frames = extract_limited_frames(video_path)
        if len(frames) < SEQUENCE_LENGTH:
            return "\u274c Error: Video too short (needs at least 30 frames)", None, None

        sequences = create_sequences(frames)
        if not sequences:
            return "\u274c Error: Could not create valid sequences", None, None

        features = extract_features(sequences, feature_extractor)
        if features.size == 0:
            return "\u274c Error: Feature extraction failed", None, None

        preds = model.predict(features, verbose=0)
        avg_pred = np.mean(preds, axis=0)
        class_idx = int(np.argmax(avg_pred))

        label_map = {
            0: 'No Seizure Detected',
            1: 'Partial Seizure',
            2: 'Partial to Generalized Seizure'
        }
        label = label_map.get(class_idx, str(class_idx))
        confidence = float(avg_pred[class_idx])

        return f"Prediction: **{label}**", label, confidence

    except Exception as e:
        return f"\u274c Error during prediction: {str(e)}", None, None

# ---------------- Main App ----------------
def main():
    st.set_page_config(page_title="EpilepSee – Seizure Detection", layout="wide", page_icon="⚕️")
    st.sidebar.image("RNN/IMG_6502.png", width=200)
    st.sidebar.title("\ud83e\udde0 EpilepSee")

    if 'go_to_model' in st.session_state and st.session_state['go_to_model']:
        page = "Model"
        st.session_state['go_to_model'] = False
    else:
        page = st.sidebar.radio("Navigation", ["Homepage", "Model"])

    with st.spinner("Loading models..."):
        model = load_seizure_model()
        feature_extractor = load_feature_extractor()

    if model is None:
        st.error("Critical Error: Could not load required resources")
        return

    if page == "Homepage":
        st.image("RNN/IMG_6502.png", width=120)
        st.markdown("<h1 style='text-align: center; color: #3e64ff;'>EpilepSee</h1>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center; color: gray;'>Your AI Assistant for Seizure Detection</h4>", unsafe_allow_html=True)
        st.markdown("---")

        st.markdown("### \ud83d\udd23 What is EpilepSee?")
        st.markdown("""
        **EpilepSee** is an intelligent video analysis tool designed to support the **early detection of epileptic seizures**.
        It uses state-of-the-art **deep learning techniques** combining:

        - \ud83d\udcf9 **Video Input** – Upload clips of observed activity  
        - \ud83e\udde0 **Model** – MobileNetV2 + LSTM to analyze spatial-temporal motion  
        - \ud83c\udfaf **Prediction Output** –  
          - `No Seizure Detected`  
          - `Partial Seizure`  
          - `Partial to Generalized Seizure`
        """)

        st.markdown("### \ud83d\udca1 Why EpilepSee?")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success("\ud83d\udd0e Automated Monitoring\n\nReduce the need for manual review of long surveillance videos.")
        with col2:
            st.info("\ud83e\udde0 Deep Learning Powered\n\nAccurate frame-by-frame motion analysis.")
        with col3:
            st.warning("\ud83c\udfe5 Clinical Support Tool\n\nHelps caregivers and researchers in early screening.")

        st.markdown("---")
        st.markdown("### \ud83e\uddec What is Epilepsy?")
        st.markdown("""
        **Epilepsy** is a neurological disorder characterized by **recurrent seizures**—sudden surges of electrical activity in the brain.  
        These seizures can range from mild staring spells to intense convulsions and loss of consciousness.

        \ud83d\udd0d **Did you know?**  
        - Over **50 million people** globally live with epilepsy  
        - **Early detection and monitoring** are critical for better management  
        - Many patients go **undiagnosed in low-resource settings**

        EpilepSee aims to contribute toward making **preliminary screening more accessible** through AI-driven video analysis.
        """)

        st.markdown("---")
        st.markdown("### \ud83c\udfa5 Demo: How EpilepSee Works")
        st.components.v1.html(
            """
            <video width="700" autoplay loop muted style="border-radius: 10px;">
              <source src="RNN/demo_video.mp4" type="video/mp4">
              Your browser does not support the video tag.
            </video>
            """,
            height=400,
        )

        st.markdown("---")
        st.markdown("### \ud83d\ude80 Try It Now!")
        st.markdown("Upload your video and let EpilepSee analyze it in seconds.")
        if st.button("Start Detection"):
            st.session_state['go_to_model'] = True
            st.experimental_rerun()

        st.markdown("""
        <div style='text-align: center; font-size: 0.9em; color: gray; padding-top: 20px;'>
            <p><b>EpilepSee</b> is a research prototype built for educational and awareness purposes only.</p>
            <p>This is not a medical device and should not replace clinical diagnosis.</p>
            <p>Made with ❤️ using Python, TensorFlow, and Streamlit</p>
        </div>
        """, unsafe_allow_html=True)

    elif page == "Model":
        st.markdown("## \ud83d\udcc4 Upload Video for Seizure Prediction")
        uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

        if uploaded_file:
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)
            video_path = os.path.join(temp_dir, uploaded_file.name)

            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            col1, col2 = st.columns(2)
            with col1:
                st.video(video_path)

            with col2:
                with st.spinner("Analyzing video..."):
                    result, label, confidence = predict_seizure(video_path, model, feature_extractor)

                if label:
                    st.success(result)
                    st.metric("Prediction", label)
                else:
                    st.error(result)

            try:
                os.remove(video_path)
            except:
                pass

if __name__ == "__main__":
    main()
