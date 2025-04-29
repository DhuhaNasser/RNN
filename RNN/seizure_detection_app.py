import streamlit as st 
import numpy as np
import tensorflow as tf
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import Sequential

# Constants
SEQUENCE_LENGTH = 30
IMAGE_SIZE = (224, 224)
MODEL_PATH = os.path.join("models", "seizure_lstm_model.h5")

# ---------- Model Loading ----------
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

# ---------- Video Processing ----------
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
            return "❌ Error: Video too short (needs at least 30 frames)", None, None

        sequences = create_sequences(frames)
        if not sequences:
            return "❌ Error: Could not create valid sequences", None, None

        features = extract_features(sequences, feature_extractor)
        if features.size == 0:
            return "❌ Error: Feature extraction failed", None, None

        preds = model.predict(features, verbose=0)
        avg_pred = np.mean(preds, axis=0)
        class_idx = int(np.argmax(avg_pred))

        label_map = {0: 'No_Seizure', 1: 'P', 2: 'PG'}
        label = label_map.get(class_idx, str(class_idx))
        confidence = float(avg_pred[class_idx])

        return f"Prediction: **{label}**", label, confidence

    except Exception as e:
        return f"❌ Error during prediction: {str(e)}", None, None

# ---------- Streamlit Interface ----------
def main():
    st.set_page_config(page_title="EpilepSee – Seizure Detection", layout="wide", page_icon="🧠")
    st.sidebar.image("RNN/IMG_6502.png", width=200)
    st.sidebar.title("🧠 EpilepSee")

    page = st.sidebar.radio("Navigation", ["Homepage", "Model"])

    with st.spinner("Loading models..."):
        model = load_seizure_model()
        feature_extractor = load_feature_extractor()

    if model is None:
        st.error("Critical Error: Could not load required resources")
        return

    if page == "Homepage":
        st.markdown("## 🧠 Welcome to EpilepSee")
        st.markdown("""
        EpilepSee is an AI-powered system designed to analyze **video recordings** and detect signs of epileptic seizures.  
        It uses a combination of **MobileNetV2 for feature extraction** and **LSTM for temporal analysis** of video frame sequences.

        ### 🔍 Prediction Categories:
        - **No_Seizure**: Normal movement  
        - **P**: Partial seizure  
        - **PG**: Partial to Generalized seizure
        """)

    elif page == "Model":
        st.markdown("## 📤 Upload Video for Seizure Prediction")
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
