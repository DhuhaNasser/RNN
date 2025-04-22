import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import os
import psutil
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import Sequential
import firebase_admin
from firebase_admin import credentials, firestore

# -------------------- INIT FIREBASE --------------------
if not firebase_admin._apps:
    cred = credentials.Certificate("resources/serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
db = firestore.client()

# -------------------- CONSTANTS --------------------
SEQUENCE_LENGTH = 30
IMAGE_SIZE = (224, 224)
MAX_FRAMES = 90  # LIMIT to reduce memory usage
MODEL_PATH = os.path.join("models", "seizure_lstm_model.h5")

# -------------------- DATABASE CLASS --------------------
class SeizureDatabase:
    def __init__(self):
        self.collection = db.collection('seizure_predictions')

    def add_prediction(self, video_name, label, confidence):
        try:
            self.collection.add({
                'video_name': os.path.basename(video_name),
                'predicted_label': label,
                'confidence': float(confidence),
                'timestamp': firestore.SERVER_TIMESTAMP
            })
        except Exception as e:
            st.error(f"Failed to save prediction: {e}")

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_seizure_model():
    model = load_model(MODEL_PATH, custom_objects={"LSTM": LSTM})
    return model

@st.cache_resource
def load_feature_extractor():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    return Sequential([base_model, GlobalAveragePooling2D()])

# -------------------- FRAME EXTRACTION --------------------
def extract_limited_frames(video_path, max_frames=MAX_FRAMES, step=2):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret or len(frames) >= max_frames:
            break
        if frame_count % step == 0:
            resized = cv2.resize(frame, IMAGE_SIZE)
            preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(resized)
            frames.append(preprocessed)
        frame_count += 1
    cap.release()
    return frames

# -------------------- SEQUENCES --------------------
def create_sequences(frames):
    return [frames[i:i+SEQUENCE_LENGTH] for i in range(0, len(frames) - SEQUENCE_LENGTH + 1, SEQUENCE_LENGTH)]

# -------------------- FEATURE EXTRACTION --------------------
def extract_features(sequences, extractor):
    features = []
    for seq in sequences:
        input_batch = np.array(seq).reshape((-1, *IMAGE_SIZE, 3))
        feats = extractor.predict(input_batch, verbose=0)
        features.append(feats.reshape(SEQUENCE_LENGTH, -1))
    return np.array(features)

# -------------------- PREDICTION --------------------
def predict(video_path, model, extractor, db):
    try:
        st.write(f"🧠 RAM usage: {psutil.virtual_memory().percent}%")
        frames = extract_limited_frames(video_path)
        if len(frames) < SEQUENCE_LENGTH:
            return "Video too short.", None, None

        sequences = create_sequences(frames)
        features = extract_features(sequences, extractor)

        preds = model.predict(features, verbose=0)
        avg = np.mean(preds, axis=0)
        label_map = {0: "No_Seizure", 1: "P", 2: "PG"}
        label = label_map[np.argmax(avg)]
        conf = float(np.max(avg))

        db.add_prediction(video_path, label, conf)
        return f"Prediction: {label} (Confidence: {conf:.2%})", label, conf

    except Exception as e:
        return f"Error during prediction: {e}", None, None

# -------------------- STREAMLIT UI --------------------
def main():
    st.set_page_config(page_title="Seizure Detection from Video", layout="wide")
    st.title("⚡ Seizure Detection from Video")
    st.markdown("Upload a video file to analyze for seizure activity.")

    model = load_seizure_model()
    extractor = load_feature_extractor()
    db = SeizureDatabase()

    uploaded = st.file_uploader("Choose video", type=["mp4", "avi", "mov"])
    if uploaded:
        path = os.path.join("temp", uploaded.name)
        os.makedirs("temp", exist_ok=True)
        with open(path, "wb") as f:
            f.write(uploaded.getbuffer())

        col1, col2 = st.columns(2)
        with col1: st.video(path)
        with col2:
            with st.spinner("Analyzing..."):
                result, label, conf = predict(path, model, extractor, db)
            if label:
                st.success(result)
                st.metric("Prediction", label)
                st.metric("Confidence", f"{conf:.2%}")
            else:
                st.error(result)

        os.remove(path)

    st.sidebar.title("🧾 History")
    if st.sidebar.button("Refresh"):
        df = db.collection.order_by('timestamp', direction=firestore.Query.DESCENDING).stream()
        records = pd.DataFrame([doc.to_dict() for doc in df])
        if not records.empty:
            st.sidebar.dataframe(records)

if __name__ == "__main__":
    main()
