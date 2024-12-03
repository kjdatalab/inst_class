import requests
import streamlit as st
import base64
import json
import librosa
import numpy as np
from pathlib import Path

API_URL = "http://localhost:8000/predict"

# Convert image file to base64 for display
def convert_image_to_base64(file_path):
    image_path = Path(file_path)
    if image_path.exists():
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    return ""

# Load instrument data (for images and labels)
def load_instruments(json_file):
    with open(json_file, "r") as f:
        return json.load(f)

# Detect the tempo from the .wav file using Librosa
def detect_tempo(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    if isinstance(tempo, (np.ndarray, list)):
        tempo = tempo[0]
    return tempo

# Display predictions
def display_predictions(predicted_instruments, instrument_data, tempo, rotate):
    if not predicted_instruments:
        st.write("No instruments predicted.")
        return

    rotation_speed = np.clip(tempo / 60, 0.1, 2)

    cols = st.columns(len(predicted_instruments))
    for col, instrument_info in zip(cols, predicted_instruments):
        instrument_name = instrument_info.get("instrument")
        probability = instrument_info.get("probability", 0)
        # Find instrument details in the JSON data
        instrument_data_item = next((item for item in instrument_data if item["instrument"] == instrument_name), None)
        if instrument_data_item:
            file_path = instrument_data_item["file_path"]
            file_path_v2 = instrument_data_item["file_path_v2"]
            base64_front = convert_image_to_base64(file_path)
            base64_back = convert_image_to_base64(file_path_v2)

            with col:
                animation_style = f"animation: rotate {rotation_speed}s infinite alternate;" if rotate else ""
                st.markdown(
                    f"""
                    <style>
                        @keyframes rotate {{
                            0% {{ transform: rotate(10deg); }}
                            50% {{ transform: rotate(-15deg); }}
                            100% {{ transform: rotate(10deg); }}
                        }}
                        .flip-card {{
                            background-color: transparent;
                            width: 238px;
                            height: 238px;
                            perspective: 1000px;
                            margin: 25px auto;
                        }}
                        .flip-card-inner {{
                            position: relative;
                            width: 100%;
                            height: 100%;
                            text-align: center;
                            transform-style: preserve-3d;
                            transition: transform 0.6s;
                        }}
                        .flip-card:hover .flip-card-inner {{
                            transform: rotateY(180deg);
                        }}
                        .flip-card-front, .flip-card-back {{
                            position: absolute;
                            width: 100%;
                            height: 100%;
                            backface-visibility: hidden;
                            border-radius: 50%;
                            display: flex;
                            justify-content: center;
                            align-items: center;
                        }}
                        .flip-card-front {{
                            background-color: #51BE85;
                        }}
                        .flip-card-back {{
                            background-color: #FFE433;
                            transform: rotateY(180deg);
                        }}
                        .flip-card img {{
                            width: 100%;
                            height: 100%;
                            object-fit: cover;
                            border-radius: 50%;
                        }}
                    </style>
                    <div class="flip-card">
                        <div class="flip-card-inner" style="{animation_style}">
                            <div class="flip-card-front">
                                <img src="data:image/png;base64,{base64_front}" alt="Instrument Image Front" />
                            </div>
                            <div class="flip-card-back">
                                <img src="data:image/png;base64,{base64_back}" alt="Instrument Image Back" />
                            </div>
                        </div>
                    </div>
                    <div style="margin-top: 10px; font-size: 28px; font-weight: bold; text-align: center;">
                        {instrument_name} ({probability:.1f}%)
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# Main Streamlit app
def main():
    st.markdown(
        """
        <style>
        .stApp { background-color: #FEFFEF; }
        .stFileUploader > label { background-color: #51BE85; color: white; border-radius: 10px; padding: 10px; }
        .stButton > button { display: block; margin: 50px auto; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Classical Music Instrument Identifier")
    st.markdown("This app displays predicted instruments with their images and labels.")

    instrument_data = load_instruments("instruments.json")
    uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])
    rotate_images = False
    prediction = []

    if uploaded_file is not None:
        tempo = detect_tempo(uploaded_file)
        st.audio(uploaded_file)

        try:
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(API_URL, files=files)
            if response.status_code == 200:
                prediction = response.json().get("predictions", [])
            else:
                st.error("Error fetching predictions from the API")
        except Exception as e:
            st.error(f"Error with the prediction request: {e}")

        if tempo:
            st.write(f"Detected tempo: {tempo:.2f} BPM")
        else:
            st.write("Tempo detection failed.")


        display_predictions(prediction, instrument_data, tempo, rotate_images)

        # if st.button("Animate Play/Pause"):
        #     rotate_images = not rotate_images
        if st.button("Animate Play/Pause"):
                st.session_state.is_playing = not st.session_state.is_playing  # Toggle play/pause state

if __name__ == "__main__":
    main()
