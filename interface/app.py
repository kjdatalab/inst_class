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
                animation_style = f"animation: rotate {rotation_speed}s infinite alternate ease-in-out;" if rotate else ""
                st.markdown(
                    f"""
                    <style>
                        @keyframes rotate {{
                            0% {{ transform: rotateY(0deg); }}
                            25% {{ transform: rotateY(10deg); }}
                            75% {{ transform: rotateY(-10deg); }}
                            100% {{ transform: rotateY(0deg); }}
                        }}
                        .flip-card {{
                            background-color: transparent;
                            width: 238px;
                            height: 438px;
                            perspective: 1000px;
                            margin: 25px auto;
                            border-radius: 10px;
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
                            border-radius: 10px;
                            border: 3px solid #31333f;
                            display: flex;
                            justify-content: center;
                            align-items: center;
                        }}
                        .flip-card-front {{
                            background-color: #51BE85;
                        }}
                        .flip-card-back {{
                            background-color: #67d2e4;
                            transform: rotateY(180deg);
                        }}
                        .flip-card img {{
                            width: 100%;
                            height: 100%;
                            object-fit: cover;
                            border-radius: 10px;
                        }}
                        .instrument-info {{
                            margin-top: 10px;
                            font-size: 24px;
                            font-weight: bold;
                            text-align: center;
                            line-height: 1.5;
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
                    <div class="instrument-info" style="font-size: 30px; font-weight: bold; text-align: center; line-height: 1.5; margin-bottom: 30px;">
                        {instrument_name}<br>
                    <span style="font-size: 24px; font-weight: normal;">(prob: {probability:.1f}%)</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# Main Streamlit app
def main():
    # Custom CSS to change the font to Lexend
    custom_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lexend:wght@400&display=swap');

    html, body, [class*="css"] {
        font-family: 'Lexend', sans-serif;
    }

    .stButton > button {
        font-size: 1.5em; /* Adjust the font size for all buttons */
        padding: 10px 20px; /* Adjust padding for better visual alignment */
    }

    .stButton.animate-play-pause > button {
        margin-top: 10px; /* Add margin-top specifically to the Animate Play/Pause button */
    }
    </style>
    """
    # Apply the custom CSS
    st.markdown(custom_css, unsafe_allow_html=True)

    st.markdown(
        """
        <style>
        .stApp { background-color: #FEFFEF; }
        .stFileUploader > label { border-radius: 10px; padding: 10px; }
        .stButton > button {
            display: block;
            margin: 30px auto;
            background-color: #ffe433;
            font-size: 1.5em !important; /* Override font size for buttons */
            padding: 15px 30px;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
    """
    <h1 style="text-align: center;">Chromatic Maestro</h1>
    """,
    unsafe_allow_html=True
    )

    st.markdown(
    """
    <style>
    .custom-text {
        text-align: center;
        font-size: 1.5em;
        margin-bottom: 1em;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )

    st.markdown('<div class="custom-text">Envoke your sense of musical virtuosity and mastery</div>', unsafe_allow_html=True)

    instrument_data = load_instruments("instruments.json")

    # Initialize session state variables
    if 'is_playing' not in st.session_state:
        st.session_state.is_playing = False
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    if 'prediction' not in st.session_state:
        st.session_state.prediction = []
    if 'tempo' not in st.session_state:
        st.session_state.tempo = None
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None

    uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

    # Clear previous prediction if a new file is uploaded
    if uploaded_file is not None and uploaded_file != st.session_state.current_file:
        st.session_state.prediction_made = False
        st.session_state.prediction = []
        st.session_state.tempo = None
        st.session_state.current_file = uploaded_file

    if uploaded_file is not None:
        st.audio(uploaded_file)

        if st.button("Prediction"):
            st.session_state.tempo = detect_tempo(uploaded_file)
            try:
                files = {"file": uploaded_file.getvalue()}
                response = requests.post(API_URL, files=files)
                if response.status_code == 200:
                    st.session_state.prediction = response.json().get("predictions", [])
                    st.session_state.prediction_made = True
                else:
                    st.error("Error fetching predictions from the API")
            except Exception as e:
                st.error(f"Error with the prediction request: {e}")

        if st.session_state.prediction_made:
            display_predictions(st.session_state.prediction, instrument_data, st.session_state.tempo, st.session_state.is_playing)

            # Add specific class to the Play/Pause button to target its style
            if st.button("Animate Play/Pause", key="play_pause"):
                st.session_state.is_playing = not st.session_state.is_playing  # Toggle play/pause state
                st.experimental_rerun()


if __name__ == "__main__":
    main()
