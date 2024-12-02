import streamlit as st
import requests
from io import BytesIO

st.title("Piano Detector")
st.write("Upload an audio file to detect if it contains piano")

# File uploader
audio_file = st.file_uploader("Choose an audio file", type=['wav'])

if audio_file is not None:
    # Display audio file
    st.audio(audio_file)

    # Make prediction when button is clicked
    if st.button("Detect Piano"):
        # Prepare the file for the API request
        files = {"file": ("audio.wav", audio_file, "audio/wav")}

        # Make API request
        response = requests.post("http://api:8000/predict", files=files)
        result = response.json()

        # Display results
        st.write("### Results")
        st.write(f"Contains piano: {'Yes' if result['contains_piano'] else 'No'}")
        st.write(f"Confidence: {result['confidence']:.1%}")
