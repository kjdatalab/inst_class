import streamlit as st
import requests
import json
import base64
from pathlib import Path
# FastAPI server URL (assumes FastAPI is running locally on port 8000)
API_URL = "http://localhost:8000/predict"
# Convert image file to base64 for display
def convert_image_to_base64(file_path):
    image_path = Path(file_path)
    if image_path.exists():
        with open(image_path, "rb") as image_file:
            base64_str = base64.b64encode(image_file.read()).decode("utf-8")
        return base64_str
    else:
        return ""
# Load instrument data (for images and labels)
def load_instruments(json_file):
    with open(json_file, "r") as f:
        return json.load(f)
# Display instrument predictions
def display_predictions(predicted_instruments, instrument_data):
    # Define background colors for the circles
    colors = ["#51BE85", "#67D2E4", "#FFE433"]
    # Create columns dynamically based on the number of predicted instruments
    cols = st.columns(len(predicted_instruments))
    for col, instrument in zip(cols, predicted_instruments):
        # Find instrument details in the JSON data
        instrument_info = next((item for item in instrument_data if item["instrument"] == instrument), None)
        if instrument_info:
            file_path = instrument_info["file_path"]
            with col:
                st.markdown(
                    f"""
                    <div style="width: 235px; height: 235px; border-radius: 50%; overflow: hidden; display: flex; justify-content: center; align-items: center; margin: 25px auto; background-color: {colors[0]}">
                        <img src="data:image/png;base64,{convert_image_to_base64(file_path)}" style="width: 100%; height: 100%; object-fit: cover; border-radius: 50%;" />
                    </div>
                    <div style="margin-top: 10px; font-size: 28px; font-weight: bold; text-align: center;">
                        {instrument}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
# Main Streamlit app
def main():
    # Change background color of the whole page
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #FEFFEF;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Title and description
    st.title("Classical Music Instrument Identifier")
    st.markdown("This app displays predicted instruments with their images and labels.")
    # Load instrument data
    instrument_data = load_instruments("instruments.json")
    # Upload audio file
    uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])
    # Check if an audio file was uploaded
    if uploaded_file is not None:
        # Save the uploaded file in a temporary location for playback
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.getvalue())
        # Play the uploaded audio file for the user
        st.audio(uploaded_file)
        # Send the audio file to FastAPI for prediction
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(API_URL, files=files)
        # Process response and display predictions
        if response.status_code == 200:
            prediction = response.json().get("predictions", [])
            display_predictions(prediction, instrument_data)
        else:
            st.error("Error fetching predictions from the API")
if __name__ == "__main__":
    main()
