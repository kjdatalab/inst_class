from fastapi import FastAPI, UploadFile, File, HTTPException
from model_loader import ModelLoader
import tensorflow as tf
import librosa
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import os
from fastapi.responses import JSONResponse

app = FastAPI()

# Load the model when the API starts
model_loader = ModelLoader('./model/CNN_model1_sliced2.keras')
model = model_loader.load_model()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    This endpoint handles the actual piano detection. We keep it because it's our main
    functionality - it receives audio files and returns predictions using our model.
    """

    unique_instruments = [1, 41, 42, 43, 61, 71, 72]

    mlb = MultiLabelBinarizer(classes=unique_instruments)
    mlb.fit([unique_instruments])  # Fit the binarizer

    try:
        #new_file_path = 'raw_data/test_data/3333.wav'
        audio_data = await file.read()
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as f:
            f.write(audio_data)
        predicted_instruments = predict_instruments(temp_file, model, mlb)
        print(f"Predicted instruments for '{temp_file}': {predicted_instruments}")

        INSTRUMENT_MAP = {
        1: 'Piano',
        41: 'Violin',
        42: 'Viola',
        43: 'Cello',
        61: 'Horn',
        71: 'Bassoon',
        72: 'Clarinet',
        }

        instrument_names = [INSTRUMENT_MAP.get(i, "Unknown") for i in predicted_instruments[0]]
        print(f"Predicted instruments for '{temp_file}': {instrument_names}")

        os.remove(temp_file)

        return JSONResponse(content={'predictions':instrument_names})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/")
async def root():
    """
    We keep this endpoint as a health check - it lets users verify the API is running
    and helps with debugging. Think of it like a doorbell - it lets you know the
    system is powered on and responsive.
    """
    print(model)
    return {"message": "Instrument Classifier API is running"}

# Function to predict instruments from a new .wav file
def predict_instruments(file_path, model, mlb):
    # Extract Mel-spectrogram from the input file
    mel_spec = extract_mel_spectrogram(file_path)
    # Pad or reshape the spectrogram to match the model input shape
    mel_spec = np.expand_dims(mel_spec, axis=-1)  # Add channel dimension
    mel_spec = tf.keras.preprocessing.sequence.pad_sequences([mel_spec],
                                                             padding="post",
                                                             dtype="float32",
                                                             value=0)
    # Predict the instruments
    prediction = model.predict(mel_spec)
    # Get the instrument labels
    predicted_instruments = mlb.inverse_transform((prediction > 0.5).astype(int))  # Threshold at 0.5 for multi-label
    return predicted_instruments


# Function to extract Mel-spectrogram from the audio file
def extract_mel_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=None, duration=5.0)  # Limit to 5 seconds to keep data consistent
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to decibels for better visualization
    return mel_spec_db
