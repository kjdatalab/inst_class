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

# Print model summary for debugging
model.summary()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    This endpoint handles audio file uploads and returns per-second instrument predictions.
    """
    unique_instruments = [1, 41, 42, 43, 61, 71, 72]

    mlb = MultiLabelBinarizer(classes=unique_instruments)
    mlb.fit([unique_instruments])  # Fit the binarizer

    try:
        audio_data = await file.read()
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as f:
            f.write(audio_data)

        # Predict per-second instruments
        predictions_per_second = predict_per_second_instruments(temp_file, model, mlb)
        os.remove(temp_file)

        # Map instrument codes to names
        INSTRUMENT_MAP = {
            1: 'Piano',
            41: 'Violin',
            42: 'Viola',
            43: 'Cello',
            61: 'Horn',
            71: 'Bassoon',
            72: 'Clarinet',
        }

        # Convert predictions to instrument names
        predictions_with_names = [
            [INSTRUMENT_MAP.get(i, "Unknown") for i in prediction]
            for prediction in predictions_per_second
        ]

        return JSONResponse(content={'predictions': predictions_with_names})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """
    Health check endpoint for debugging.
    """
    return {"message": "Instrument Classifier API is running"}

def predict_per_second_instruments(file_path, model, mlb):
    """
    Processes the audio file with a sliding window and makes predictions for each segment.
    """
    # Load audio file
    y, sr = librosa.load(file_path, sr=None)

    # Sliding window parameters
    window_size = 5  # 5 seconds
    step_size = 1    # 1 second
    num_samples_per_window = int(window_size * sr)
    step_samples = int(step_size * sr)

    # Generate predictions for each window
    predictions = []
    for start in range(0, len(y) - num_samples_per_window + 1, step_samples):
        window = y[start:start + num_samples_per_window]
        mel_spec = extract_mel_spectrogram(window, sr)

        # Ensure shape compatibility
        mel_spec = np.expand_dims(mel_spec, axis=-1)  # Add channel dimension (128, 431, 1)
        mel_spec = np.expand_dims(mel_spec, axis=0)   # Add batch dimension (1, 128, 431, 1)

        # Predict and decode
        prediction = model.predict(mel_spec)
        predicted_labels = mlb.inverse_transform((prediction > 0.5).astype(int))
        predictions.append(predicted_labels[0])

    return predictions

def extract_mel_spectrogram(y, sr):
    """
    Extracts Mel-spectrogram from raw audio samples.
    """
    # Compute the Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

    # Convert to decibels
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Pad or truncate to ensure the correct number of time steps
    target_time_steps = 431  # Expected by the model
    if mel_spec_db.shape[1] < target_time_steps:
        # Pad with zeros if fewer time steps
        pad_width = target_time_steps - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        # Truncate if too many time steps
        mel_spec_db = mel_spec_db[:, :target_time_steps]

    return mel_spec_db
