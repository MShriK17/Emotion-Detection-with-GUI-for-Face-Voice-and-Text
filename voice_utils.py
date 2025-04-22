import librosa
import numpy as np
import pickle
from fastapi import UploadFile

model = pickle.load(open("backend/models/voice_emotion_model.pkl", "rb"))

async def predict_emotion_from_audio(file: UploadFile):
    contents = await file.read()
    with open("temp.wav", "wb") as f:
        f.write(contents)
    y, sr = librosa.load("temp.wav", duration=3, offset=0.5)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    prediction = model.predict([mfccs])
    return prediction[0]
