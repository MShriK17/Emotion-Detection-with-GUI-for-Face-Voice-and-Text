import cv2
import numpy as np
import tensorflow as tf
from fastapi import UploadFile

model = tf.keras.models.load_model("backend/models/face_emotion_model.h5")

async def predict_emotion_from_image(file: UploadFile):
    contents = await file.read()
    img_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (48, 48))
    image = image.reshape(1, 48, 48, 1) / 255.0
    prediction = model.predict(image)
    classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    return classes[np.argmax(prediction)]
