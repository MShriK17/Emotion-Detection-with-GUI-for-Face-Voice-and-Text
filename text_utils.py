from transformers import pipeline
classifier = pipeline("text-classification", model="nateraw/bert-base-uncased-emotion")

def predict_emotion_from_text(text: str):
    result = classifier(text)
    return result[0]['label']
