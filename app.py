from fastapi import FastAPI
from backend.api import face, voice, text

app = FastAPI(title="Emotion Detection API")

app.include_router(face.router, prefix="/face")
app.include_router(voice.router, prefix="/voice")
app.include_router(text.router, prefix="/text")

@app.get("/")
def read_root():
    return {"message": "Emotion Detection API is up"}
