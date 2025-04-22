from fastapi import APIRouter, UploadFile, File
from backend.utils.voice_utils import predict_emotion_from_audio

router = APIRouter()

@router.post("/detect")
async def detect_voice_emotion(file: UploadFile = File(...)):
    result = await predict_emotion_from_audio(file)
    return {"emotion": result}
