from fastapi import APIRouter, UploadFile, File
from backend.utils.face_utils import predict_emotion_from_image

router = APIRouter()

@router.post("/detect")
async def detect_face_emotion(file: UploadFile = File(...)):
    result = await predict_emotion_from_image(file)
    return {"emotion": result}
