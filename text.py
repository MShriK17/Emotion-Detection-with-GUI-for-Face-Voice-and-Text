from fastapi import APIRouter
from pydantic import BaseModel
from backend.utils.text_utils import predict_emotion_from_text

router = APIRouter()

class TextInput(BaseModel):
    text: str

@router.post("/detect")
async def detect_text_emotion(input: TextInput):
    result = predict_emotion_from_text(input.text)
    return {"emotion": result}
