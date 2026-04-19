from fastapi import APIRouter

from app.api.v1.endpoints.detection import detect_image
from app.schemas.detection import DetectionResponse

router = APIRouter()

router.post("/detect-image", response_model=DetectionResponse)(detect_image)
