import cv2
import numpy as np
from fastapi import Request, HTTPException

from app.schemas.detection import SUPPORTED_CONTENT_TYPES, DetectionResponse
from app.services.plate_detector import PlateDetector


async def detect_image(request: Request):
    content_type = request.headers.get("content-type", "")
    if content_type not in SUPPORTED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{content_type}'. Expected one of: {SUPPORTED_CONTENT_TYPES}",
        )

    detector: PlateDetector = request.app.state.plate_detector

    contents = await request.body()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    plates = await detector.detect_plates(img)

    return DetectionResponse(plates=plates)
