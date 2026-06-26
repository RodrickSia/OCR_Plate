import cv2
import numpy as np
from fastapi import Request, HTTPException, UploadFile, File
from typing import Optional

from app.schemas.detection import SUPPORTED_CONTENT_TYPES, DetectionResponse
from app.services.plate_detector import PlateDetector

MAX_BODY_SIZE = 16 * 1024 * 1024  # 16 MB


async def detect_image(
    request: Request,
    image: Optional[UploadFile] = File(None),
    image_bytes: Optional[bytes] = File(None),
):
    detector: PlateDetector = request.app.state.plate_detector

    contents = b""

    if image is not None:
        contents = await image.read()
    elif image_bytes is not None:
        contents = image_bytes
    else:
        contents = await request.body()

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty image data")

    if len(contents) > MAX_BODY_SIZE:
        raise HTTPException(status_code=413, detail="Request body too large")

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    plates = await detector.detect_plates(img)

    return DetectionResponse(plates=plates)
