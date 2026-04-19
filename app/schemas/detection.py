from pydantic import BaseModel


SUPPORTED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


class DetectionResponse(BaseModel):
    plates: list[str]
