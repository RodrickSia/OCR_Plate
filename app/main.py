from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.services.plate_detector import PlateDetector
from app.api.v1.routes import router as v1_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.plate_detector = PlateDetector()
    yield
    app.state.plate_detector.close()


app = FastAPI(title="OCR Plate API", version="1.0.0", lifespan=lifespan)

app.include_router(v1_router, prefix="/api/v1", tags=["detection"])
