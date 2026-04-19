import asyncio

import cv2
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from transformers import AutoModel, AutoTokenizer

from app.core.config import (
    YOLO_MODEL_PATH,
    VINTERN_MODEL_NAME,
    YOLO_CONFIDENCE,
    YOLO_IMGSZ,
    OCR_MAX_NEW_TOKENS,
    OCR_NUM_BEAMS,
    OCR_PROMPT,
)


class PlateDetector:

    def __init__(self) -> None:
        self.yolo = YOLO(YOLO_MODEL_PATH)

        self.vintern = AutoModel.from_pretrained(
            VINTERN_MODEL_NAME,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).eval().cuda()

        self.tokenizer = AutoTokenizer.from_pretrained(
            VINTERN_MODEL_NAME,
            trust_remote_code=True,
            use_fast=False,
        )

        self._executor = ThreadPoolExecutor(max_workers=1)

    # ── image pre-processing ──

    @staticmethod
    def _build_transform(input_size: int = 448) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _prepare_image(self, img: np.ndarray) -> torch.Tensor:
        image = Image.fromarray(img).convert("RGB")
        transform = self._build_transform()
        return transform(image).unsqueeze(0)

    # ── OCR (runs in thread pool) ──

    def _recognize_plate_sync(self, plate_img: np.ndarray) -> str:
        try:
            plate_img = cv2.resize(plate_img, (448, 448))
            pixel_values = self._prepare_image(plate_img).to(torch.float16).cuda()

            generation_config = dict(
                max_new_tokens=OCR_MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=OCR_NUM_BEAMS,
            )

            with torch.no_grad():
                text = self.vintern.chat(
                    self.tokenizer,
                    pixel_values,
                    OCR_PROMPT,
                    generation_config,
                )
            result = text.strip()
            print("OCR:", result)
            return result
        except Exception as e:
            print("OCR ERROR:", e)
            return ""

    async def _recognize_plate(self, plate_img: np.ndarray) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._recognize_plate_sync, plate_img)

    # ── public API ──

    async def detect_plates(self, image: np.ndarray) -> list[str]:
        with torch.no_grad():
            results = self.yolo.predict(image, conf=YOLO_CONFIDENCE, imgsz=YOLO_IMGSZ, verbose=False)

        plates: list[str] = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = image[y1:y2, x1:x2]

                if crop.size > 0:
                    text = await self._recognize_plate(crop)
                    if text:
                        plates.append(text)

        return plates
