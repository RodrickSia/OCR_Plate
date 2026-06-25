import asyncio
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

from app.core.config import (
    YOLO_MODEL_PATH,
    VINTERN_MODEL_NAME,
    YOLO_CONFIDENCE,
    YOLO_IMGSZ,
    OCR_MAX_NEW_TOKENS,
    OCR_NUM_BEAMS,
    OCR_PROMPT,
)


# ── Workaround for Vintern config bug ──────────────────────────
import transformers.configuration_utils as _cfg_utils

_orig_info = _cfg_utils.logger.info


def _safe_info(msg, *args, **kwargs):
    try:
        _orig_info(msg, *args, **kwargs)
    except (KeyError, AttributeError):
        pass


_cfg_utils.logger.info = _safe_info
# ── End workaround ─────────────────────────────────────────────


class _VinternEngine(nn.Module):
    """GPU-preferring OCR engine using Vintern VLM."""

    def __init__(self, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype
        from transformers import AutoModel, AutoTokenizer

        self.model = AutoModel.from_pretrained(
            VINTERN_MODEL_NAME,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            VINTERN_MODEL_NAME,
            trust_remote_code=True,
            use_fast=False,
        )

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

    def recognize(self, plate_img: np.ndarray) -> str:
        try:
            plate_img = cv2.resize(plate_img, (448, 448))
            pixel_values = self._prepare_image(plate_img).to(self.dtype).to(self.device)

            generation_config = dict(
                max_new_tokens=OCR_MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=OCR_NUM_BEAMS,
            )

            with torch.no_grad():
                text = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    OCR_PROMPT,
                    generation_config,
                )
            result = text.strip()
            return result
        except Exception as e:
            print("Vintern OCR ERROR:", e)
            return ""


class _EasyOCREngine(nn.Module):
    """CPU-friendly OCR engine using EasyOCR."""

    def __init__(self) -> None:
        super().__init__()
        import easyocr

        self.reader = easyocr.Reader(["vi"], gpu=False)

    def recognize(self, plate_img: np.ndarray) -> str:
        try:
            result = self.reader.readtext(plate_img)
            if result:
                return result[0][1].strip().replace(".", "").replace(" ", "")
            return ""
        except Exception as e:
            print("EasyOCR ERROR:", e)
            return ""


class _DummyEngine(nn.Module):
    def recognize(self, plate_img: np.ndarray) -> str:
        return ""


class PlateDetector:

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.yolo = YOLO(YOLO_MODEL_PATH)

        self.ocr_engine = self._init_ocr_engine()

        self._executor = ThreadPoolExecutor(max_workers=1)

    def _init_ocr_engine(self) -> nn.Module:
        if torch.cuda.is_available():
            try:
                print("[OCR] Trying Vintern (GPU) ...")
                dtype = torch.float16
                return _VinternEngine(self.device, dtype)
            except (ImportError, RuntimeError) as e:
                print(f"[OCR] Vintern GPU failed ({e}), falling back ...")
        else:
            print("[OCR] No CUDA available, skipping Vintern ...")

        try:
            print("[OCR] Loading EasyOCR (CPU) ...")
            return _EasyOCREngine()
        except ImportError as e:
            print(f"[OCR] EasyOCR also unavailable ({e}), using dummy engine")
            return _DummyEngine()

    def close(self):
        self._executor.shutdown(wait=True)

    # ── OCR (runs in thread pool) ──

    def _recognize_plate_sync(self, plate_img: np.ndarray) -> str:
        return self.ocr_engine.recognize(plate_img)

    async def _recognize_plate(self, plate_img: np.ndarray) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._recognize_plate_sync, plate_img)

    # ── public API ──

    async def detect_plates(self, image: np.ndarray) -> list[str]:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            self._executor,
            lambda: self.yolo.predict(image, conf=YOLO_CONFIDENCE, imgsz=YOLO_IMGSZ, verbose=False),
        )

        plates: list[str] = []
        h, w = image.shape[:2]
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                crop = image[y1:y2, x1:x2]

                if crop.shape[0] > 10 and crop.shape[1] > 10:
                    text = await self._recognize_plate(crop)
                    if text:
                        plates.append(text)

        return plates
