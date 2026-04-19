from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

YOLO_MODEL_PATH = str(BASE_DIR / "models" / "license-plate.pt")
VINTERN_MODEL_NAME = "5CD-AI/Vintern-1B-v3_5"

YOLO_CONFIDENCE = 0.5
YOLO_IMGSZ = 640

OCR_MAX_NEW_TOKENS = 32
OCR_NUM_BEAMS = 2

OCR_PROMPT = (
    "<image>\n"
    "Đọc biển số xe trong ảnh này. Chỉ viết liền các chữ cái và chữ số, "
    "không chứa ký tự đặc biệt. Bỏ các dấu chấm trong hình. "
    "Trong trường hợp bị lóa sáng hãy cố nhận ra đó là kí tự gì cho chính xác nhất."
)
