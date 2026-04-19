import cv2
import numpy as np
import base64
import asyncio
import json
import torch
import sqlite3
import csv
import io
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta

VN_TZ = timezone(timedelta(hours=7))

from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from ultralytics import YOLO
from transformers import AutoModel, AutoTokenizer
from PIL import Image
from torchvision import transforms

app = FastAPI()

# ================= DATABASE =================

def init_db():
    conn = sqlite3.connect("parking.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS parking_log (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            plate     TEXT NOT NULL,
            event     TEXT NOT NULL CHECK(event IN ('IN','OUT')),
            timestamp TEXT NOT NULL
        )
    """)
    # Bảng lưu xe đang đỗ trong bãi
    c.execute("""
        CREATE TABLE IF NOT EXISTS parking_active (
            plate      TEXT PRIMARY KEY,
            entry_time TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

def db_conn():
    conn = sqlite3.connect("parking.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

# ================= LOAD MODELS =================

yolo_model = YOLO("license-plate.pt")

model_name = "5CD-AI/Vintern-1B-v3_5"

model_vintern = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    trust_remote_code=True
).eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_fast=False
)

# Thread pool để chạy OCR (blocking) mà không block event loop
ocr_executor = ThreadPoolExecutor(max_workers=1)

# ================= PREPROCESS =================

def build_transform(input_size=448):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def load_image(img):
    image = Image.fromarray(img).convert('RGB')
    transform = build_transform()
    return transform(image).unsqueeze(0)

# ================= OCR =================

def recognize_plate(plate_img):
    """Chạy OCR đồng bộ — được gọi trong thread pool."""
    try:
        plate_img = cv2.resize(plate_img, (448, 448))
        pixel_values = load_image(plate_img).to(torch.float16).cuda()

        generation_config = dict(
            max_new_tokens=32,
            do_sample=False,
            num_beams=2
        )
        question = "<image>\nĐọc biển số xe trong ảnh này. Chỉ viết liền các chữ cái và chữ số, không chứa ký tự đặc biệt. Bỏ các dấu chấm trong hình. Trong trường hợp bị lóa sáng hãy cố nhận ra đó là kí tự gì cho chính xác nhất."

        with torch.no_grad():
            text = model_vintern.chat(
                tokenizer,
                pixel_values,
                question,
                generation_config
            )

        result = text.strip()
        print("OCR:", result)
        return result

    except Exception as e:
        print("OCR ERROR:", e)
        return ""

# ================= PARKING LOGIC =================

# Số lần nhận diện liên tiếp cần thiết trước khi xác nhận sự kiện
CONFIRM_FRAMES = 5

def record_parking_event(plate: str, event: str):
    """Ghi sự kiện vào/ra vào DB, trả về dict để push qua WS."""
    now = datetime.now(VN_TZ).strftime("%Y-%m-%d %H:%M:%S")
    conn = db_conn()
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO parking_log (plate, event, timestamp) VALUES (?,?,?)",
            (plate, event, now)
        )
        if event == "IN":
            c.execute(
                "INSERT OR REPLACE INTO parking_active (plate, entry_time) VALUES (?,?)",
                (plate, now)
            )
        else:
            c.execute("DELETE FROM parking_active WHERE plate=?", (plate,))
        conn.commit()
    finally:
        conn.close()
    return {"plate": plate, "event": event, "timestamp": now}

def is_plate_inside(plate: str) -> bool:
    conn = db_conn()
    row = conn.execute(
        "SELECT 1 FROM parking_active WHERE plate=?", (plate,)
    ).fetchone()
    conn.close()
    return row is not None

# ================= CACHE HELPER =================

def bbox_to_key(x1, y1, x2, y2, bucket=20):
    return f"{round(x1/bucket)}_{round(y1/bucket)}_{round(x2/bucket)}_{round(y2/bucket)}"

# ================= ROUTES =================

@app.get("/")
async def get_index():
    return FileResponse("index.html")

# ──────────────────────── PARKING API ────────────────────────

@app.get("/parking/active")
async def get_active():
    """Xe đang có mặt trong bãi."""
    conn = db_conn()
    rows = conn.execute(
        "SELECT plate, entry_time FROM parking_active ORDER BY entry_time DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.get("/parking/log")
async def get_log(limit: int = 100):
    """Lịch sử vào/ra gần nhất."""
    conn = db_conn()
    rows = conn.execute(
        "SELECT id, plate, event, timestamp FROM parking_log ORDER BY id DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.get("/parking/stats")
async def get_stats():
    """Thống kê tổng hợp."""
    conn = db_conn()
    total_in  = conn.execute("SELECT COUNT(*) FROM parking_log WHERE event='IN'").fetchone()[0]
    total_out = conn.execute("SELECT COUNT(*) FROM parking_log WHERE event='OUT'").fetchone()[0]
    active    = conn.execute("SELECT COUNT(*) FROM parking_active").fetchone()[0]
    conn.close()
    return {"total_in": total_in, "total_out": total_out, "currently_parked": active}

@app.get("/parking/export")
async def export_csv():
    """Xuất toàn bộ lịch sử ra CSV."""
    conn = db_conn()
    rows = conn.execute(
        "SELECT id, plate, event, timestamp FROM parking_log ORDER BY id"
    ).fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Biển số", "Sự kiện", "Thời gian"])
    for r in rows:
        writer.writerow([r["id"], r["plate"], r["event"], r["timestamp"]])

    output.seek(0)
    filename = f"parking_log_{datetime.now(VN_TZ).strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.post("/parking/manual")
async def manual_entry(body: dict):
    """
    Cho phép nhân viên ghi tay sự kiện vào/ra.
    Body: { "plate": "51A12345", "event": "IN" | "OUT" }
    """
    plate = body.get("plate", "").strip().upper()
    event = body.get("event", "").strip().upper()
    if not plate or event not in ("IN", "OUT"):
        from fastapi import HTTPException
        raise HTTPException(400, "plate và event (IN/OUT) là bắt buộc")
    rec = record_parking_event(plate, event)
    return rec

# ================= IMAGE API =================

@app.post("/detect-image")
async def detect_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    loop = asyncio.get_event_loop()
    parking_events = []

    with torch.no_grad():
        results = yolo_model.predict(img, conf=0.5)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]

            if crop.size > 0:
                text = await loop.run_in_executor(ocr_executor, recognize_plate, crop)

                if text:
                    # Ảnh tĩnh: nếu xe chưa có trong bãi → ghi IN
                    event = "OUT" if is_plate_inside(text) else "IN"
                    rec = record_parking_event(text, event)
                    parking_events.append(rec)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return {
        "image": f"data:image/jpeg;base64,{img_base64}",
        "events": parking_events
    }

# ================= REALTIME WORKER =================

async def inference_worker(websocket: WebSocket, queue: asyncio.Queue):
    """
    Worker riêng cho từng kết nối WebSocket.
    Thêm logic xác nhận vào/ra:
      - Theo dõi số frame liên tiếp nhìn thấy biển số.
      - Chỉ ghi sự kiện khi đủ CONFIRM_FRAMES → tránh false-positive.
      - Xe rời khung hình N frame → ghi OUT (nếu đang ở IN).
    """
    loop = asyncio.get_event_loop()

    plate_cache: dict[str, str]   = {}
    plate_hit_count: dict[str, int] = {}
    REOCR_INTERVAL = 20

    # Đếm frame liên tiếp thấy/không thấy biển số
    plate_seen_count:   dict[str, int] = {}   # frame liên tiếp thấy
    plate_unseen_count: dict[str, int] = {}   # frame liên tiếp không thấy
    plate_confirmed:    set[str]       = set() # đã xác nhận (đã ghi IN/OUT)

    CONFIRM_IN    = CONFIRM_FRAMES       # frame thấy liên tiếp → ghi IN
    CONFIRM_OUT   = CONFIRM_FRAMES * 2   # frame mất liên tiếp → ghi OUT

    try:
        while True:
            frame = await queue.get()
            frame_small = cv2.resize(frame, (960, 540))

            with torch.no_grad():
                results = yolo_model.predict(
                    frame_small, conf=0.5, imgsz=640, verbose=False
                )

            seen_keys: set[str] = set()
            seen_plates: dict[str, str] = {}   # key → plate text trong frame này

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    plate_crop = frame_small[y1:y2, x1:x2]
                    plate_text = ""

                    if plate_crop.shape[0] > 20 and plate_crop.shape[1] > 40:
                        key = bbox_to_key(x1, y1, x2, y2)
                        seen_keys.add(key)
                        plate_hit_count[key] = plate_hit_count.get(key, 0) + 1

                        if (key not in plate_cache or
                                plate_hit_count[key] % REOCR_INTERVAL == 0):
                            plate_text = await loop.run_in_executor(
                                ocr_executor, recognize_plate, plate_crop
                            )
                            plate_cache[key] = plate_text
                        else:
                            plate_text = plate_cache[key]

                        if plate_text:
                            seen_plates[key] = plate_text
                            # Đếm frame liên tiếp thấy
                            plate_seen_count[plate_text]   = plate_seen_count.get(plate_text, 0) + 1
                            plate_unseen_count[plate_text] = 0   # reset bộ đếm mất

                    cv2.rectangle(frame_small, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame_small, plate_text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                    )

            # ── Xử lý sự kiện VÀO ──
            parking_events = []
            for key, plate_text in seen_plates.items():
                count = plate_seen_count.get(plate_text, 0)
                if count == CONFIRM_IN and plate_text not in plate_confirmed:
                    plate_confirmed.add(plate_text)
                    event = "OUT" if is_plate_inside(plate_text) else "IN"
                    rec = record_parking_event(plate_text, event)
                    parking_events.append(rec)
                    print(f"[PARKING] {event}: {plate_text}")

            # ── Xử lý sự kiện RA (xe mất khỏi khung hình) ──
            stale_keys = [k for k in plate_cache if k not in seen_keys]
            for k in stale_keys:
                lost_plate = plate_cache.get(k, "")
                if lost_plate:
                    plate_unseen_count[lost_plate] = plate_unseen_count.get(lost_plate, 0) + 1
                    if (plate_unseen_count[lost_plate] >= CONFIRM_OUT
                            and lost_plate in plate_confirmed
                            and is_plate_inside(lost_plate)):
                        rec = record_parking_event(lost_plate, "OUT")
                        parking_events.append(rec)
                        plate_confirmed.discard(lost_plate)
                        plate_seen_count.pop(lost_plate, None)
                        plate_unseen_count.pop(lost_plate, None)
                        print(f"[PARKING] OUT (lost): {lost_plate}")

                # Dọn cache
                del plate_cache[k]
                plate_hit_count.pop(k, None)

            _, buffer = cv2.imencode('.jpg', frame_small)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            payload = {"image": f"data:image/jpeg;base64,{img_base64}"}
            if parking_events:
                payload["events"] = parking_events

            await websocket.send_text(json.dumps(payload))

    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"[inference_worker] Lỗi: {e}")

# ================= WEBSOCKET =================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Client kết nối")

    queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    task = asyncio.create_task(inference_worker(websocket, queue))

    try:
        while True:
            data = await websocket.receive_text()

            img_bytes = base64.b64decode(data.split(",")[1])
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass

            await queue.put(frame)

    except Exception as e:
        print(f"[WS] Ngắt kết nối: {e}")
    finally:
        task.cancel()
        await task
        try:
            await websocket.close()
        except Exception:
            pass
        print("[WS] Client đã ngắt")   