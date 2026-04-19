# Parking License Plate Detection & OCR Service

## Overview

This project provides a backend service for detecting vehicle license plates and performing OCR to extract plate numbers. It also includes a parking management system to track vehicle entry and exit events.

The system supports both single image processing via HTTP API and real-time processing via WebSocket.

---

## Base URL

When running locally:

http://localhost:8000

---

## Technologies Used

- FastAPI (backend API)
- YOLO (license plate detection)
- ViNTern model (OCR)
- SQLite (database)
- OpenCV (image processing)
- WebSocket (realtime communication)

---

## API Endpoints

### 1. Detect License Plate from Image

Endpoint:
POST /detect-image

Protocol:
HTTP (REST)

Request:
- Content-Type: multipart/form-data
- Field:
  - file: image (jpg or png)

Response:
```json
{
  "image": "data:image/jpeg;base64,...",
  "events": [
    {
      "plate": "51A12345",
      "event": "IN",
      "timestamp": "2026-04-19 10:00:00"
    }
  ]
}