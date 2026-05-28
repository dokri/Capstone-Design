from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import time
import cv2
import numpy as np
import requests
from ultralytics import YOLO
from datetime import datetime, timezone
import threading

model = YOLO("yolov8n.pt")

SERVER_URL = "http://127.0.0.1:8000/detections"
last_send_time = 0

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

latest_frames = {}
frame_lock = threading.Lock()

HTML = """
<!DOCTYPE html>
<html>
<body>
<h2>iPhone Camera Upload</h2>

<video id="v" autoplay playsinline width="640"></video>
<p id="status">starting...</p>

<script>
const video = document.getElementById("v");
const statusText = document.getElementById("status");

async function start() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: {
            width: 416,
            height: 312,
            facingMode: "environment"
        },
        audio: false
    });

    video.srcObject = stream;

    const c = document.createElement("canvas");
    const ctx = c.getContext("2d");

    c.width = 416;
    c.height = 312;

    setInterval(async () => {
        ctx.drawImage(video, 0, 0, c.width, c.height);

        const blob = await new Promise(resolve =>
            c.toBlob(resolve, "image/jpeg", 0.6)
        );

        const fd = new FormData();
        fd.append("file", blob, "frame.jpg");

        try {
            const res = await fetch("/upload", {
                method: "POST",
                body: fd
            });

            if (res.ok) {
                statusText.innerText = "frame sent";
            } else {
                statusText.innerText = "upload failed";
            }
        } catch (e) {
            statusText.innerText = "connection error";
        }

    }, 333);
}

start();
</script>
</body>
</html>
"""

@app.get("/")
def index():
    return HTMLResponse(HTML)


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    global last_send_time

    camera_id = "cam03"

    data = await file.read()
    np_img = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if frame is None:
        return {"ok": False}

    results = model.predict(
        frame,
        classes=[0],
        verbose=False
    )

    detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            foot_x = float((x1 + x2) / 2)
            foot_y = float(y2)

            detections.append({
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "confidence": float(box.conf[0].cpu().numpy())
            })

            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )

            cv2.circle(
                frame,
                (int(foot_x), int(foot_y)),
                5,
                (0, 0, 255),
                -1
            )

    cv2.putText(
        frame,
        f"Detecting: {len(detections)}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    with frame_lock:
        latest_frames[camera_id] = frame.copy()

    now = time.time()

    if now - last_send_time >= 0.5:
        try:
            payload = {
                "camera_id": camera_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "detections": detections
            }

            response = requests.post(
                SERVER_URL,
                json=payload,
                timeout=0.5
            )

            print("Main Server 응답:", response.status_code, response.text)

            if response.status_code == 200:
                print(f"전송 완료: {len(detections)}명")

        except Exception as e:
            print(f"서버 연결 실패: {e}")

        last_send_time = now

    return {
        "ok": True,
        "detections": detections
    }


def mjpeg_generator(camera_id: str):
    while True:
        with frame_lock:
            frame = latest_frames.get(camera_id)

        if frame is None:
            blank = np.zeros((312, 416, 3), dtype=np.uint8)
            cv2.putText(
                blank,
                "Waiting for camera frame...",
                (30, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            frame = blank

        ret, buffer = cv2.imencode(".jpg", frame)

        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )

        time.sleep(0.05)


@app.get("/video/{camera_id}")
def video_feed(camera_id: str):
    return StreamingResponse(
        mjpeg_generator(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001
    )