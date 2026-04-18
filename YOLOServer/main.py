import asyncio
import os
import time
from datetime import datetime, timezone

import aiohttp
import cv2
from ultralytics import YOLO

# ── 설정 ──────────────────────────────────────────────────────────────────────
MAIN_SERVER_URL = os.getenv("MAIN_SERVER_URL", "http://localhost:8000")
CAMERA_ID       = os.getenv("CAMERA_ID", "cam_01")
CAMERA_SOURCE   = os.getenv("CAMERA_SOURCE", "0")       # 0=웹캠, rtsp://... or 영상 파일 경로
MODEL_PATH      = os.getenv("MODEL_PATH", "yolov8n.pt")
INTERVAL        = float(os.getenv("INTERVAL", "1.0"))   # 탐지 주기 (초)
CONFIDENCE      = float(os.getenv("CONFIDENCE", "0.5")) # 탐지 신뢰도 임계값


def load_model() -> YOLO:
    print(f"[YOLO] 모델 로드 중: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("[YOLO] 모델 로드 완료")
    return model


def open_capture(source: str) -> cv2.VideoCapture:
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"카메라를 열 수 없습니다: {source}")
    print(f"[카메라] 연결 성공: {source}")
    return cap


def detect_people(model: YOLO, frame) -> list[dict]:
    """
    YOLO로 사람(class 0)만 탐지하여 bounding box 목록 반환.
    반환: [{"x1": float, "y1": float, "x2": float, "y2": float, "confidence": float}, ...]
    """
    results = model(frame, classes=[0], conf=CONFIDENCE, verbose=False)
    boxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append({
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2,
                "confidence": float(box.conf[0]),
            })
    return boxes


async def send_detections(session: aiohttp.ClientSession, detections: list[dict], timestamp: datetime):
    """메인 서버 POST /detections 호출"""
    payload = {
        "camera_id": CAMERA_ID,
        "timestamp": timestamp.isoformat(),
        "detections": detections,
        "clear_all": len(detections) == 0,  # 아무도 없으면 명시적 해제
    }
    try:
        async with session.post(
            f"{MAIN_SERVER_URL}/detections",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                print(f"[전송 오류] status={resp.status} body={body}")
    except aiohttp.ClientError as e:
        print(f"[전송 실패] 메인 서버 연결 오류: {e}")


async def run():
    model = load_model()
    cap = open_capture(CAMERA_SOURCE)

    async with aiohttp.ClientSession() as session:
        print(f"[시작] camera_id={CAMERA_ID}, interval={INTERVAL}s")
        while True:
            loop_start = time.monotonic()

            ret, frame = cap.read()
            if not ret:
                print("[경고] 프레임 읽기 실패 — 재연결 시도")
                cap.release()
                await asyncio.sleep(2)
                cap = open_capture(CAMERA_SOURCE)
                continue

            timestamp = datetime.now(timezone.utc)
            detections = detect_people(model, frame)
            print(f"[탐지] {timestamp.strftime('%H:%M:%S')} — {len(detections)}명")

            await send_detections(session, detections, timestamp)

            # 다음 주기까지 대기
            elapsed = time.monotonic() - loop_start
            wait = max(0.0, INTERVAL - elapsed)
            await asyncio.sleep(wait)

    cap.release()


if __name__ == "__main__":
    asyncio.run(run())
