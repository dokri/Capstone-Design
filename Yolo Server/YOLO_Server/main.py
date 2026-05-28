import asyncio
import os
import time
from datetime import datetime, timezone

import aiohttp
import cv2
import numpy as np
import sys
from ultralytics import YOLO
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# ── 환경 변수 설정 ────────────────────────────────────────────────────────────
MAIN_SERVER_URL = os.getenv("MAIN_SERVER_URL", "http://localhost:8000")

# 1순위: 터미널 입력값 (python main.py cam02)
# 2순위: .env 파일 설정값
# 3순위: 기본값 "cam01"
if len(sys.argv) > 1:
    CAMERA_ID = sys.argv[1]
else:
    CAMERA_ID = os.getenv("CAMERA_ID", "cam01")

# 카메라 소스(영상 경로 또는 웹캠 번호)도 인자로 받을 수 있게 구성
if len(sys.argv) > 2:
    CAMERA_SOURCE = sys.argv[2]
else:
    CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")

MODEL_PATH      = os.getenv("MODEL_PATH", "yolov8n.pt")
INTERVAL        = float(os.getenv("INTERVAL", "0.5"))   # 0.5초마다 전송
CONFIDENCE      = float(os.getenv("CONFIDENCE", "0.5"))


def load_model() -> YOLO:
    print(f"📦 [YOLO] 모델 로드 중: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    return model


def open_capture(source: str) -> cv2.VideoCapture:
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"❌ [카메라] 연결 실패: {source}")
    print(f"✅ [카메라] 연결 성공: {source}")
    return cap


def get_foot_points(model: YOLO, frame) -> list:
    results = model(frame, classes=[0], conf=CONFIDENCE, verbose=False)
    
    # YOLO 텐서를 Numpy 배열로 변환
    boxes = results[0].boxes.xyxy.cpu().numpy()
    
    if len(boxes) == 0:
        return []

    # 벡터 연산으로 발바닥 중앙점 계산 (속도 향상)
    feets_x = (boxes[:, 0] + boxes[:, 2]) / 2
    feets_y = boxes[:, 3]
    
    # (N, 2) 형태의 좌표 리스트로 병합
    points = np.stack([feets_x, feets_y], axis=1).tolist()
    
    # 시각화 (선택 사항)
    for pt in points:
        cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
            
    return points


async def send_detections(session: aiohttp.ClientSession, points: list):
    """메인 서버가 기대하는 DetectionPayload 형식으로 전송"""
    url = f"{MAIN_SERVER_URL}/detections"  # 엔드포인트 통합
    
    # 서버의 DetectionPayload 스키마에 맞춘 데이터 구성
    payload = {
        "camera_id": CAMERA_ID,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "detections": points,  # [[x,y], [x,y]]
        "clear_all": False     # 특정 상황(카메라 가려짐 등)이 아니면 False
    }
    
    try:
        async with session.post(
            url,
            json=payload, 
            timeout=aiohttp.ClientTimeout(total=2),
        ) as resp:
            if resp.status == 200:
                # 메인 서버 응답에서 업데이트된 좌석 수를 받아옴
                res_data = await resp.json()
                print(f"📡 [전송] {len(points)}명 감지 -> {res_data.get('seats_updated')}개 좌석 업데이트 완료")
            else:
                body = await resp.text()
                print(f"⚠️ [오류] 서버 응답 코드: {resp.status}, 메시지: {body}")
    except Exception as e:
        print(f"❌ [실패] 메인 서버 연결 불가: {e}")


async def run():
    model = load_model()
    cap = open_capture(CAMERA_SOURCE)

    # aiohttp 세션 하나를 생성해서 재사용 (성능 최적화)
    async with aiohttp.ClientSession() as session:
        print(f"🚀 [시작] 감지 루프 가동 (Camera: {CAMERA_ID}, Interval: {INTERVAL}s)")
        
        while True:
            loop_start = time.monotonic()

            ret, frame = cap.read()
            if not ret:
                print("🔄 [경고] 프레임 읽기 실패 - 재연결 중...")
                cap.release()
                await asyncio.sleep(2)
                cap = open_capture(CAMERA_SOURCE)
                continue

            # 1. 사람 발바닥 좌표 추출
            points = get_foot_points(model, frame)
            
            # 2. 서버 전송
            await send_detections(session, points)

            # 시각화 창 띄우기 (시연용)
            cv2.imshow(f"YOLO Detection - {CAMERA_ID}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # 주기(Interval) 맞추기
            elapsed = time.monotonic() - loop_start
            wait = max(0.01, INTERVAL - elapsed)
            await asyncio.sleep(wait)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n👋 프로그램을 종료합니다.")