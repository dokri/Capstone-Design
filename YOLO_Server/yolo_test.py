import cv2
import requests
from ultralytics import YOLO
from datetime import datetime

# 1. 설정 (본인의 환경에 맞게 수정)
model = YOLO('yolov8n.pt')  # YOLO 모델 (없으면 자동으로 다운로드됨)
video_path = 'CCTV_data/demo_video.mov'  # <<< 여기에 유저님이 가지고 계신 영상 파일 이름을 넣으세요!
server_url = "http://127.0.0.1:8000/detections"
camera_id = "cam01"

cap = cv2.VideoCapture(video_path)

print("영상을 분석하여 서버로 데이터를 전송합니다... (종료하려면 'q' 키)")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 2. YOLO 탐지 (사람만)
    results = model(frame, classes=[0], verbose=False)
    
    detections = []
    for r in results[0].boxes:
        x1, y1, x2, y2 = r.xyxy[0].tolist()
        detections.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "confidence": float(r.conf)
        })

    # 3. 메인 서버로 전송
    payload = {
        "camera_id": camera_id,
        "timestamp": datetime.now().isoformat(),
        "detections": detections,
        "clear_all": False
    }
    
    try:
        # 실시간성을 위해 너무 자주 보내지 않으려면 아래 주석을 해제하세요.
        # if len(detections) > 0: 
        requests.post(server_url, json=payload)
    except Exception as e:
        print(f"서버 연결 오류: {e}")

    # 화면에 분석 결과 표시
    cv2.imshow("YOLO Real-time Test", results[0].plot())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()