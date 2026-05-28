import cv2
import requests
import time
import numpy as np
from ultralytics import YOLO

# 1. 설정
MODEL_PATH = "yolov8n.pt"  # 모델 파일 (자동 다운로드됨)
SERVER_URL = "http://127.0.0.1:8000/detections/cam01" # 서버 주소 (카메라 ID 확인!)
VIDEO_SOURCE = 0 # 0은 웹캠, 영상 파일이면 "video.mp4" 경로 입력

# 2. 모델 로드
model = YOLO(MODEL_PATH)

def run_client():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    print(f"🚀 실시간 감지 시작... (서버: {SERVER_URL})")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # YOLO 감지 (사람만 찾기: classes=[0])
        results = model.predict(frame, classes=[0], verbose=False)
        
        detections = []
        for r in results:
            for box in r.boxes:
                # 사람의 발 위치 계산 (바운딩 박스 하단 중앙)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                foot_x = float((x1 + x2) / 2)
                foot_y = float(y2)
                detections.append([foot_x, foot_y])

                # 시각화 (화면에 박스와 점 그리기)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(frame, (int(foot_x), int(foot_y)), 5, (0, 0, 255), -1)

        # 3. 서버로 데이터 전송 (0.5초 간격 권장)
        try:
            response = requests.post(SERVER_URL, json=detections, timeout=0.5)
            if response.status_code == 200:
                print(f"📡 전송 완료: {len(detections)}명 감지 중")
        except Exception as e:
            print(f"❌ 서버 연결 실패: {e}")

        # 화면 출력
        cv2.putText(frame, f"Detecting: {len(detections)}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("AutoReturn Vision Client", frame)

        # 'q' 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # 서버 부하 방지를 위한 미세한 대기 (약 10FPS)
        time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_client()