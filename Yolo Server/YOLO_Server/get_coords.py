import cv2

# 시연용 영상 파일 경로를 넣으세요
video_path = "CCTV_data/demo_video.mov" 
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

if not ret:
    print("영상을 불러올 수 없습니다.")
    exit()

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"클릭된 좌표: [ {x}, {y} ]")
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Select Points", frame)

print("영상 화면에서 호모그래피 기준점 4곳을 순서대로 클릭하세요.")
print("다 하셨으면 아무 키나 누르세요.")

cv2.imshow("Select Points", frame)
cv2.setMouseCallback("Select Points", mouse_callback)
cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()