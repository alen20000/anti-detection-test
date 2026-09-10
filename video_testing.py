import cv2
import numpy as np
import time
from ultralytics import YOLO

'''
目標: 解決遊戲檢測
'''

# 1. 讀取測試影片 & 初始化
video_path = "videos/test_lie_detector.mp4"
cap = cv2.VideoCapture(video_path)

# 載入訓練好的模型
model = YOLO("runs/detect/train-4/weights/best.pt")

if not cap.isOpened():
    print(f"無法開啟影片：{video_path}，請檢查檔案路徑。")
    exit()

print("按 'q' 鍵可關閉視窗。")

# 開啟影片迴圈
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("影片播放完畢。")
        break

    # 2. 切割取 ROI
    h, w, _ = frame.shape
    ymin, ymax = int(h * 0.22), int(h * 0.70)
    xmin, xmax = int(w * 0.05), int(w * 0.95)
    roi_frame = frame[ymin:ymax, xmin:xmax]

    # 3. 使用 YOLO 模型對當前 ROI 畫面進行檢測
    # conf=0.5 代表信心水準大於 50% 才顯示
    results = model(roi_frame, conf=0.5, verbose=False)
    
    # 4. 把檢測到的框畫在畫面上 (results[0].plot() 會回傳畫好框的影像)
    annotated_frame = results[0].plot()

    # 5. 顯示即時追蹤畫面
    cv2.imshow("Lie Detector Tracking Test", annotated_frame)

    # 6. 偵測鍵盤按鍵，按 'q' 離開
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()