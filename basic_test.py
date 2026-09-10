from ultralytics import YOLO
from pathlib import Path
import cv2

# 
img = Path("test_1.jpg")
# 載入 YOLO 模型
model = YOLO("yolov8n.pt")

# 隨便丟一張你電腦裡的圖片路徑測試（或者放一張截圖）

if img.exists():
    results = model("test_1.jpg")

# 把結果印出來，只要沒報錯、能看到座標，就代表 YOLO 環境架設成功了！
    print(results[0].boxes)

# 顯示圖片
cv2.imshow("Result", results[0].plot())
cv2.waitKey(0)
cv2.destroyAllWindows()