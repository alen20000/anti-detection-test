from ultralytics import YOLO
from pathlib import Path

# 路徑處理
current_dir = Path(__file__)
data_dir = current_dir.parent / "lie_detector-v1" / "data.yaml"


print(data_dir)
# 1. 載入基底模型（建議從輕量又快的 yolov8n.pt 開始）
model = YOLO("yolov8n.pt")

if data_dir.exists():

    # 2. 開始訓練
    results = model.train(
        data=data_dir,  # 
        epochs=50,  # 訓練輪數
        imgsz=640,  # 圖片大小
        batch=16,  # 批次大小
    )
else:
    print("檔案不存在！")