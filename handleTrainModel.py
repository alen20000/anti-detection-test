from ultralytics import YOLO
from pathlib import Path

# conda activate yolo_env # 激活虛擬環境

FOLDER_PATH = Path(__file__).parent
# 數據集目錄
DATASET_FOLDER_PATH  = FOLDER_PATH / "dataset"

# 基礎模型路徑設定
YOLO_MODEL_DICT = {
    "yolov8n": "yolov8n.pt",}
YOLO_MODEL_CHOOSE = "yolov8n" # 預設模型
BASIC_MODEL = FOLDER_PATH / "models" / YOLO_MODEL_DICT[YOLO_MODEL_CHOOSE]

'''
@brief: 訓練模型腳本
@details : 若要調用GPU訓練，要安裝對應版本的 python 與 PyTorch。

@Note: 

檢查有沒有GPU: CMD -> python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
顯示沒有，但其實有 -> 移除舊版並安裝 GPU 專用 PyTorch ; 清除 Torch -> pip uninstall torch torchvision torchaudio -y
然後安裝對python版本支援GPU的PyTorch

沒辦法的話，只能建立虛擬環境，安裝穩定版本的 python 3.12 然後在虛擬機環境安裝 -> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
'''
# # 路徑處理
# current_dir = Path(__file__)
# data_dir = current_dir.parent / "Maple_yolo_v1" / "data.yaml" # 訓練資料
# basic_yolo_model = current_dir.parent /"models"/ "yolov8n.pt"

# print(data_dir)
# # 1. 載入基底模型 (y8比較經典)
# model = YOLO(basic_yolo_model)

# if data_dir.exists():

#     # 2. 開始訓練
#     results = model.train(
#         data=data_dir,  # 
#         epochs=50,  # 訓練輪數
#         imgsz=640,  # 圖片大小
#         batch=16,  # 批次大小
#     )
# else:
#     print("檔案不存在！")

class TrainModel:
    def __init__(self) -> None:
        # self.basic_model = 
        pass

if __name__ == "__main__":
    print(BASIC_MODEL)