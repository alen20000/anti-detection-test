from ultralytics import YOLO
from pathlib import Path

# conda activate yolo_env # 激活虛擬環境

FOLDER_PATH = Path(__file__).parent
# 數據集目錄
DATASETS_ROOT_FOLDER_PATH  = FOLDER_PATH / "dataset"
DATASET_FOLDER_NAME = "LD_yolov8"  # 填入要使用數據集的資料夾名稱
DATASET_FOLDER_PATH = DATASETS_ROOT_FOLDER_PATH / DATASET_FOLDER_NAME # 數據集路徑
DATA_YAML_PATH = DATASET_FOLDER_PATH / "data.yaml" # 數據集的設定檔


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

class TrainModel:
    def __init__(self, resume: bool = False):
        
        # 上次訓練模型的路徑
        last_model_path = (
        FOLDER_PATH / "runs" / "detect" / "train" / "weights" / "last.pt"
        )

        if resume and last_model_path.exists():
            print(f"發現上次的訓練進度：{last_model_path}")
            print("是否繼續上次的訓練")
            answer = input("y / n: ")
            if answer == "y":
                print("繼續上次訓練")
                self.basic_model = last_model_path
                self.model = YOLO(self.basic_model)
            else:
                resume = False

        if not resume:
            self.basic_model = BASIC_MODEL
            if not self.basic_model.exists():
                raise FileNotFoundError(f"錯誤回報：找不到基礎模型檔案：{YOLO_MODEL_CHOOSE}")
            else:
                self.model = YOLO(self.basic_model)

        self.traing_config = DATA_YAML_PATH
        if not self.traing_config.exists():
            raise FileNotFoundError(f"錯誤回報：找不到 {DATASET_FOLDER_NAME} 數據集的設定檔案")
        else:
            print(f"載入{DATASET_FOLDER_NAME} 數據集的設定檔案；路徑：{DATA_YAML_PATH}")
        
    def train(self,
            epochs: int = 50,
            imgsz: int = 640,
            batch: int = 16,
            workers: int=4,
            save_period: int = None,
            resume: bool = False,
            ):
        
        train_args = {
            "data":self.traing_config,  # 
            "epochs":epochs,  # 訓練輪數:同張圖訓練幾輪
            "imgsz":imgsz,   # 統一圖片大小:太大很吃內存
            "batch":batch,   # 批次大小:炸顯存不要調太大，每次調整習慣用2的冪次方
            "workers":workers, # 類似線程的東西，有餘裕在開大
            "device": 0, # 調用GPU跑
        }
        # 動態加載參數:幾輪儲存一次
        if save_period is not None and save_period > 0:
            train_args["save_period"] = save_period
        # 動態加載參數:繼續上一次訓練
        if resume:
            train_args["resume"] = resume

        self.model.train(
            **train_args
        )
if __name__ == "__main__":

    # 接續訓練參數開關:True or False
    resume = False
    runTrain = TrainModel(resume)
    runTrain.train()