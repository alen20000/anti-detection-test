import cv2
import numpy as np
import time
from ultralytics import YOLO


MODEL_PATH = "models/Maple_yolo_v1.pt"

class AntiDetector:
    '''
    解測謊模塊
    '''
    def __init__(self):

        # 模型載入
        self.model = YOLO(MODEL_PATH)  # <- 加載模型
        self.names = self.model.names
        print(f"模型載入成功，模型名稱：{self.names}")

        # 容器初始化
        self.locked_id = None #鎖定的目標ID

        self.trajectory = [] # 路徑記錄

        # 預熱:空推論
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy_frame, verbose=False)

    def _lock_initial_target(self, binary_frame, boxes_xyxy, boxes_ids, boxes_confs, 
                                conf_thresh=0.5, min_area=50):
        """抓出目標

        利用全域二值化找出最大的白色區塊，並判斷它屬於哪個 YOLO 目標 ID
        
        改進: 去算多目標的中心點，以中西點中心範圍內為白色，則為我們要的追蹤目標

        Args:
            binary_frame (np.ndarray): 二值化影像
            boxes_xyxy (np.ndarray): YOLO 目標的坐標 (x1, y1, x2, y2)
            boxes_ids (np.ndarray): YOLO 目標的 ID
            boxes_confs (np.ndarray): YOLO 目標的置信度
            conf_thresh (float, optional): 置信度閾值. Defaults to 0.5.
            min_area (int, optional): 最小面積閾值. Defaults to 50.
        """

        for i, box in enumerate(boxes_xyxy):
            if boxes_confs[i] < conf_thresh:
                continue
            
            x1, y1, x2, y2 = map(int, box)

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # 中心點附近
            x_min,x_max = cx - 10 , cx + 10
            y_min,y_max = cy - 100 , cy + 10

            patch = binary_frame[y_min:y_max, x_min:x_max]
            if patch.size > 0:
                has_whitee = np.any(patch == 255)
                if has_whitee:
                    print(f"找到目標，ID：{boxes_ids[i]}")
                    return boxes_ids[i]


    def _tracking_target_position(self,target_ID,boxes):
        '''追蹤目標的cx,xy'''
        #防呆
        if (
            target_ID is None 
            or boxes is None 
            or boxes.id is None
        ):
            return
        
        boxes_xyxy = boxes.xyxy.cpu().numpy()
        boxes_ids = boxes.id.cpu().numpy().astype(int)
        
        for i, track_id in enumerate(boxes_ids):
            if track_id == self.locked_id:
                x1, y1, x2, y2 = map(int, boxes_xyxy[i])

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                return cx,cy  # 回傳該目標的座標

        return  None

    def draw_BBOX(self, results, target_position=None, trajectory=None):
        '''繪製BBOX；繪製路徑
        '''
        #  YOLO 的標記畫面
        result_img = results[0].plot()
        
        # 繪製移動軌跡連線
        if trajectory and len(trajectory) > 1:

            pts = np.array(trajectory, dtype=np.int32).reshape((-1, 1, 2)) # list轉 numpy陣列格式 (Points, 1, 2) 

            cv2.polylines(result_img, [pts], isClosed=False, color=(0, 255, 0), thickness=2) # 參數 isClosed=False 代表不要頭尾相連

        return result_img
    
    def solving_problem(self,frame):
        ''' 開始解題

        Args: 
            frame: 畫面
        '''

        # 防呆
        if frame is None:
            return
        
        results = self.model.track(
            frame, 
            conf=0.2, 
            iou=0.7,
            persist=True, 
            tracker="ocsort.yaml", 
            verbose=False
        )

        boxes = results[0].boxes

        # frame ->灰階 -> 二值
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary_frame = cv2.threshold(gray_frame, 240, 255, cv2.THRESH_BINARY)

        '''定位目標
        '''
        if self.locked_id is None :
            self.locked_id = self._lock_initial_target(
                binary_frame=binary_frame,
                boxes_xyxy=boxes.xyxy.cpu().numpy(),
                boxes_ids=boxes.id.cpu().numpy().astype(int), # 從CPU轉numpy再以整數儲存
                boxes_confs=boxes.conf.cpu().numpy()
            )
            if self.locked_id is not None:
                print("鎖定目標:", self.locked_id)
            
            # 沒有抓到就跳出
            else:
                print("找不到目標")
                return False
            
        print(f"追蹤目標:", self.locked_id)
        # 目標座標追蹤
        current_target_position = self._tracking_target_position(self.locked_id,boxes)
        if current_target_position is not None:
            print(current_target_position)
            self.trajectory.append(current_target_position)

        # bbox display
        annotated_frame  = self.draw_BBOX(results, current_target_position, self.trajectory)

        
        cv2.imshow("Lie Detector Tracking Test", annotated_frame) # 測試用：顯示目標追蹤畫面
        # cv2.imshow("test", binary_frame) # 測試用：顯示二值化後的畫面





