import cv2
import numpy as np
import time
from ultralytics import YOLO


MODEL_PATH = "models/Maple_yolo_v2.pt"

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

        self.last_objects_position = {} # 放上一幀的目標座標
        self.current_objects_position = {} # 放本幀目標座標
        # 預熱:空推論
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy_frame, verbose=False)

    def reset(self):
        '''每次解題開始前呼叫，把狀態還原'''
        self.locked_id = None
        self.trajectory = []
        self.last_objects_position = {}
        self.current_objects_position = {}

        # 防呆檢查 + 清掉 YOLO tracker 的內部狀態
        if self.model.predictor is not None and hasattr(self.model.predictor, "trackers"):
            for tracker in self.model.predictor.trackers:
                tracker.reset()


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

    def record_object_position(self, boxes):
        '''紀錄物件資訊
        Args: boxes (目標bbox容器)
        '''
        # 容器初始化
        self.current_objects_position = {}

        # 防呆:這一幀沒有追蹤 ID 就不記錄
        if boxes is None or boxes.id is None:
            return

        # 紀錄中心座標； cpu() 預防資料存gpu而拉回cpu ；numpy() 將tensor轉為numpy； astype() 轉整數儲存
        xywh = boxes.xywh.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)

        # 先測試，只儲存前幀資訊
        for i in range(len(ids)):
            tracknig_id = int(ids[i])
            self.current_objects_position[tracknig_id] = (int(xywh[i][0]), int(xywh[i][1]))

    def copy_to_last_object_position(self):
        '''把當前幀的物件資料儲存給上一幀容器'''
        self.last_objects_position = self.current_objects_position.copy()


    def handleMoveDirection(self,frame):
        '''
        判斷每個物件的移動方向
        '''

        for tracknig_id, (cx, cy) in self.current_objects_position.items():

            # 有些號碼會消失、不存在，遇到消失的編號做跳過處理
            if tracknig_id not in self.last_objects_position:
                continue

            last_cx, last_cy = self.last_objects_position[tracknig_id]

            # 計算位移
            dx = cx - last_cx
            dy = cy - last_cy

            # 怕數字太小，畫不出來
            upscale = 1
            dx = dx * upscale
            dy = dy * upscale

            # 畫箭頭
            cv2.arrowedLine(frame, (cx, cy), (cx + dx, cy + dy), (180, 100, 0), 2)

    def solving_problem(self,frame):
        ''' 開始解題

        Args: 
            frame: 畫面
        '''

        # 防呆
        if frame is None:
            return

        # Yolo物件偵測與多目標追蹤
        results = self.model.track(
            frame, 
            conf=0.2, 
            iou=0.7,
            persist=True, 
            tracker="ocsort.yaml", 
            verbose=False
        )
        # 偵測與追蹤的框體結果
        boxes = results[0].boxes

        # 紀錄該幀的物件資訊
        self.record_object_position(boxes)

        # 防呆:該幀沒找到任何物件
        if boxes.id is None:
            self.copy_to_last_object_position()   # 等於清空容器
            return False
        print("test:目前內容", self.current_objects_position)
        print("test:前幀內容", self.last_objects_position)


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
        self.handleMoveDirection(annotated_frame)

        #迴圈結束前把該幀資訊除存
        self.copy_to_last_object_position()

        cv2.imshow("Lie Detector Tracking Test", annotated_frame) # 測試用：顯示目標追蹤畫面
        # cv2.imshow("test", binary_frame) # 測試用：顯示二值化後的畫面





