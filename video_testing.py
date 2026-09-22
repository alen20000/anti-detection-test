import cv2
import numpy as np
import time
from ultralytics import YOLO

'''
目標: 解決遊戲檢測
'''


class Tracker:
    def __init__(self,model=None,source=None):
        self.model = YOLO(model)  
        self.names = self.model.names
        print(f"模型載入成功，模型名稱：{self.names}")
        self.cap =  cv2.VideoCapture(source)
        assert self.cap.isOpened(), "讀取串流失敗"

        self.locked_id = None #鎖定的目標ID
        self.first_lock_time = None # 紀錄第一次觸發 lock_initial_target 的時間

    def draw_BBOX(self,results):

        result = results[0].plot()
        return result

    def lock_initial_target(self, binary_frame, boxes_xyxy, boxes_ids, boxes_confs, 
                                conf_thresh=0.5, min_area=50):
        """
        利用全域二值化找出最大的白色區塊，並判斷它屬於哪個 YOLO 目標 ID
        """
        # 1. 在全域的二值化影像中尋找所有輪廓
        contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        # 2. 找出全域面積最大的輪廓
        largest_contour = max(contours, key=cv2.contourArea)
        largest_area = cv2.contourArea(largest_contour)
        
        if largest_area < min_area:
            return None
            
        # 計算這個最大輪廓的中心點座標 (cx, cy)
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        best_id = None

        # 3. 遍歷多目標:以中心座標判斷，落在哪個yolo目標的 box 裡面
        for i, box in enumerate(boxes_xyxy):
            if boxes_confs[i] < conf_thresh:
                continue
            
            x1, y1, x2, y2 = map(int, box)
            
            # 判斷點 (cx, cy) 是否在該 box 範圍內
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                best_id = int(boxes_ids[i])
                break

        return best_id

    def run(self):
        while self.cap.isOpened():

            success, frame = self.cap.read()

            if not success:
                print("影片播放完畢")
                break

            # 對測試的影片畫面進行裁切
            h, w, _ = frame.shape
            ymin, ymax = int(h * 0.22), int(h * 0.70)
            xmin, xmax = int(w * 0.05), int(w * 0.95)
            roi_frame = frame[ymin:ymax, xmin:xmax]



            results = self.model.track(
                roi_frame, 
                conf=0.2, 
                iou=0.7,
                persist=True, 
                tracker="ocsort.yaml", 
                verbose=False
            )

            boxes = results[0].boxes

            # frame ->灰階 -> 二值
            gray_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            _, binary_frame = cv2.threshold(gray_frame, 240, 255, cv2.THRESH_BINARY)
            '''
            步驟 I:
                找到「白色像素最多」的目標，作為鎖定目標
            Note: 
                只追蹤兩秒
            '''
            if self.first_lock_time is None and boxes is not None and boxes.id is not None:
                self.locked_id = self.lock_initial_target(
                    binary_frame=binary_frame,
                    boxes_xyxy=boxes.xyxy.cpu().numpy(),
                    boxes_ids=boxes.id.cpu().numpy().astype(int), # 從CPU轉numpy再以整數儲存
                    boxes_confs=boxes.conf.cpu().numpy()
                )
                if self.locked_id is not None:
                    self.first_lock_time = time.time()
                    print("初次鎖定目標:", self.locked_id)

            if self.first_lock_time and self.first_lock_time + 2 > time.time():
                    self.locked_id = self.lock_initial_target(
                    binary_frame=binary_frame,
                    boxes_xyxy=boxes.xyxy.cpu().numpy(),
                    boxes_ids=boxes.id.cpu().numpy().astype(int), # 從CPU轉numpy再以整數儲存
                    boxes_confs=boxes.conf.cpu().numpy()
                )
            print("追蹤目標:", self.locked_id)


            # bbox display
            annotated_frame  = self.draw_BBOX(results)
            cv2.imshow("Lie Detector Tracking Test", annotated_frame)
            cv2.imshow("test", binary_frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break


        # 釋放資源
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    my_model = "models\Maple_yolo_v1.pt"
    video_path = "videos/test_lie_detector.mp4"
    tracker = Tracker(model=my_model,
            source=video_path)
    tracker.run()