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

    def lock_initial_target(self, bgr_frame, boxes_xyxy, boxes_ids, boxes_confs, 
                                conf_thresh=0.5, white_thresh=100, min_area=50):
            """
            找到「白色像素最多」的目標，作為鎖定的目標
            """
            best_idx = None
            best_area = 0  # 修正 1：正確宣告區域變數初始化

            # 白色目標範圍
            lower_white = np.array([200, 200, 200], dtype=np.uint8)
            upper_white = np.array([255, 255, 255], dtype=np.uint8)

            for i, box in enumerate(boxes_xyxy):
                if boxes_confs[i] < conf_thresh:
                    continue  # 信心值太低的框，直接跳過，不列入候選
                x1, y1, x2, y2 = map(int, box)
                roi = bgr_frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                # 先用顏色篩出白色區域的遮罩
                white_mask = cv2.inRange(roi, lower_white, upper_white)
                # 用輪廓找出白色區域
                contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if not contours: # 防呆處理
                    continue
                largest_contour = max(contours, key=cv2.contourArea)
                largest_area = cv2.contourArea(largest_contour)

                if largest_area > best_area:
                    best_area = largest_area
                    best_idx = i

            # 修正 2：迴圈全部跑完、選出全畫面最大的白色候選後，才檢查門檻
            if best_idx is None or best_area < min_area:
                return None

            self.best_area = best_area  # 同步更新實例變數（若後續需要跨幀比較）
            return int(boxes_ids[best_idx])  # 回傳這個框對應的 track ID，之後就一路追這個ID

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
                tracker="bytetrack.yaml", 
                verbose=False
            )

            boxes = results[0].boxes


            '''
            步驟 I:
                找到「白色像素最多」的目標，作為鎖定目標
            Note: 
                只追蹤兩秒
            '''
            if self.first_lock_time is None and boxes is not None and boxes.id is not None:
                self.locked_id = self.lock_initial_target(
                    bgr_frame=roi_frame,
                    boxes_xyxy=boxes.xyxy.cpu().numpy(),
                    boxes_ids=boxes.id.cpu().numpy().astype(int), # 從CPU轉numpy再以整數儲存
                    boxes_confs=boxes.conf.cpu().numpy()
                )
                if self.locked_id is not None:
                    self.first_lock_time = time.time()
                    print("初次鎖定目標:", self.locked_id)
            if self.first_lock_time and self.first_lock_time + 2 > time.time():
                    self.locked_id = self.lock_initial_target(
                    bgr_frame=roi_frame,
                    boxes_xyxy=boxes.xyxy.cpu().numpy(),
                    boxes_ids=boxes.id.cpu().numpy().astype(int), # 從CPU轉numpy再以整數儲存
                    boxes_confs=boxes.conf.cpu().numpy()
                )
            print("追蹤目標:", self.locked_id)

            annotated_frame  = self.draw_BBOX(results)
            cv2.imshow("Lie Detector Tracking Test", annotated_frame)
            
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