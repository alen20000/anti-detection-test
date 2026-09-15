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


    def draw_BBOX(self,results):

        result = results[0].plot()
        return result

    def lock_initial_target(self, bgr_frame, boxes_xyxy, boxes_ids, boxes_confs, 
                            conf_thresh=0.5, white_thresh=100):
        """
        找到「白色像素最多」的目標，作為鎖定的目標

        """
        best_idx = None
        best_white_ratio = 0  # 記錄目前找到的框裡，白色像素佔比最高的數值



        for i, box in enumerate(boxes_xyxy):
            if boxes_confs[i] < conf_thresh:
                continue  # 信心值太低的框，直接跳過，不列入候選

            x1, y1, x2, y2 = map(int, box)
            roi = bgr_frame[y1:y2, x1:x2]  # 從原圖裁出這個框的範圍
            if roi.size == 0:
                continue

            # 算這個框裡面，有多少比例的像素是接近白色的
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            white_pixels = np.sum(gray_roi > white_thresh)
            white_ratio = white_pixels / gray_roi.size  # 白色像素佔整個框的比例

            # 背景假目標大多是沙色，白色比例會很低
            # 真目標是全白，白色比例會明顯高很多
            if white_ratio > best_white_ratio:
                best_white_ratio = white_ratio
                best_idx = i

        if best_idx is None or best_white_ratio < 0.1:
            # 一個夠白的框都找不到，代表這一幀目標可能被擋住了，先不要亂鎖
            return None

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
                conf=0.8, 
                persist=True, 
                tracker="botsort.yaml", 
                verbose=False
            )

            boxes = results[0].boxes


            '''
            鎖定目標
            '''

            if boxes is not None and boxes.id is not None:
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
    my_model = "runs/detect/train-4/weights/best.pt"
    video_path = "videos/test_lie_detector.mp4"
    tracker = Tracker(model=my_model,
            source=video_path)
    tracker.run()