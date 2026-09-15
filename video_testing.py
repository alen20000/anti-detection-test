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
        pass

    def draw_BBOX(self):
        pass
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
            self.draw_BBOX(results[0].plot())

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break


        # 釋放資源
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    my_model = "runs/detect/train-4/weights/best.pt"
    video_path = "videos/test_lie_detector.mp4"
    Tracker(model=my_model,
            source=video_path)
    Tracker.run()