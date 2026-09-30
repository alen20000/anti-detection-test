import time

import cv2
import numpy as np
from PIL import ImageGrab

import AntiDetector
import TriggerDetector
'''
$brief 測試 檢測是否測謊、若測謊則進入解題

測試trigger的影片素材
1. https://www.youtube.com/shorts/FizX6Ti5E6A
2. https://www.youtube.com/watch?v=UQzZI4hf7xU
3. https://www.youtube.com/shorts/qB7RdlTa84c


'''

#======= 偵測範圍
WINDOW_X_0 = 1038
WINDOW_X_1 = 1737
WINDOW_Y_0 = 260
WINDOW_Y_1 = 1447
DETCTION_RANGE = (WINDOW_X_0, WINDOW_Y_0, WINDOW_X_1, WINDOW_Y_1)
# ======  偵查頻率(秒)
CHECK_INTERVAL = 1
# ===================

class Main():
    def __init__(self):
        # self.AntiDetector = AntiDetector.AntiDetector()
        self.TriggerDetector = TriggerDetector.TriggerDetector()

        # 計時器
        self.last_check_time = time.time()

    def _capture_screen(self):
        '''抓指定區域畫面'''

        current_frame_res = ImageGrab.grab(DETCTION_RANGE)
        if current_frame_res is None:
            return None
        current_frame_np = np.array(current_frame_res)
        current_frame_bgr = cv2.cvtColor(current_frame_np, cv2.COLOR_BGR2RGB)

        return current_frame_bgr

    def run(self):
        # init
        current_frame = None
        try:
            while True:
                current_frame = self._capture_screen()
                if current_frame is not None:
                    cv2.imshow("DISPLAY", current_frame)

                # detect 
                if current_frame is not None:

                    if time.time() - self.last_check_time > CHECK_INTERVAL:  # 計時器
                        self.last_check_time = time.time()
                        is_detecting = self.TriggerDetector.check_lie_detector(current_frame)

                        if is_detecting:
                            print("檢測到人物正在被測謊!!!")
                        if not is_detecting:
                            print("Null")


                # EXIT(press "Q")
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(e)
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main = Main()
    main.run()
    