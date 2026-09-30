import time

import cv2
import numpy as np
from PIL import ImageGrab
from enum import Enum, auto

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



#[!] 應該弄個簡單的狀態機，如果用一堆 flag,toggle 再加一堆判斷，感覺以後會很難看
#特別是如果要把這模塊放進 maplestory-opencv-automation 的repo內，應該會亂七八糟

class State(Enum):
    MONITORING = auto()
    SOLVING    = auto()


class Main():
    def __init__(self):
        # self.AntiDetector = AntiDetector.AntiDetector()
        self.TriggerDetector = TriggerDetector.TriggerDetector()

        # 計時器
        self.last_check_time = time.time()

        # State
        self.state = State.MONITORING

    #====
    # 狀態組
    #====
    def _change_state(self, new_state):
        '''切換狀態
        '''
        print(f"{self.state} -> {new_state}")
        self.state = new_state

    def _on_monitoring(self):
        '''監測狀態'''
        pass
    def _on_solving(self):
        '''解題狀態'''
        pass
    #===
    # 其他
    #===
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
    