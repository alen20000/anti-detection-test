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
WINDOW_X_0 = 577
WINDOW_Y_0 = 672
WINDOW_X_1 = 800
WINDOW_Y_1 = 700
DETCTION_RANGE = (WINDOW_X_0, WINDOW_Y_0, WINDOW_X_1, WINDOW_Y_1)
# ====== 時間計時常數相關 
CHECK_INTERVAL = 1 # 偵查頻率(秒)
SOLVING_TIMEOUT = 30 # 解題狀態持續時間
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
        self.last_monitoring_time = time.time() #監測狀態使用
        self._on_solving_start_time = None # 解題狀態使用:紀錄開始時間

        # State
        self.state = State.MONITORING

    #====
    # 狀態組
    #====
    def _change_state(self, new_state):
        '''切換狀態
        '''
        print(f"狀態切換: {self.state} -> {new_state}")
        self.state = new_state
        
        if self.state == State.SOLVING:
            self.solving_start_time = time.monotonic() # 紀錄解題開始時間

        elif self.state == State.MONITORING:
            self.last_check_time = time.monotonic() # 更新上次檢測時間(其實影響也不大 頂多多跑一次OC檢測)

    def _on_monitoring(self, current_frame):
        '''監測狀態
        
        若 is_detecting 為 True 則轉狀態至解題
        args:
            current_frame: 畫面
        note:
            is_detecting: bool
        '''
        
        
        if time.time() - self.last_monitoring_time > CHECK_INTERVAL:  # 計時器
            print("監測狀態中...")
            self.last_monitoring_time = time.time()
            is_detecting = self.TriggerDetector.check_lie_detector(current_frame)

            if is_detecting:

                print("檢測到人物正在被測謊!!!")
                self._change_state(State.SOLVING)
            if not is_detecting:
                print("Null")

    def _on_solving(self):
        '''解題狀態
        想法: 解題狀態用時間來結束先預設大概30秒，
        '''
        # 
        elapsed = time.monotonic() - self.solving_start_time
        if elapsed > SOLVING_TIMEOUT:
            print("解題狀態結束")

            self._change_state(State.MONITORING) # 轉狀態:解題結束轉回監測
        print("解題狀態中...")



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

                # 抓畫面
                current_frame = self._capture_screen()

                if current_frame is not None:
                    cv2.imshow("DISPLAY", current_frame)

                # 防呆
                if current_frame is None:
                    continue

                #依狀態判斷

                if self.state == State.MONITORING:
                    self._on_monitoring(current_frame)
                elif self.state == State.SOLVING:
                    self._on_solving()


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
    