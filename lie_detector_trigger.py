import cv2
from pathlib import Path
'''
# brief:
觸發偵測啟動yolo過檢測

'''
THRESHOLD = 0.8

class Lie_Detector_Trigger():
    def __init__(self):
        self.trigger_template = Path("img/trigger_img.png")

        if self.trigger_template_path.exists():
            self.trigger_template = cv2.imread(str(self.trigger_template_path))
        else:
            raise FileNotFoundError(f"沒有檢測模板: {self.trigger_template_path.resolve()}")

    def check_trigger(self,frame):
        '''
        @ brief: 檢查觸發條件
        @ return: True or False
        '''
        try:
            result = cv2.matchTemplate(frame, self.trigger_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > THRESHOLD:
                return True
            else:
                return False

        except Exception as e:
            print(e)
            return




if __name__ == "__main__":
    trigger = Lie_Detector_Trigger()
    trigger.run()