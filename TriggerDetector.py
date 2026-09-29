import cv2
from pathlib import Path
from paddleocr import PaddleOCR
'''
# brief:
觸發偵測啟動yolo過檢測

'''
THRESHOLD = 0.8

class TriggerDetector():
    def __init__(self):
        self.trigger_template_path = Path("img/trigger_img.png")

        if self.trigger_template_path.exists():
            self.trigger_template = cv2.imread(str(self.trigger_template_path))
        else:
            raise FileNotFoundError(f"沒有檢測模板: {self.trigger_template_path.resolve()}")

        # 初始化 PaddleOCR，設定使用繁體中文 (ch) 或中英文混合
        self.ocr = PaddleOCR(use_angle_cls=True, lang='ch')

    def check_lie_detector(self,frame):
        '''啟動偵測

        Returns: True | False

        '''
        # TODO: cv2的方法
        # try:
        #     result = cv2.matchTemplate(frame, self.trigger_template, cv2.TM_CCOEFF_NORMED)
        #     _, max_val, _, _ = cv2.minMaxLoc(result)
        #     if max_val > THRESHOLD:
        #         return True
        #     else:
        #         return False

        # except Exception as e:
        #     print(e)
        #     return

        # TODO: OCR方法

        try:
        
            result = self.ocr.predict (frame, cls=True)

            if not result or not result[0]:
                return False
            # 解析辨識出的文字
            detected_texts = [line[1][0] for line in result[0]]
            
            # 組合所有抓到的文字以便做關鍵字比對
            full_text = "".join(detected_texts)
            print("測試內容捕獲",full_text)

            if "透明的圖形" in full_text or "LIE DETECTOR" in full_text:
                return True


            return False
        except Exception as e:
            print(e)
            return



if __name__ == "__main__":
    trigger = TriggerDetector()
    trigger.run()