import cv2
from pathlib import Path
from paddleocr import PaddleOCR
'''
# brief:
觸發偵測啟動yolo過檢測

'''
THRESHOLD = 0.8
TARGET_TEXTS = ["透明的圖形"]
class TriggerDetector():
    def __init__(self):
        self.trigger_template_path = Path("img/trigger_img.png")

        if self.trigger_template_path.exists():
            self.trigger_template = cv2.imread(str(self.trigger_template_path))
        else:
            raise FileNotFoundError(f"沒有檢測模板: {self.trigger_template_path.resolve()}")

        # 初始化 PaddleOCR，設定使用繁體中文
        self.ocr = PaddleOCR(

            lang='chinese_cht',
            text_detection_model_name="PP-OCRv5_mobile_det", # 偵測模型
            text_recognition_model_name="PP-OCRv5_mobile_rec", # 辨識模型
            use_doc_orientation_classify=False, # 判斷整張圖是否被旋轉: True or False
            use_doc_unwarping=False,    # 判斷文字彎曲扭曲: True or False
            use_textline_orientation=False, # 判斷文字上下顛倒: True or False

            )

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

            result = self.ocr.predict (frame)

            # 防呆
            if not result or not result[0]:
                print('沒有東西')
                return False

            # 匹配
            for res in result:
                text = res["rec_texts"]
                
                # 直接拼成一個長文本作為目標
                full_text = "".join(text)

                for target in TARGET_TEXTS:
                    if target in full_text:
                        return True
                    

            return False
        except Exception as e:
            print(u"發生錯誤:", e)
            return



# if __name__ == "__main__":
#     trigger = TriggerDetector()
#     trigger.run()