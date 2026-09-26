
import cv2
import numpy as np

from PIL import ImageGrab
'''
$brief 測試 檢測是否測謊、若測謊則進入解題

'''
WINDOW_X_0 = 0
WINDOW_X_1 = 800
WINDOW_Y_0 = 0
WINDOW_Y_1 = 600 

detection_range = (WINDOW_X_0, WINDOW_Y_0, WINDOW_X_1, WINDOW_Y_1)


def capture_screen():
    '''抓指定區域畫面'''

    current_frame_res = ImageGrab.grab(detection_range)
    if current_frame_res is None:
        return None
    current_frame_np = np.array(current_frame_res)
    current_frame_bgr = cv2.cvtColor(current_frame_np, cv2.COLOR_BGR2RGB)

    return current_frame_bgr

def run():
    # init
    current_frame = None
    try:
        while True:
            current_frame = capture_screen()
            if current_frame is not None:
                cv2.imshow("DISPLAY", current_frame)


            # EXIT
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(e)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
    