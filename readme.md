# Experiment Note

* 訓練basic_model : yolo_8
* 訓練樣本: 一種圖形的三張標註圖片
* 訓練步數: epochs=50,batch=16
* [NOTE] <br>cv2 + 光流 + 卡爾曼濾波 : False; 卡爾曼濾波是慣性預測，隨機運動抓不到；光流會累積誤差，幾次後就飄了<br>yolo+ ByteTrack: nullptr ;<br> yolo + OC-SORT: 效果比較好，ocsort有追朔修正(失聯前一幀)，多目標時以運動慣性做短追蹤+分辨

# Experiment Img
<table>
  <tr>
    <td align="center">
      <img src="./assets/log_img.png" width="400">
      <br>
      <em></em>
    </td>
  </tr>
</table>