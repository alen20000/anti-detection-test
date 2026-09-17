# Experiment Note

* 訓練basic_model : yolo_8
* 訓練樣本: 一種圖形的三張標註圖片
* 訓練步數: epochs=50,batch=16
* [NOTE] <br>cv2 + 光流 + 卡爾曼濾波 : False; 卡爾曼濾波是慣性預測，隨機運動抓不到；光流會累積誤差，幾次後就飄了<br>yolo+ ByteTrack: nullptr ;<br> yolo + OC-SORT: nullptr

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