from inference import get_roboflow_model
import supervision as sv
import cv2
import time
import queue
import numpy as np
import threading
import re

# INIT
class CPRS() :
  def __init__(self):
    self.PLATE_MODEL = get_roboflow_model(model_id="taiwan-license-plate-recognition-research-tlprr/7",api_key="9d2tqEGBk4q34SiofV0Q")
    self.LETTER_MODEL =get_roboflow_model(model_id="license-bha52-bssnw/2",api_key="9d2tqEGBk4q34SiofV0Q")
    self.INFERENCE_TIME = 100
    self.image_queue = queue.Queue(maxsize=self.INFERENCE_TIME)
    self.car_plate_queue = queue.Queue(maxsize=2)
    self.plate_counts = dict()
    self.image_event = threading.Event()

  def correct_skew(self, image):
      # 旋轉裁切的車牌
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      edges = cv2.Canny(gray, 50, 150, apertureSize=3)
      lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

      if lines is not None:
          angles = []
          for line in lines:
              x1, y1, x2, y2 = line[0]
              angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
              angles.append(angle)

          median_angle = np.median(angles)

          rows, cols = image.shape[:2]
          M = cv2.getRotationMatrix2D((cols / 2, rows / 2), median_angle, 1)
          corrected_image = cv2.warpAffine(image, M, (cols, rows))
          return corrected_image
      else:
          return image # No lines detected, return original image
      
  def INFER(self,image):
    model = self.PLATE_MODEL
    carplate_box_result = model.infer(image)
    detections = sv.Detections.from_inference(carplate_box_result[0].model_dump(by_alias=True, exclude_none=True))
    # create supervision annotators
    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # annotate the image with our inference results
    annotated_image = bounding_box_annotator.annotate(
        scene=image, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)
    detection_label = 0
    for i in range (len(detections.class_id)):
      # Get bounding box coordinates (might be named differently)
      if detections.class_id[i] == 0 and detections.confidence[i] > 0.9 :
        detection_label+=1
        #print(detections.confidence[i])
      # Crop the image based on coordinates
      # upper left corner
        x1 = int(detections.xyxy[i][0])
        y1 = int(detections.xyxy[i][1])
      # lower right corner
        x2 = int(detections.xyxy[i][2])
        y2 = int(detections.xyxy[i][3])    
        #print("Cropping car plate image\n")
        cropped_image = image[y1:y2, x1:x2]
        cropped_image = self.correct_skew(cropped_image)
        #cv2.imshow("cropped", cropped_image)
        #cv2.waitKey(1)
        if not self.image_queue.full():
          self.image_queue.put(cropped_image,block=True)

  def filter_formatted_match(self,string):
    # AAA-1234 OR 123-AAA
    pattern1 = r"^[A-Z]{3}-\d{4}$"
    pattern2 = r"^\d{3}-[A-Z]{3}$"
    if re.match(pattern1, string) or re.match(pattern2, string):
      #print("CAR PLATE NUMBER : " + string + " MATCH ")
      if string in self.plate_counts:
        self.plate_counts[string]+=1
      else:
        self.plate_counts[string]=1
      return True
    #print("CAR PLATE NUMBER : " + string + " DOES NOT MATCH")
    return False

  def predict_frame_letter(self):
    model=self.LETTER_MODEL
    img_num = 1
    self.image_event.wait()
    while not self.image_queue.empty() :
      image = self.image_queue.get()
      hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
      # 取出 V 通道 (亮度)
      v = hsv[:, :, 2]
      # 二值化 V 通道
      ret, thresh = cv2.threshold(v, 127, 255, cv2.THRESH_BINARY)
      # 將二值化後的 V 通道合併成三通道影像
      image = cv2.merge([thresh, thresh, thresh])
      carplate_box_result = model.infer(image)
      detections = sv.Detections.from_inference(carplate_box_result[0].model_dump(by_alias=True, exclude_none=True)) 
      bounding_box_annotator = sv.BoxAnnotator()
      label_annotator = sv.LabelAnnotator()
      # annotate the image with our inference results
      annotated_image = bounding_box_annotator.annotate(
        scene=image, detections=detections)
      annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)
      
      #cv2.imshow("letter crop", annotated_image)
      cv2.waitKey(1)
      cv2.imwrite(f"cropped_letter/cropped_letter_box_{img_num}.jpg",annotated_image)
      img_num +=1
      
      xyxy = detections.xyxy # array of bounding box left corner
      data = detections.data["class_name"]
      # 將數字和座標合併成一個列表
      list_xy = []
      for xy in xyxy :
        list_xy.append(int(xy[0]))
      zipped = zip(list_xy,data) # 合併 字母或數字x座標與對應之字母或數字
      # 根據 x 座標排序
      list_zip = list(zipped)
      #print(f'zipped : {list_zip}')
      # 分離排序後的數字和座標
      car_plate_number = ''.join([l[1] for l in sorted(list_zip, key=lambda x: x[0])])
      self.filter_formatted_match(car_plate_number)
        
    # VOTING CAR PLATE
    if len(self.plate_counts) > 0:
      #print(plate_counts)
      sorted_plate_counts = sorted(self.plate_counts.items(), key=lambda x: x[1],reverse=True)
      #print(sorted_plate_counts)
      voted_plate_number, voted_plate_count = sorted_plate_counts[0]
      print("[CPR_SYS_MSG] VOTE FOR CAR PLATE NUMBER : " + voted_plate_number)
      self.car_plate_queue.put(voted_plate_number)
      self.plate_counts.clear()
    else :
      print("[CPR_SYS_MSG] NOT RECOGNIZE ANY CAR PLATE, PLEASE COME CLOSER")
    self.image_event.clear()  
    
  def process_frame(self,video_path):
    cap = cv2.VideoCapture(video_path)
    input_size = 640
    cap.set(cv2.CAP_PROP_FPS,30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,input_size)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,input_size)
    infer_time = self.INFERENCE_TIME
    while True:
      start_time = time.time()
      ret, frame = cap.read()
      if not ret:
        break
      self.INFER(frame)
      end_time = time.time()
      fps = 1 / (end_time - start_time)
      framefps = "FPS: {:.2f}".format(fps)
      cv2.rectangle(frame, (10,1), (120,20), (0,0,0), -1)
      cv2.putText(frame, framefps, (15,17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255),2)
      cv2.imshow('anotated_frame', frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
      infer_time-=1
      #print(infer_time)
    cap.release()

  def car_plate_result(self):
    if not self.car_plate_queue.empty():
      return True, self.car_plate_queue.get()
    return False, ""
  
def start_predict(path):
  cprs = CPRS()
  cprs.__init__()
  producer_thread = threading.Thread(target=cprs.process_frame,args=(path,))
  consumer_thread = threading.Thread(target=cprs.predict_frame_letter)
  start_time = time.time()
  producer_thread.start()
  consumer_thread.start()
  producer_thread.join()
  cprs.image_event.set() # consumer 等待 producer通知
  consumer_thread.join()
  end_time = time.time()
  total_time = end_time-start_time
  msg = "[CPR_SYS_MSG] Inference time : {:.2f} seconds".format(total_time)
  print(msg)
  return cprs.car_plate_result()
  