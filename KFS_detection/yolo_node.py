#!/usr/bin/env python3 

import rclpy 
from rclpy.node import Node 
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
import os
from pathlib import Path


#YOLO and Attribute classifier 
import cv2
import torch
import numpy as np 
from ultralytics import YOLO
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm


class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')

        BASE_DIR = Path(__file__).resolve().parent
        weights_dir = BASE_DIR / "weights"
        detector_path = weights_dir / "Initial_detection.pt"
        authenticator_path = weights_dir / "Authenticity.pt"

        self.main_detector = YOLO(str(detector_path))
        self.second_detector = YOLO(str(authenticator_path))


        self.Main_classes = ["Red_KFS", "Blue_KFS", #boxes
                             "Spear", "Palm", "Fist"] #spear heads
        self.Second_classes = ["r1", "r2", "fake"]

        self.bridge = CvBridge()
        self.sub = self.create_subscription(CompressedImage, "/camera1/image_compressed", self.image_callback, 10)
        self.pub_debug = self.create_publisher(Image, "detection/debug_image", 10)
        self.pub_info = self.create_publisher(String, "debugging/info", 10)

       
    def image_callback(self ,msg):
        
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


        # frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        results = self.main_detector(frame)
        #results = self.main_detector.track(frame, tracker="bytetrack.yaml", conf=0.8)

        info_msgs = []

        for box in results[0].boxes:
            cls_id =int(box.cls[0])
            cls_name = self.Main_classes[cls_id]

            

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            h, w, _ = frame.shape
            x1 , y1 = max(0,x1) , max(0,y1)
            x2 , y2 = min(w,x2) , min(h,y2)

            crop = frame[y1:y2, x1:x2]

            label = cls_name
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0) , 1)

            if cls_name in  ["Red_KFS","Blue_KFS"]:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

                Authentications = self.second_detector(crop_rgb)

                
                if len(Authentications[0].boxes) > 0:
                    best = max(Authentications[0].boxes, key=lambda b: float(b.conf[0]))
                    auth_id = int(best.cls[0])
                    auth_label = self.Second_classes[auth_id]
                else:
                    auth_label = "Unknown"

                cv2.putText(frame, f"{label}:{auth_label}", (x1,y1-10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,255,0), 2 )

            else:
                
                cv2.putText(frame, label, (x1,y1-10), cv2.FONT_HERSHEY_PLAIN, 0.6, (0,255,0), 2 )

            
            
            if cls_name in ["Red_KFS", "Blue_KFS"]:
                info_msgs.append(f"{label}:{auth_label}")
            else:
                info_msgs.append(label)

        img_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
        self.pub_debug.publish(img_msg)

        if info_msgs:
            self.pub_info.publish(String(data="; ".join(info_msgs)))
        
        #if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to quit
        #    self.get_logger().info("Exiting...")
         #   self.cap.release()
          #  cv2.destroyAllWindows()
           # self.timer.cancel()
           # rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    main()