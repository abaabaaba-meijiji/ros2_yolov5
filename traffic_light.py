
import cv2
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rclpy
import os
import sys
from pathlib import Path
import torch
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from ultralytics.utils.plotting import Annotator, colors
from models.common import DetectMultiBackend
from utils.general import (
    Profile,
    check_img_size,
    cv2,
    non_max_suppression,
    scale_boxes,
)
from utils.torch_utils import select_device
from utils.augmentations import (
    letterbox,
)

import pred_process

class ImageSubscriber(Node):    
	def __init__(self):
		self.Object=[]
		self.device = select_device("")#cuda device, i.e. 0 or 0,1,2,3 or cpu
		self.model = DetectMultiBackend("yolov5s.pt", device=self.device, dnn=False, data="data/coco128.yaml", fp16=False)
		self.conf_thres=0.25,  # confidence threshold
		self.iou_thres=0.45,  # NMS IOU threshold


		self.stride, self.names, pt = self.model.stride, self.model.names, self.model.pt
		self.imgsz = check_img_size((640, 640), s=self.stride)  # check image size
		self.model.warmup(imgsz=(1 if pt or self.model.triton else 1, 3, *self.imgsz))  # warmup
		self.seen, self.windows, self.dt = 0, [], (Profile(device=self.device), Profile(device=self.device), Profile(device=self.device))

		self.classes=None


		# initiate the Node class's constructor and give it a name        
		super().__init__("image_subsriber")        
		# Create the subscriber. This subsciber will receive an image        
		self.subscription = self.create_subscription(Image, "/camera/color/image_raw", self.image_callback, 10)        
		self.subscription # prevent unused variable warning        
		self.br = CvBridge()    


	
	def image_callback(self, data):        
		self.get_logger().info("Receiving realsense frame")        
		current_img = self.br.imgmsg_to_cv2(data)#get img from realsense       
		b,g,r=cv2.split(current_img)
		current_img=cv2.merge([r,g,b])

		im0,pred=self.yolo_detect(current_img)#im0 , pred is the information list[[cls,conf,*xyxy],]

		cv2.imshow("camera", im0 )        
		cv2.waitKey(1)


	def yolo_detect(self,current_img):
		im0s=current_img.copy()
		im0=[current_img.copy(),]
		im=[current_img,]
		#im=np.array(im)
		im = np.stack([letterbox(x, 640, stride=self.stride, auto=True)[0] for x in im0])  # resize
		#print("immmm:",im.shape)
		im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
		im = np.ascontiguousarray(im)  # contiguous
		
		augment=False

		with self.dt[0]:
			#print(im.shape)
			im = torch.from_numpy(np.array(im)).to(self.model.device)
			im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
			im /= 255  # 0 - 255 to 0.0 - 1.0
			if len(im.shape) == 3:
				im = im[None]  # expand for batch dim
			if self.model.xml and im.shape[0] > 1:
				ims = torch.chunk(im, im.shape[0], 0)

        # Inference
		with self.dt[1]:
			visualize = False #increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
			print(im.shape)
			pred = self.model(im, augment=augment, visualize=visualize)
        # NMS
		with self.dt[2]:
			pred = non_max_suppression(pred, 0.25, 0.45, self.classes, False, max_det=1000)

		for i, det in enumerate(pred):
			im0= im0s.copy()

			annotator = Annotator(im0, line_width=3, example=str(self.names))

			if len(det):
                # Rescale boxes from img_size to im0 size
				det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()



                # Write results
				for *xyxy, conf, cls in reversed(det):
					confidence = float(conf)
					confidence_str = f"{confidence:.2f}"
					c = int(cls)  # integer class
					label = f"{self.names[c]} {conf:.2f}"
					annotator.box_label(xyxy, label, color=colors(c, True))
			im0 = annotator.result()
		return im0,pred

def main(args=None):    
	rclpy.init(args=args)    
	yolo_node = ImageSubscriber()    
	# yolo_node.get_logger().info("print inof")   
	rclpy.spin(yolo_node)    
	rclpy.shutdown()
		
if __name__ == "__main__":       
	 main()

