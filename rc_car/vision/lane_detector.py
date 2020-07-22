import numpy as np
import cv2
import tensorflow as tf
import os.path as ops
import torch
import scipy.special
import torchvision.transforms as transforms

from PIL import Image

from rc_car.vision.utils import *
from abc import ABCMeta, abstractmethod




from rc_car.vision.uf_detector.model.model import parsingNet
from rc_car.vision.uf_detector.utils.common import merge_config


class LaneDetector(metaclass=ABCMeta):
    
    @abstractmethod
    def detect_lanes(self, image: np.ndarray)->np.ndarray:
        pass
    
class UFNetLaneDetector(LaneDetector):

    def __init__(self, weights_path:str):
        cls_num_per_lane = 18
        backbone = '18'
        self.griding_num = 200
        self.net = net = parsingNet(pretrained = False, backbone=backbone,cls_dim = (self.griding_num+1,cls_num_per_lane,4),
                    use_aux=False).cuda()
        state_dict = torch.load(weights_path, map_location='cpu')['model']
        compatible_state_dict = {}
        for k, v in state_dict.items():
            if 'module.' in k:
                compatible_state_dict[k[7:]] = v
            else:
                compatible_state_dict[k] = v
        self.net.load_state_dict(state_dict)
        self.net.eval()
        self.img_transform = transforms.Compose([
                                #transforms.Resize((288, 800)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                            ])
    
    def _postprocess(self, orig_img: np.ndarray, detections: np.ndarray)->np.ndarray:
        col_sample = np.linspace(0, 800-1, self.griding_num)
        col_sample_w = col_sample[1] - col_sample[0]

        out_j = detections[0].data.cpu().numpy()
        out_j = out_j[:, ::-1, :]
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
        idx = np.arange(self.griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == self.griding_num] = 0
        out_j = loc
            
        vis = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2BGR)
        lanes = []
        last_y = -1

        for i in range(out_j.shape[1]):
            if np.sum(out_j[:, i] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        ppp = (int(out_j[k, i] * col_sample_w * 800 / 800) - 1, int(288 - k * 10) - 1)
                        x, y = ppp
                        if abs(y - last_y) > 11:
                            lanes.append([ppp])
                        last_y = y
                        #print(ppp)
                        lanes[-1].append(ppp)
                        cv2.circle(vis,ppp,5,(0,255,0),-1)

        for i,lane in enumerate(lanes):
            print(f"Lane {i}")
            print(lane)

        return vis


    def detect_lanes(self, img: np.ndarray)->np.ndarray:
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        orig_img = img
        img = self.img_transform(img).resize(1,3,288,800)

        with torch.no_grad():
            out = self.net(img.cuda())
        return self._postprocess(orig_img, out)
    


class SimpleLaneDetector(LaneDetector):
    """ Taken from CarND Udacity course. Simple as it can be """
     
    def __init__(self, image_size = (160,120,3)):
        self.image_size = (160,120,3)
    
    def detect_lanes(self, image: np.ndarray)->np.ndarray:
        
        imshape = image.shape
               
        # convert to grayscale
        greyscale_image = grayscale(image)
    
        # apply blur filter
        blured_image = gaussian_blur(greyscale_image, gaussian_kernel_size)
    
        # canny edge detection
        canny_image = canny(blured_image, canny_low_threshold, canny_high_threshold)

        #return cv2.cvtColor(canny_image,cv2.COLOR_GRAY2RGB)
        # mask the detected edges
        imshape = canny_image.shape
        vertices = np.array([[(0+imshape[1]/20, imshape[0]),
                              (imshape[1]-20, imshape[0]+crop_from_top),
                              (imshape[1]+20, imshape[0]+crop_from_top),
                              (imshape[1]-imshape[1]/20,imshape[0])]], dtype=np.int32)
        masked_image = region_of_interest(canny_image,vertices)
        return cv2.cvtColor(masked_image, cv2.COLOR_GRAY2RGB)
    
        # hough transformation
        hough_image = hough_lines(masked_image, 
                                hough_rho, 
                                hough_theta, 
                                hough_threshold, 
                                hough_min_line_len, 
                                hough_max_line_gap,
                                thickness, 
                                crop_from_top)

        result = weighted_img(hough_image, image, 0.9, 0.9)
 
        return result 