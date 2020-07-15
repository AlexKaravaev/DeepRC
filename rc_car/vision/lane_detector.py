import numpy as np
import cv2

from rc_car.vision.utils import *
from abc import ABCMeta, abstractmethod


class LaneDetector(metaclass=ABCMeta):
    
    @abstractmethod
    def detect_lanes(self, image: np.ndarray)->np.ndarray:
        pass
    

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
        vertices = np.array([[(0+imshape[1]/20,imshape[0]),(imshape[1]/2-20, imshape[0]/2+crop_from_top),
                              (imshape[1]/2+20, imshape[0]/2+crop_from_top), (imshape[1]-imshape[1]/2,imshape[0])]], dtype=np.int32)
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