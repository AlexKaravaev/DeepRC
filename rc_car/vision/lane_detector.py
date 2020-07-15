import numpy as np
import cv2
import tensorflow as tf
import os.path as ops


from rc_car.vision.utils import *
from abc import ABCMeta, abstractmethod


from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess
from local_utils.config_utils import parse_config_utils
from local_utils.log_util import init_logger


CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix='lanenet_test')


class LaneDetector(metaclass=ABCMeta):
    
    @abstractmethod
    def detect_lanes(self, image: np.ndarray)->np.ndarray:
        pass
    

class LaneNetLaneDetector(LaneDetector):
    
    def __init__(self, weights_path:str):
        
        self.input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

        self.net = lanenet.LaneNet(phase='test')
        self.binary_seg_ret, self.instance_seg_ret = self.net.inference(input_tensor=self.input_tensor, name='LaneNet')

        self.postprocessor = lanenet_postprocess.LaneNetPostProcessor()
            # Set sess configuration
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
        sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
        sess_config.gpu_options.allocator_type = 'BFC'

        self.sess = tf.Session(config=sess_config)
          # define moving average version of the learned variables for eval
        with tf.variable_scope(name_or_scope='moving_avg'):
            variable_averages = tf.train.ExponentialMovingAverage(
                CFG.SOLVER.MOVING_AVE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()

        # define saver
        self.saver = tf.train.Saver(variables_to_restore)
    
        self.saver.restore(sess = self.sess, save_path = weights_path)
    
    def detect_lanes(self, image: np.ndarray)->np.ndarray:
        
        image_vis = cv2.resize(image, (512,256), interpolation=cv2.INTER_LINEAR)
        image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
        image = image / 127.5 - 1.0
        binary_seg_image, instance_seg_image = self.sess.run(
                [self.binary_seg_ret, self.instance_seg_ret],
                feed_dict={self.input_tensor: [image]}
            )
        postprocess_result = self.postprocessor.postprocess(
            binary_seg_result=binary_seg_image[0],
            instance_seg_result=instance_seg_image[0],
            source_image=image_vis
        )
        mask_image = postprocess_result['mask_image']
        for i in range(CFG.MODEL.EMBEDDING_FEATS_DIMS):
            instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
        embedding_image = np.array(instance_seg_image[0], np.uint8)[:, :, (0, 1, 2)]
        
        return weighted_img(mask_image, image_vis)
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
        print(vertices)
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