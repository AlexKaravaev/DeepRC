# Original code https://github.com/MarkBroerkens/CarND-LaneLines-P1/blob/master/P1.ipynb

import math
import cv2

import numpy as np

global previous_right
global previous_left
previous_right = [-1,-1]
previous_left = [-1,-1]

# parameters
gaussian_kernel_size = 3
canny_low_threshold = 50
canny_high_threshold = 150
hough_rho = 1
hough_theta = np.pi/120 #120
hough_threshold = 20 #20
hough_min_line_len = 5 #15
hough_max_line_gap = 2 #40

thickness = 15
crop_from_top = 20

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            
   
# Find X for point if we know Y1, other point and slope
def findX(slope, y1, x2, y2):
    # slope = (y2 - y1) / (x2 - x1)
    x1 = -(((y2 - y1)/slope) - x2)
    return int(x1)


# heloper that draws the lane line
# Inputs are the sets of X values where the extrapolated lines from the hough transformation cross the top of the
# region of interest and the X values where they cross the lower part of the image.
# From those sets of X values, the median value is calculated in order to identify the X value of the lane line
def draw_line_extended2(img, top_y, top_Xs, bottom_Xs, is_right, color, thickness):

    global previous_right
    global previous_left    
    
    if (len(top_Xs) != 0 and len(bottom_Xs) != 0):
        top_X = int(np.median(top_Xs))
        bottom_X = int(np.median(bottom_Xs))
        cv2.line(img, (bottom_X, img.shape[0]), (top_X, top_y), color, thickness)
        if (is_right):
            previous_right = (bottom_X, top_X)
        else:
            previous_left = (bottom_X, top_X)
    else:
        if (is_right):
            cv2.line(img, (previous_right[0], img.shape[0]), (previous_right[1], top_y), color, thickness)
        else:
            cv2.line(img, (previous_left[0], img.shape[0]), (previous_left[1], top_y), color, thickness)
        
    
    
def draw_lines_extended2(img, lines, color, thickness, crop_from_top):
    right_top_Xs = []
    right_bottom_Xs = []
    left_top_Xs = []
    left_bottom_Xs = []
    
    top_y = int(img.shape[0]/2 + crop_from_top)
    
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y2 - y1)/(x2 - x1)
            
            if (slope != 0):
                top_x = findX(slope, top_y, x2, y2)
                bottom_x = findX(slope, img.shape[0], x1,y1)

                # filter out lines that are most likely not lane lines and assigne the line to the right or left lane line
                if (slope > 0.5 and slope <2):
                    right_top_Xs.append(top_x)
                    right_bottom_Xs.append(bottom_x)
                elif (slope < -0.5 and slope >-2):
                    left_top_Xs.append(top_x)
                    left_bottom_Xs.append(bottom_x)
    
    #draw right line
    draw_line_extended2(img, top_y, right_top_Xs,  right_bottom_Xs, True, color, thickness)
    
    #draw left line
    draw_line_extended2(img, top_y, left_top_Xs,  left_bottom_Xs, False, color, thickness)
    

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, thickness=5, crop_from_top=60):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines_extended2(line_img, lines, [255, 0, 0], thickness, crop_from_top)
    return line_img    
    
    
    
# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)