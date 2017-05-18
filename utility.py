__author__ = 'zhiyong_wang'

import numpy as np
import cv2

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap,solid=False):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """

    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    if lines is None:
        return
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), [255, 0, 0], 1)
    # draw_lines(line_img, lines,solid=solid)



    return line_img

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



def process_image_file(image_file_path):
    img = cv2.imread(image_file_path)
    return process_image(img)

def process_image(img):
    crop_img = img[60:160, 0:320]
    gray_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    blur_image=cv2.GaussianBlur(gray_img, (5, 5), 0)
    edges_image = cv2.Canny(blur_image, 50, 150)
    vertices = np.array([[(0,0),(0,100), (160, 50), (320,100), (320, 0)]], dtype=np.int32)
    region_image = region_of_interest(edges_image,vertices)
    hough_image = hough_lines(region_image,1,1*np.pi/180,50,50,20,False)
    if hough_image is None:
        return None

    final_image = np.reshape(hough_image,(100,320,1))

    return final_image
