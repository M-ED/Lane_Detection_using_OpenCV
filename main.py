import cv2
## In openCV we use imread and imshow function
import numpy as np
import matplotlib.pyplot as plt

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1, y1, x2, y2])
    

def average_slope_intercept(image, lines):
    left_fit=[]
    right_fit=[]
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        ## it is going to fit first degree polynomial to x and y points and return a vector of coefficients which describes a slope of y_intercept.
   
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope<0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])
    

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny  

## Deviation is 0, kernel is 5x5

## The edge corresponds to the region of the image where there is sharp change in image.
## The change in brightness over the series of pixel is gradient.
 
## A function of pixel intensities, in all directions, x and y.
## What canny filter do, it measures adjacent changes in intensity in all directions, x and y. The derivative is small change in intensity and big derivative is big change in intensity.
## The gradient is change in brightness over the series of pixels.
## Area where there is completely black corresponds to low changes in intensity between adjacent pixels.
## Whereas the white line represents a region in the image where there is a high change in intensity, exceeding the threshold.
## cv2.Canny(image, low_threshold, high_threshold)

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image       
     ## each line is a 2D array containing our line coordinated in the form 
     ## [[x1, y1, x2, y2]]. These coordinates specify the line's parameters, as well
     ## as the location of the lines with respect to the image space, ensuring that they are placed in the correct position.       
     

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)
        ]])
    mask = np.zeros_like(image) ## black
    cv2.fillPoly(mask, polygons, 255) ## color of polygon white
    masked_image = cv2.bitwise_and(image, mask)
    ## Computing the bitwise & of both images
    return masked_image

## Finding Lane Lines (Hough Transform)


# image = cv2.imread(r'C:\Users\Fame\Desktop\Projects_2024\Computer_Vision_Intermediate_Projects\Find_Lanes_for_Self-Driving_Cars\test_image.jpg')
# lane_image = np.copy(image)
# canny_image = canny(lane_image)
# cropped_image = region_of_interest(canny_image)
# ## precision of 2 pixel
# ## threshold: minimum number of votes needed to accept a candidate line.
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) 

# averaged_lines = average_slope_intercept(lane_image, lines)
# line_image = display_lines(lane_image, averaged_lines)

# ## multiply all elements into 0.8, 
# combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
# cv2.imshow('result', combo_image)
# #cv2.imshow('result', region_of_interest(canny))
# ## plt.imshow(canny)
# cv2.waitKey(0)

cap = cv2.VideoCapture('test2.mp4')
while(cap.isOpened()):
    _, frame =  cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    ## precision of 2 pixel
    ## threshold: minimum number of votes needed to accept a candidate line.
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) 

    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)

    ## multiply all elements into 0.8, 
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow('result', combo_image)
    cv2.waitKey(1)
    

## 2. Using Canny Edge Detector, we identify sharp changes in intensity in adjacent pixels.