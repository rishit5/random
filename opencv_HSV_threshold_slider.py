import cv2
import sys
import dlib
import numpy as np

# Starting values for HSV Threshold
H_MAX = 110
H_MIN = 94
S_MAX = 256
S_MIN = 122
V_MAX = 256
V_MIN = 45

# Defining the kernel for the morphological transformations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
# TrackBar Callback Methods
def h_max_val_changed(x):
    global H_MAX
    H_MAX = x
def h_min_val_changed(x):
    global H_MIN
    H_MIN = x
def s_max_val_changed(x):
    global S_MAX
    S_MAX = x
def s_min_val_changed(x):
    global S_MIN
    S_MIN = x
def v_max_val_changed(x):
    global V_MAX
    V_MAX = x
def v_min_val_changed(x):
    global V_MIN
    V_MIN = x


# Creating a Named Window
cv2.namedWindow("params")

# Creating TrackBars for threshold values on the Window
cv2.createTrackbar('H_max', 'params', 0, 179, h_max_val_changed)
cv2.createTrackbar('H_min', 'params', 0, 179, h_min_val_changed)
cv2.createTrackbar('S_max', 'params', 0, 256, s_max_val_changed)
cv2.createTrackbar('S_min', 'params', 0, 256, s_min_val_changed)
cv2.createTrackbar('V_max', 'params', 0, 256, v_max_val_changed)
cv2.createTrackbar('V_min', 'params', 0, 256, v_min_val_changed)

cam = cv2.VideoCapture(0)
while True:
    ret, img = cam.read()

    img = cv2.flip(img, 1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([H_MIN, S_MIN, V_MIN])
    upper_bound = np.array([H_MAX, S_MAX, V_MAX])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    pre_mask = mask
    #Operations on the Binary Mask to reomve noise and highlight the foreground
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel=kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel=kernel)
    mask = cv2.GaussianBlur(mask, (3,3), 0)

    #Working with contours and isolating the largest contour
    mask_c = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel = kernel)
    contours, hierarchy = cv2.findContours(mask_c, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = 0
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) >= cv2.contourArea(contours[largest_contour]):
            largest_contour = i
    cv2.drawContours(mask_c, contours, largest_contour,(255, 255, 255), thickness=-200)
    M = cv2.moments(mask_c, True)

    output = cv2.bitwise_and(img, img, mask=mask)  # Applying mask to Original image
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.circle(output, (cx, cy),  6, (0, 0, 255), -1)
        cv2.circle(img, (cx, cy), 6, (0, 0, 255), -1)

    #Displaying the images
    cv2.imshow("Original", img)
    #cv2.imshow("HSV", hsv)
    cv2.imshow("Binary  Mask - Before Processing", pre_mask)
    cv2.imshow("Binary Mask- After Processing", mask_c)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()

