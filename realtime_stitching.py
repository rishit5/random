from __future__ import print_function
from stitcher import Stitcher
from imutils.video import VideoStream
from imutils.object_detection import non_max_suppression
import numpy as np
import datetime
from imutils import paths 
import time
import cv2
import dlib 

# initialize the video streams and allow them to warmup
print("[INFO] starting cameras...")

#Initializing Objects
leftStream = VideoStream(src=1).start()
rightStream = VideoStream(src=0).start()
time.sleep(2.0)
stitcher = Stitcher()
predictor = None 
faceCascade = cv2.CascadeClassifier('face_cascade.xml')
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def cleanup():
    print("[INFO] cleaning up...")
    cv2.destroyAllWindows()
    leftStream.stop()
    rightStream.stop()

def detect_faces(image):
    gray =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    print("Found faces! ", str(len(faces)))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)


def detect_people(image):
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),padding=(8, 8), scale=1.05)
    # draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        
def rect2bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return x, y, w, h

while True:
    left = leftStream.read()
    right = rightStream.read()
    print(np.shape(left))
    result = stitcher.stitch([left,right])

    # no homograpy could be computed
    if result is None:
        print("[INFO] homography could not be computed")
        break


    # show the output images
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    detect_people(gray)
    cv2.imshow("Result", result)

    cv2.imshow("Left Frame", left)
    cv2.imshow("Right Frame", right)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

#Cleaning Up
cleanup()

