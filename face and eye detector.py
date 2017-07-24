import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
from pprint import pprint
from copy import deepcopy
import argparse

### cv2 take BGR as input we need ot convert it to RGB before plotting it
def plotBGR2RGB(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.show()
    return(1)

# Given an image matrix and a cascade, detects a face,
# draws a rectangle around it, and returns it
def face_detector(image, cascade="cascades/haarcascade_frontalface_default.xml"):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detector = cv2.CascadeClassifier(cascade)
    rects = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return(image)


# Given an image matrix and a cascade, detects the eyes,
# draws a rectangle around it, and returns it
def eye_detector(image, cascade="cascades/haarcascade_eye.xml"):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detector = cv2.CascadeClassifier(cascade)
    rects = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return(image)

# Given path to an image, executes the face detection pipeline and plots resulting images
def face_plot(imgpath, cascade1,cascade2):
    image_p = cv2.imread(imgpath)
    image_f = deepcopy(image_p)
    image_f = face_detector(image_f, cascade1)
    image_f = eye_detector(image_f, cascade2)
    res = np.hstack((image_p, image_f))
    plt.figure(figsize=(20,10))
    plotBGR2RGB(res)
    return(res)

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True)
    ap.add_argument("-c", "--cascade", required=False)
    ap.add_argument("-o", "--output", required=False)
    args = vars(ap.parse_args())

    if args["cascade"] is None:
        args["cascade1"] = "cascades/haarcascade_frontalface_default.xml"
        args["cascade2"] = "cascades/haarcascade_eye.xml"

        

    img = face_plot(args["image"], args["cascade1"],args["cascade2"])

    if args["output"] is not None:
        cv2.imwrite(args["output"], img)
