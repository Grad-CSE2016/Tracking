import cv2
import numpy as np
import imutils
from random import randint
from imutils.object_detection import non_max_suppression

cv2.namedWindow("tracking")
camera = cv2.VideoCapture("t.avi")

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
tracker = cv2.MultiTracker("KCF")

FirstType=False
colors = [(255,0,0),(0,255,0),(0,0,255),
        (255,255,0),(0,255,255),(255,0,255),(0,0,0),(255,255,255),(0,0,0),(120,120,120)]


boxes=  np.array([[0,0,0,0]])
j=0
while camera.isOpened():
    ok, image=camera.read()
    #image = imutils.resize(image, width=min(600, image.shape[1]))
    if FirstType==False:
        # detect people in the image
        (rects, weights) = hog.detectMultiScale(image, winStride=(4,4), padding=(8,8), scale=0.5)
        # draw the original bounding boxes
        for (x, y, w, h) in rects:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            bsbox=(x,y,w,h)
            for newbox in boxes:
                px = int(newbox[0])-x
                py = int(newbox[1])-y
                if(abs(px)>30 and abs(py)>30):
                    tracker.add(image,bsbox)
        FirstType=True
    j=j+1
    if j==20:
        FirstType=False
        print("Reset")
        j=0
    ok, boxes = tracker.update(image)
    #print(boxes)
    COrder=0
    for newbox in boxes:
        if COrder>9:
            COrder=0
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.rectangle(image, p1, p2, colors[COrder])
        COrder=COrder+1
    COrder=0
    print(boxes)
    cv2.imshow("tracking", image)
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break # esc pressed
