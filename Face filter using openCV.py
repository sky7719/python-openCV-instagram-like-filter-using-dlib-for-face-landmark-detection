# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 01:37:03 2021

@author: shivam kumar
"""

import cv2
import numpy as np
import dlib

def empty(a):
    pass

cv2.namedWindow("BGR")
cv2.resizeWindow("BGR",400,240)
cv2.createTrackbar("Blue","BGR",0,255,empty)
cv2.createTrackbar("Green","BGR",0,255,empty)
cv2.createTrackbar("Red","BGR",0,255,empty)

def cropimg(img,points,scale=5):
    
    mask = np.zeros_like(img)
    mask = cv2.fillPoly(mask,[points],(255,255,255))
    return mask

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture(1)

while True:
    _,img = cap.read()
#     imgColor = np.zeros_like(img)
    if _:
        imgorig = img.copy()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for face in faces:
            x1,y1 = face.left(),face.top()
            x2,y2 = face.right(),face.bottom()
            landmarks = predictor(gray,face)
            myPoints = []
            for n in range(0,68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                myPoints.append([x,y])
            myPoints = np.array(myPoints)
            leye = cropimg(imgorig,myPoints[36:42])
            reye = cropimg(imgorig,myPoints[42:48])
            eyes = cv2.bitwise_or(leye,reye)
            b = cv2.getTrackbarPos("Blue","BGR")
            g = cv2.getTrackbarPos("Green","BGR")
            r = cv2.getTrackbarPos("Red","BGR")
            imgColor = np.zeros_like(eyes)
            imgColor[:] = b,g,r
            imgColor = cv2.bitwise_and(imgColor,eyes)
            imgColor = cv2.GaussianBlur(imgColor,(7,7),10)
            imgColor = cv2.addWeighted(imgorig,0.4,imgColor,1,0)
        cv2.imshow("BGR",imgColor)
        if cv2.waitKey(1)==13:
            break
    else:
        print("Image not found")
cap.release()
cv2.destroyAllWindows()