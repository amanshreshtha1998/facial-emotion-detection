# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 07:53:28 2018

@author: RAJDEEP PAL
"""
#%%
#Import required modules
import cv2
import dlib

#%%
#Set up some required objects
cap = cv2.VideoCapture(0) #Webcam object
detector = dlib.get_frontal_face_detector() #Face detector
predictor = dlib.shape_predictor('F:/year 2/udyam/mosaic/this_will_work/lib/shape_predictor_68_face_landmarks.dat') #Landmark identifier. Set the filename to whatever you named the downloaded file
ret = True
while ret:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray)

    detections = detector(clahe_image, 1) #Detect the faces in the image

    for k,d in enumerate(detections): #For each detected face
        
        shape = predictor(clahe_image, d) #Get coordinates
        for i in range(1,68): #There are 68 landmark points on each face
            cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2) #For each point, draw a red circle with thickness2 on the original frame

    cv2.imshow("frame", frame) #Display the frame

    if cv2.waitKey(1)  == ord('q'): #Exit program when the user presses 'q'
        break

cap.release()
cv2.destroyAllWindows()
