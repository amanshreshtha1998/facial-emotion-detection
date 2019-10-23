# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 10:35:51 2018

@author: RAJDEEP PAL
"""

import cv2

#%%
cap = cv2.VideoCapture(0)

count = 0
ret = True
path = 'F:/year 2/udyam/mosaic/this_will_work/dataset/Aman_dataset/anger/'
while ret:
    
  ret, frame = cap.read()
  cv2.imwrite(path + "frame%d.jpg" % count, frame)     # save frame as JPEG file
  if cv2.waitKey(1) == ord('q') :                     # exit if Escape is hit
      break
  count += 1
  cv2.imshow('frame', frame)
  
cap.release()
cv2.destroyAllWindows()

#%%
cap = cv2.VideoCapture(0)

count = 0
ret = True
path = 'F:/year 2/udyam/mosaic/this_will_work/dataset/Aman_dataset/happy/'
while ret:
    
  ret, frame = cap.read()
  cv2.imwrite(path + "frame%d.jpg" % count, frame)     # save frame as JPEG file
  if cv2.waitKey(1) == ord('q'):                     # exit if Escape is hit
      break
  count += 1
  cv2.imshow('frame', frame)
  
cap.release()
cv2.destroyAllWindows()

#%%
cap = cv2.VideoCapture(0)

count = 0
ret = True
path = 'F:/year 2/udyam/mosaic/this_will_work/dataset/Aman_dataset/sad/'
while ret:
    
  ret, frame = cap.read()
  cv2.imwrite(path + "frame%d.jpg" % count, frame)     # save frame as JPEG file
  if cv2.waitKey(1) == ord('q'):                     # exit if Escape is hit
      break
  count += 1
  cv2.imshow('frame', frame)
  
cap.release()
cv2.destroyAllWindows()
#%%

cap = cv2.VideoCapture(0)

count = 0
ret = True
path = 'F:/year 2/udyam/mosaic/this_will_work/dataset/Aman_dataset/surprise/'
while ret:
    
  ret, frame = cap.read()
  cv2.imwrite(path + "frame%d.jpg" % count, frame)     # save frame as JPEG file
  if cv2.waitKey(1) == ord('q'):                     # exit if Escape is hit
      break
  count += 1
  cv2.imshow('frame', frame)
  
cap.release()
cv2.destroyAllWindows()