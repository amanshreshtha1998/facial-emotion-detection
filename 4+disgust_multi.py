# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 11:04:06 2018

@author: RAJDEEP PAL
"""
#%%
import cv2
import glob
import random
import math
import numpy as np
import dlib
import itertools
from sklearn.svm import SVC
import pandas as pd
import _pickle as cPickle
from sklearn.externals import joblib

#%%
emotions = ["anger", "happy", "sad","surprise", "disgust"] #Emotion list

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('F:/year 2/udyam/mosaic/this_will_work/lib/shape_predictor_68_face_landmarks.dat') #Or set this to whatever you named the downloaded file
#clf = SVC(kernel='linear', probability=True, tol=1e-3)#, verbose = True) #Set the classifier as a support vector machines with polynomial kernel

data = {} #Make dictionary for all values
#data['landmarks_vectorised'] = []

data_direc = 'F:/year 2/udyam/mosaic/this_will_work/dataset/4_emotions/'


#%%
def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
 
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)
#%%
def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob(data_direc + '%s/*' % emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, prediction


def get_landmarks(image):
    detections = detector(image, 1)
    all_faces = []
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(1,68): #Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
            
        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        xcentral = [(x-xmean) for x in xlist]
        ycentral = [(y-ymean) for y in ylist]

        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
            meannp = np.asarray((ymean,xmean))
            coornp = np.asarray((z,w))
            dist = np.linalg.norm(coornp-meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(y, x)*360)/(2*math.pi))

        data['landmarks_vectorised'] = landmarks_vectorised
        all_faces.append(landmarks_vectorised)
    if len(detections) < 1: 
        data['landmarks_vestorised'] = "error"
        #return "error"
    #return landmarks_vectorised
    return (detections, all_faces)


def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        print(" working on %s" %emotion)
        training, prediction = get_files(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                training_data.append(data['landmarks_vectorised']) #append image array to training data list
                training_labels.append(emotions.index(emotion))
    
        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                prediction_data.append(data['landmarks_vectorised'])
                prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels   
#%%
clf = SVC(kernel = 'linear', gamma = 0.01, C = 0.0001, probability=True, tol=1e-3)
accur_lin = []
for i in range(0,1):
    print("Making sets %s" %i) #Make sets
    training_data, training_labels, prediction_data, prediction_labels = make_sets()

    npar_train = np.array(training_data) #numpy array for the classifier
    npar_trainlabs = np.array(training_labels)
    print("training SVM linear %s" %i) #train SVM
    clf.fit(npar_train, training_labels)

    print("getting accuracies %s" %i) #accuracy
    npar_pred = np.array(prediction_data)
    pred_lin = clf.score(npar_pred, prediction_labels)
    print ("linear: ", pred_lin)
    accur_lin.append(pred_lin) #Store accuracy in a list

print("Mean value lin svm: %s" %np.mean(accur_lin)) # mean accuracy 
print (accur_lin)
print (clf.classes_)
print (npar_train.shape)
#print (data)
#print (clf.predict_proba(npar_pred))
#%%                      
                               """ HYPER PARAMETER TUNING """

training_data, training_labels, prediction_data, prediction_labels = make_sets()

X_train = pd.DataFrame(training_data)
y_train = pd.DataFrame(training_labels)
X_test = pd.DataFrame(prediction_data)
y_test = pd.DataFrame(prediction_labels)

X = X_train.append(X_test, ignore_index = True)
y = y_train.append(y_test, ignore_index = True)


print (X_train.shape, X_test.shape, X.shape)
#%%                            
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score

param_range = {'kernel' : ['linear', 'rbf'], 'gamma' :np.logspace(-2, 2, 10), 'C' : np.logspace(-5,0,5)}
grid_clf = GridSearchCV(clf, param_grid = param_range, scoring = 'accuracy'
                        )
grid_clf.fit(X, y)


print (grid_clf.best_params_)
print (grid_clf.best_score_)
#%%

clf = SVC(kernel = 'linear', gamma = 0.01, C = 0.0031622776601683794, probability=True, tol=1e-3)
clf.fit(X, y)

#%%
#%%
path = 'F:/year 2/udyam/mosaic/this_will_work/classifiers/filename'
with open(filename, 'wb') as fid:
    cPickle.dump(clf, fid)    
#%%
# load it again
with open(path, 'rb') as fid:
    clf = cPickle.load(fid)

print (clf)


#%%
'gamma' = 0.01 , 'kernel' = linear, 'C' = 0.000177
scoring = accuracy .91
#%%
'gamma' = 0.001 , 'kernel' = linear
scoring = accracy .90
#%%
{'C': 0.00017782794100389227, 'gamma': 100.0, 'kernel': 'linear'}
scoring = neg_log_loss   -0.28531888998837884
#%%
{'C': 0.00017782794100389227, 'gamma': 0.01, 'kernel': 'linear'}
scoring = precision_macro   0.8791267334549655
#%%
{'C': 0.00017782794100389227, 'gamma': 0.01, 'kernel': 'linear'}
scoring = precision_micro   0.9144144144144144
#%%
{'C': 0.00017782794100389227, 'gamma': 0.01, 'kernel': 'linear'}
scoring = precision_weighted   0.9155896883600868
#%%

                                """ LIVE TESTING """
                                
from statistics import mode

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input 
                               
                                
# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)                               
face_cascade = cv2.CascadeClassifier('F:/year 2/udyam/mosaic/emotion_detection2/models/haarcascade_frontalface_default.xml')




#%%
cap = cv2.VideoCapture(0)
ret = True
emotion_window = []


while ret:
    
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert to grayscale
    clahe_image = clahe.apply(gray)
    rects, all_faces = get_landmarks(clahe_image)
    
    if (len(all_faces) == 0):
        continue
    
    #print (emotion_prediction, emotion_text)
    #print (emotion_text)
    
    """ putting on screen """
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    

    for rect, x_test in zip (rects, all_faces):
        
     #   if (i >= len(faces)):
     #       break
        
    #if data['landmarks_vectorised'] == "error":
    #    print("no face detected on this one")
    #    continue
    #else:
    #    x_test=data['landmarks_vectorised'] #append image array to training data list
        #training_labels.append(emotions.index(emotion))

        #try:
        #    x_test=data['landmarks_vectorised']
        #except:
        #    continue
        
        #print (x_test.shape)
        x_test = np.array( x_test ).reshape(1, 268)
    #print (x_test[0, :5])
        emotion_prediction = clf.predict_proba(x_test)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotions[emotion_label_arg]
        emotion_window.append(emotion_text)
        
        
        
        
        face_coordinates = rect_to_bb(rect)
        #print (emotion_prediction, emotion_text)
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        
        
        #if len(emotion_window) > frame_window:
         #   emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue
        
        
        if (emotion_text == 'anger' ):
            color = emotion_probability * np.asarray((255, 0, 0))  # RED
            #print ('anger')
        elif (emotion_text == 'sad'):
            color = emotion_probability * np.asarray((0, 0, 255))  # BLUE
        elif (emotion_text == 'happy'):
            color = emotion_probability * np.asarray((255, 255, 0)) # YELLOW
        elif (emotion_text == 'surprise'):
            color = emotion_probability * np.asarray((0, 255, 255)) # CYAN
        else:
            color = emotion_probability * np.asarray((0, 255, 0))  # GREEN
            #print ('else')

        color = color.astype(int)
        color = color.tolist()
        
        draw_bounding_box(face_coordinates, rgb_image, color)
        
        
        color = color / emotion_probability
        meter = emotion_probability * 1000
        emotion_text = emotion_text + str(int (meter))
        draw_text(face_coordinates, rgb_image, emotion_text
                  ,
                  color, 0, -45, 1, 1)
        
        
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) == ord('q'):
        break

        

cap.release()
cv2.destroyAllWindows()


#%%
print (x_test.shape)
x_test = x_text.reshape(1, 268)
print (x_test.shape)







#%%
print (1)










