
from imutils import paths
import numpy as np
import shutil
import imutils
import pickle
import cv2
import os
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import os
#print(os.listdir())
#from FaсeDetector.main_fase import *


class Fase(object):
   def __init__(self):
        detetor ='face_detection_model'
        em_model = 'openface_nn4.small2.v1.t7'
        rec = 'output/recognizer.pickle'
        lee ='output/le.pickle'
 
        print("[INFO] loading face detector...")
        protoPath = os.path.sep.join([detetor, "deploy.prototxt"])
        modelPath = os.path.sep.join([detetor,"res10_300x300_ssd_iter_140000.caffemodel"])
        self.detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
        print("[INFO] loading face recognizer...")
        self.embedder = cv2.dnn.readNetFromTorch(em_model)

 
        self.recognizer = pickle.loads(open(rec, "rb").read())
        self.le = pickle.loads(open(lee, "rb").read())
        print('fase detector activare')
   def update(self):
        build()
        train()
   def main(self,img):
    frame = img
    frame = imutils.resize(frame, width=1000)
    (h, w) = frame.shape[:2]
    
    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    self.detector.setInput(imageBlob)
    detections = self.detector.forward()
    inframe = []
    obj ={}
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > 0.2:
            # compute the (x, y)-coordinates of the bounding box for
            # the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                (96, 96), (0, 0, 0), swapRB=True, crop=False)
            self.embedder.setInput(faceBlob)
            vec = self.embedder.forward()

            # perform classification to recognize the face
            preds = self.recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = self.le.classes_[j]
           
            # draw the bounding box of the face along with the
            # associated probability
            if proba > 0.2:
              
              obj['name']= name
              obj['ver'] = proba
              obj['kord']=[startX,startY,endX,endY]
              inframe.append(obj)
            else:
              name = 'unknown'
              
              text = "{}: {:.2f}%".format(name, proba * 100)
              y = startY - 10 if startY - 10 > 10 else startY + 10
              #cv2.rectangle(frame, (startX, startY), (endX, endY),
              #    (0, 0, 255), 2)
              #cv2.putText(frame, text, (startX, y),
              #    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
              obj['name']= name
              obj['ver'] = proba
              inframe.append(obj)
              obj['kord']=[startX,startY,endX,endY]
              
            #self.vizual(inframe,frame) 
   
    
    #print('next_frane')
    # update the FPS counter
    #fps.update()
    #print(inframe)
    return [frame,inframe]
   def vizual(self,inframe,frame):
       for i in range(len(inframe)):
              text = "{}: {:.2f}%".format(inframe[i]['name'], inframe[i]['ver'] * 100)
             
              y = inframe[i]['kord'][1] - 10 
              cv2.rectangle(frame, (inframe[i]['kord'][0], inframe[i]['kord'][1]), (inframe[i]['kord'][2], inframe[i]['kord'][3]),
                  (0, 0, 255), 2)
              cv2.putText(frame, text, (inframe[i]['kord'][0], y),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
       return frame
   
class drow(object):
 def drowzone(self,frame,x,y,x1,y1):
       output = frame.copy()
       '''mas = os.listdir('ico')
       for i in range(len(mas)):
           mas[i]=cv2.imread('ico/'+str(mas[i]),cv2.IMREAD_UNCHANGED)
       img2 =cv2.imread('1.png',cv2.IMREAD_UNCHANGED)'''
       h , w = frame.shape[:2]
       
       cv2.rectangle(frame, (0, 0), (w, y), (128, 128, 128), -1) #верх
       
       cv2.rectangle(frame, (0, y), (x, y1), (128, 128, 128), -1) #лево центр
       cv2.rectangle(frame, (x1, y), (w, y1), (128, 128, 128), -1) #правао центр
       
       cv2.rectangle(frame, (0, y1), (w, h), (128, 128, 128), -1) #низ
       
       cv2.addWeighted(frame, 0.7, output, 1 - 0.7, 0, output)
       
       #rez = imgmasdrow(output,mas)
       return output
 def drowimgs(self,background, overlays):
    x = 125 
    y = 5
    for i in range(len(overlays)):
        background = self.drowimg(background, overlays[i],x,y)
        y+=90
    return background
 def drowimg(self,background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background

#img2 = cv2.imread('1.png',cv2.IMREAD_UNCHANGED)
'''f =Fase()
vs = VideoStream(src=0).start()
time.sleep(2.0)
while True:
    frame=vs.read()
    frame = f.main(frame)
    cv2.imshow("f",frame[0])
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break'''