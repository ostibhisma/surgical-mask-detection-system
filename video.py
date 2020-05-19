import cv2
from tensorflow.keras.models import load_model
import numpy as np


model = load_model('mask.model')
label_dict = {1:'with mask',0:'without mask'}
color_dict = {1:(0,255,0),0:(0,0,255)}

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")




class VideoCamera(object):
    def __init__(self):
       #capturing video
       self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        #releasing camera
        self.video.release()


    def get_frame(self):
        _,frame = self.video.read()
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(image,1.3,5)
        for (x,y,w,h) in faces:
            face_img = image[x:x+w,y:y+h]
            img = cv2.resize(face_img,(100,100))
            resized = img/255.0
            reshaped = np.reshape(img,(1,100,100,1))
            result = model.predict(reshaped)
            label = np.argmax(result,axis=1)[0]
            cv2.rectangle(frame,(x,y),(x+w,y+h),color_dict[label],2)
            cv2.rectangle(frame,(x,y-40),(x+w,y),color_dict[label],-1)
            cv2.putText(frame,label_dict[label],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
            break
        # encode OpenCV raw frame to jpg and displaying it
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()