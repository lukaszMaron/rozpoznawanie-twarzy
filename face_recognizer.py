from PIL import Image
from keras.applications.vgg16 import preprocess_input
import cv2
from keras.models import load_model
import numpy as np


model = load_model('trained_face_model.h5')
name = ""
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_cropper(img):
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    
    if isinstance(faces, tuple):
        return None
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0, 255, 180), 5)
        cv2.putText(frame,name, (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)
        face_cropped = img[y:y+h, x:x+w]

    return face_cropped


webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    
    face=face_cropper(frame)
    
    if type(face) is np.ndarray:
        face = cv2.resize(face, (224, 224))
        img = Image.fromarray(face, 'RGB')

        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        print(prediction)
        
        if(prediction[0][0]>0.9):
            name='Andrzej Grabowski'
        elif(prediction[0][1]>0.9):
            name='Krzysztof Krawczyk'
        elif(prediction[0][2]>0.9):
            name='Lukasz'
        elif(prediction[0][3]>0.9):
            name='Robert Maklowicz'
        elif(prediction[0][4]>0.9):
            name='Adam Malysz'
        else:
            name='nie rozpoznano'
        
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()