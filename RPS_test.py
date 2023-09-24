import cv2
import numpy as np
from keras.models import model_from_json


RPS_dict = {0: "Paper", 1: "Rock", 2: "Scissors"}

# load json and create model
json_file = open('RPS_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
RPS_model = model_from_json(loaded_model_json)

# load weights into new model
RPS_model.load_weights("RPS_model.h5")
print("Loaded model from disk")

# start the webcam feed
cap = cv2.VideoCapture(0)

# pass here your video path
#cap = cv2.VideoCapture("D:\\Akshay\\BITS\\Sample_videos\\emotion_sample6.mp4")

while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
        
    # Create a list to hold cascade classifiers
    cascade_list = []

# Load each cascade classifier
    cascade1 = cv2.CascadeClassifier('haarcascades/Hand_haar_cascade.xml')
    cascade_list.append(cascade1)

    cascade2 = cv2.CascadeClassifier('haarcascades/palm.xml')
    cascade_list.append(cascade2)
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    for cascade in cascade_list:
            num_faces = cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in num_faces:
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
                roi_gray_frame = gray_frame[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
                RPS_prediction = RPS_model.predict(cropped_img)
                maxindex = int(np.argmax(RPS_prediction))
                cv2.putText(frame, RPS_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('RPS Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
cap.release()
cv2.destroyAllWindows()





#    face_detector = cv2.CascadeClassifier('haarcascades/Hand_haar_cascade.xml')
#    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#    # detect faces available on camera
#    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
#
#    # take each face available on the camera and Preprocess it
#    for (x, y, w, h) in num_faces:
#        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
#        roi_gray_frame = gray_frame[y:y + h, x:x + w]
#        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
#
#        # predict RPS
#        RPS_prediction = RPS_model.predict(cropped_img)
#        maxindex = int(np.argmax(RPS_prediction))
#        cv2.putText(frame, RPS_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#
#    cv2.imshow('RPS Detection', frame)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
#
#cap.release()
#cv2.destroyAllWindows()
