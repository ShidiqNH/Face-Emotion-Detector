import numpy as np
import cv2
import time
from keras_preprocessing import image
from tensorflow.keras.models import model_from_json

# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the trained emotion recognition model
model = model_from_json(open('facial_expression_model_structure.json').read())
model.load_weights('facial_expression_model_weights.h5')

emotions = ('Marah', 'Jijik', 'Takut', 'Senang', 'Sedih', 'Terkejut', 'Biasa')

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        detected_face = img[y:y + h, x:x + w]
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
        detected_face = cv2.resize(detected_face, (48, 48))
        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255
        
        # Get predictions for each emotion category
        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        print("Max Index:", max_index)
        emotion = emotions[max_index]
        
        # Display the predicted emotion on the face
        cv2.putText(img, emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Display the image with predicted emotions
    cv2.imshow('img', img)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
