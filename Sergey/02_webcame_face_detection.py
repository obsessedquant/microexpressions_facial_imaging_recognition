import cv2
import sys
import os 
import numpy as np
import tensorflow as tf
from keras.models import model_from_json
from keras.preprocessing import image

#get classifier 
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)


#load weights and model 
model = model_from_json(open("cnn_model.json", "r").read())
model.load_weights("cnn_weights.h5")


video_capture = cv2.VideoCapture(0) #start capturing from default 0 (webcam) change 0 to a file name if you want to check emotions on a video

#while true keep capturing frames 
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    #convert to grayscale
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #pass the image scale factor and minneighbors 
    faces = faceCascade.detectMultiScale(
        gray_img,
        scaleFactor=1.1,
        minNeighbors=5,
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h),(0, 0, 255), 1) #change the color and size of the box 
        #area of interest inside the box
        aoi_gray = gray_img[y:y+w,x:x+h]
        #convert aoi to 48X48
        aoi_gray = cv2.resize(aoi_gray,(48,48))
        #Image Processing 
        image_pixels = tf.keras.preprocessing.image.img_to_array(aoi_gray) #convert to array
        image_pixels = np.expand_dims(image_pixels, axis = 0) 
        

        prediction = model.predict(image_pixels)
        max_index = np.argmax(prediction[0]) #take the highest probability

        #Create the emotion array 
        emotion_names = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotions = emotion_names[max_index]
        print(predicted_emotions)

        #print emotion on the frame
        cv2.putText(frame, predicted_emotions, (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX,2, (0,0,255), 3 ) #Change the color of the emotion text 

    #resize 
    img_resize = cv2.resize(frame,(1000,700))

    # Display the resulting frame
    cv2.imshow('Emotion Detection', img_resize)

    #Quit when q button is clicked
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()