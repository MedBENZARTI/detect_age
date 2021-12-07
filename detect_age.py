import cv2
import numpy as np
import os
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from model import predict_age

# load json and create model
json_file = open('regression_model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("regression_model1.h5")
print("Loaded model from disk")


# def predict_age(image, model):
#     #   im = plt.imread(path)
#     # im = image.img_to_array(image)
#     im = image * 1./255
#     im = np.resize(im, (120, 120, 3))
#     im = np.expand_dims(im, axis=0)
#     return "{:.0f}".format(model.predict(im)[0][0] - 10)


# detect face
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX


camera_number = 0
cam = cv2.VideoCapture(camera_number + cv2.CAP_DSHOW)
# this picks the LARGEST image possible
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)


# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:

    ret, img = cam.read()
    img = cv2.flip(img, 1)  # Flip vertically

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect faces contours
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for(x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        predicted_age = predict_age(img[y-50:y+h+50, x-50:x+w+50], model)
        # confidence = recognizer.predict(gray[y-50:y+h+50,x-50:x+w+50])

        cv2.putText(img, predicted_age,
                    (x+5, y-5), font, 1, (255, 0, 0), 2)
        # cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)

    cv2.imshow('camera', img)

    k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
