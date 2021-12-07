from inspect import classify_class_attrs
from flask import Flask, render_template, Response, request
import cv2
import datetime
import time
import os
import sys
import numpy as np
from threading import Thread
import os
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

global capture, rec_frame, grey, switch, neg, face, rec, out
capture = 0
grey = 0
neg = 0
face = 0
switch = 1
rec = 0
model_json = "./saved_model/models/relumodel.json"
model_weights = "./saved_model/models/relumodel.h5"
size = (120, 120)
offset = 0


def classify_age(age):
    if(age < 18):
        return "You are a kid"
    elif(age < 25):
        return "You are a young"
    elif(age < 35):
        return "You are a middle aged"
    else:
        return "You are a old"


# load json and create model
json_file = open(model_json, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(model_weights)
print("Loaded model from disk")

# make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

# Load pretrained face detection model
net = cv2.dnn.readNetFromCaffe('./saved_model/deploy.prototxt.txt',
                               './saved_model/res10_300x300_ssd_iter_140000.caffemodel')

# instatiate flask app
app = Flask(__name__, template_folder='./templates')


camera = cv2.VideoCapture(0)


def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)


def predict_age(image, model):
    im = image * 1./255
    im = np.resize(im, size + (3,))
    im = np.expand_dims(im, axis=0)
    age = model.predict(im)[0][0] - offset
    return "{0}".format(classify_age(age))


def detect_face(frame):
    global net
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]

    if confidence < 0.5:
        return frame

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    try:
        frame = frame[startY:endY, startX:endX]
        (h, w) = frame.shape[:2]
        r = 480 / float(h)
        dim = (int(w * r), 480)
        frame = cv2.resize(frame, dim)
    except Exception as e:
        pass
    return frame


def gen_frames():  # generate frame by frame from camera
    global out, capture, rec_frame
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
        success, frame = camera.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces contours
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for(x, y, w, h) in faces:

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            predicted_age = predict_age(frame[y-50:y+h+50, x-50:x+w+50], model)

            cv2.putText(frame, predicted_age,
                        (x+5, y-5), font, 1, (255, 0, 0), 2)
        frame = cv2.flip(frame, 1)  # Flip vertically

        if success:
            if(face):
                frame = detect_face(frame)
                age = predict_age(frame, model)
                print(age)
            if(grey):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if(neg):
                frame = cv2.bitwise_not(frame)
            if(capture):
                capture = 0
                now = datetime.datetime.now()
                p = os.path.sep.join(
                    ['shots', "shot_{}.png".format(str(now).replace(":", ''))])
                cv2.imwrite(p, frame)

            if(rec):
                rec_frame = frame
                frame = cv2.putText(cv2.flip(
                    frame, 1), "Recording...", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
                frame = cv2.flip(frame, 1)

            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass

        else:
            pass


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global switch, camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture = 1
        elif request.form.get('grey') == 'Grey':
            global grey
            grey = not grey
        elif request.form.get('neg') == 'Negative':
            global neg
            neg = not neg
        elif request.form.get('face') == 'Face Only':
            global face
            face = not face
            if(face):
                time.sleep(4)
        elif request.form.get('stop') == 'Stop/Start':

            if(switch == 1):
                switch = 0
                camera.release()
                cv2.destroyAllWindows()

            else:
                camera = cv2.VideoCapture(0)
                switch = 1
        elif request.form.get('rec') == 'Start/Stop Recording':
            global rec, out
            rec = not rec
            if(rec):
                now = datetime.datetime.now()
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('vid_{}.avi'.format(
                    str(now).replace(":", '')), fourcc, 20.0, (640, 480))
                # Start new thread for recording the video
                thread = Thread(target=record, args=[out, ])
                thread.start()
            elif(rec == False):
                out.release()

    elif request.method == 'GET':
        return render_template('index.html')
    return render_template('index.html')


if __name__ == '__main__':
    app.run()

camera.release()
cv2.destroyAllWindows()
