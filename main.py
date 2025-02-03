import os
from urllib.request import Request, urlopen

import cv2
import numpy as np
from flask import Flask, render_template, Response, request, redirect, flash
from tensorflow.keras.models import model_from_json, Sequential
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

with open("model/model_architecture.json", "r") as json_file:
    json_model = json_file.read()

model = model_from_json(json_model, custom_objects={"Sequential": Sequential})
model.load_weights('model/model_weights.h5')

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def Start():
    """ Renders the Home Page """
    return render_template('index.html')


camera = cv2.VideoCapture(0)


def gen_frames():
    while True:
        success, test_img = camera.read()
        if not success:
            break
        else:
            gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
            faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
            for (x, y, w, h) in faces_detected:
                roi_gray = gray_img[y:y + w, x:x + h]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                img_pixels = image.img_to_array(roi_gray)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels /= 255
                predictions = model.predict(img_pixels)
                emotions = ("Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise")
                predicted_emotion = emotions[np.argmax(predictions)]
                rec_col = {"Happy": (0, 255, 0), "Sad": (255, 0, 0), "Surprise": (255, 204, 55), "Angry": (0, 0, 255),
                           "Disgust": (230, 159, 0), "Neutral": (0, 255, 255), "Fear": (128, 0, 128)}
                Text = str(predicted_emotion)
                cv2.rectangle(test_img, (x, y), (x + w, y + h), rec_col[str(predicted_emotion)], 2)
                cv2.rectangle(test_img, (x, y - 40), (x + w, y), rec_col[str(predicted_emotion)], -1)
                cv2.putText(test_img, Text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            resized_img = cv2.resize(test_img, (1000, 700))
            ret, buffer = cv2.imencode('.jpg', test_img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/RealTime', methods=['POST'])
def RealTime():
    """ Video streaming """
    return render_template('real_time.html')


def Emotion_Analysis(img):
    """ It does prediction of Emotions found in the Image provided, saves as Images and returns them """
    path = "static/" + str(img)
    image = cv2.imread(path)
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_haar_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    if len(faces) == 0:
        return [img]
    for (x, y, w, h) in faces:
        roi = gray_frame[y:y + h, x:x + w]
        roi = cv2.resize(roi, (48, 48))
        prediction = model.predict(roi[np.newaxis, :, :, np.newaxis])
        EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
        rec_col = {"Happy": (0, 255, 0), "Sad": (255, 0, 0), "Surprise": (255, 204, 55), "Angry": (0, 0, 255),
                   "Disgust": (230, 159, 0), "Neutral": (0, 255, 255), "Fear": (128, 0, 128)}
        pred_emotion = EMOTIONS_LIST[np.argmax(prediction)]
        Text = str(pred_emotion)
        cv2.rectangle(image, (x, y), (x + w, y + h), rec_col[str(pred_emotion)], 2)
        cv2.rectangle(image, (x, y - 40), (x + w, y), rec_col[str(pred_emotion)], 2)
        cv2.putText(image, Text, (x, y - 10), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 2)
        path = "static/" + "pred" + str(img)
        cv2.imwrite(path, image)
    return ([img, "pred" + img, pred_emotion])


def allowed_file(filename):
    """ Checks the file format when file is uploaded"""
    return ('.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)


@app.route('/ImageUpload', methods=['POST'])
def ImageUpload():
    """ Manual Uploading of Images via URL or Upload """
    return render_template('image.html')


@app.route('/UrlUpload', methods=['POST'])
def UrlUpload():
    """ Manual Uploading of Images via URL or Upload """
    return render_template('url.html')


@app.route('/uploadimage', methods=['POST'])
def uploadimage():
    """ Loads Image from System, does Emotion Analysis & renders."""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            result = Emotion_Analysis(filename)
            if len(result) == 1:
                return render_template('no_prediction.html', orig=result[0])
            return render_template('prediction.html', orig=result[0], pred=result[1])


@app.route('/imageurl', methods=['POST'])
def imageurl():
    """ Fetches Image from URL Provided, does Emotion Analysis & renders."""
    url = request.form['url']
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(req).read()
    arr = np.asarray(bytearray(webpage), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    save_to = "static/"
    cv2.imwrite(save_to + "url.jpg", img)
    result = Emotion_Analysis("url.jpg")
    if len(result) == 1:
        return render_template('no_prediction.html', orig=result[0])
    return render_template('prediction.html', orig=result[0], pred=result[1])


if __name__ == '__main__':
    app.run()
