import urllib.request
import cv2
from flask import Flask, jsonify, request
import face_recognition

app = Flask(__name__)


@app.route('/')
def welcome():
    return 'Server is Running.' \
           'The endpoint to verify face is /authenticate and pass links for both images as' \
           'image and testImage in params'


@app.route('/authenticate', methods=['POST'])
def check_face():
    with app.app_context():
        image1 = urllib.request.urlopen(request.args.get('image'))
        image2 = urllib.request.urlopen(request.args.get('testImage'))
        img = cv2.cvtColor(face_recognition.load_image_file(image1), cv2.COLOR_BGR2RGB)
        img_test = cv2.cvtColor(face_recognition.load_image_file(image2), cv2.COLOR_BGR2RGB)
        encodeElon = face_recognition.face_encodings(img)[0]
        encodeElonTest = face_recognition.face_encodings(img_test)[0]
        results = face_recognition.compare_faces([encodeElon], encodeElonTest)
        response = {"result": bool(results[0])}
        return jsonify(response)
