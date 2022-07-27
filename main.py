import cv2
import numpy as np
import face_recognition

elon_img = cv2.cvtColor(face_recognition.load_image_file('elon.png'), cv2.COLOR_BGR2RGB)
elon_test = cv2.cvtColor(face_recognition.load_image_file('jackma.png'), cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(elon_img)[0]
encodeElon = face_recognition.face_encodings(elon_img)[0]
cv2.rectangle(elon_img, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(elon_test)[0]
encodeElonTest = face_recognition.face_encodings(elon_test)[0]
cv2.rectangle(elon_test, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeElon], encodeElonTest)
print(results)

cv2.imshow('Elon Musk', elon_img)
cv2.imshow('Elon Test', elon_test)
cv2.waitKey(0)
