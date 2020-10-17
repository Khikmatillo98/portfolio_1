import cv2
import numpy as np 

capture = cv2.VideoCapture(0)
face_cascader = cv2.CascadeClassifier('Cascade_files/haarcascade_frontalface_default.xml')
eye_cascader = cv2.CascadeClassifier('Cascade_files/haarcascade_eye.xml')

while (capture.isOpened()):
    ret, frame = capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = face_cascader.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(40, 40)
    )

    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2 )
        roi_color = frame[y:y+h, x:x+w]
        roi_gray = gray[y:y+h, x:x+w]


        eye = eye_cascader.detectMultiScale(
            roi_gray,
            scaleFactor=1.2,
            minNeighbors=15,
            minSize=(5, 5)
        )
        for (xx, yy, ww, hh) in eye:
            cv2.rectangle(roi_color, (xx, yy), (xx+ww, yy+hh), (0, 255, 0), 2)


        cv2.imshow('video', frame)

    t = cv2.waitKey(0) & 0xFF
    if t == 27:
        break


capture.release()
cv2.destroyAllWindows()
