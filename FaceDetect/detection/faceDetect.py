import cv2

face_cascader = cv2.CascadeClassifier('Cascade_files/haarcascade_frontalface_default.xml')
video = cv2.VideoCapture(0)

while (video.isOpened()):

    ret, duplicate = video.read()
    gray = cv2.cvtColor(duplicate, cv2.COLOR_BGR2GRAY)

    face = face_cascader.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(40, 40)

    )
    for (x, y, w, h) in face:
        cv2.rectangle(duplicate, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_color = duplicate[y:y+h, x:x+w]
        roi_gray = gray[y:y+h, x:x+w]


        cv2.imshow('face_detect', duplicate)
    l = cv2.waitKey(0) & 0xFF
    if l == 27:
        break 


video.release()
cv2.destroyAllWindows()

