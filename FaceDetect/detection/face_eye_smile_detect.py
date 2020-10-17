import cv2

video = cv2.VideoCapture(0)
f_cascace = cv2.CascadeClassifier('Cascade_files/haarcascade_frontalface_default.xml')
e_cascade = cv2.CascadeClassifier('Cascade_files/haarcascade_eye.xml')
s_cascade = cv2.CascadeClassifier('Cascade_files/haarcascade_smile.xml')

while (video.isOpened()):

    ret, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = f_cascace.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(40, 40)
    )

    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_color = frame[y:y+h, x:x+w]
        roi_gray = gray[y:y+h, x:x+w]

        eye = e_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.5,
            minNeighbors=15,
            minSize=(5, 5)
        )
        for (xx, yy, ww, hh) in eye:
            cv2.rectangle(roi_color, (xx, yy), (xx+ww, yy+hh), (0, 0, 255), 2)

        smile = s_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.5,
            minNeighbors=15,
            minSize=(25, 25)
        )

        for (xxx, yyy, www, hhh) in smile:
            cv2.rectangle(roi_color, (xxx, yyy), (xxx+www, yyy+hhh), (0, 255, 0), 2)

        cv2.imshow('video', frame)


        
    if cv2.waitKey(30) & 0xFF == ord('q'):

        break 

video.release()
cv2.destroyAllWindows()
