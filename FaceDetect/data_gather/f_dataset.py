import cv2
import os 


cap = cv2.VideoCapture(0)
cap.set(3, 700)
cap.set(4, 600)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_id = input('\n enter user id end press <return> ==>  ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")

count = 0
while True:

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        #count += 1

        cv2.imwrite("dataset/User." + str(face_id) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('img', frame)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
 
    if k == 27:
        break
    #elif count >= 30: # Take 30 face sample and stop video
         #break


print("\n [INFO] Exiting Program and cleanup stuff")
cap.release()
cv2.destroyAllWindows()