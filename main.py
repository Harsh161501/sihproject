import cv2 as cv

face_case=cv.CascadeClassifier("vidimg/haarcascade_frontalface_default.xml")
img=cv.VideoCapture(0)

while True:
    success,face=img.read()
    img_grey = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
    faces=face_case.detectMultiScale(img_grey,scaleFactor=1.5,minNeighbors=5)
    for x,y,w,h in faces:
        cv.rectangle(face,(x,y),(x+w,y+h),(0,255,0),1)
    cv.imshow("face",face)
    if cv.waitKey(1) & 0xFF==ord('q'):
        break

