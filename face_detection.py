
import cv2


file = "haarcascade_frontalface_default.xml"
classifier = cv2.CascadeClassifier(file)

cam =cv2.VideoCapture(0)
while True:
    _,img = cam.read()
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = classifier.detectMultiScale(grayimg, 1.3, 4)

    for (x, y, w, h) in face:
        cv2.rectangle(img,(x,y), (x+w,y+h), (0,255,0), 2)
    cv2.imshow("face detections",img)
    key = cv2.waitKey(10)
    if key == 27:
        break


cam.release()
cam.destroyAllWindows()


    

