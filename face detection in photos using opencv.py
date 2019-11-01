import cv2
#read input image
img=cv2.imread('Google.jpg')
#Use a trained classifier for face detection
face_classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#The classifier used takes as input gray images so we need to convert our image to gray
gray_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#The next step is to detect the faces inside our image
faces=face_classifier.detectMultiScale(gray_img,1.1,4)
for (x, y, w, h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),4)

cv2.imshow('img',img)
cv2.waitKey()