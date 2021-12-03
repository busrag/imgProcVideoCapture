import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
while True:

    _,image = cap.read()

    gauss = np.random.normal(0, 1, image.size)
    gauss = gauss.reshape(image.shape[0], image.shape[1], image.shape[2]).astype('uint8')

    img_gauss = cv2.add(image, gauss)

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    yuzler = face_cascade.detectMultiScale(image_gray, 2.3, 5)


    for x, y, width, height in yuzler:
        cv2.rectangle(image, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=2)
        cv2.imshow("image", image)
        if cv2.waitKey(1) == ord("q"):
            break

print("görüntüde tespit edilen yüz sayisi=")
cap.release()
cv2.destroyAllWindows()
