import cv2
import numpy as np

#이미지 읽어오기
img = cv2.imread("1st.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, dsize=(640, 480))


#코너 patch 지정 및 저장
for i in range(4):
    x,y,w,h = cv2.selectROI('img', img, False)
    if w and h:
        roi = img[y:y+h, x:x+w]
        cv2.imwrite('./'+'1st_corner'+str(i)+'.jpg', roi)


#Blurring and Computing Gradient
img = cv2.GaussianBlur(img, (5, 5), 0)
img = np.float32(img)/255.0

gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)


cv2.waitKey(0)
cv2.destroyAllWindows()
 
