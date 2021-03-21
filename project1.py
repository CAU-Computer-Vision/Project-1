import cv2
import numpy as np

#이미지 읽어오기
img = cv2.imread("1st.jpg", cv2.INTER_AREA)
img = cv2.resize(img, dsize=(640, 480), interpolation=cv2.INTER_AREA)

#ROI
for i in range(4):
    x,y,w,h = cv2.selectROI('img', img, False)
    if w and h:
        roi = img[y:y+h, x:x+w]
        cv2.imwrite('./'+'1st_corner'+str(i)+'.jpg', roi)   # ROI 영역만 파일로 저장

#아래 코드 한 줄을 통해 가우시안 블러를 미리 넣어줄 수 있습니다.
img = cv2.GaussianBlur(img, (11, 11), 0)

#소벨, 라플라스, 캐니 필터를 적용시킵니다.
#sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
#sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)
laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=5)

#여러 필터 처리된 결과물을 보여줍니다.
cv2.imshow("Image", img)
#cv2.imshow("Sobelx", sobelx)
#cv2.imshow("Sobely", sobely)
cv2.imshow("Laplacian", laplacian)

cv2.waitKey(0)
cv2.destroyAllWindows()
