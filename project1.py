import cv2
import numpy as np


# Mouse Click Event(Drawing rectangle and appending coordinates clicked to list)
def on_click_1st(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points1.append([x - rect_side // 2, y - rect_side // 2])
        idx = len(points1) - 1
        cv2.rectangle(img1, (points1[idx][0], points1[idx][1]),
                      (points1[idx][0] + rect_side, points1[idx][1] + rect_side), 2)
        cv2.imshow('img1', img1)
        print(points1)


def on_click_2nd(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points2.append([x - rect_side // 2, y - rect_side // 2])
        idx = len(points2) - 1
        cv2.rectangle(img2, (points2[idx][0], points2[idx][1]),
                      (points2[idx][0] + rect_side, points2[idx][1] + rect_side), 2)
        cv2.imshow('img2', img2)
        print(points2)


points1 = list()
points2 = list()
rect_side = 16

# Read Image
img1 = cv2.imread('1st.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img1 = cv2.resize(img1, dsize=(640, 480))

img2 = cv2.imread('2nd.jpg')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img2 = cv2.resize(img2, dsize=(640, 480))

cv2.namedWindow('img1')
cv2.setMouseCallback('img1', on_click_1st)

cv2.namedWindow('img2')
cv2.setMouseCallback('img2', on_click_2nd)

# Blurring
img1 = cv2.GaussianBlur(img1, (5, 5), 0)
img1 = np.float32(img1) / 255.0

img2 = cv2.GaussianBlur(img2, (5, 5), 0)
img2 = np.float32(img2) / 255.0

cv2.imshow('img1', img1)
cv2.imshow('img2', img2)

# Computing Gradient
gx = cv2.Sobel(img1, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(img1, cv2.CV_32F, 0, 1, ksize=1)
mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

cv2.waitKey(0)
cv2.destroyAllWindows()
