import cv2
import numpy as np


# Mouse Click Event(Drawing rectangle and appending coordinates clicked to list)
def on_click_1st(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points1.append([x - rect_side // 2, y - rect_side // 2])
        idx = len(points1) - 1
        x = points1[idx][0]
        y = points1[idx][1]

        roi = img1[y: y + rect_side, x: x + rect_side]
        compute_gradient(roi)
        cv2.rectangle(img1, (x, y), (x + rect_side, y + rect_side), 2)
        cv2.imshow('img1', img1)


def on_click_2nd(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points2.append([x - rect_side // 2, y - rect_side // 2])
        idx = len(points2) - 1
        cv2.rectangle(img2, (points2[idx][0], points2[idx][1]),
                      (points2[idx][0] + rect_side, points2[idx][1] + rect_side), 2)
        cv2.imshow('img2', img2)
        print(points2)


def compute_gradient(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (11, 11), 0)
    img = np.float32(img) / 255.0
    tmp = cv2.resize(img, dsize=(500, 500), interpolation=cv2.INTER_AREA)
    cv2.imshow('img', tmp)

    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    print('angle')
    print(angle)
    merge_by_degree(angle)


def merge_by_degree(angle):
    degree_arr = [0 for _ in range(360 // unit_of_degree)]
    for angle_i in angle:
        for a in angle_i:
            degree_arr[int(a) // unit_of_degree] += 1
    print('degree_arr')
    print(degree_arr)
    return degree_arr


points1 = list()
points2 = list()
rect_side = 16
unit_of_degree = 30

# Read Image
img1 = cv2.imread('1st.jpg')
img1 = cv2.resize(img1, dsize=(640, 480))
img2 = cv2.imread('2nd.jpg')
img2 = cv2.resize(img2, dsize=(640, 480))

cv2.namedWindow('img1')
cv2.setMouseCallback('img1', on_click_1st)
cv2.namedWindow('img2')
cv2.setMouseCallback('img2', on_click_2nd)

cv2.imshow('img1', img1)
cv2.imshow('img2', img2)

cv2.waitKey(0)
cv2.destroyAllWindows()
