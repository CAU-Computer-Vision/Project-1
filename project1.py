import cv2
import numpy as np
import matplotlib.pyplot as plt


# Mouse Click Event(Drawing rectangle and appending coordinates clicked to list)
def on_click_1st(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if param == 1:
            arr = points1
            img = img1
        else:
            arr = points2
            img = img2
        arr.append([x - rect_side // 2, y - rect_side // 2])
        idx = len(arr) - 1
        x = arr[idx][0]
        y = arr[idx][1]

        roi = img[y: y + rect_side, x: x + rect_side]
        compute_gradient(roi)
        cv2.rectangle(img, (x, y), (x + rect_side, y + rect_side), 2)
        cv2.imshow(str(param), img)


def compute_gradient(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (11, 11), 0)
    img = np.float32(img) / 255.0
    tmp = cv2.resize(img, dsize=(500, 500), interpolation=cv2.INTER_AREA)
    cv2.imshow('img', tmp)

    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    merged_arr = merge_by_degree(angle)
    x_line = [i * unit_of_degree for i in range(360 // unit_of_degree)]
    plt.xticks(x_line)
    plt.hist(merged_arr, x_line)
    plt.show()


def merge_by_degree(angle):
    global unit_of_degree
    merged_angle = []
    angle = angle // unit_of_degree * unit_of_degree
    for angle_i in angle:
        for a in angle_i:
            merged_angle.append(a)
    return merged_angle


def get_degree_arr(angle):
    degree_arr = [0 for _ in range(360 // unit_of_degree)]
    for angle_i in angle:
        for a in angle_i:
            degree_arr[int(a) // unit_of_degree] += 1
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

cv2.namedWindow('1')
cv2.setMouseCallback('1', on_click_1st, 1)
cv2.namedWindow('2')
cv2.setMouseCallback('2', on_click_1st, 2)

cv2.imshow('1', img1)
cv2.imshow('2', img2)

cv2.waitKey(0)
cv2.destroyAllWindows()
