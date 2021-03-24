import cv2
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# Mouse Click Event(Drawing rectangle and appending coordinates clicked to list)
def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global count_1, count_2
        if param == 1:
            if count_1 >= 4:
                return
            count_1 += 1
            arr = points1
            brr = degrees1
            img = img1
        else:
            if count_2 >= 4:
                return
            count_2 += 1
            arr = points2
            brr = degrees2
            img = img2

        arr.append([x - rect_side // 2, y - rect_side // 2])
        print(arr)
        idx = len(arr) - 1
        x = arr[idx][0]
        y = arr[idx][1]

        roi = img[y: y + rect_side, x: x + rect_side]
        brr.append(compute_gradient(roi))
        cv2.rectangle(img, (x, y), (x + rect_side, y + rect_side), 2)
        cv2.imshow(str(param), img)
        if count_1 + count_2 == 8:
            store_point3()
            show_addedimg()


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
    return merged_arr


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


# MSE 함수 정의
def mean_squared_error(y,t):
    return ((y-t)**2).mean(axis=None)

# 유사도 가장 높은기 배열 찾기
def find_mini(answer,list):
    mini = float('inf')
    mini_total = float('inf')
    np_answer = np.array(answer)
    np_list = np.array(list)

    for i in range(4):
        for j in range(12):
            if mean_squared_error(np_answer,np_list[i]) < mini:
                mini = mean_squared_error(np_answer,np_list[i])
            np_list[i] = np.roll(np_list[i],1)

        if mini < mini_total:
            mini_total = mini
            result = i
    return result



# img1의 각 꼭짓점별 img2의 최소거리 좌표 저장
def store_point3():
    for k in range(4):
        answer = degrees1[k]
        min_index = find_mini(answer,degrees2)
        temp = deepcopy(points2[min_index])
        points3.append((temp))


    #사진 합쳤을때를 위해 x 좌표를 img1 width 만큼 늘려줌
    for m in range(4):
        points3[m][0] = points3[m][0] + 640



# img1과 img2 합치기
def show_addedimg():
    added_img = cv2.hconcat([img1, img2])
    for i in range(4):
        cv2.line(added_img,tuple(points1[i]),tuple(points3[i]), (0, 0, 255), 2)
    cv2.imshow('added', added_img)



points1 = list()
points2 = list()
points3 = list()
degrees1 = list()
degrees2 = list()


rect_side = 16
unit_of_degree = 30
count_1 = 0
count_2 = 0

# Read Image
img1 = cv2.imread('/Users/joon/downloads/1st.jpg')
img1 = cv2.resize(img1, dsize=(640, 480))
img2 = cv2.imread('/Users/joon/downloads/2nd.jpg')
img2 = cv2.resize(img2, dsize=(640, 480))

cv2.namedWindow('1')
cv2.setMouseCallback('1', on_click, 1)
cv2.namedWindow('2')
cv2.setMouseCallback('2', on_click, 2)

cv2.imshow('1', img1)
cv2.imshow('2', img2)

cv2.waitKey(0)
cv2.destroyAllWindows()
