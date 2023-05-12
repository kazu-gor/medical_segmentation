import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob
import os

import math


def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None


def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False


###################トリミングを行うコード#############


# for file in glob.glob(r"C:/Users/student/PycharmProjects/lab/dataset/CD12切り抜き/*.png"):
#     os.remove(file)
# for file in glob.glob(r"C:/Users/student/PycharmProjects/lab/dataset/CD13_14切り抜き/*.png"):
#     os.remove(file)

# files = glob.glob(r"C:/Users/student/PycharmProjects/lab/歯科パノラマ/CD13_14/*.jpg")################

# files = glob.glob(r"C:/Users/student/PycharmProjects/lab/歯科パノラマ/CD12(健常者データ）/*.jpg")  ################
files = glob.glob(r"C:/Users/student/PycharmProjects/lab/歯科パノラマ/CD13_14/**/*.jpg", recursive=True)##############

for i in files:###################
    img = imread(i)
    basename = os.path.splitext(os.path.basename(i))[0]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    img = img[math.floor(h * 1 / 5) - 1:, :]
    h, w = img.shape

    h_list = []
    for height in range(h):
        img_h = img[height, :]
        img_list = [img_h[:w // 5], img_h[w // 5:int(w / 5 * 2)], img_h[int(w / 5 * 2):int(w / 5 * 3)],
                    img_h[int(w / 5 * 3):int(w / 5 * 4)], img_h[int(w / 5 * 4):]]
        count = 0
        for img_w in img_list:
            if np.std(img_w) < 5:
                count += 1
        if count >= 3:
            h_list.append(height)
        else:
            break
    img = np.delete(img, h_list, 0)
    h, w = img.shape
    w_list = []

    for width in range(w):
        img_w = img[:, width]
        img_list = [img_w[:h // 5], img_w[h // 5:int(h / 5 * 2)], img_w[int(h / 5 * 2):int(h / 5 * 3)],
                    img_w[int(h / 5 * 3):int(h / 5 * 4)], img_w[int(h / 5 * 4):]]
        count = 0
        for img_h in img_list:
            if np.std(img_h) < 5:
                count += 1
        if count >= 3:
            w_list.append(width)
        else:
            break

    for width in reversed(range(w)):
        img_w = img[:, width]
        img_list = [img_w[:h // 5], img_w[h // 5:int(h / 5 * 2)], img_w[int(h / 5 * 2):int(h / 5 * 3)],
                    img_w[int(h / 5 * 3):int(h / 5 * 4)], img_w[int(h / 5 * 4):]]
        count = 0
        for img_h in img_list:
            if np.std(img_h) < 5:
                count += 1
        if count >= 3:
            w_list.append(width)
        else:
            break
    img = np.delete(img, w_list, 1)

    # imwrite(os.path.join("C:/Users/student/PycharmProjects/lab/dataset/CD12切り抜き/", basename + ".png"), img)  ############
    imwrite(os.path.join("C:/Users/student/PycharmProjects/lab/dataset/CD13_14切り抜き/", basename + ".png"), img)  ############
