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

############前処理を行うコード####################
###########右と左に分けてリサイズした画像を作る#############

for file in glob.glob(r"C:/Users/student/pythonProject6/前処理2/image/*.png"):
    os.remove(file)
for file in glob.glob(r"C:/Users/student/pythonProject6/前処理2/mask/*.png"):
    os.remove(file)
# # 前処理2のimage, maskからpngファイルを削除し，中身を掃除

files = glob.glob("C:/Users/student/pythonProject6/前処理1/image/*.png")
for i in files:
    img = imread(i)
    basename = os.path.basename(i)
    mask = imread("C:/Users/student/pythonProject6/前処理1/mask/" + basename)  ##############
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)  ##################
    h, w = img.shape
    img1 = img[:, :w // 2]
    img2 = img[:, w // 2:]
    mask1 = mask[:, :w // 2]  ###############
    mask2 = mask[:, w // 2:]  ###################

    h, w = img1.shape
    img1 = img1[:, :int(w - 0.375 * (w - h))]
    img2 = img2[:, int(0.375 * (w - h)):]
    mask1 = mask1[:, :int(w - 0.375 * (w - h))]  ##################
    mask2 = mask2[:, int(0.375 * (w - h)):]  ####################

    # img1 = cv2.resize(img1, (416, 416))
    # img2 = cv2.resize(img2, (416, 416))
    # mask1 = cv2.resize(mask1, (416, 416))  ################
    # mask2 = cv2.resize(mask2, (416, 416))  ####################
    img1 = cv2.resize(img1, (352, 352))
    img2 = cv2.resize(img2, (352, 352))
    mask1 = cv2.resize(mask1, (352, 352))  ################
    mask2 = cv2.resize(mask2, (352, 352))  ####################

    ## 前処理2のimage,maskにリサイズ後の画像を出力 ##
    imwrite(os.path.join("C:/Users/student/pythonProject6/前処理2/image", "left_" + basename), img1)
    imwrite(os.path.join("C:/Users/student/pythonProject6/前処理2/mask", "left_" + basename), mask1)  #####

    imwrite(os.path.join("C:/Users/student/pythonProject6/前処理2/image", "right_" + basename), img2)
    imwrite(os.path.join("C:/Users/student/pythonProject6/前処理2/mask", "right_" + basename), mask2)  #####

