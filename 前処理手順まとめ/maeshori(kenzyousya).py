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
###########正解画像に石灰化がないとき真っ黒のマスク画像を作る#############
# 前処理1,前処理2実行後に行う

for file in glob.glob(r"C:/Users/student/PycharmProjects/lab/dataset/前処理3/mask/*.png"):
    os.remove(file)
files = glob.glob("C:/Users/student/PycharmProjects/lab/dataset/前処理2/mask/*.png")
for i in files:
    img = imread(i)
    basename = os.path.basename(i)
    h, w, c = img.shape

    img = np.zeros((h, w, c))

    imwrite(os.path.join("C:/Users/student/PycharmProjects/lab/dataset/前処理3/mask", basename), img)
