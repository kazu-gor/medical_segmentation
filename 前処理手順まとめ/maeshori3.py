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
###########正解画像の塗りつぶしからマスク画像を作る#############


for file in glob.glob(r"C:/Users/student/pythonProject6/前処理3/mask/*.png"):
    os.remove(file) ###########pngファイルを削除

files = glob.glob("C:/Users/student/pythonProject6/前処理/mask/*.jpg")
for i in files:
    img = imread(i)
    basename = os.path.splitext(os.path.basename(i))[0]
    h, w, c = img.shape

    for i in range(h):
        for j in range(w):
            if img[i][j][0] < 100 and img[i][j][1] > 150 and img[i][j][2] < 100:#########BGR#######
                img[i][j] = 255
                # img[i][j][2] = 255
            else:
                img[i][j] = 0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    imwrite(os.path.join("C:/Users/student/pythonProject6/前処理3/mask", basename + ".png"), img)

