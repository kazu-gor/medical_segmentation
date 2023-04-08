import numpy as np

import cv2
import glob
import os

from PIL import Image


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


files = glob.glob(r"/home/student/src2/藤林/プログラム/PraNet-master/dataset_matome/dataset_ver6/CD12_前処理済み(リサイズあり)/mask/*.png")
for i in files:
    img = imread(i)
    basename = os.path.basename(i)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (352, 352))
    # 画像の拡張
    imwrite(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/dataset_matome/dataset_ver5/CD12_前処理済み(リサイズあり)/mask", os.path.basename(i)), img)
