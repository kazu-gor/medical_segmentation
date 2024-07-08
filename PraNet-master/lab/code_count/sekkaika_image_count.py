import glob
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

###石灰化の個数を数える


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
            with open(filename, mode="w+b") as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False


files1 = glob.glob(
    r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/TrainDataset/masks/*.png"
)
# files1 = glob.glob(r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/TestDataset/masks/*.png")
# files1 = glob.glob(r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/ValDataset/masks/*.png")
cnt1 = 0
for i in files1:
    img = imread(i)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    nlabels1, labels1, stats1, centroids1 = cv2.connectedComponentsWithStats(gray)
    if nlabels1 > 1:
        cnt1 += 1

print(cnt1)
