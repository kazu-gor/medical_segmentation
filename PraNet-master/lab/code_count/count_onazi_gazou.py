import glob
import os

import cv2
import numpy as np
from sklearn.model_selection import train_test_split


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
    r"/home/student/src2/藤林/プログラム/PraNet-master/lab/nana_roku/test/*.png"
)
files2 = glob.glob(
    r"/home/student/src2/藤林/プログラム/PraNet-master/lab/nana_roku/train/*.png"
)
files3 = glob.glob(
    r"/home/student/src2/藤林/プログラム/PraNet-master/lab/nana_roku/val/*.png"
)
files4 = glob.glob(
    r"/home/student/src2/藤林/プログラム/PraNet-master/lab/nana_roku/test7/*.png"
)
files5 = glob.glob(
    r"/home/student/src2/藤林/プログラム/PraNet-master/lab/nana_roku/test3/*.png"
)
count = 0
for i in files2:
    for j in files3:
        if os.path.basename(i) == os.path.basename(j):
            count += 1
print(count)
