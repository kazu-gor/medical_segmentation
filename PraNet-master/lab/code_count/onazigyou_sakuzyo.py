import numpy as np

import cv2
import glob
import os

from sklearn.model_selection import train_test_split
import shutil


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


with open("/home/student/src2/藤林/プログラム/PraNet-master/lab/1/easy_val.txt", "w", encoding='utf-8') as a:
    with open("/home/student/src2/藤林/プログラム/PraNet-master/lab/1/val.txt", "r", encoding='utf-8') as f:
        datalist1 = f.readlines()
        for data1 in datalist1:
            with open("/home/student/src2/藤林/プログラム/PraNet-master/lab/easy_txt/train_easy_nrlr_05.txt", "r",
                      encoding='utf-8') as f2:
                datalist2 = f2.readlines()
                k = 0
                for data2 in datalist2:
                    if data1 == data2:
                        k = 1
            if k == 0:
                a.write(data1)
