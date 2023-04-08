import numpy as np

import cv2
import glob
import os

from sklearn.model_selection import train_test_split


###SSDを学習するためのtxtファイルをlab/ssd_txt/に保存


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


files1 = glob.glob(r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/TrainDataset/images/*.png")

with open("/home/student/src2/藤林/プログラム/PraNet-master/lab/ssd_txt/train.txt", "w", encoding="utf-8") as f:
    for i in files1:
        f.write(os.path.splitext(os.path.basename(i))[0] + '\n')

files2 = glob.glob(r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/ValDataset/images/*.png")

with open("/home/student/src2/藤林/プログラム/PraNet-master/lab/ssd_txt/val.txt", "w", encoding="utf-8") as f:
    for i in files2:
        f.write(os.path.splitext(os.path.basename(i))[0] + '\n')

files3 = glob.glob(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/train73/3/images/*.png")

with open("/home/student/src2/藤林/プログラム/PraNet-master/lab/ssd_txt/test3.txt", "w", encoding="utf-8") as f:
    for i in files3:
        f.write(os.path.splitext(os.path.basename(i))[0] + '\n')

files4 = glob.glob(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/train73/7/images/*.png")

with open("/home/student/src2/藤林/プログラム/PraNet-master/lab/ssd_txt/test7.txt", "w", encoding="utf-8") as f:
    for i in files4:
        f.write(os.path.splitext(os.path.basename(i))[0] + '\n')


files5 = glob.glob(r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/TestDataset/images/*.png")

with open("/home/student/src2/藤林/プログラム/PraNet-master/lab/ssd_txt/test.txt", "w", encoding="utf-8") as f:
    for i in files5:
        f.write(os.path.splitext(os.path.basename(i))[0] + '\n')
