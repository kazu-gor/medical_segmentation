import numpy as np

import cv2
import glob
import os

from sklearn.model_selection import train_test_split


###yolov3を学習するためのtxtファイルをlab/yolov3_txt/に保存
###ただしtrainは石灰化を含む画像のみ

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


files1 = glob.glob(r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/TrainDataset/masks/*.png")

with open("/home/student/src2/藤林/プログラム/PraNet-master/lab/yolov3_txt/train.txt", "w", encoding="utf-8") as f:
    for i in files1:
        img = imread(i)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        label = cv2.connectedComponentsWithStats(gray)
        nlabels = label[0]
        if nlabels > 1:
            f.write(os.path.join(r"/home/student/src2/藤林/プログラム/darknet/cfg/task/datasets",
                                 os.path.splitext(os.path.basename(i))[0] + ".jpg") + '\n')

files2 = glob.glob(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/train73/3/masks/*.png")

with open("/home/student/src2/藤林/プログラム/PraNet-master/lab/yolov3_txt/test3.txt", "w", encoding="utf-8") as f:
    for i in files2:
        img = imread(i)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        label = cv2.connectedComponentsWithStats(gray)
        nlabels = label[0]
        if nlabels > 1:
            f.write(os.path.join(r"/home/student/src2/藤林/プログラム/darknet/cfg/task/datasets",
                                 os.path.splitext(os.path.basename(i))[0] + ".jpg") + '\n')

files3 = glob.glob(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/train73/7/masks/*.png")

with open("/home/student/src2/藤林/プログラム/PraNet-master/lab/yolov3_txt/test7.txt", "w", encoding="utf-8") as f:
    for i in files3:
        img = imread(i)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        label = cv2.connectedComponentsWithStats(gray)
        nlabels = label[0]
        if nlabels > 1:
            f.write(os.path.join(r"/home/student/src2/藤林/プログラム/darknet/cfg/task/datasets",
                                 os.path.splitext(os.path.basename(i))[0] + ".jpg") + '\n')



files4 = glob.glob(r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/ValDataset/masks/*.png")

with open("/home/student/src2/藤林/プログラム/PraNet-master/lab/yolov3_txt/val.txt", "w", encoding="utf-8") as f:
    for i in files4:
        f.write(os.path.join(r"/home/student/src2/藤林/プログラム/darknet/cfg/task/datasets",
                             os.path.splitext(os.path.basename(i))[0] + ".jpg") + '\n')

files3 = glob.glob(r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/TestDataset/masks/*.png")

with open("/home/student/src2/藤林/プログラム/PraNet-master/lab/yolov3_txt/test.txt", "w", encoding="utf-8") as f:
    for i in files3:
        f.write(os.path.join(r"/home/student/src2/藤林/プログラム/darknet/cfg/task/datasets",
                             os.path.splitext(os.path.basename(i))[0] + ".jpg") + '\n')
