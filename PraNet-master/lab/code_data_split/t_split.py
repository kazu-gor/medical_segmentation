import glob
import os
import shutil

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

###txtファイルからtrain,test,valを分ける


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


for file in glob.glob(
    "/home/student/src2/藤林/プログラム/PraNet-master/dataset/TestDataset/images/*.png"
):
    os.remove(file)
for file in glob.glob(
    "/home/student/src2/藤林/プログラム/PraNet-master/dataset/TestDataset/masks/*.png"
):
    os.remove(file)
for file in glob.glob(
    "/home/student/src2/藤林/プログラム/PraNet-master/dataset/TrainDataset/images/*.png"
):
    os.remove(file)
for file in glob.glob(
    "/home/student/src2/藤林/プログラム/PraNet-master/dataset/TrainDataset/masks/*.png"
):
    os.remove(file)
for file in glob.glob(
    "/home/student/src2/藤林/プログラム/PraNet-master/lab/test_img/*.png"
):
    os.remove(file)
for file in glob.glob(
    "/home/student/src2/藤林/プログラム/PraNet-master/lab/train_img/*.png"
):
    os.remove(file)
for file in glob.glob(
    "/home/student/src2/藤林/プログラム/PraNet-master/lab/test_mask/*.png"
):
    os.remove(file)
for file in glob.glob(
    "/home/student/src2/藤林/プログラム/PraNet-master/lab/train_mask/*.png"
):
    os.remove(file)
for file in glob.glob(
    "/home/student/src2/藤林/プログラム/PraNet-master/dataset/ValDataset/images/*.png"
):
    os.remove(file)
for file in glob.glob(
    "/home/student/src2/藤林/プログラム/PraNet-master/dataset/ValDataset/masks/*.png"
):
    os.remove(file)

with open(
    "/home/student/src2/藤林/プログラム/PraNet-master/lab/1/test3.txt",
    "r",
    encoding="utf-8",
) as f:
    datalist = f.readlines()
    for data in datalist:
        data = data.rstrip("\n")
        shutil.copy(
            os.path.join(
                r"/home/student/src2/藤林/プログラム/PraNet-master/lab/images",
                data + ".png",
            ),
            os.path.join(
                r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/TestDataset/images",
                data + ".png",
            ),
        )
        shutil.copy(
            os.path.join(
                r"/home/student/src2/藤林/プログラム/PraNet-master/lab/masks",
                data + ".png",
            ),
            os.path.join(
                r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/TestDataset/masks",
                data + ".png",
            ),
        )
with open(
    "/home/student/src2/藤林/プログラム/PraNet-master/lab/1/train.txt",
    "r",
    encoding="utf-8",
) as f:
    # with open("/home/student/src2/藤林/プログラム/PraNet-master/lab/easy_txt/train_easy_nrlr_05.txt", "r", encoding='utf-8') as f:
    datalist = f.readlines()
    for data in datalist:
        data = data.rstrip("\n")
        shutil.copy(
            os.path.join(
                r"/home/student/src2/藤林/プログラム/PraNet-master/lab/images",
                data + ".png",
            ),
            os.path.join(
                r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/TrainDataset/images",
                data + ".png",
            ),
        )
        shutil.copy(
            os.path.join(
                r"/home/student/src2/藤林/プログラム/PraNet-master/lab/images",
                data + ".png",
            ),
            os.path.join(
                r"/home/student/src2/藤林/プログラム/PraNet-master/lab/train_img",
                data + ".png",
            ),
        )
        shutil.copy(
            os.path.join(
                r"/home/student/src2/藤林/プログラム/PraNet-master/lab/masks",
                data + ".png",
            ),
            os.path.join(
                r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/TrainDataset/masks",
                data + ".png",
            ),
        )
        shutil.copy(
            os.path.join(
                r"/home/student/src2/藤林/プログラム/PraNet-master/lab/masks",
                data + ".png",
            ),
            os.path.join(
                r"/home/student/src2/藤林/プログラム/PraNet-master/lab/train_mask",
                data + ".png",
            ),
        )
with open(
    "/home/student/src2/藤林/プログラム/PraNet-master/lab/1/val.txt",
    "r",
    encoding="utf-8",
) as f:
    datalist = f.readlines()
    for data in datalist:
        data = data.rstrip("\n")
        shutil.copy(
            os.path.join(
                r"/home/student/src2/藤林/プログラム/PraNet-master/lab/images",
                data + ".png",
            ),
            os.path.join(
                r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/ValDataset/images",
                data + ".png",
            ),
        )
        shutil.copy(
            os.path.join(
                r"/home/student/src2/藤林/プログラム/PraNet-master/lab/masks",
                data + ".png",
            ),
            os.path.join(
                r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/ValDataset/masks",
                data + ".png",
            ),
        )
