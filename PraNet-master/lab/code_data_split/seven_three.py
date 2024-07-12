import glob
import os
import shutil

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

#########画像を7:3に分けるtrain73に保存
###SVMの実験するときに使う


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
    "/home/student/src2/藤林/プログラム/PraNet-master/lab/train73/7/images/*.png"
):
    os.remove(file)
for file in glob.glob(
    "/home/student/src2/藤林/プログラム/PraNet-master/lab/train73/7/masks/*.png"
):
    os.remove(file)
for file in glob.glob(
    "/home/student/src2/藤林/プログラム/PraNet-master/lab/train73/3/images/*.png"
):
    os.remove(file)
for file in glob.glob(
    "/home/student/src2/藤林/プログラム/PraNet-master/lab/train73/3/masks/*.png"
):
    os.remove(file)


x = 42  # random_state
files1 = glob.glob(
    r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/TrainDataset/images/*.png"
)
train7, train3 = train_test_split(files1, train_size=0.7, random_state=x)

for i in train7:
    img = imread(i)
    imwrite(
        os.path.join(
            r"/home/student/src2/藤林/プログラム/PraNet-master/lab/train73/7/images",
            os.path.basename(i),
        ),
        img,
    )
    shutil.copy(
        os.path.join(
            r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/TrainDataset/masks",
            os.path.splitext(os.path.basename(i))[0] + ".png",
        ),
        os.path.join(
            r"/home/student/src2/藤林/プログラム/PraNet-master/lab/train73/7/masks",
            os.path.splitext(os.path.basename(i))[0] + ".png",
        ),
    )

for i in train3:
    img = imread(i)
    imwrite(
        os.path.join(
            r"/home/student/src2/藤林/プログラム/PraNet-master/lab/train73/3/images",
            os.path.basename(i),
        ),
        img,
    )
    shutil.copy(
        os.path.join(
            r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/TrainDataset/masks",
            os.path.splitext(os.path.basename(i))[0] + ".png",
        ),
        os.path.join(
            r"/home/student/src2/藤林/プログラム/PraNet-master/lab/train73/3/masks",
            os.path.splitext(os.path.basename(i))[0] + ".png",
        ),
    )
