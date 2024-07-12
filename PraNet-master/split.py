import glob
import os

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

###/PraNet-master/dataset/TestDataset/imagesと
###/PraNet-master/lab/test_imgに
###学習、検証、テストをそれぞれ分ける


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
    "/home/student/研究/プログラム/PraNet-master/dataset/TestDataset/images/*.png"
):
    os.remove(file)
for file in glob.glob(
    "/home/student/研究/プログラム/PraNet-master/dataset/TestDataset/masks/*.png"
):
    os.remove(file)
for file in glob.glob(
    "/home/student/研究/プログラム/PraNet-master/dataset/TrainDataset/images/*.png"
):
    os.remove(file)
for file in glob.glob(
    "/home/student/研究/プログラム/PraNet-master/dataset/TrainDataset/masks/*.png"
):
    os.remove(file)
for file in glob.glob(
    "/home/student/研究/プログラム/PraNet-master/dataset/ValDataset/images/*.png"
):
    os.remove(file)
for file in glob.glob(
    "/home/student/研究/プログラム/PraNet-master/dataset/ValDataset/masks/*.png"
):
    os.remove(file)

x = 42  # random_state
files1 = glob.glob(r"/home/student/研究/プログラム/PraNet-master/dataset/images/*.png")
# files1 = glob.glob(r"/home/student/研究/プログラム/PraNet-master/lab/sekkai_images/*.png")
train0, test = train_test_split(files1, train_size=0.8, random_state=x)
train, val = train_test_split(train0, train_size=0.9, random_state=x)

for i in test:
    img = imread(i)
    basename = os.path.basename(i)
    img2 = imread(
        os.path.join(
            r"/home/student/研究/プログラム/PraNet-master/dataset/masks", basename
        )
    )
    imwrite(
        os.path.join(
            r"/home/student/研究/プログラム/PraNet-master/dataset/TestDataset/images",
            basename,
        ),
        img,
    )
    imwrite(
        os.path.join(
            r"/home/student/研究/プログラム/PraNet-master/dataset/TestDataset/masks",
            basename,
        ),
        img2,
    )

for i in train:
    img = imread(i)
    basename = os.path.basename(i)
    img2 = imread(
        os.path.join(
            r"/home/student/研究/プログラム/PraNet-master/dataset/masks", basename
        )
    )
    imwrite(
        os.path.join(
            r"/home/student/研究/プログラム/PraNet-master/dataset/TrainDataset/images",
            os.path.basename(i),
        ),
        img,
    )
    imwrite(
        os.path.join(
            r"/home/student/研究/プログラム/PraNet-master/dataset/TrainDataset/masks",
            basename,
        ),
        img2,
    )

for i in val:
    img = imread(i)
    basename = os.path.basename(i)
    img2 = imread(
        os.path.join(
            r"/home/student/研究/プログラム/PraNet-master/dataset/masks", basename
        )
    )
    imwrite(
        os.path.join(
            r"/home/student/研究/プログラム/PraNet-master/dataset/ValDataset/images",
            os.path.basename(i),
        ),
        img,
    )

    imwrite(
        os.path.join(
            r"/home/student/研究/プログラム/PraNet-master/dataset/ValDataset/masks",
            basename,
        ),
        img2,
    )
