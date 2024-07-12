import glob
import os

import cv2
import numpy as np


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


def create_gamma_img(gamma, img):
    gamma_cvt = np.zeros((256, 1), dtype=np.uint8)
    for i in range(256):
        gamma_cvt[i][0] = 255 * (float(i) / 255) ** (1.0 / gamma)
    return cv2.LUT(img, gamma_cvt)


# files = glob.glob(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/train_img/*.png")
files = glob.glob(
    r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/TrainDataset/images/*.png"
)
for i in files:
    img = imread(i)

    # 画像の拡張
    img = create_gamma_img(1.5, img)
    imwrite(
        os.path.join(
            r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/TrainDataset/images/",
            "gamma1.5" + os.path.basename(i),
        ),
        img,
    )


# files = glob.glob(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/train_mask/*.png")
files = glob.glob(
    r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/TrainDataset/masks/*.png"
)
for i in files:
    img = imread(i)

    # 画像の拡張

    imwrite(
        os.path.join(
            r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/TrainDataset/masks/",
            "gamma1.5" + os.path.basename(i),
        ),
        img,
    )
