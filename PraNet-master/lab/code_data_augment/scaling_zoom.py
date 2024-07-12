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


def scaling_zoom(src, ksize):
    d = int((ksize - 1) / 2)
    src1 = src[d : 416 - d, d : 416 - d, 0:3]
    dst = cv2.resize(src1, (416, 416))

    return dst


# files = glob.glob(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/train_img/*.png")
files = glob.glob(
    r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/TrainDataset/images/*.png"
)
for i in files:
    img = imread(i)

    # 画像の拡張

    img = scaling_zoom(img, 21)
    imwrite(
        os.path.join(
            r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/TrainDataset/images/",
            "scaling_zoom" + os.path.basename(i),
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
    img = scaling_zoom(img, 21)

    imwrite(
        os.path.join(
            r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/TrainDataset/masks/",
            "scaling_zoom" + os.path.basename(i),
        ),
        img,
    )
