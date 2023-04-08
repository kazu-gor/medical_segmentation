import numpy as np

import cv2
import glob
import os

from PIL import Image


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


files = glob.glob(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/train_img/*.png")
# files = glob.glob(r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/TrainDataset/images/*.png")
for i in files:
    file, ext = os.path.splitext(i)
    img = Image.open(i)
    img = img.rotate(10)
    # 画像の拡張
    img.save(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/TrainDataset/images/",
                         "rot10" + os.path.basename(i)), "PNG")


files = glob.glob(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/train_mask/*.png")
# files = glob.glob(r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/TrainDataset/masks/*.png")
for i in files:
    img = Image.open(i)
    img = img.rotate(10)

    # 画像の拡張
    img.save(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/TrainDataset/masks/",
                          "rot10" + os.path.basename(i)), "PNG")

