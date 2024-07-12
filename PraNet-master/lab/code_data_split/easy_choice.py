import glob
import os

import cv2
import numpy as np

#########easyの画像を/lab/easy_imagesに保存する


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


files = glob.glob(
    r"/home/student/src2/藤林/プログラム/PraNet-master/lab/sekkai_images/*.png"
)  ###正解画像
for i in files:
    basename = os.path.basename(i)
    img = imread(i)
    if "easy" in basename:
        imwrite(
            os.path.join(
                r"/home/student/src2/藤林/プログラム/PraNet-master/lab/easy_images",
                os.path.basename(i),
            ),
            img,
        )
        img2 = imread(
            "/home/student/src2/藤林/プログラム/PraNet-master/lab/sekkai_masks/"
            + basename
        )
        imwrite(
            os.path.join(
                r"/home/student/src2/藤林/プログラム/PraNet-master/lab/easy_masks",
                os.path.basename(i),
            ),
            img2,
        )
