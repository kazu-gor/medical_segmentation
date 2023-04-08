import numpy as np

import cv2
import glob
import os



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


# files = glob.glob(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/train_img/*.png")
files = glob.glob(r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/TrainDataset/images/*.png")
for i in files:
    img = imread(i)

    # 画像の拡張
    img = cv2.flip(img, 1)     #第一引数に元のndarray、第二引数flipCodeに反転方向を示す値を指定する。
                               # flipcode = 0: 上下反転、flipcode > 0: 左右反転、flipcode < 0: 上下左右反転
    imwrite(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/TrainDataset/images/", "reverse"+os.path.basename(i)), img)


# files = glob.glob(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/train_mask/*.png")
files = glob.glob(r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/TrainDataset/masks/*.png")
for i in files:
    img = imread(i)

    # 画像の拡張
    img = cv2.flip(img, 1)     #第一引数に元のndarray、第二引数flipCodeに反転方向を示す値を指定する。
                               # flipcode = 0: 上下反転、flipcode > 0: 左右反転、flipcode < 0: 上下左右反転
    imwrite(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/TrainDataset/masks/", "reverse"+os.path.basename(i)), img)
