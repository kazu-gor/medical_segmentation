import numpy as np

import cv2
import glob
import os

from sklearn.model_selection import train_test_split


###/nana_roku/にある石灰化76か所の画像をPraNet-master/dataset/TestDatasetとPraNet-master/lab/test_imgにコピーする


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


for file in glob.glob('/home/student/src2/藤林/プログラム/PraNet-master/dataset/TestDataset/images/*.png'):
    os.remove(file)
for file in glob.glob('/home/student/src2/藤林/プログラム/PraNet-master/dataset/TestDataset/masks/*.png'):
    os.remove(file)
for file in glob.glob('/home/student/src2/藤林/プログラム/PraNet-master/dataset/TrainDataset/images/*.png'):
    os.remove(file)
for file in glob.glob('/home/student/src2/藤林/プログラム/PraNet-master/dataset/TrainDataset/masks/*.png'):
    os.remove(file)
for file in glob.glob('/home/student/src2/藤林/プログラム/PraNet-master/dataset/ValDataset/images/*.png'):
    os.remove(file)
for file in glob.glob('/home/student/src2/藤林/プログラム/PraNet-master/dataset/ValDataset/masks/*.png'):
    os.remove(file)
for file in glob.glob('/home/student/src2/藤林/プログラム/PraNet-master/lab/test_img/*.png'):
    os.remove(file)
for file in glob.glob('/home/student/src2/藤林/プログラム/PraNet-master/lab/train_img/*.png'):
    os.remove(file)
for file in glob.glob('/home/student/src2/藤林/プログラム/PraNet-master/lab/test_mask/*.png'):
    os.remove(file)
for file in glob.glob('/home/student/src2/藤林/プログラム/PraNet-master/lab/train_mask/*.png'):
    os.remove(file)

files1 = glob.glob(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/nana_roku/test3/*.png")

for i in files1:
    img = imread(i)
    imwrite(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/TestDataset/images", os.path.basename(i)),
            img)
    imwrite(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/test_img", os.path.basename(i)),
            img)
files2 = glob.glob(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/nana_roku/train/*.png")
for i in files2:
    img = imread(i)
    imwrite(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/TrainDataset/images", os.path.basename(i)),
            img)
    imwrite(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/train_img", os.path.basename(i)),
            img)
files3 = glob.glob(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/nana_roku/val/*.png")
for i in files3:
    img = imread(i)
    imwrite(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/ValDataset/images", os.path.basename(i)),
            img)

files4 = glob.glob(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/nana_roku/test3_mask/*.png")

for i in files4:
    img = imread(i)
    imwrite(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/TestDataset/masks", os.path.basename(i)),
            img)
    imwrite(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/test_mask", os.path.basename(i)),
            img)
files5 = glob.glob(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/nana_roku/train_mask/*.png")
for i in files5:
    img = imread(i)
    imwrite(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/TrainDataset/masks", os.path.basename(i)),
            img)
    imwrite(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/train_mask", os.path.basename(i)),
            img)
files6 = glob.glob(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/nana_roku/val_mask/*.png")
for i in files6:
    img = imread(i)
    imwrite(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/ValDataset/masks", os.path.basename(i)),
            img)
