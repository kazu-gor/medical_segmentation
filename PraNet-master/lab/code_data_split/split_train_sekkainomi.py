import numpy as np

import cv2
import glob
import os
import shutil
from sklearn.model_selection import train_test_split


###trainのみ石灰化を含む画像のみにして分ける
###その他はsplit.pyと同じ

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
for file in glob.glob('/home/student/src2/藤林/プログラム/PraNet-master/lab/test_img/*.png'):
    os.remove(file)
for file in glob.glob('/home/student/src2/藤林/プログラム/PraNet-master/lab/train_img/*.png'):
    os.remove(file)
for file in glob.glob('/home/student/src2/藤林/プログラム/PraNet-master/lab/test_mask/*.png'):
    os.remove(file)
for file in glob.glob('/home/student/src2/藤林/プログラム/PraNet-master/lab/train_mask/*.png'):
    os.remove(file)
for file in glob.glob('/home/student/src2/藤林/プログラム/PraNet-master/dataset/ValDataset/images/*.png'):
    os.remove(file)
for file in glob.glob('/home/student/src2/藤林/プログラム/PraNet-master/dataset/ValDataset/masks/*.png'):
    os.remove(file)

x = 0  # random_state

files1 = glob.glob(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/images/*.png")
# files1 = glob.glob(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/sekkai_images/*.png")
train0, test = train_test_split(files1, train_size=0.8, random_state=x)
train, val = train_test_split(train0, train_size=0.9, random_state=x)

for i in test:
    img = imread(i)
    imwrite(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/TestDataset/images", os.path.basename(i)),
            img)
    imwrite(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/test_img", os.path.basename(i)),
            img)

for i in val:
    img = imread(i)
    imwrite(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/ValDataset/images", os.path.basename(i)),
            img)
    imwrite(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/val_img", os.path.basename(i)), img)

files2 = glob.glob(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/masks/*.png")
# files2 = glob.glob(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/sekkai_masks/*.png")
train0, test = train_test_split(files2, train_size=0.8, random_state=x)
train, val = train_test_split(train0, train_size=0.9, random_state=x)

for i in test:
    img = imread(i)
    imwrite(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/TestDataset/masks", os.path.basename(i)),
            img)
    imwrite(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/test_mask", os.path.basename(i)),
            img)

for i in train:
    img = imread(i)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    label = cv2.connectedComponentsWithStats(gray)
    nlabels = label[0]
    if nlabels > 1:
        imwrite(
            os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/TrainDataset/masks", os.path.basename(i)),
            img)
        imwrite(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/train_mask", os.path.basename(i)),
                img)
        shutil.copy(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/images",
                                 os.path.splitext(os.path.basename(i))[0] + ".png")
                    , os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/TrainDataset/images",
                                   os.path.splitext(os.path.basename(i))[0] + ".png"))
        shutil.copy(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/images",
                                 os.path.splitext(os.path.basename(i))[0] + ".png")
                    , os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/train_img",
                                   os.path.splitext(os.path.basename(i))[0] + ".png"))

for i in val:
    img = imread(i)
    imwrite(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/dataset/ValDataset/masks", os.path.basename(i)),
            img)
    imwrite(os.path.join(r"/home/student/src2/藤林/プログラム/PraNet-master/lab/val_mask", os.path.basename(i)), img)
