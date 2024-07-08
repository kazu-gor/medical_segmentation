import argparse
import glob
import os
import time

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from lib.Trans_CaraNet import Trans_CaraNet_L
from skimage import img_as_ubyte
from utils.dataloader import test_dataset


def mean_iou_np(y_true, y_pred, **kwargs):
    """
    compute mean iou for binary segmentation map via numpy
    """
    axes = (0, 1)
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    union = mask_sum - intersection

    smooth = 0.001
    iou = (intersection + smooth) / (union + smooth)
    return iou


def mean_dice_np(y_true, y_pred, **kwargs):
    """
    compute mean dice for binary segmentation map via numpy
    """
    axes = (0, 1)  # W,H axes of each image
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)

    smooth = 0.001
    dice = 2 * (intersection + smooth) / (mask_sum + smooth)
    return dice


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


parser = argparse.ArgumentParser()
parser.add_argument("--testsize", type=int, default=352, help="testing size")
parser.add_argument("--normalization", type=bool, default=False)
parser.add_argument("--epoch", type=str, default="69", help="epoch number")
parser.add_argument(
    "--train_weight", type=str, default="Transfuse_S", help="load model weight"
)

opt = parser.parse_args()

norm = opt.normalization

save_path = "./results/preprocessing/"

model = Trans_CaraNet_L()

model.load_state_dict(
    torch.load(f"./snapshots/{opt.train_weight}/Transfuse-{opt.epoch}.pth")
)

model.cuda()
model.eval()

os.makedirs(save_path, exist_ok=True)
for file in glob.glob("./results/preprocessing/*.png"):
    os.remove(file)

img_path = "./datasets/preprocessing/images/"
gt_path = "./datasets/preprocessing/masks/"
test_loader = test_dataset(img_path, gt_path, opt.testsize)

dice_bank = []
iou_bank = []
acc_bank = []

no = 0
time_start = time.time()
for i in range(test_loader.size):
    no += 1
    image, gt, name = test_loader.load_data()
    gt = np.asarray(gt, np.float32)

    if norm:
        gt /= gt.max() + 1e-8  ##########################
    else:
        gt = 1.0 * (gt > 0.5)  ########################

    image = image.cuda()
    with torch.no_grad():
        res5, res4, res3, res2 = model(image)
    res = res2
    # res = res5
    res = F.upsample(res, size=gt.shape, mode="bilinear", align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()

    if norm:
        res = (res - res.min()) / (
            res.max() - res.min() + 1e-8
        )  ############################
    else:
        res = 1.0 * (res > 0.5)  ############################

    dice = mean_dice_np(gt, res)
    iou = mean_iou_np(gt, res)
    acc = np.sum(res == gt) / (res.shape[0] * res.shape[1])

    acc_bank.append(acc)
    dice_bank.append(dice)
    iou_bank.append(iou)
    imageio.imsave(save_path + name, img_as_ubyte(res))

time_finish = time.time()
print(f"epoch: {opt.epoch}")
print("timer: {:.4f} sec.".format((time_finish - time_start) / no))
print(
    "Dice: {:.4f}, IoU: {:.4f}, Acc: {:.4f}".format(
        np.mean(dice_bank), np.mean(iou_bank), np.mean(acc_bank)
    )
)
