import argparse
import glob
import os
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.nn.functional as F
# from lib.TransFuse_l import AttnTransFuse_L
from lib.TransFuse_l import TransFuse_L
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


parser = argparse.ArgumentParser()
parser.add_argument("--testsize", type=int, default=352, help="testing size")
parser.add_argument(
    "--pth_path", type=str, default="./snapshots/polyp491_10/TransFuse-best.pth"
)
parser.add_argument("--normalization", type=bool, default=False)

opt = parser.parse_args()
data_path = "./dataset_attn/sekkai_TestDataset/"
# data_path = "./dataset/dataset_v2/sekkai/sekkai_TestDataset"
norm = opt.normalization
save_path = "./results/AttnTransFuse_L"

# model = AttnTransFuse_L()
model = TransFuse_L()
model.load_state_dict(torch.load(opt.pth_path))
model.cuda()
model.eval()

os.makedirs(save_path, exist_ok=True)
for file in glob.glob(f"./{save_path}/*.png"):
    os.remove(file)

image_root = Path(f"{data_path}/images/")
gt_root = Path(f"{data_path}/masks/")
attn_map_root_1 = Path(f"{data_path}/attention_1/")
attn_map_root_2 = Path(f"{data_path}/attention_2/")
test_loader = test_dataset(
    image_root=image_root,
    gt_root=gt_root,
    testsize=opt.testsize,
    attn_map_root_1=attn_map_root_1,
    attn_map_root_2=attn_map_root_2,
)

dice_bank = []
iou_bank = []
acc_bank = []

for _ in range(test_loader.size):
    image, gt, name = test_loader.load_attn_data()
    gt = np.asarray(gt, np.float32)

    if norm:
        gt /= gt.max() + 1e-8  ##########################
    else:
        gt = 1.0 * (gt > 0.5)  ########################

    image = image.cuda()

    with torch.no_grad():
        _, _, res = model(image)

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
    imageio.imsave(f"{save_path}/{name}", img_as_ubyte(res))

print(
    "Dice: {:.4f}, IoU: {:.4f}, Acc: {:.4f}".format(
        np.mean(dice_bank), np.mean(iou_bank), np.mean(acc_bank)
    )
)
