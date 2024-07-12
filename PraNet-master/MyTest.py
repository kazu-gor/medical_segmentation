import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from lib.PraNet_Res2Net import PraNet
from scipy import misc
from utils.dataloader import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--testsize", type=int, default=352, help="testing size")
parser.add_argument(
    "--test_path",
    type=str,
    default="./dataset/TestDataset/",
    help="path to test dataset",
)
parser.add_argument("--normalization", type=bool, default=True)

parser.add_argument(
    "--pth_path", type=str, default="./snapshots/PraNet_Res2Net/PraNet-100.pth"
)
# parser.add_argument('--pth_path', type=str, default='./weights/new_PraNet/PraNet-100.pth')

opt = parser.parse_args()

data_path = opt.test_path
# data_path = './dataset/TestDataset/'
# data_path = './dataset/ValDataset/'

norm = opt.normalization

save_path = "./results/PraNet/"
model = PraNet()
model.load_state_dict(torch.load(opt.pth_path))
model.cuda()
model.eval()

os.makedirs(save_path, exist_ok=True)
for file in glob.glob("./results/PraNet/*.png"):
    os.remove(file)
image_root = "{}/images/".format(data_path)
gt_root = "{}/masks/".format(data_path)
test_loader = test_dataset(image_root, gt_root, opt.testsize)
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
    misc.imsave(save_path + name, res)
time_finish = time.time()
print("timer: {:.4f} sec.".format((time_finish - time_start) / no))
