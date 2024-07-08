import argparse

import torch
from torchvision import transforms
from utils.dataloader import test_dataset

#################################################
# epoch1~20の重みを全部テストする。

parser = argparse.ArgumentParser()
parser.add_argument("--testsize", type=int, default=352, help="testing size")
# parser.add_argument('--test_path1', type=str, default='./dataset/TestDataset', help='path to test dataset')
parser.add_argument(
    "--test_path1",
    type=str,
    default="./dataset/crop_TestDataset",
    help="path to test dataset",
)

opt = parser.parse_args()

data_path1 = opt.test_path1

save_path = "./results/Transfuse_S/"

image_root1 = "{}/images/".format(data_path1)
gt_root1 = "{}/masks/".format(data_path1)
test_loader1 = test_dataset(image_root1, gt_root1, opt.testsize)

mask_exist = 0
mask_not_exist = 0

for i in range(test_loader1.size):
    image, gt, name = test_loader1.load_data()
    label = transforms.functional.to_tensor(gt)
    label = torch.einsum("ijk->i", label) > 0
    label = torch.where(label > 0, torch.tensor(1), torch.tensor(0))

    if label == 1:
        mask_exist += 1
    else:
        mask_not_exist += 1

print("mask_exist:", mask_exist)
print("mask_not_exist:", mask_not_exist)
