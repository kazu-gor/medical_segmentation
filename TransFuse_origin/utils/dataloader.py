import os
import cv2
import copy
import random

import torch
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import albumentations as albu
from data_augumentation import Compose, Scale, RandomRotation, RandomMirror, Resize, Normalize_Tensor


class ImageTransform():
    def __init__(self):
        self.data_transform = {'train': Compose([
            # Scale(scale=[0.5, 1.5]),
            # RandomRotation(angle=[-10, 10]),
        ]),
            'val': Compose([])}

    def __call__(self, img, mask, phase='train'):
        return self.data_transform[phase](img, mask)


def get_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        # albu.Rotate(limit=[-10, 10], p=1.0),
        # albu.ShiftScaleRotate(shift_limit=[-0.0625, 0.0625], scale_limit=[-0.1, 0.1], rotate_limit=[-30, 30],
        #                       interpolation=1, border_mode=4, value=None, mask_value=None, p=0.5),
        # albu.RandomGamma(gamma_limit=[50, 150], p=1.0),
        # albu.RandomSizedCrop([300, 400], 416, 416, p=0.5),
        # albu.RandomGridShuffle(grid=(2, 2), p=0.5),
        # albu.RandomBrightness(limit=0.2,p=0.5),
        # albu.RandomContrast(limit=0.2, p=0.5),
        # albu.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, brightness_by_max=True, p=0.5),
        # albu.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
        # albu.CoarseDropout(max_holes=16, max_height=32, max_width=32, min_holes=1, min_height=8, min_width=8,
        #                    fill_value=0, p=0.5)
    ]
    return albu.Compose(train_transform)


class PolypAttnDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """

    def __init__(self, image_root, gt_root, attn_map_root_1, attn_map_root_2, trainsize, phase='train'):
        self.trainsize = trainsize
        self.images = [str(image_root / f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.attn_maps_1 = [str(attn_map_root_1 / f) for f in os.listdir(attn_map_root_1) if f.endswith('.jpg') or f.endswith('.png')]
        self.attn_maps_2 = [str(attn_map_root_2 / f) for f in os.listdir(attn_map_root_2) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [str(gt_root / f) for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.attn_maps_1 = sorted(self.attn_maps_1)
        self.attn_maps_2 = sorted(self.attn_maps_2)
        self.phase = phase

        print(f">>> Number of images: {len(self.images)}")
        self.filter_files()
        print(f">>> Number of images after filtering: {len(self.images)}")

        self.size = len(self.images)

        self._alpha = 0.2

        self.transform = ImageTransform()
        self.transform2 = get_augmentation()

        # self.augmentations = [
        #     albu.Rotate(limit=[-10, 10], p=1.0),
        #     albu.ShiftScaleRotate(shift_limit=[-0.0625, 0.0625], scale_limit=[-0.1, 0.1], rotate_limit=[-30, 30],
        #                           interpolation=1, border_mode=4, value=None, mask_value=None, p=1.0),
        #     albu.RandomBrightness(limit=0.2, p=1.0),
        #     albu.RandomContrast(limit=0.2, p=1.0),
        #     # albu.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, brightness_by_max=True, p=1.0),
        #     albu.RandomGamma(gamma_limit=[50, 150], p=1.0),
        #     # albu.RandomGridShuffle(grid=(2, 2), p=1.0),
        #     # albu.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
        #     albu.RandomSizedCrop([300, 400], 416, 416, p=1.0),
        #     # albu.CoarseDropout(max_holes=16, max_height=32, max_width=32, min_holes=1, min_height=8,
        #     #                    min_width=8, fill_value=0, p=1.0)
        # ]

        self.transform3 = albu.Compose(
            [
               albu.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=25, p=0.5, border_mode=0),
                albu.ColorJitter(),
                albu.HorizontalFlip(),
                # albu.VerticalFlip()
            ]
        )

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

        self.index_attn_1 = 0
        self.index_attn_2 = 0

    def __getitem__(self, index):
        self.exception_count_1 = 0
        self.exception_count_2 = 0

        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])

        name = self.images[index].split('/')[-1]
        name_gt = self.gts[index].split('/')[-1]

        def condition(x, name):
            return x.split('/')[-1] == name

        try:
            # nameが一致する要素をself.attn_map_1から抽出
            attn_path = [a for a in self.attn_maps_1 if condition(a, name)][0]
            attn_map_1 = self.binary_loader(attn_path)
            attn_map_1 = np.stack([attn_map_1, attn_map_1, attn_map_1], axis=2)
            name_attn = attn_path.split('/')[-1]
            assert name == name_gt == name_attn, f"{name} == {name_gt} == {name_attn}"
            self.index_attn_1 += 1
        except Exception:
            attn_map_1 = np.zeros_like(image)
            attn_map_1 = Image.fromarray(attn_map_1)
            self.exception_count_1 += 1

        try:
            # nameが一致する要素をself.attn_map_1から抽出
            attn_path = [a for a in self.attn_maps_2 if condition(a, name)][0]
            attn_map_2 = self.binary_loader(attn_path)
            attn_map_2 = np.stack([attn_map_2, attn_map_2, attn_map_2], axis=2)
            name_attn = attn_path.split('/')[-1]
            assert name == name_gt == name_attn, f"{name} == {name_gt} == {name_attn}"
            self.index_attn_2 += 1
        except Exception:
            attn_map_2 = np.zeros_like(image)
            attn_map_2 = Image.fromarray(attn_map_2)
            self.exception_count_2 += 1

        image = np.array(image)
        attn_map_1 = np.array(attn_map_1)
        attn_map_2 = np.array(attn_map_2)
        # image[:, :, 1] = attn_map_1[:, :, 0]
        # image[:, :, 2] = attn_map_2[:, :, 0]

        image[:, :, 2] = attn_map_1[:, :, 0]

        image = Image.fromarray(image)
        image = image.convert('RGB')

        if self.phase == 'train':

            image = np.array(image)
            gt = np.array(gt)

            augmented = self.transform3(image=image, mask=gt)

            image, gt = augmented['image'], augmented['mask']

            image = Image.fromarray(image)
            gt = Image.fromarray(gt)

            image = image.convert('RGB')
            gt = gt.convert('L')

        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        return image, gt

    # def __getitem__(self, index):
    #     self.exception_count = 0

    #     image = self.rgb_loader(self.images[index])
    #     gt = self.binary_loader(self.gts[index])

    #     name = self.images[index].split('/')[-1]
    #     name_gt = self.gts[index].split('/')[-1]

    #     try:
    #         attn_map = self.rgb_loader(self.attn_maps[self.index_attn])
    #         name_attn = self.attn_maps[self.index_attn].split('/')[-1]
    #         assert name == name_gt == name_attn, f"{name} == {name_gt} == {name_attn}"
    #         self.index_attn += 1
    #     except Exception:
    #         attn_map = np.zeros_like(image)
    #         attn_map = Image.fromarray(attn_map)

    #     if self.phase == 'train':

    #         image = np.array(image)
    #         gt = np.array(gt)
    #         attn_map = np.array(attn_map)
    #         # attn_map = self._minmax_normalize(attn_map)

    #         augmented = self.transform3(image=image, masks=[gt, attn_map])

    #         image, masks = augmented['image'], augmented['masks']
    #         gt, attn_map = masks[0], masks[1]

    #         image = Image.fromarray(image)
    #         gt = Image.fromarray(gt)
    #         attn_map = attn_map.astype(np.uint8)
    #         attn_map = Image.fromarray(attn_map)

    #         image = image.convert('RGB')
    #         gt = gt.convert('L')
    #         attn_map = attn_map.convert('RGB')

    #     image = self.img_transform(image)
    #     gt = self.gt_transform(gt)
    #     attn_map = self.gt_transform(attn_map)
    #     return image, gt, attn_map

    def _minmax_normalize(self, attention_map):
        min_val = np.min(attention_map)
        max_val = np.max(attention_map)

        if min_val == max_val:
            # すべての値が同じ場合、値を0.5に設定する
            return np.ones_like(attention_map) * 0.5
        else:
            return (attention_map - min_val) / (max_val - min_val)

    def _apply_mixup(self, image1, gt1, idx1):
        image1 = np.array(image1)
        gt1 = np.array(gt1)
        # mixする画像のインデックスを拾ってくる
        idx2 = self._get_pair_index(idx1)
        # 画像の準備
        image2 = self.rgb_loader(self.images[idx2])
        # ラベルの準備（アノテーションファイルは1,0,0,0のように所属するクラスが記されている）
        gt2 = self.binary_loader(self.gts[idx2])
        image2 = np.array(image2)
        gt2 = np.array(gt2)
        # 混ぜる割合を決めて
        # r = np.random.beta(self._alpha, self._alpha, 1)[0]

        r = np.random.normal(loc=0, scale=3, size=1)
        r = 1 / (1 + np.exp(-r))
        # 画像、ラベルを混ぜる（クリップしないと範囲外になることがある）
        mixed_image = np.clip(r * image1 + (1 - r) * image2, 0, 255)
        mixed_gt = np.clip(r * gt1 + (1 - r) * gt2, 0, 255)
        mixed_image = Image.fromarray(mixed_image.astype(np.uint8))
        mixed_gt = Image.fromarray(mixed_gt.astype(np.uint8))
        mixed_image = mixed_image.convert('RGB')
        mixed_gt = mixed_gt.convert('L')
        return mixed_image, mixed_gt

    # Datasetの__get_item__のidx以外のindexを取得する
    def _get_pair_index(self, idx):
        r = list(range(0, idx)) + list(range(idx + 1, len(self.images)))
        return random.choice(r)

    def cutmix(self, image1, gt1, idx1):
        image1 = np.array(image1)
        gt1 = np.array(gt1)
        # mixする画像のインデックスを拾ってくる
        idx2 = self._get_pair_index(idx1)
        # 画像の準備
        image2 = self.rgb_loader(self.images[idx2])
        # ラベルの準備（アノテーションファイルは1,0,0,0のように所属するクラスが記されている）
        gt2 = self.binary_loader(self.gts[idx2])
        image2 = np.array(image2)
        gt2 = np.array(gt2)

        lam = np.random.beta(self._alpha, self._alpha)

        image_h, image_w, _ = image1.shape
        cx = np.random.uniform(0, image_w)
        cy = np.random.uniform(0, image_h)
        w = image_w * np.sqrt(1 - lam)
        h = image_h * np.sqrt(1 - lam)
        x0 = int(np.round(max(cx - w / 2, 0)))
        x1 = int(np.round(min(cx + w / 2, image_w)))
        y0 = int(np.round(max(cy - h / 2, 0)))
        y1 = int(np.round(min(cy + h / 2, image_h)))

        image1[y0:y1, x0:x1, :] = image2[y0:y1, x0:x1, :]
        gt1[y0:y1, x0:x1] = gt2[y0:y1, x0:x1]

        return image1, gt1

    def augment_and_mix(self, image, gt, width=3, depth=-1, alpha=1.):
        """Perform AugMix augmentations and compute mixture.
        Args:
          image: Raw input image as float32 np.ndarray of shape (h, w, c)
          severity: Severity of underlying augmentation operators (between 1 to 10).
          width: Width of augmentation chain
          depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
            from [1, 3]
          alpha: Probability coefficient for Beta and Dirichlet distributions.
        Returns:
          mixed: Augmented and mixed image.
        """
        ws = np.float32(
            np.random.dirichlet([alpha] * width))
        m = np.float32(np.random.beta(alpha, alpha))

        mix = np.zeros_like(image).astype('float32')
        mix_gt = np.zeros_like(gt).astype('float32')
        for i in range(width):
            image_aug = image.copy()
            gt_aug = gt.copy()
            d = depth if depth > 0 else np.random.randint(1, 4)
            for _ in range(d):
                op = np.random.choice(self.augmentations)
                image_aug = op(image=image_aug)['image']
                gt_aug = op(image=gt_aug)['image']
            # Preprocessing commutes since all coefficients are convex
            mix += ws[i] * image_aug
            mix_gt = ws[i] * gt_aug

        mixed = (1 - m) * image + m * mix
        mixed_gt = (1 - m) * gt + m * mix_gt
        return mixed.astype('uint8'), mixed_gt.astype('uint8')

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)

            # if img.size == gt.size:
            #     # If gt does not contain any labels (total value is 0), do not include it in the data set
            #     if self.phase in ['train', 'val']:
            #         if np.sum(np.array(gt)) > 0:
            #             images.append(img_path)
            #             gts.append(gt_path)
            #     else:
            #         images.append(img_path)
            #         gts.append(gt_path)

            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts


    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size

class PolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """

    def __init__(self, image_root, gt_root, trainsize, phase='train'):
        self.trainsize = trainsize
        self.images = [str(image_root / f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [str(gt_root / f) for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.phase = phase

        print(f">>> Number of images: {len(self.images)}")
        self.filter_files()
        print(f">>> Number of images after filtering: {len(self.images)}")
        self.size = len(self.images)

        self._alpha = 0.2

        self.transform = ImageTransform()
        self.transform2 = get_augmentation()

        # self.augmentations = [
        #     albu.Rotate(limit=[-10, 10], p=1.0),
        #     albu.ShiftScaleRotate(shift_limit=[-0.0625, 0.0625], scale_limit=[-0.1, 0.1], rotate_limit=[-30, 30],
        #                           interpolation=1, border_mode=4, value=None, mask_value=None, p=1.0),
        #     albu.RandomBrightness(limit=0.2, p=1.0),
        #     albu.RandomContrast(limit=0.2, p=1.0),
        #     # albu.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, brightness_by_max=True, p=1.0),
        #     albu.RandomGamma(gamma_limit=[50, 150], p=1.0),
        #     # albu.RandomGridShuffle(grid=(2, 2), p=1.0),
        #     # albu.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
        #     albu.RandomSizedCrop([300, 400], 416, 416, p=1.0),
        #     # albu.CoarseDropout(max_holes=16, max_height=32, max_width=32, min_holes=1, min_height=8,
        #     #                    min_width=8, fill_value=0, p=1.0)
        # ]

        self.transform3 = albu.Compose(
            [
                albu.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=25, p=0.5, border_mode=0),
                albu.ColorJitter(),
                albu.HorizontalFlip(),
                # albu.VerticalFlip()
            ]
        )

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        if self.phase == 'train':

            # if random.random() < 1.0:
            #     image, gt = self._apply_mixup(image, gt, index)

            # if random.random() < 1.0:
            #     image, gt = self.cutmix(image, gt, index)

            image = np.array(image)
            gt = np.array(gt)

            # image, gt = self.augment_and_mix(image, gt)
            # augmented = self.transform2(image=image, mask=gt)
            augmented = self.transform3(image=image, mask=gt)

            image, gt = augmented['image'], augmented['mask']
            image = Image.fromarray(image)
            gt = Image.fromarray(gt)
            image = image.convert('RGB')
            gt = gt.convert('L')

        # image, gt = self.transform(image, gt, phase=self.phase)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        return image, gt

    def _apply_mixup(self, image1, gt1, idx1):
        image1 = np.array(image1)
        gt1 = np.array(gt1)
        # mixする画像のインデックスを拾ってくる
        idx2 = self._get_pair_index(idx1)
        # 画像の準備
        image2 = self.rgb_loader(self.images[idx2])
        # ラベルの準備（アノテーションファイルは1,0,0,0のように所属するクラスが記されている）
        gt2 = self.binary_loader(self.gts[idx2])
        image2 = np.array(image2)
        gt2 = np.array(gt2)
        # 混ぜる割合を決めて
        # r = np.random.beta(self._alpha, self._alpha, 1)[0]

        r = np.random.normal(loc=0, scale=3, size=1)
        r = 1 / (1 + np.exp(-r))
        # 画像、ラベルを混ぜる（クリップしないと範囲外になることがある）
        mixed_image = np.clip(r * image1 + (1 - r) * image2, 0, 255)
        mixed_gt = np.clip(r * gt1 + (1 - r) * gt2, 0, 255)
        mixed_image = Image.fromarray(mixed_image.astype(np.uint8))
        mixed_gt = Image.fromarray(mixed_gt.astype(np.uint8))
        mixed_image = mixed_image.convert('RGB')
        mixed_gt = mixed_gt.convert('L')
        return mixed_image, mixed_gt

    # Datasetの__get_item__のidx以外のindexを取得する
    def _get_pair_index(self, idx):
        r = list(range(0, idx)) + list(range(idx + 1, len(self.images)))
        return random.choice(r)

    def cutmix(self, image1, gt1, idx1):
        image1 = np.array(image1)
        gt1 = np.array(gt1)
        # mixする画像のインデックスを拾ってくる
        idx2 = self._get_pair_index(idx1)
        # 画像の準備
        image2 = self.rgb_loader(self.images[idx2])
        # ラベルの準備（アノテーションファイルは1,0,0,0のように所属するクラスが記されている）
        gt2 = self.binary_loader(self.gts[idx2])
        image2 = np.array(image2)
        gt2 = np.array(gt2)

        lam = np.random.beta(self._alpha, self._alpha)

        image_h, image_w, _ = image1.shape
        cx = np.random.uniform(0, image_w)
        cy = np.random.uniform(0, image_h)
        w = image_w * np.sqrt(1 - lam)
        h = image_h * np.sqrt(1 - lam)
        x0 = int(np.round(max(cx - w / 2, 0)))
        x1 = int(np.round(min(cx + w / 2, image_w)))
        y0 = int(np.round(max(cy - h / 2, 0)))
        y1 = int(np.round(min(cy + h / 2, image_h)))

        image1[y0:y1, x0:x1, :] = image2[y0:y1, x0:x1, :]
        gt1[y0:y1, x0:x1] = gt2[y0:y1, x0:x1]

        return image1, gt1

    def augment_and_mix(self, image, gt, width=3, depth=-1, alpha=1.):
        """Perform AugMix augmentations and compute mixture.
        Args:
          image: Raw input image as float32 np.ndarray of shape (h, w, c)
          severity: Severity of underlying augmentation operators (between 1 to 10).
          width: Width of augmentation chain
          depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
            from [1, 3]
          alpha: Probability coefficient for Beta and Dirichlet distributions.
        Returns:
          mixed: Augmented and mixed image.
        """
        ws = np.float32(
            np.random.dirichlet([alpha] * width))
        m = np.float32(np.random.beta(alpha, alpha))

        mix = np.zeros_like(image).astype('float32')
        mix_gt = np.zeros_like(gt).astype('float32')
        for i in range(width):
            image_aug = image.copy()
            gt_aug = gt.copy()
            d = depth if depth > 0 else np.random.randint(1, 4)
            for _ in range(d):
                op = np.random.choice(self.augmentations)
                image_aug = op(image=image_aug)['image']
                gt_aug = op(image=gt_aug)['image']
            # Preprocessing commutes since all coefficients are convex
            mix += ws[i] * image_aug
            mix_gt = ws[i] * gt_aug

        mixed = (1 - m) * image + m * mix
        mixed_gt = (1 - m) * gt + m * mix_gt
        return mixed.astype('uint8'), mixed_gt.astype('uint8')

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)

            # if img.size == gt.size:
            #     # If gt does not contain any labels (total value is 0), do not include it in the data set
            #     if self.phase in ['train', 'val']:
            #         if np.sum(np.array(gt)) > 0:
            #             images.append(img_path)
            #             gts.append(gt_path)
            #     else:
            #         images.append(img_path)
            #         gts.append(gt_path)

            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_attn_loader(image_root, gt_root, attn_map_root_1, attn_map_root_2, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True, phase='train',
               droplast=False):
    dataset = PolypAttnDataset(image_root, gt_root, attn_map_root_1, attn_map_root_2, trainsize, phase=phase)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  drop_last=droplast)
    return data_loader


def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True, phase='train',
               droplast=False):
    dataset = PolypDataset(image_root, gt_root, trainsize, phase=phase)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  drop_last=droplast)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, testsize, attn_map_root_1=None, attn_map_root_2=None):
        self.testsize = testsize
        self.images = [str(image_root / f) for f in os.listdir(image_root) if
                       f.endswith('.jpg') or f.endswith('.png') or f.endswith('.tif')]
        self.gts = [str(gt_root / f) for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]

        try:
            self.attn_maps_1 = [str(attn_map_root_1 / f) for f in os.listdir(attn_map_root_1) if f.endswith('.jpg') or f.endswith('.png')]
            self.attn_maps_2 = [str(attn_map_root_2 / f) for f in os.listdir(attn_map_root_2) if f.endswith('.jpg') or f.endswith('.png')]
        except Exception:
            pass

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.resize = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0
        self.index_attn_1 = 0
        self.index_attn_2 = 0

    def load_data(self):

        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        # gt = self.resize(gt)

        name = self.images[self.index].split('/')[-1]
        name_gt = self.gts[self.index].split('/')[-1]

        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1

        return image, gt, name

    def load_attn_data(self):

        image = self.rgb_loader(self.images[self.index])
        image_copy = copy.deepcopy(image)
        gt = self.binary_loader(self.gts[self.index])
        # gt = self.resize(gt)

        name = self.images[self.index].split('/')[-1]
        name_gt = self.gts[self.index].split('/')[-1]

        def condition(x, name):
            return x.split('/')[-1] == name

        try:
            # nameが一致する要素をself.attn_map_1から抽出
            attn_path = [a for a in self.attn_maps_1 if condition(a, name)][0]
            attn_map_1 = self.binary_loader(attn_path)
            attn_map_1 = np.stack([attn_map_1, attn_map_1, attn_map_1], axis=2)
            name_attn = attn_path.split('/')[-1]
            assert name == name_gt == name_attn, f"{name} == {name_gt} == {name_attn}"
            self.index_attn_1 += 1
        except Exception:
            attn_map_1 = np.zeros_like(image)
            attn_map_1 = Image.fromarray(attn_map_1)

        try:
            # nameが一致する要素をself.attn_map_1から抽出
            attn_path = [a for a in self.attn_maps_2 if condition(a, name)][0]
            attn_map_2 = self.binary_loader(attn_path)
            attn_map_2 = np.stack([attn_map_2, attn_map_2, attn_map_2], axis=2)
            name_attn = attn_path.split('/')[-1]
            assert name == name_gt == name_attn, f"{name} == {name_gt} == {name_attn}"
            self.index_attn_2 += 1
        except Exception:
            attn_map_2 = np.zeros_like(image)
            attn_map_2 = Image.fromarray(attn_map_2)

        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1

        image = np.array(image)
        attn_map_1 = np.array(attn_map_1)
        attn_map_2 = np.array(attn_map_2)
        # image[:, :, 1] = attn_map_1[:, :, 0]
        # image[:, :, 2] = attn_map_2[:, :, 0]

        image[:, :, 2] = attn_map_1[:, :, 0]

        image = Image.fromarray(image)
        image = image.convert('RGB')

        image = self.transform(image).unsqueeze(0)

        return image, gt, name


    def load_data_mixup(self):

        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])

        idx2 = self._get_pair_index(self.index)
        image2 = self.rgb_loader(self.images[idx2])
        image2 = self.transform(image2).unsqueeze(0)
        gt2 = self.binary_loader(self.gts[idx2])

        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, image2, gt2, name

    def _get_pair_index(self, idx):
        r = list(range(0, idx)) + list(range(idx + 1, len(self.images)))
        return random.choice(r)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


class test_dataset_crop:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if
                       f.endswith('.jpg') or f.endswith('.png') or f.endswith('.tif')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):

        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
 
