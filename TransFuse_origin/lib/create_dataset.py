import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


class Mask2BBox(object):
    """mask画像からbboxを生成するクラス

    Args:
        object (_type_): _description_
    """

    def __init__(self, classes=None):

        self.classes = classes
        self.space = 5

    def __call__(self, mask_list, height=352, width=352):
        """
        masksからbboxを生成

        Parameters
        ----------
        mask_list: list
            masksのリスト
        """
        bndbox = []
        SPACE = self.space
        if not isinstance(mask_list, list):
            mask_list = [mask_list]
        for mask in mask_list:
            org_img = cv2.imread(str(mask))
            height, width, _ = org_img.shape

            lower = np.array([0, 0, 0], dtype="uint8")
            upper = np.array([255, 50, 255], dtype="uint8")
            img = cv2.inRange(org_img, lower, upper)

            img = cv2.bitwise_not(img)

            contours, _ = cv2.findContours(
                img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            contour_arr = np.array([])

            for i in range(len(contours)):
                x, y, w, h = cv2.boundingRect(contours[i])

                xmin = (x - SPACE)
                ymin = (y - SPACE)
                xmax = (x + w + SPACE)
                ymax = (y + h + SPACE)

                coordinate_list = np.clip(
                    np.array([[xmin, ymin, xmax, ymax, i]]), 
                    0, [width, height, width, height, 0]
                )[0]

                contour_arr = np.concatenate(
                    (contour_arr, coordinate_list), axis=0)

            # # バウンディングボックスを統合する
            # if contour_arr.size > 0:
            #     contour_arr = contour_arr.reshape(-1, 5)
            #     contour_arr = np.concatenate(
            #         (contour_arr.min(axis=0)[:2],
            #          contour_arr.max(axis=0)[2:4],
            #          contour_arr[0][4]), axis=None).astype('int64')

            bndbox.append(contour_arr)
        return np.array(bndbox).reshape(-1, 5).astype('float64')


class CreateDataset:
    def __init__(self, root_path, output_path):
        self.root_path = Path(root_path)
        self.image_path = self.root_path / 'images'
        self.mask_path = self.root_path / 'masks'

        self.image_list = sorted(list(self.image_path.glob('*.png')))
        self.mask_list = sorted(list(self.mask_path.glob('*.png')))

        self.output_path = output_path

    def get_no_parameter_dataset(self):
        for image_path, mask_path in tqdm(zip(self.image_list, self.mask_list)):
            assert image_path.stem == mask_path.stem, 'imageとmaskの名前が一致しません'
            image = cv2.imread(str(image_path))
            mask = cv2.imread(str(mask_path))
            _, w, _ = image.shape

            # 2. Split the image into left and right
            image_left = image[:, :w//2]
            image_right = image[:, w//2:]
            mask_left = mask[:, :w//2]
            mask_right = mask[:, w//2:]

            # 3. Save the image
            cv2.imwrite(str(self.output_path / 'images' / f'left_{image_path.stem}.png'), image_left)
            cv2.imwrite(str(self.output_path / 'images' / f'right_{image_path.stem}.png'), image_right)
            cv2.imwrite(str(self.output_path / 'masks' / f'left_{mask_path.stem}.png'), mask_left)
            cv2.imwrite(str(self.output_path / 'masks' / f'right_{mask_path.stem}.png'), mask_right)


    def _create_dir(self, mode, dir_name):
        os.makedirs(self.output_path / f'{mode}/images/{dir_name}', exist_ok=True)
        os.makedirs(self.output_path / f'{mode}/masks/{dir_name}', exist_ok=True)
        os.makedirs(self.output_path / f'{mode}/labels/{dir_name}', exist_ok=True)

    def _get_image_list(self, image_path):
        if isinstance(image_path, str):
            image_path = Path(image_path)
        return sorted([image.stem for image in image_path.glob('*.png')])

    def get_divide_data(self):

        self._create_dir('all', 'TrainDataset')
        self._create_dir('all', 'TestDataset')
        self._create_dir('all', 'ValDataset')

        self._create_dir('sekkai', 'sekkai_TrainDataset')
        self._create_dir('sekkai', 'sekkai_TestDataset')
        self._create_dir('sekkai', 'sekkai_ValDataset')

        train_image_list = self._get_image_list('../../../dataset/TrainDataset/images/')
        test_image_list = self._get_image_list('../../../dataset/TestDataset/images/')
        val_image_list = self._get_image_list('../../../dataset/ValDataset/images/')

        train_sekkai_image_list = self._get_image_list('../../../dataset/sekkai_TrainDataset/images/')
        test_sekkai_image_list = self._get_image_list('../../../dataset/sekkai_TestDataset/images/')
        val_sekkai_image_list = self._get_image_list('../../../dataset/sekkai_ValDataset/images/')

        no_para_data_path = Path('../../../dataset/original_images/preprocessed/')
        no_para_image_list = self._get_image_list(no_para_data_path / 'images/')
        no_para_mask_list = self._get_image_list(no_para_data_path / 'masks/')

        for image_path, mask_path in tqdm(zip(no_para_image_list, no_para_mask_list)):
            if image_path in train_image_list:
                mode = 'TrainDataset'
            elif image_path in test_image_list:
                mode = 'TestDataset'
            elif image_path in val_image_list:
                mode = 'ValDataset'
            else:
                raise ValueError('modeが正しくありません')

            image = cv2.imread(str(no_para_data_path / f'images/{image_path}.png'))
            mask = cv2.imread(str(no_para_data_path / f'masks/{mask_path}.png'))
            cv2.imwrite(str(self.output_path / f'all/images/{mode}/{image_path}.png'), image)
            cv2.imwrite(str(self.output_path / f'all/masks/{mode}/{mask_path}.png'), mask)

            if image_path in train_sekkai_image_list:
                mode = 'sekkai_TrainDataset'
            elif image_path in test_sekkai_image_list:
                mode = 'sekkai_TestDataset'
            elif image_path in val_sekkai_image_list:
                mode = 'sekkai_ValDataset'
            else:
                continue

            cv2.imwrite(str(self.output_path / f'sekkai/images/{mode}/{image_path}.png'), image)
            cv2.imwrite(str(self.output_path / f'sekkai/masks/{mode}/{mask_path}.png'), mask)

    def _plot_box(self, image_path, mask_path, anno):
        os.makedirs('./output/plot/images', exist_ok=True)
        os.makedirs('./output/plot/masks', exist_ok=True)
        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path))
        for a in anno.astype('int64'):
            cv2.rectangle(image,
                          (a[0], a[1]),
                          (a[2], a[3]),
                          (0, 255, 0), 2)
            cv2.imwrite(f'./output/plot/images/{image_path.stem}.png', image)
            cv2.rectangle(mask,
                          (a[0], a[1]),
                          (a[2], a[3]),
                          (0, 255, 0), 2)
            cv2.imwrite(f'./output/plot/masks/{mask_path.stem}.png', mask)

    def _make_label(self, image_list, mask_list, mode, dir_name):
        for image_path, mask_path in tqdm(zip(image_list, mask_list)):
            transform_anno = Mask2BBox()
            anno = transform_anno(mask_list=mask_path)

            image = cv2.imread(str(image_path))
            height, width, _ = image.shape

            self._plot_box(image_path, mask_path, anno)

            with open(f'{self.output_path}/{mode}/labels/{dir_name}/{image_path.stem}.txt', 'w') as f:
                for a in anno:
                    f.write(f'{a[4]} '
                            f'{(a[0] + a[2]) / 2 / width} '
                            f'{(a[1] + a[3]) / 2 / height} '
                            f'{(a[2] - a[0]) / width} '
                            f'{(a[3] - a[1]) / height}\n')

    def get_labels(self):
        self._make_label(
            (self.output_path / 'sekkai/images/sekkai_TrainDataset/').glob('*.png'), 
            (self.output_path / 'sekkai/masks/sekkai_TrainDataset/').glob('*.png'),
            'sekkai', 'sekkai_TrainDataset')

        self._make_label(
            (self.output_path / 'sekkai/images/sekkai_TestDataset/').glob('*.png'),
            (self.output_path / 'sekkai/masks/sekkai_TestDataset/').glob('*.png'),
            'sekkai', 'sekkai_TestDataset')

        self._make_label(
            (self.output_path / 'sekkai/images/sekkai_ValDataset/').glob('*.png'),
            (self.output_path / 'sekkai/masks/sekkai_ValDataset/').glob('*.png'),
            'sekkai', 'sekkai_ValDataset')


if __name__ == '__main__':

    root_path = Path('../../../dataset/original_images/preprocessed')
    output_path = Path('../../../dataset/original_images/edited/')

    os.makedirs(output_path / 'sekkai', exist_ok=True)
    os.makedirs(output_path / 'all', exist_ok=True)
    create_dataset = CreateDataset(root_path, output_path)

#     print('>>>>>>>> no parameter dataset')
#     create_dataset.get_no_parameter_dataset()
#     print('>>>>>>>> divide data')
#     create_dataset.get_divide_data()
    print('>>>>>>>> get labels')
    create_dataset.get_labels()
