import os
import cv2
import pathlib
import numpy as np
from pathlib import Path

from ultralytics import YOLO


class Predictor:

    def __init__(self, 
                 weights=None, 
                 mode='sekkai', 
                 dataset_root='./datasets/dataset_v0/',
                 yolo_runs_root='./ultralytics/runs/detect/',
                 verbose=True):
        self.weights = weights
        self.mode = mode
        self.verbose = verbose
        
        self.dataset_root = Path(dataset_root)
        self.yolo_runs_root = Path(yolo_runs_root)
        
        self.train_epoch = 1

    def _get_latest_train_weight_dir(self):

        directory = self.yolo_runs_root

        train_weight_dirs = []
        for item in os.listdir(directory):
            if (item.startswith("polyp491_") \
                    and item != "polyp491_" \
                    and os.path.isdir(os.path.join(directory, item))) \
                    or item == "polyp491_":
                train_weight_dirs.append(item)

        train_weight_dirs.sort(key=lambda x: int(x[9:]) if x != "polyp491_" else 0)

        if train_weight_dirs:
            return train_weight_dirs[-1]
        else:
            return None

    def _get_latest_predict_dir(self):

        directory = self.yolo_runs_root

        predict_dirs = []
        for item in os.listdir(directory):
            if (item.startswith("predict") \
                    and item != "predict" \
                    and os.path.isdir(os.path.join(directory, item))) \
                    or item == "predict":
                predict_dirs.append(item)

        predict_dirs.sort(key=lambda x: int(x[7:]) if x != "predict" else 0)

        if predict_dirs:
            return predict_dirs[-1]
        else:
            return None

    def predict_yolo_forPolyp(self):
        if self.weights is None and self.mode == 'train':
            self.weights = self.yolo_runs_root / f"{self._get_latest_train_weight_dir()}/weights/last.pt"
        else: 
            raise ValueError('Weights is None is allowed only for training.')

        if self.mode == 'sekkai':
            root_path = self.dataset_root / 'sekkai/images/sekkai_TestDataset'
        elif self.mode == 'all':
            root_path = self.dataset_root / 'all/images/TestDataset'
        elif self.mode == 'train':
            root_path = self.dataset_root / 'sekkai/images/sekkai_TrainDataset'
        else:
            raise ValueError('Invalid mode')

        model = YOLO(self.weights)
        img_files = root_path.glob('*.png')

        for img_file in img_files:
            model.predict(
                img_file,
                imgsz=640,
                data='polyp491.yaml',
                max_det=1,
                # conf=0.01,
                single_cls=True,
                save=True,
                save_txt=True,
                save_conf=True,
                save_crop=True,
            )

        if self.verbose:
            print('Prediction is done.')
        self.crop_images('image')
        if self.verbose:
            print('Image Cropping is done.')
        self.crop_images('mask')
        if self.verbose:
            print('Mask Cropping is done.')


    def _crop_image(self, img_path, label_path, img_type):

        if isinstance(img_path, pathlib.PosixPath):
            img_path = str(img_path)
        if isinstance(label_path, pathlib.PosixPath):
            label_path = str(label_path)
        assert Path(img_path).stem == Path(label_path).stem, \
                f"{Path(img_path).stem}, {Path(label_path).stem}"

        img = cv2.imread(img_path)
        img_h, img_w, _ = img.shape
        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip().split(' ')
            _, x, y, w, h, _ = line
            x, y, w, h = float(x), float(y), float(w), float(h)
            x, y, w, h = int(x * img_w), int(y * img_h), int(w *
                                                             img_w), int(h * img_h)
            x1, y1 = x - w // 2, y - h // 2
            x2, y2 = x + w // 2, y + h // 2
            crop_img = img[y1:y2, x1:x2]
            crop_img = cv2.resize(crop_img, (352, 352))

            if self.verbose:
                print(f"{x1 = }, {y1 = }, {x2 = }, {y2 = }")

            if img_type == 'masks':
                lower = np.array([0, 0, 0], dtype="uint8")
                upper = np.array([255, 50, 255], dtype="uint8")
                crop_img = cv2.inRange(crop_img, lower, upper)

                crop_img = cv2.bitwise_not(crop_img)

            cv2.imwrite(
                f'./datasets/preprocessing/{img_type}/{Path(img_path).name}', crop_img)

            # draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite(
                f'./datasets/preprocessing/plottings_{img_type}/{Path(img_path).name}', img)

    def _delete_existing_files(self, img_path):
        if isinstance(img_path, pathlib.PosixPath):
            img_path = str(img_path)
        if os.path.exists(img_path):
            flag = input(f'Do you want to delete {img_path}? [y/n]: ')
            if flag == 'y':
                os.system(f'rm -r {img_path}')
            elif flag == 'n':
                return
            else:
                raise ValueError('Invalid input')

    def crop_images(self, img_type):
        pred_path = self.yolo_runs_root / f"{self._get_latest_predict_dir()}/labels"
        if self.mode == 'sekkai':
            gt_path = self.dataset_root / f'sekkai/{img_type}/sekkai_TestDataset'
        elif self.mode == 'all':
            gt_path = self.dataset_root / f'all/{img_type}/TestDataset'
        elif self.mode == 'train':
            gt_path = self.dataset_root / f'sekkai/{img_type}/sekkai_TrainDataset'
        else:
            raise ValueError('Invalid mode')

        check_path = Path(f'./datasets/preprocessing/{img_type}')
        self._delete_existing_files(check_path)
        check_path = Path(f'./datasets/preprocessing/plottings_{img_type}')
        self._delete_existing_files(check_path)
        check_path = Path('./datasets/preprocessing/train')
        self._delete_existing_files(check_path)

        if self.mode == 'train':
            try:
                os.makedirs(f'./datasets/preprocessing/train/epoch_{self.train_epoch}/{img_type}')
            except FileExistsError:
                self.train_epoch += 1
                os.makedirs(f'./datasets/preprocessing/train/epoch_{self.train_epoch}/{img_type}')

        # create directories
        os.makedirs(f'./datasets/preprocessing/{img_type}', exist_ok=True)
        os.makedirs(f'./datasets/preprocessing/plottings_{img_type}', exist_ok=True)

        gt_path_list = list(gt_path.glob('*.png'))
        gt_path_list_len = len(gt_path_list)
        num_img = 0
        for num_img, label_file in enumerate(sorted((pred_path).glob('*.txt')), start=1):
            img_file = gt_path / f"{label_file.stem}.png"
            gt_path_list.remove(img_file) # remove the image from the list
            self._crop_image(img_file, label_file, img_type)
            if self.verbose:
                print(f"{img_file = }, {label_file = }")

        if self.verbose:
            print(f"{num_img = }")
            print(f"num_img_non = {len(gt_path_list)}")
            print(f"{gt_path_list_len = }")

        assert num_img + len(gt_path_list) == gt_path_list_len, \
            f"{num_img = }, {len(gt_path_list) = }, {gt_path_list_len = }"

        # copy the remaining images
        for gt_file in gt_path_list:
            img = cv2.imread(str(gt_file))
            img = cv2.resize(img, (352, 352))
            cv2.imwrite(f'./datasets/preprocessing/{img_type}/{gt_file.name}', img)


if __name__ == '__main__':
    
    predictor = Predictor(
        weights='ultralytics/runs/detect/polyp491_62/weights/last.pt',
        mode='sekkai',
        dataset_root='./datasets/dataset_v0/',
        yolo_runs_root='./ultralytics/runs/detect/',
        verbose=True,
    )

    predictor.predict_yolo_forPolyp()
    predictor.crop_images(img_type='images')
    predictor.crop_images(img_type='masks')

