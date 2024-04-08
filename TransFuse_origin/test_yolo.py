import os
import cv2
import pathlib
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageFilter

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

    def predict_yolo_forTest(self):

        if self.weights is None:
            raise ValueError('Please provide the weight path')
        if self.mode not in ['sekkai', 'all']:
            raise ValueError('Invalid mode')

        if self.mode == 'sekkai':
            root_path = self.dataset_root / 'sekkai/images/sekkai_TestDataset'
        elif self.mode == 'all':
            root_path = self.dataset_root / 'all/images/TestDataset'
        else:
            raise ValueError('Invalid mode')

        model = YOLO(self.weights)
        img_files = root_path.glob('*.png')
        img_files = [str(img_file) for img_file in img_files]

        model.predict(
            img_files,
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
        self.crop_images('images')
        self.crop_images('masks')

    def predict_yolo_forSegTrain(self):

        if self.mode == 'sekkai':
            root_path = self.dataset_root / 'sekkai/images/sekkai_TestDataset'
        elif self.mode == 'all':
            root_path = self.dataset_root / 'all/images/TestDataset'
        elif self.mode == 'train':
            root_path = self.dataset_root / 'sekkai/images/sekkai_TrainDataset'
        elif self.mode == 'val':
            root_path = self.dataset_root / 'sekkai/images/sekkai_ValDataset'
        else:
            raise ValueError('Invalid mode')

        train_weight_dir = self._get_latest_train_weight_dir()
        if train_weight_dir is None:
            raise ValueError('There is no trained model.')

        train_weight_epochs = Path(self.yolo_runs_root / train_weight_dir / 'weights').glob('*')
        img_files = root_path.glob('*.png')
        img_files = [str(img_file) for img_file in img_files]
        for weight in tqdm(list(train_weight_epochs)):
            model = YOLO(str(weight))
            model.predict(
                img_files,
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
            self.crop_images('images', sub_dir=str(weight.stem))
            self.crop_images('masks', sub_dir=str(weight.stem))

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

            while crop_img.shape[0] < 32:
                x1, x2 = x1 - (32 - crop_img.shape[0]) // 2, x2 + (32 - crop_img.shape[0]) // 2
                crop_img = img[y1:y2, x1:x2]
            while crop_img.shape[1] < 32:
                y1, y2 = y1 - (32 - crop_img.shape[0]) // 2, y2 + (32 - crop_img.shape[0]) // 2
                crop_img = img[y1:y2, x1:x2]

            crop_img = cv2.resize(crop_img, (352, 352))

            if self.verbose:
                print(f"{x1 = }, {y1 = }, {x2 = }, {y2 = }")

            if img_type == 'masks':
                # lower = np.array([0, 0, 0], dtype="uint8")
                # upper = np.array([255, 50, 255], dtype="uint8")
                # crop_img = cv2.inRange(crop_img, lower, upper)
                # crop_img = cv2.bitwise_not(crop_img)
                    
                gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (15, 15), 0)
                _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                crop_img = np.zeros_like(crop_img)
                cv2.drawContours(crop_img, contours, -1, (255, 255, 255), -1)

            if self.mode in ['train', 'val']:
                cv2.imwrite(
                    f'./datasets/{self.output_dir}/{self.mode}/{self.sub_dir}/{img_type}/{str(Path(img_path).name)}', crop_img)
            else:
                cv2.imwrite(
                    f'./datasets/{self.output_dir}/{img_type}/{Path(img_path).name}', crop_img)

            # draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if self.mode not in ['train', 'val']:
                cv2.imwrite(
                    f'./datasets/{self.output_dir}/plottings_{img_type}/{Path(img_path).name}', img)

    def _delete_existing_files(self, img_path, force=False):
        if isinstance(img_path, pathlib.PosixPath):
            img_path = str(img_path)
        if os.path.exists(img_path):
            if force:
                os.system(f'rm -r {img_path}')
                return
            flag = input(f'Do you want to delete {img_path}? [y/n]: ')
            if flag == 'y':
                os.system(f'rm -r {img_path}')
            elif flag == 'n':
                return
            else:
                raise ValueError('Invalid input')

    def crop_images(self, img_type, output_dir='preprocessing', sub_dir='.'):
        pred_path = self.yolo_runs_root / f"{self._get_latest_predict_dir()}/labels"
        self.output_dir = output_dir
        self.sub_dir = sub_dir

        if self.mode == 'sekkai':
            gt_path = self.dataset_root / f'sekkai/{img_type}/sekkai_TestDataset'
        elif self.mode == 'all':
            gt_path = self.dataset_root / f'all/{img_type}/TestDataset'
        elif self.mode == 'train':
            gt_path = self.dataset_root / f'sekkai/{img_type}/sekkai_TrainDataset'
        elif self.mode == 'val':
            gt_path = self.dataset_root / f'sekkai/{img_type}/sekkai_ValDataset'
        else:
            raise ValueError('Invalid mode')

        if self.mode in ['train', 'val']:
            self._delete_existing_files(
                Path(f'./datasets/{self.output_dir}/{self.mode}/{self.sub_dir}/{img_type}'),
                force=True)
            os.makedirs(f'./datasets/{self.output_dir}/{self.mode}/{self.sub_dir}/{img_type}')
        else:
            # delete existing files
            self._delete_existing_files(Path(f'./datasets/{self.output_dir}/{img_type}'), force=True)
            self._delete_existing_files(Path(f'./datasets/{self.output_dir}/plottings_{img_type}'), force=True)
            # create directories
            os.makedirs(f'./datasets/{self.output_dir}/{img_type}', exist_ok=True)
            os.makedirs(f'./datasets/{self.output_dir}/plottings_{img_type}', exist_ok=True)

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
            if self.mode in ['train', 'val']:
                cv2.imwrite(f'./datasets/{self.output_dir}/{self.mode}/{self.sub_dir}/{img_type}/{gt_file.name}', img)
            else:
                cv2.imwrite(f'./datasets/{self.output_dir}/{img_type}/{gt_file.name}', img)


if __name__ == '__main__':

    # predictor = Predictor(
    #     mode='train',
    #     dataset_root='./datasets/dataset_v1/',
    #     yolo_runs_root='./ultralytics/runs/detect/',
    #     verbose=False,
    # )
    # predictor.predict_yolo_forSegTrain()

    # predictor = Predictor(
    #     mode='val',
    #     dataset_root='./datasets/dataset_v1/',
    #     yolo_runs_root='./ultralytics/runs/detect/',
    #     verbose=False,
    # )
    # predictor.predict_yolo_forSegTrain()

    predictor = Predictor(
        weights='./ultralytics/runs/detect/polyp491_85/weights/epoch100.pt',
        mode='sekkai',
        dataset_root='./datasets/dataset_v1/',
        yolo_runs_root='./ultralytics/runs/detect/',
        verbose=False,
    )
    predictor.predict_yolo_forTest()
