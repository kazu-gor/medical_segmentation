import os
import cv2
import pathlib
from pathlib import Path

from ultralytics import YOLO


class Predictor:

    def __init__(self, 
                 weights, 
                 mode='sekkai', 
                 dataset_root='./datasets/dataset_v0/',
                 yolo_runs_root='./ultralytics/runs/detect/',
                 verbose=True):
        self.weights = weights
        self.mode = mode
        self.verbose = verbose
        
        self.dataset_root = Path(dataset_root)
        self.yolo_runs_root = Path(yolo_runs_root)


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
        model = YOLO(self.weights)

        if self.mode == 'sekkai':
            root_path = self.dataset_root / 'sekkai/images/sekkai_TestDataset'
        elif self.mode == 'all':
            root_path = self.dataset_root / 'all/images/TestDataset'
        else:
            raise ValueError('Invalid mode')

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

    def _crop_image(self, img_path, label_path):

        if isinstance(img_path, pathlib.PosixPath):
            img_path = str(img_path)
        if isinstance(label_path, pathlib.PosixPath):
            label_path = str(label_path)

        assert Path(img_path).stem == Path(
            label_path).stem, f"{Path(img_path).stem}, {Path(label_path).stem}"
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
            print(f"{x1 = }, {y1 = }, {x2 = }, {y2 = }")
            crop_img = img[y1:y2, x1:x2]
            cv2.imwrite(
                f'./datasets/preprocessing/images/{Path(img_path).name}', crop_img)
            # draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite(
                f'./datasets/preprocessing/plottings/{Path(img_path).name}', img)
    
    def crop_images(self, img_type):
        pred_path = self.yolo_runs_root / f"{self._get_latest_predict_dir()}/labels"
        if self.mode == 'sekkai':
            gt_path = self.dataset_root / f'sekkai/{img_type}/sekkai_TestDataset'
        elif self.mode == 'all':
            gt_path = self.dataset_root / f'all/{img_type}/TestDataset'
        else:
            raise ValueError('Invalid mode')

        if os.path.exists(f'./datasets/preprocessing/{img_type}'):
            flag = input('Do you want to delete the existing files? [y/n]: ')
            if flag == 'y':
                os.system(f'rm -r ./datasets/preprocessing/{img_type}')
                os.system('rm -r ./datasets/preprocessing/plottings')
            elif flag == 'n':
                return
            else:
                raise ValueError('Invalid input')

        os.makedirs(f'./datasets/preprocessing/{img_type}', exist_ok=True)
        os.makedirs('./datasets/preprocessing/plottings', exist_ok=True)

        gt_path_list = list(gt_path.glob('*.png'))
        gt_path_list_len = len(gt_path_list)

        num_img = 0
        for num_img, label_file in enumerate(sorted((pred_path).glob('*.txt')), start=1):
            img_file = gt_path / f"{label_file.stem}.png"
            gt_path_list.remove(img_file)
            if self.verbose:
                print(f"{img_file = }, {label_file = }")
            self._crop_image(img_file, label_file)

        if self.verbose:
            print(f"{num_img = }")
            print(f"num_img_non = {len(gt_path_list)}")
            print(f"{gt_path_list_len = }")

        assert num_img + len(gt_path_list) == gt_path_list_len

        for gt_file in gt_path_list:
            img = cv2.imread(str(gt_file))
            cv2.imwrite(f'./datasets/preprocessing/{img_type}/{gt_file.name}', img)


if __name__ == '__main__':
    
    predictor = Predictor(
            weights='ultralytics/runs/detect/polyp491_34/weights/last.pt',
            mode='sekkai',
            dataset_root='./datasets/dataset_v0/',
            yolo_runs_root='./ultralytics/runs/detect/',
            verbose=True,
            )

    predictor.predict_yolo_forPolyp()
    predictor.crop_images(img_type='masks')
    predictor.crop_images(img_type='images')

