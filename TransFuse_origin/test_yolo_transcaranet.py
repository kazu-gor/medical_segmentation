import os
import glob
import cv2
import pathlib
from pathlib import Path

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionPredictor


def test_yolo_polyp491():
    args = dict(
        model='ultralytics/runs/detect/polyp491_34/weights/last.pt',
        data='polyp491.yaml',
        single_cls=True,
        imgsz=640,
        batch=1,
        workers=4,
        name='polyp491_',
        save=True,
        max_det=1,
        # conf=0.01,
        source='../../../dataset_v0/TestDataset/images',
        save_txt=True,
        save_conf=True,
        save_crop=True,
    )

    predictor = DetectionPredictor(overrides=args)
    predictor.predict_cli()


def predict_yolo_polyp491(weights):
    model = YOLO(weights)

    root_path = '../../../dataset_v0/TestDataset/images'
    img_files = glob.glob(f'{root_path}/*.png')

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


def crop_image(img_path, label_path):

    if isinstance(img_path, pathlib.PosixPath):
        img_path = str(img_path)
    if isinstance(label_path, pathlib.PosixPath):
        label_path = str(label_path)

    assert Path(img_path).stem == Path(label_path).stem, f"{Path(img_path).stem}, {Path(label_path).stem}"
    img = cv2.imread(img_path)
    img_h, img_w, _ = img.shape
    with open(label_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split(' ')
        _, x, y, w, h, _ = line
        x, y, w, h = float(x), float(y), float(w), float(h)
        x, y, w, h = int(x * img_w), int(y * img_h), int(w * img_w), int(h * img_h)
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


if __name__ == '__main__':
    # test_yolo_polyp491()

    # predict_yolo_polyp491(
    #     'ultralytics/runs/detect/polyp491_34/weights/last.pt')

    pred_path = Path('./ultralytics/runs/detect/predict9/labels/')
    # gt_path = Path('./datasets/dataset_v0/all/TestDataset/images/')
    gt_path = Path('./datasets/dataset_v0/all/TestDataset/masks/')
    gt_path_list = list(gt_path.glob('*.png'))
    gt_path_list_len = len(gt_path_list)
    print('Start gt (ropping...')

    print(len(list(pred_path.glob('*.txt'))))
    print(len(list(gt_path.glob('*.png'))))

    os.makedirs('./datasets/preprocessing/images', exist_ok=True)
    os.makedirs('./datasets/preprocessing/plottings', exist_ok=True)

    num_img = 0
    for num_img, label_file in enumerate(sorted((pred_path).glob('*.txt')), start=1):
        img_file = gt_path / f"{label_file.stem}.png"
        gt_path_list.remove(img_file)
        print(f"{img_file = }, {label_file = }")
        crop_image(img_file, label_file)

    print(f"{num_img = }")
    print(f"num_img_non = {len(gt_path_list)}")
    print(f"{gt_path_list_len = }")

    assert num_img + len(gt_path_list) == gt_path_list_len

    for gt_file in gt_path_list:
        img = cv2.imread(str(gt_file))
        cv2.imwrite(f'./datasets/preprocessing/images/{gt_file.name}', img)
