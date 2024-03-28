import glob
import cv2
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

    assert Path(img_path).stem == Path(label_path).stem
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    with open(label_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split(' ')
        _, x, y, w, h = line
        x, y, w, h = float(x), float(y), float(w), float(h)
        x, y, w, h = int(x * w), int(y * h), int(w ** 2), int(h ** 2)
        x1, y1 = x - w // 2, y - h // 2
        x2, y2 = x + w // 2, y + h // 2
        crop_img = img[y1:y2, x1:x2]
        cv2.imwrite(
            f'./datasets/preprocessing/images/{Path(img_path).name}', crop_img)
        # draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(
            f'./datasets/preprocessing/plottings/{Path(img_path).name}', img)


if __name__ == '__main__':
    # test_yolo_polyp491()

    predict_yolo_polyp491(
        'ultralytics/runs/detect/polyp491_34/weights/last.pt')

    pred_path = Path('ultralytics/runs/detect/predict7')
    gt_path = Path('./dataset_v0/sekkai_TestDataset/images/')
    for img_file, label_file in zip(sorted(gt_path.glob('*.png')), sorted((pred_path / 'labels').glob('*.txt'))):
        crop_image(img_file, label_file)
