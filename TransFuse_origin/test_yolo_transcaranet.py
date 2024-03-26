import glob

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionPredictor


def test_yolo_polyp491():
    args = dict(
        model='ultralytics/runs/detect/polyp491_18/weights/last.pt',
        data='polyp491.yaml',
        single_cls=True,
        imgsz=640,
        batch=1,
        workers=4,
        name='polyp491_',
        save=True,
        # max_det=1,
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
            single_cls=True,
            imgsz=640,
            max_det=1,
            conf=0.01,
            save=True,
            save_txt=True,
            save_conf=True,
            save_crop=True,
        )


if __name__ == '__main__':
    # test_yolo_polyp491()

    predict_yolo_polyp491(
        'ultralytics/runs/detect/polyp491_18/weights/last.pt')
