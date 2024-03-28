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

    root_path = '../../../dataset_v0/sekkai_TestDataset/images'
    img_files = glob.glob(f'{root_path}/*.png')

    model.predict(
        img_files,
        # imgsz=640,
        data='polyp491.yaml',
        # max_det=1,
        # conf=0.01,
        single_cls=True,
        save=True,
        save_txt=True,
        save_conf=True,
        save_crop=True,
        device='cpu'
    )


if __name__ == '__main__':
    # test_yolo_polyp491()

    predict_yolo_polyp491('/home/student/git/laboratory/python/py/murano_program/yolobv8/runs/detect/train5/weights/best.pt')
