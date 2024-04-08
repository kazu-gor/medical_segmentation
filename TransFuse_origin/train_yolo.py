from ultralytics import YOLO


if __name__ == '__main__':

    yolov8 = YOLO(model='yolov8n.pt', task='detect', verbose=True)
    yolov8.train(
        data='polyp491.yaml',
        epochs=100,
        # imgsz=640,
        # batch=8,
        workers=4,
        name='polyp491_',
        save=True,
        save_period=1,
        save_dir='preprocessing',
        single_cls=True,
    )
