from ultralytics.models.yolo.detect import DetectionTrainer


def get_yolo_trainer() -> DetectionTrainer:

    args = dict(
        model='yolov8n.pt',
        data='polyp491.yaml',
        epochs=100,
        single_cls=True,
        imgsz=640,
        batch=8,
        workers=4,
        name='polyp491_',
        save=True,
    )
    return DetectionTrainer(overrides=args)


if __name__ == '__main__':
    trainer = get_yolo_trainer()
    trainer.train()
