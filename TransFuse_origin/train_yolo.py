import cv2
from ultralytics.models.yolo.detect import DetectionTrainer


def train_yolo(mode, pretrainer: DetectionTrainer):
    for phase in mode:
        if phase == 'train':
            preds, score, img_file_list, stop_flag = pretrainer.train()
            pretrainer.save_model()
        else:
            validator = pretrainer.get_validator()
            preds, score, img_file_list, stop_flag = validator(
                trainer=pretrainer)

        top1_score, top1_index = score.max(dim=1)
        top1_score = top1_score.squeeze()
        top1_index = top1_index.squeeze()
        top1_box = preds[range(preds.shape[0]), top1_index]

        for j, img_file in enumerate(img_file_list):
            image = cv2.imread(img_file)
            x1, y1, x2, y2 = map(int, top1_box[j])
            x1, y1, x2, y2 = max(0, x1-5), max(0, y1-5), min(
                image.shape[1], x2+5), min(image.shape[0], y2+5)
            image = image[y1:y2, x1:x2]
            image = cv2.resize(image, (352, 352))

            gt_path = f"../../../dataset_v0/TrainDataset/masks/{img_file.split('/')[-1]}"
            gt = cv2.imread(gt_path, 0)
            gt = gt[y1:y2, x1:x2]
            gt = cv2.resize(gt, (352, 352))

            # if image is empty, save the original image
            if image.shape[0] == 0 or image.shape[1] == 0:
                original_img_path = \
                    f"./dataset/sekkai_TrainDataset/images/{img_file.split('/')[-1]}"
                original_gt_path = original_img_path.replace('images', 'masks')
                image = cv2.imread(original_img_path)
                gt = cv2.imread(original_gt_path, 0)

            cv2.imwrite(
                f'./dataset/preprocessing/images/{img_file.split("/")[-1]}', image)
            cv2.imwrite(
                f'./dataset/preprocessing/masks/{img_file.split("/")[-1]}', gt)

        if phase == 'train':
            return stop_flag


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
    train_yolo(['train'], trainer)
