# =====================================================================
# Git Update part
# =====================================================================

echo ">>> git pull"
git pull

cd /home/student/git/laboratory/python/py/murano_program/TransFuse_origin/ultralytics/
echo ">>> git pull"
git pull

cd /home/student/git/laboratory/python/py/murano_program/TransFuse_origin


for i in {103..107}
do
    # =====================================================================
    # Setting env
    # =====================================================================

    TRAIN_WEIGHT="polyp491_$i"
    # SAVE_DIR="./logs/preprocess"
    # LATEST_WEIGHT_DIR="./ultralytics/runs/detect/polyp491"

    echo ">>> TRAIN_WEIGHT=$TRAIN_WEIGHT"
    # echo ">>> SAVE_DIR=$SAVE_DIR"
    # echo ">>> LATEST_WEIGHT_DIR=$LATEST_WEIGHT_DIR"

    python3 ../../tools/slack_bot.py --text ">>> TRAIN_WEIGHT=$TRAIN_WEIGHT"

    # =====================================================================
    # Create Dir part
    # =====================================================================

    # if [ -d $SAVE_DIR ]; then
    #     i=1
    #     while [ -d $SAVE_DIR"_"$i ]
    #     do
    #         i=$((i+1))
    #     done
    #     SAVE_DIR=$SAVE_DIR"_"$i
    # else
    #     mkdir $SAVE_DIR
    # fi

    # if [ -d $LATEST_WEIGHT_DIR ]; then
    #     i=1
    #     while [ -d $LATEST_WEIGHT_DIR"_"$i ]
    #     do
    #         i=$((i+1))
    #     done
    #     LATEST_WEIGHT_DIR=$LATEST_WEIGHT_DIR"_"$i
    # else
    #     mkdir $LATEST_WEIGHT_DIR
    # fi

    # echo "save dir: $SAVE_DIR"
    # echo "LATEST_WEIGHT_DIR: $LATEST_WEIGHT_DIR"

    # =====================================================================
    # Training part mtl
    # =====================================================================

    python3 ../../tools/slack_bot.py --text ">>> MTL Training YOLOv8 and TransCaraNet"

    echo ">>> python3 ./train_yolo.py"
    python3 ./train_yolo.py | tee ./logs/train_yolo.log
    python3 ../../tools/slack_bot.py --text "YOLO Training is done"

    echo ">>> python3 ./test_yolo.py"
    python3 ./test_yolo.py --mode mtl | tee ./logs/test_yolo.log
    python3 ../../tools/slack_bot.py --text "YOLO Test is done"

    echo ">>> Rename file from last.pt to epoch100.pt"
    mv ./ultralytics/runs/detect/$TRAIN_WEIGHT/weights/last.pt ./ultralytics/runs/detect/$TRAIN_WEIGHT/weights/epoch100.pt || echo ">>> Skipped file renaming process"

    echo ">>> python3 ./train_trans_caranet.py"
    python3 ./train_yolo_caranet_nash.py --train_save $TRAIN_WEIGHT | tee ./logs/train_yolo_caranet_nash.log
    python3 ../../tools/slack_bot.py --text "TransCaraNet Training is done"

    python3 ../../tools/slack_bot.py --text "`cat ./logs/train_yolo_caranet_nash.log`"

    # =====================================================================
    # Test part mtl
    # =====================================================================

    python3 ../../tools/slack_bot.py --text ">>> MTL Test YOLOv8 and TransCaraNet"

    echo ">>> python3 ./test_yolo.py"
    python3 ./test_yolo.py --mode all --weight $TRAIN_WEIGHT | tee ./logs/test_yolo.log
    python3 ../../tools/slack_bot.py --text "YOLO Test is done"

    echo ">>> python3 ./test_yolo_caranet_nash.py --epoch best"
    python3 ./test_yolo_caranet_nash.py \
        --pth_path "./snapshots/$TRAIN_WEIGHT/Transfuse-best.pth" \
        --pth_path2  "./snapshots/$TRAIN_WEIGHT/Discriminator-best.pth" \
        --test_path1 "./datasets/preprocessing" | tee ./logs/test_yolo_caranet_nash.log
    python3 ../../tools/slack_bot.py --text "`cat ./logs/test_yolo_caranet_nash.log`"

    # =====================================================================
    # Training part
    # =====================================================================

    # python3 ../../tools/slack_bot.py --text ">>> Training YOLOv8 and TransCaraNet"

    # echo ">>> python3 ./train_yolo.py"
    # python3 ./train_yolo.py | tee ./logs/train_yolo.log
    # python3 ../../tools/slack_bot.py --text "YOLO Training is done"

    # echo ">>> python3 ./test_yolo.py"
    # python3 ./test_yolo.py --mode train | tee ./logs/test_yolo.log
    # python3 ../../tools/slack_bot.py --text "YOLO Test is done"

    # echo ">>> Rename file from last.pt to epoch100.pt"
    # mv ./ultralytics/runs/detect/$TRAIN_WEIGHT/weights/last.pt ./ultralytics/runs/detect/$TRAIN_WEIGHT/weights/epoch100.pt || echo ">>> Skipped file renaming process"

    # echo ">>> python3 ./train_trans_caranet.py"
    # python3 ./train_trans_caranet.py --train_save $TRAIN_WEIGHT | tee ./logs/train_trans_caranet.log
    # python3 ../../tools/slack_bot.py --text "TransCaraNet Training is done"

    # =====================================================================
    # Test part
    # =====================================================================

    # python3 ../../tools/slack_bot.py --text ">>> Test YOLOv8 and TransCaraNet"

    # echo ">>> python3 ./test_yolo.py"
    # python3 ./test_yolo.py --mode sekkai --weight $TRAIN_WEIGHT | tee ./logs/test_yolo.log
    # python3 ../../tools/slack_bot.py --text "YOLO Test is done"

    # echo ">>> python3 ./test_trans_caranet.py --epoch best"
    # python3 ./test_trans_caranet.py --train_weight $TRAIN_WEIGHT --epoch best | tee ./logs/test_trans_caranet_best.log
    # python3 ../../tools/slack_bot.py --text "`cat ./logs/test_trans_caranet_best.log`"
    # python3 ../../tools/slack_bot.py --text "`cat ./logs/test_trans_caranet_best.log`"
    # python3 ../../tools/slack_bot.py --text "`cat ./logs/test_trans_caranet_best.log`"
    # python3 ../../tools/slack_bot.py --text "`cat ./logs/test_trans_caranet_best.log`"
    # python3 ../../tools/slack_bot.py --text "`cat ./logs/test_trans_caranet_best.log`"
    # python3 ../../tools/slack_bot.py --text "`cat ./logs/test_trans_caranet_best.log`"
    # python3 ../../tools/slack_bot.py --text "`cat ./logs/test_trans_caranet_best.log`"
    # python3 ../../tools/slack_bot.py --text "`cat ./logs/test_trans_caranet_best.log`"

    # echo ">>> python3 ./test_trans_caranet.py --epoch 39"
    # python3 ./test_trans_caranet.py --train_weight $TRAIN_WEIGHT --epoch 39 | tee ./logs/test_trans_caranet_39.log
    # python3 ../../tools/slack_bot.py --text "`cat ./logs/test_trans_caranet_39.log`"

    # echo ">>> python3 ./test_trans_caranet.py --epoch 49"
    # python3 ./test_trans_caranet.py --train_weight $TRAIN_WEIGHT --epoch 49 | tee ./logs/test_trans_caranet_49.log
    # python3 ../../tools/slack_bot.py --text "`cat ./logs/test_trans_caranet_49.log`"

    # echo ">>> python3 ./test_trans_caranet.py --epoch 59"
    # python3 ./test_trans_caranet.py --train_weight $TRAIN_WEIGHT --epoch 59 | tee ./logs/test_trans_caranet_59.log
    # python3 ../../tools/slack_bot.py --text "`cat ./logs/test_trans_caranet_59.log`"

    # echo ">>> python3 ./test_trans_caranet.py --epoch 69"
    # python3 ./test_trans_caranet.py --train_weight $TRAIN_WEIGHT --epoch 69 | tee ./logs/test_trans_caranet_69.log
    # python3 ../../tools/slack_bot.py --text "`cat ./logs/test_trans_caranet_69.log`"

    # echo ">>> python3 ./test_trans_caranet.py --epoch 99"
    # python3 ./test_trans_caranet.py --train_weight $TRAIN_WEIGHT --epoch 99 | tee ./logs/test_trans_caranet_99.log
    # python3 ../../tools/slack_bot.py --text "`cat ./logs/test_trans_caranet_99.log`"

    # =====================================================================

    # Original TransCaraNet
    # =====================================================================

    # python3 ../../tools/slack_bot.py --text ">>> Train and Test Original TransCaraNet"

    # echo ">>> python3 ./train_trans_caranet_origin.py"
    # python3 ./train_trans_caranet_origin.py | tee ./logs/train_trans_caranet_origin.log
    # python3 ../../tools/slack_bot.py --text "TransCaraNet Training is done"

    # echo ">>> python3 ./test_trans_caranet_origin.py --epoch best"
    # python3 ./test_trans_caranet_origin.py  --epoch best | tee ./logs/test_trans_caranet_origin_best.log
    # python3 ../../tools/slack_bot.py --text "`cat ./logs/test_trans_caranet_origin_best.log`"

    # echo ">>> python3 ./test_trans_caranet_origin.py --epoch 39"
    # python3 ./test_trans_caranet_origin.py  --epoch 39 | tee ./logs/test_trans_caranet_origin_39.log
    # python3 ../../tools/slack_bot.py --text "`cat ./logs/test_trans_caranet_origin_39.log`"

    # echo ">>> python3 ./test_trans_caranet_origin.py --epoch 49"
    # python3 ./test_trans_caranet_origin.py  --epoch 49 | tee ./logs/test_trans_caranet_origin_49.log
    # python3 ../../tools/slack_bot.py --text "`cat ./logs/test_trans_caranet_origin_49.log`"

    # echo ">>> python3 ./test_trans_caranet_origin.py --epoch 59"
    # python3 ./test_trans_caranet_origin.py  --epoch 59 | tee ./logs/test_trans_caranet_origin_59.log
    # python3 ../../tools/slack_bot.py --text "`cat ./logs/test_trans_caranet_origin_59.log`"

    # echo ">>> python3 ./test_trans_caranet_origin.py --epoch 69"
    # python3 ./test_trans_caranet_origin.py  --epoch 69 | tee ./logs/test_trans_caranet_origin_69.log
    # python3 ../../tools/slack_bot.py --text "`cat ./logs/test_trans_caranet_origin_69.log`"

    # echo ">>> python3 ./test_trans_caranet_origin.py --epoch 99"
    # python3 ./test_trans_caranet_origin.py  --epoch 99 | tee ./logs/test_trans_caranet_origin_99.log
    # python3 ../../tools/slack_bot.py --text "`cat ./logs/test_trans_caranet_origin_99.log`"
done
