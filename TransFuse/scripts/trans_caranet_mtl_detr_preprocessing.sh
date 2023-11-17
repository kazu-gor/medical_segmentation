export PYTHONPATH="/home/student/git/laboratory/python/py/tools:$PYTHONPATH"
TRAIN_SAVE_PATH='trans_caranet_nash_mtl_random_crop_v2'
MTL='nashmtl'

python3 ../tools/slack_bot.py \
    --text "`cat ./trans_caranet_mtl_detr_preprocessing.sh`"

# python train_trans_caranet_nash.py \
#     --tuning True \
#     --mtl $MTL \
#     --train_save $TRAIN_SAVE_PATH \
#     --train_path './dataset/clipping_20230514/TrainDataset' \
#     --val_path './dataset/clipping_20230514/ValDataset' || \
#     python ../../tools/slack_bot.py \
#         --text "Error: TransCaraNet ($MTL) training Failed."

python test_trans_caranet_nash.py \
    --pth_path "./snapshots/$TRAIN_SAVE_PATH/Transfuse-best.pth" \
    --pth_path2 "./snapshots/$TRAIN_SAVE_PATH/Discriminator-best.pth" \
    --test_path1 "../../../dataset/detr_preprocessing/crop_resize" \
    --test_path2 "../../../dataset/detr_preprocessing/crop_resize" \
    --save_path "./results/$TRAIN_SAVE_PATH/" || \
    python ../../tools/slack_bot.py \
        --text "Error: TransCaraNet ($MTL) evaluation Failed."

