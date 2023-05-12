TRAIN_SAVE_PATH='trans_caranet_nash_clip_v2'
# TRAIN_SAVE_PATH='trans_caranet_stl_clip'
MTL='nashmtl'

python ../../git/laboratory/python/py/tools/slack_bot.py \
    --text "[`date --rfc-3339=seconds`] Training and Evaluation: $MTL"

# python train_trans_caranet_nash.py \
#     --tuning True \
#     --mtl $MTL \
#     --train_save $TRAIN_SAVE_PATH \
#     --train_path './dataset/clipping_20230415/TrainDataset' \
#     --val_path './dataset/clipping_20230415/ValDataset' || \
#     python ../../git/laboratory/python/py/tools/slack_bot.py \
#         --text "Error: TransCaraNet ($MTL) training Failed."

python test_trans_caranet_nash.py \
    --pth_path "./snapshots/$TRAIN_SAVE_PATH/Transfuse-best.pth" \
    --pth_path2 "./snapshots/$TRAIN_SAVE_PATH/Discriminator-best.pth" \
    --test_path1 "/home/student/git/laboratory/python/dataset/detr_preprocessing/crop_resize" \
    --test_path2 "/home/student/git/laboratory/python/dataset/detr_preprocessing/crop_resize" \
    --save_path "./results/Transfuse_S/" || \
    python ../../git/laboratory/python/py/tools/slack_bot.py \
        --text "Error: TransCaraNet ($MTL) evaluation Failed."

