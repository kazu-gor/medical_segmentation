export PYTHONPATH="/home/student/git/laboratory/python/py/tools:$PYTHONPATH"
TRAIN_SAVE_PATH='trans_caranet_nash_clip_v3'
# TRAIN_SAVE_PATH='trans_caranet_stl_clip'
MTL='nashmtl'

python ../../tools/slack_bot.py \
    --text "[`date --rfc-3339=seconds`] Training and Evaluation: $MTL"

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
    --test_path1 "./dataset/clipping_20230514/TestDataset" \
    --test_path2 "./dataset/clipping_20230514/sekkai_TestDataset" \
    --save_path "./results/Transfuse_S/" || \
    python ../../tools/slack_bot.py \
        --text "Error: TransCaraNet ($MTL) evaluation Failed."

