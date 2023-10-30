# config
export PYTHONPATH="/home/student/git/laboratory/python/py/tools:/home/student/git/laboratory/python/py/detr:$PYTHONPATH"
TRAIN_SAVE_PATH='trans_caranet_nash_clip_v4'
MTL='nashmtl'

python ../../tools/slack_bot.py \
    --text "[`date --rfc-3339=seconds`] Training and Evaluation: $MTL"

python ./train_detr_trans_caranet_nash.py \
    --tuning True \
    --mtl $MTL \
    --train_save $TRAIN_SAVE_PATH \
    --train_path './dataset/TrainDataset' \
    --val_path './dataset/ValDataset' \
    --dataset_file panorama \
    --panorama_path ../../../dataset \
    --sekkai \
    --num_queries 1 \
    --bbox_loss_coef 0 \
    --giou_loss_coef 1 \
    --backbone resnet101 \
    --resume https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth \
    --epochs 500 \
    --batch_size 8 \
    --dilation \
    --device cuda \
    --no_aux_loss \
    --output_dir ./logs/cotr_v10 \
    --transformer cotr \
    || python ../../tools/slack_bot.py \
        --text "Error: TransCaraNet ($MTL) training Failed."

# python test_trans_caranet_nash.py \
#     --pth_path "./snapshots/$TRAIN_SAVE_PATH/Transfuse-best.pth" \
#     --pth_path2 "./snapshots/$TRAIN_SAVE_PATH/Discriminator-best.pth" \
#     --test_path1 "../../../dataset/detr_preprocessing/crop_resize" \
#     --test_path2 "../../../dataset/detr_preprocessing/crop_resize" \
#     --save_path "./results/Transfuse_S/" || \
#     python ../../tools/slack_bot.py \
#         --text "Error: TransCaraNet ($MTL) evaluation Failed."

