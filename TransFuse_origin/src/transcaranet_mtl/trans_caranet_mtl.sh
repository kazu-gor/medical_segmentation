export PYTHONPATH="../../"

SAVE_PATH='trans_caranet_nash_mtl'
MTL='nashmtl'
mkdir -p ./logs/$SAVE_PATH

python ./train_trans_caranet_nash.py \
    --batchsize 16 \
    --tuning True \
    --mtl $MTL \
    --train_save $SAVE_PATH
    | tee ./logs/$SAVE_PATH/training_result.txt

python ./test_trans_caranet_nash.py \
    --save_path "./results/$SAVE_PATH/" \
    --pth_path "./snapshots/$SAVE_PATH/Transfuse-best.pth" \
    --pth_path2 "./snapshots$SAVE_PATH/Discriminator-best.pth" \
    | tee ./logs/$SAVE_PATH/evaluation_results.txt
