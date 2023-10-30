# SAVE_PATH='Transfuse_nashmtl_crop_x_30percent_v1'
# SAVE_PATH='Transfuse_nashmtl_crop_x_40percent_v1'
SAVE_PATH='Transfuse_nashmtl_crop_x1_5_x2_30_v1'
# SAVE_PATH='Transfuse_Transtransfuse_MTL_nashent_v1'
# SAVE_PATH='Transfuse_MTL_nash'

# DATA_PATH="../../../dataset/cropping_x_30percent/TestDataset/"
# DATA_PATH2="../../../dataset/cropping_x_30percent/TestDataset/"

# DATA_PATH="../../../dataset/cropping_x_40percent/TestDataset/"
# DATA_PATH2="../../../dataset/cropping_x_40percent/TestDataset/"

TRAIN_PATH="../../../dataset/cropping_x1_5_x2_30/TrainDataset/"
VAL_PATH="../../../dataset/cropping_x1_5_x2_30/ValDataset/"

DATA_PATH="../../../dataset/cropping_x1_5_x2_30/TestDataset/"
DATA_PATH2="../../../dataset/cropping_x1_5_x2_30/TestDataset/"

# DATA_PATH="./dataset/TestDataset/"
# DATA_PATH2="./dataset/sekkai_TestDataset/"

python3 ./train_trans_nash.py \
    --train_save $SAVE_PATH \
    --train_path $TRAIN_PATH \
    --val_path $VAL_PATH

PTH_NAME='best'
python3 ./test_trans_nash.py \
    --pth_path "./snapshots/$SAVE_PATH/Transfuse-$PTH_NAME.pth" \
    --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-$PTH_NAME.pth" \
    --data_path1 $DATA_PATH \
    --data_path2 $DATA_PATH2

echo
echo '---------------------------------------------------'
echo

PTH_NAME='best2'
python3 ./test_trans_nash.py \
    --pth_path "./snapshots/$SAVE_PATH/Transfuse-$PTH_NAME.pth" \
    --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-$PTH_NAME.pth" \
    --data_path1 $DATA_PATH \
    --data_path2 $DATA_PATH2

echo
echo '---------------------------------------------------'
echo

PTH_NAME='59'
python3 ./test_trans_nash.py \
    --pth_path "./snapshots/$SAVE_PATH/Transfuse-$PTH_NAME.pth" \
    --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-$PTH_NAME.pth" \
    --data_path1 $DATA_PATH \
    --data_path2 $DATA_PATH2

echo
echo '---------------------------------------------------'
echo

PTH_NAME='99'
python3 ./test_trans_nash.py \
    --pth_path "./snapshots/$SAVE_PATH/Transfuse-$PTH_NAME.pth" \
    --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-$PTH_NAME.pth" \
    --data_path1 $DATA_PATH \
    --data_path2 $DATA_PATH2

echo
