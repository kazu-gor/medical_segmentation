SAVE_PATH='TransFuse_nash_mtl_crop_original'
python3 ./train_trans_nash.py --train_save $SAVE_PATH --batchsize 8 --train_path ../../dataset/clipping_20231212_original/TrainDataset --val_path ../../dataset/clipping_20231212_original/ValDataset
python3 ./test_trans_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best.pth" --data_path1 ../../dataset/clipping_20231212_original/TestDataset --data_path2 ../../dataset/clipping_20231212_original/sekkai_TestDataset
echo '---------------------------------------------------'
python3 ./test_trans_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best2.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best2.pth" --data_path1 ../../dataset/clipping_20231212_original/TestDataset --data_path2 ../../dataset/clipping_20231212_original/sekkai_TestDataset
echo '---------------------------------------------------'
python3 ./test_trans_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-59.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-59.pth" --data_path1 ../../dataset/clipping_20231212_original/TestDataset --data_path2 ../../dataset/clipping_20231212_original/sekkai_TestDataset
echo '---------------------------------------------------'
python3 ./test_trans_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-99.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-99.pth" --data_path1 ../../dataset/clipping_20231212_original/TestDataset --data_path2 ../../dataset/clipping_20231212_original/sekkai_TestDataset
echo '---------------------------------------------------'

SAVE_PATH='TransFuse_nash_mtl_crop_v1'
python3 ./train_trans_nash.py --train_save $SAVE_PATH --batchsize 8 --train_path ../../dataset/clipping_20231212/TrainDataset --val_path ../../dataset/clipping_20231212/ValDataset
python3 ./test_trans_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best.pth" --data_path1 ../../dataset/clipping_20231212/TestDataset --data_path2 ../../dataset/clipping_20231212/sekkai_TestDataset
echo '---------------------------------------------------'
python3 ./test_trans_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best2.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best2.pth" --data_path1 ../../dataset/clipping_20231212/TestDataset --data_path2 ../../dataset/clipping_20231212/sekkai_TestDataset
echo '---------------------------------------------------'
python3 ./test_trans_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-59.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-59.pth" --data_path1 ../../dataset/clipping_20231212/TestDataset --data_path2 ../../dataset/clipping_20231212/sekkai_TestDataset
echo '---------------------------------------------------'
python3 ./test_trans_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-99.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-99.pth" --data_path1 ../../dataset/clipping_20231212/TestDataset --data_path2 ../../dataset/clipping_20231212/sekkai_TestDataset
echo '---------------------------------------------------'
