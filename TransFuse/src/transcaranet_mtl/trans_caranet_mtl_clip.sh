export PYTHONPATH="../../"

DATA_PATH='clipping_20231212'
TRAIN_SAVE_PATH='trans_caranet_nash_clip'
python3 train_trans_caranet_nash.py --train_save $SAVE_PATH --train_path "../../dataset/$DATA_PATH/TrainDataset" --val_path "../../dataset/$DATA_PATH/ValDataset"
python3 test_trans_caranet_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best.pth" --test_path1 "../../dataset/$DATA_PATH/TestDataset" --test_path2 "../../dataset/$DATA_PATH/sekkai_TestDataset"
python3 test_trans_caranet_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best2.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best2.pth" --test_path1 "../../dataset/$DATA_PATH/TestDataset" --test_path2 "../../dataset/$DATA_PATH/sekkai_TestDataset"
python3 test_trans_caranet_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-59.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-59.pth" --test_path1 "../../dataset/$DATA_PATH/TestDataset" --test_path2 "../../dataset/$DATA_PATH/sekkai_TestDataset"
python3 test_trans_caranet_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-99.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-99.pth" --test_path1 "../../dataset/$DATA_PATH/TestDataset" --test_path2 "../../dataset/$DATA_PATH/sekkai_TestDataset"

DATA_PATH='clipping_20231212_original'
TRAIN_SAVE_PATH='trans_caranet_nash_clip_original'
python3 train_trans_caranet_nash.py --train_save $SAVE_PATH --train_path "../../dataset/$DATA_PATH/TrainDataset" --val_path "../../dataset/$DATA_PATH/ValDataset"
python3 test_trans_caranet_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best.pth" --test_path1 "../../dataset/$DATA_PATH/TestDataset" --test_path2 "../../dataset/$DATA_PATH/sekkai_TestDataset"
python3 test_trans_caranet_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best2.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best2.pth" --test_path1 "../../dataset/$DATA_PATH/TestDataset" --test_path2 "../../dataset/$DATA_PATH/sekkai_TestDataset"
python3 test_trans_caranet_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-59.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-59.pth" --test_path1 "../../dataset/$DATA_PATH/TestDataset" --test_path2 "../../dataset/$DATA_PATH/sekkai_TestDataset"
python3 test_trans_caranet_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-99.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-99.pth" --test_path1 "../../dataset/$DATA_PATH/TestDataset" --test_path2 "../../dataset/$DATA_PATH/sekkai_TestDataset"
