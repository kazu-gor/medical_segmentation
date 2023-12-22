# export PYTHONPATH="../../"

DATA_PATH='20231221_s10px_m20px'
SAVE_PATH='trans_caranet_nash_clipping_20231221_s10px_m20px' 
python3 train_trans_caranet_nash.py --train_save $SAVE_PATH --train_path "./dataset/$DATA_PATH/clipping/TrainDataset" --val_path "./dataset/$DATA_PATH/clipping/ValDataset"
python3 test_trans_caranet_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best.pth" --test_path1 "./dataset/$DATA_PATH/clipping/TestDataset" --test_path2 "./dataset/$DATA_PATH/clipping/sekkai_TestDataset"
python3 test_trans_caranet_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best2.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best2.pth" --test_path1 "./dataset/$DATA_PATH/clipping/TestDataset" --test_path2 "./dataset/$DATA_PATH/clipping/sekkai_TestDataset"
python3 test_trans_caranet_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-59.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-59.pth" --test_path1 "./dataset/$DATA_PATH/clipping/TestDataset" --test_path2 "./dataset/$DATA_PATH/clipping/sekkai_TestDataset"
python3 test_trans_caranet_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-99.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-99.pth" --test_path1 "./dataset/$DATA_PATH/clipping/TestDataset" --test_path2 "./dataset/$DATA_PATH/clipping/sekkai_TestDataset"

DATA_PATH='20231221_s15px_m30px'
TRAIN_SAVE_PATH='trans_caranet_nash_clipping_20231221_s15px_m30px'
python3 train_trans_caranet_nash.py --train_save $SAVE_PATH --train_path "./dataset/$DATA_PATH/clipping/TrainDataset" --val_path "./dataset/$DATA_PATH/clipping/ValDataset"
python3 test_trans_caranet_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best.pth" --test_path1 "./dataset/$DATA_PATH/clipping/TestDataset" --test_path2 "./dataset/$DATA_PATH/clipping/sekkai_TestDataset"
python3 test_trans_caranet_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best2.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best2.pth" --test_path1 "./dataset/$DATA_PATH/clipping/TestDataset" --test_path2 "./dataset/$DATA_PATH/clipping/sekkai_TestDataset"
python3 test_trans_caranet_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-59.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-59.pth" --test_path1 "./dataset/$DATA_PATH/clipping/TestDataset" --test_path2 "./dataset/$DATA_PATH/clipping/sekkai_TestDataset"
python3 test_trans_caranet_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-99.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-99.pth" --test_path1 "./dataset/$DATA_PATH/clipping/TestDataset" --test_path2 "./dataset/$DATA_PATH/clipping/sekkai_TestDataset"

DATA_PATH='20231221_s20px_m35px'
TRAIN_SAVE_PATH='trans_caranet_nash_clipping_20231221_s20px_m35px'
python3 train_trans_caranet_nash.py --train_save $SAVE_PATH --train_path "./dataset/$DATA_PATH/clipping/TrainDataset" --val_path "./dataset/$DATA_PATH/clipping/ValDataset"
python3 test_trans_caranet_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best.pth" --test_path1 "./dataset/$DATA_PATH/clipping/TestDataset" --test_path2 "./dataset/$DATA_PATH/clipping/sekkai_TestDataset"
python3 test_trans_caranet_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best2.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best2.pth" --test_path1 "./dataset/$DATA_PATH/clipping/TestDataset" --test_path2 "./dataset/$DATA_PATH/clipping/sekkai_TestDataset"
python3 test_trans_caranet_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-59.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-59.pth" --test_path1 "./dataset/$DATA_PATH/clipping/TestDataset" --test_path2 "./dataset/$DATA_PATH/clipping/sekkai_TestDataset"
python3 test_trans_caranet_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-99.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-99.pth" --test_path1 "./dataset/$DATA_PATH/clipping/TestDataset" --test_path2 "./dataset/$DATA_PATH/clipping/sekkai_TestDataset"
