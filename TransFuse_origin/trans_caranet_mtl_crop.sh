DATA_PATH='crop_TestDataset'
SAVE_PATH='trans_caranet_nash_clipping_20231221_s10px_m20px' 
python3 test_trans_caranet_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best.pth" --test_path1 "./dataset/$DATA_PATH" --test_path2 "./dataset/$DATA_PATH"
python3 test_trans_caranet_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best2.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best2.pth" --test_path1 "./dataset/$DATA_PATH" --test_path2 "./dataset/$DATA_PATH"
python3 test_trans_caranet_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-59.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-59.pth" --test_path1 "./dataset/$DATA_PATH" --test_path2 "./dataset/$DATA_PATH"
python3 test_trans_caranet_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-99.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-99.pth" --test_path1 "./dataset/$DATA_PATH" --test_path2 "./dataset/$DATA_PATH"

TRAIN_SAVE_PATH='trans_caranet_nash_clipping_20231221_s15px_m30px'
python3 test_trans_caranet_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best.pth" --test_path1 "./dataset/$DATA_PATH" --test_path2 "./dataset/$DATA_PATH"
python3 test_trans_caranet_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best2.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best2.pth" --test_path1 "./dataset/$DATA_PATH" --test_path2 "./dataset/$DATA_PATH"
python3 test_trans_caranet_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-59.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-59.pth" --test_path1 "./dataset/$DATA_PATH" --test_path2 "./dataset/$DATA_PATH"
python3 test_trans_caranet_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-99.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-99.pth" --test_path1 "./dataset/$DATA_PATH" --test_path2 "./dataset/$DATA_PATH"

TRAIN_SAVE_PATH='trans_caranet_nash_clipping_20231221_s20px_m35px'
python3 test_trans_caranet_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best.pth" --test_path1 "./dataset/$DATA_PATH" --test_path2 "./dataset/$DATA_PATH"
python3 test_trans_caranet_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best2.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best2.pth" --test_path1 "./dataset/$DATA_PATH" --test_path2 "./dataset/$DATA_PATH"
python3 test_trans_caranet_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-59.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-59.pth" --test_path1 "./dataset/$DATA_PATH" --test_path2 "./dataset/$DATA_PATH"
python3 test_trans_caranet_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-99.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-99.pth" --test_path1 "./dataset/$DATA_PATH" --test_path2 "./dataset/$DATA_PATH"
