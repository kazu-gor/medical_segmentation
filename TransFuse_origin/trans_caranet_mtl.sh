SAVE_PATH='trans_caranet_nash'
python3 train_trans_caranet_nash.py --train_save $SAVE_PATH
python3 test_trans_caranet_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best.pth"
python3 test_trans_caranet_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best2.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best2.pth"
python3 test_trans_caranet_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-59.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-59.pth"
python3 test_trans_caranet_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-99.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-99.pth"
