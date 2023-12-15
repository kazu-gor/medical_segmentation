SAVE_PATH='PoltpPVT_nash'
python3 ./train_polyp_nash.py --train_save $SAVE_PATH
python3 ./test_polyp_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best.pth"
echo '---------------------------------------------------'
python3 ./test_polyp_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best2.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best2.pth"
echo '---------------------------------------------------'
python3 ./test_polyp_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-59.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-59.pth"
echo '---------------------------------------------------'
python3 ./test_polyp_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-99.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-99.pth"
echo '---------------------------------------------------'


