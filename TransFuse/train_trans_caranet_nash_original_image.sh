SAVE_PATH='Transfuse_nash_original_image'
python3 ./train_trans_caranet_nash_original_image.py --train_save $SAVE_PATH --batchsize 8
python3 ./test_trans_caranet_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best.pth"
echo '---------------------------------------------------'
python3 ./test_trans_caranet_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best2.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best2.pth"
echo '---------------------------------------------------'
python3 ./test_trans_caranet_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-59.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-59.pth"
echo '---------------------------------------------------'
python3 ./test_trans_caranet_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-99.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-99.pth"
echo '---------------------------------------------------'

# SAVE_PATH='Transfuse_nash_original_image_fuse_weight_015'
# python3 ./train_trans_caranet_nash_original_image.py --train_save $SAVE_PATH --batchsize 8 --fuse_weight 0.15
# python3 ./test_trans_caranet_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best.pth"
# echo '---------------------------------------------------'
# python3 ./test_trans_caranet_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best2.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best2.pth"
# echo '---------------------------------------------------'
# python3 ./test_trans_caranet_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-59.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-59.pth"
# echo '---------------------------------------------------'
# python3 ./test_trans_caranet_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-99.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-99.pth"
# echo '---------------------------------------------------'

# SAVE_PATH='Transfuse_nash_original_image_fuse_weight_020'
# python3 ./train_trans_caranet_nash_original_image.py --train_save $SAVE_PATH --batchsize 8 --fuse_weight 0.2
# python3 ./test_trans_caranet_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best.pth"
# echo '---------------------------------------------------'
# python3 ./test_trans_caranet_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best2.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best2.pth"
# echo '---------------------------------------------------'
# python3 ./test_trans_caranet_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-59.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-59.pth"
# echo '---------------------------------------------------'
# python3 ./test_trans_caranet_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-99.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-99.pth"
# echo '---------------------------------------------------'

# SAVE_PATH='Transfuse_nash_original_image_fuse_weight_025'
# python3 ./train_trans_caranet_nash_original_image.py --train_save $SAVE_PATH --batchsize 8 --fuse_weight 0.2
# python3 ./test_trans_caranet_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best.pth"
# echo '---------------------------------------------------'
# python3 ./test_trans_caranet_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best2.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best2.pth"
# echo '---------------------------------------------------'
# python3 ./test_trans_caranet_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-59.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-59.pth"
# echo '---------------------------------------------------'
# python3 ./test_trans_caranet_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-99.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-99.pth"
# echo '---------------------------------------------------'

# SAVE_PATH='Transfuse_nash_original_image_fuse_weight_005'
# python3 ./train_trans_caranet_nash_original_image.py --train_save $SAVE_PATH --batchsize 8 --fuse_weight 0.05
# python3 ./test_trans_caranet_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best.pth"
# echo '---------------------------------------------------'
# python3 ./test_trans_caranet_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best2.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best2.pth"
# echo '---------------------------------------------------'
# python3 ./test_trans_caranet_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-59.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-59.pth"
# echo '---------------------------------------------------'
# python3 ./test_trans_caranet_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-99.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-99.pth"
# echo '---------------------------------------------------'

# SAVE_PATH='Transfuse_nash_original_image_fuse_weight_0075'
# python3 ./train_trans_caranet_nash_original_image.py --train_save $SAVE_PATH --batchsize 8 --fuse_weight 0.075
# python3 ./test_trans_caranet_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best.pth"
# echo '---------------------------------------------------'
# python3 ./test_trans_caranet_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best2.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best2.pth"
# echo '---------------------------------------------------'
# python3 ./test_trans_caranet_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-59.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-59.pth"
# echo '---------------------------------------------------'
# python3 ./test_trans_caranet_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-99.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-99.pth"
# echo '---------------------------------------------------'

# SAVE_PATH='Transfuse_nash_original_image_fuse_weight_0125'
# python3 ./train_trans_caranet_nash_original_image.py --train_save $SAVE_PATH --batchsize 8 --fuse_weight 0.125
# python3 ./test_trans_caranet_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best.pth"
# echo '---------------------------------------------------'
# python3 ./test_trans_caranet_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best2.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best2.pth"
# echo '---------------------------------------------------'
# python3 ./test_trans_caranet_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-59.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-59.pth"
# echo '---------------------------------------------------'
# python3 ./test_trans_caranet_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-99.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-99.pth"
# echo '---------------------------------------------------'

# SAVE_PATH='Transfuse_nash_original_image_fuse_weight_01125'
# python3 ./train_trans_caranet_nash_original_image.py --train_save $SAVE_PATH --batchsize 8 --fuse_weight 0.1125
# python3 ./test_trans_caranet_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best.pth"
# echo '---------------------------------------------------'
# python3 ./test_trans_caranet_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best2.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best2.pth"
# echo '---------------------------------------------------'
# python3 ./test_trans_caranet_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-59.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-59.pth"
# echo '---------------------------------------------------'
# python3 ./test_trans_caranet_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-99.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-99.pth"
# echo '---------------------------------------------------'

# SAVE_PATH='Transfuse_nash_original_image_fuse_weight_01375'
# python3 ./train_trans_caranet_nash_original_image.py --train_save $SAVE_PATH --batchsize 8 --fuse_weight 0.1375
# python3 ./test_trans_caranet_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best.pth"
# echo '---------------------------------------------------'
# python3 ./test_trans_caranet_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best2.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best2.pth"
# echo '---------------------------------------------------'
# python3 ./test_trans_caranet_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-59.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-59.pth"
# echo '---------------------------------------------------'
# python3 ./test_trans_caranet_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-99.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-99.pth"
# echo '---------------------------------------------------'

# SAVE_PATH='Transfuse_nash_original_image_fuse_weight_00875'
# python3 ./train_trans_caranet_nash_original_image.py --train_save $SAVE_PATH --batchsize 8 --fuse_weight 0.0875
# python3 ./test_trans_caranet_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best.pth"
# echo '---------------------------------------------------'
# python3 ./test_trans_caranet_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best2.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best2.pth"
# echo '---------------------------------------------------'
# python3 ./test_trans_caranet_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-59.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-59.pth"
# echo '---------------------------------------------------'
# python3 ./test_trans_caranet_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-99.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-99.pth"
# echo '---------------------------------------------------'
