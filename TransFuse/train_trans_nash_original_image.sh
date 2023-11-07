# SAVE_PATH='Transfuse_nash_original_image'
# FUSE_WEIGHT=0.1
# # python3 ./train_trans_nash_original_image.py --train_save $SAVE_PATH --batchsize 8 --fuse_weight $FUSE_WEIGHT
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best.pth" --fuse_weight $FUSE_WEIGHT | tee log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best2.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best2.pth" --fuse_weight $FUSE_WEIGHT | tee -a log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-59.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-59.pth" --fuse_weight $FUSE_WEIGHT | tee -a log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-99.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-99.pth" --fuse_weight $FUSE_WEIGHT | tee -a log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# # echo `cat -A log_$SAVE_PATH.txt` | xargs -d '$' -IXXX python3 ../../tools/slack_bot.py --text XXX

# SAVE_PATH='Transfuse_nash_original_image_fuse_weight_005'
# FUSE_WEIGHT=0.05
# # python3 ./train_trans_nash_original_image.py --train_save $SAVE_PATH --batchsize 8 --fuse_weight $FUSE_WEIGHT
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best.pth" --fuse_weight $FUSE_WEIGHT | tee log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best2.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best2.pth" --fuse_weight $FUSE_WEIGHT | tee -a log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-59.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-59.pth" --fuse_weight $FUSE_WEIGHT | tee -a log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-99.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-99.pth" --fuse_weight $FUSE_WEIGHT | tee -a log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# # echo `cat -A log_$SAVE_PATH.txt` | xargs -d '$' -IXXX python3 ../../tools/slack_bot.py --text XXX

# SAVE_PATH='Transfuse_nash_original_image_fuse_weight_0075'
# FUSE_WEIGHT=0.075
# # python3 ./train_trans_nash_original_image.py --train_save $SAVE_PATH --batchsize 8 --fuse_weight $FUSE_WEIGHT
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best.pth" --fuse_weight $FUSE_WEIGHT | tee log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best2.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best2.pth" --fuse_weight $FUSE_WEIGHT | tee -a log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-59.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-59.pth" --fuse_weight $FUSE_WEIGHT | tee -a log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-99.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-99.pth" --fuse_weight $FUSE_WEIGHT | tee -a log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# # echo `cat -A log_$SAVE_PATH.txt` | xargs -d '$' -IXXX python3 ../../tools/slack_bot.py --text XXX

# SAVE_PATH='Transfuse_nash_original_image_fuse_weight_00875'
# FUSE_WEIGHT=0.0875
# # python3 ./train_trans_nash_original_image.py --train_save $SAVE_PATH --batchsize 8 --fuse_weight $FUSE_WEIGHT
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best.pth" --fuse_weight $FUSE_WEIGHT | tee log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best2.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best2.pth" --fuse_weight $FUSE_WEIGHT | tee -a log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-59.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-59.pth" --fuse_weight $FUSE_WEIGHT | tee -a log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-99.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-99.pth" --fuse_weight $FUSE_WEIGHT | tee -a log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# # echo `cat -A log_$SAVE_PATH.txt` | xargs -d '$' -IXXX python3 ../../tools/slack_bot.py --text XXX

# SAVE_PATH='Transfuse_nash_original_image_fuse_weight_01125'
# FUSE_WEIGHT=0.1125
# # python3 ./train_trans_nash_original_image.py --train_save $SAVE_PATH --batchsize 8 --fuse_weight $FUSE_WEIGHT
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best.pth" --fuse_weight $FUSE_WEIGHT | tee log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best2.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best2.pth" --fuse_weight $FUSE_WEIGHT | tee -a log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-59.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-59.pth" --fuse_weight $FUSE_WEIGHT | tee -a log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-99.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-99.pth" --fuse_weight $FUSE_WEIGHT | tee -a log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# # echo `cat -A log_$SAVE_PATH.txt` | xargs -d '$' -IXXX python3 ../../tools/slack_bot.py --text XXX

# SAVE_PATH='Transfuse_nash_original_image_fuse_weight_0125'
# FUSE_WEIGHT=0.125
# # python3 ./train_trans_nash_original_image.py --train_save $SAVE_PATH --batchsize 8 --fuse_weight $FUSE_WEIGHT
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best.pth" --fuse_weight $FUSE_WEIGHT | tee log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best2.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best2.pth" --fuse_weight $FUSE_WEIGHT | tee -a log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-59.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-59.pth" --fuse_weight $FUSE_WEIGHT | tee -a log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-99.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-99.pth" --fuse_weight $FUSE_WEIGHT | tee -a log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# # echo `cat -A log_$SAVE_PATH.txt` | xargs -d '$' -IXXX python3 ../../tools/slack_bot.py --text XXX

# SAVE_PATH='Transfuse_nash_original_image_fuse_weight_01375'
# FUSE_WEIGHT=0.1375
# # python3 ./train_trans_nash_original_image.py --train_save $SAVE_PATH --batchsize 8 --fuse_weight $FUSE_WEIGHT
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best.pth" --fuse_weight $FUSE_WEIGHT | tee log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best2.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best2.pth" --fuse_weight $FUSE_WEIGHT | tee -a log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-59.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-59.pth" --fuse_weight $FUSE_WEIGHT | tee -a log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-99.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-99.pth" --fuse_weight $FUSE_WEIGHT | tee -a log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# # echo `cat -A log_$SAVE_PATH.txt` | xargs -d '$' -IXXX python3 ../../tools/slack_bot.py --text XXX

# SAVE_PATH='Transfuse_nash_original_image_fuse_weight_015'
# FUSE_WEIGHT=0.15
# # python3 ./train_trans_nash_original_image.py --train_save $SAVE_PATH --batchsize 8 --fuse_weight $FUSE_WEIGHT
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best.pth" --fuse_weight $FUSE_WEIGHT | tee log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best2.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best2.pth" --fuse_weight $FUSE_WEIGHT | tee -a log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-59.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-59.pth" --fuse_weight $FUSE_WEIGHT | tee -a log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-99.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-99.pth" --fuse_weight $FUSE_WEIGHT | tee -a log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# # echo `cat -A log_$SAVE_PATH.txt` | xargs -d '$' -IXXX python3 ../../tools/slack_bot.py --text XXX

# SAVE_PATH='Transfuse_nash_original_image_fuse_weight_020'
# FUSE_WEIGHT=0.20
# # python3 ./train_trans_nash_original_image.py --train_save $SAVE_PATH --batchsize 8 --fuse_weight $FUSE_WEIGHT
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best.pth" --fuse_weight $FUSE_WEIGHT | tee log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best2.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best2.pth" --fuse_weight $FUSE_WEIGHT | tee -a log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-59.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-59.pth" --fuse_weight $FUSE_WEIGHT | tee -a log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-99.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-99.pth" --fuse_weight $FUSE_WEIGHT | tee -a log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# # echo `cat -A log_$SAVE_PATH.txt` | xargs -d '$' -IXXX python3 ../../tools/slack_bot.py --text XXX

# SAVE_PATH='Transfuse_nash_original_image_fuse_weight_025'
# FUSE_WEIGHT=0.25
# # python3 ./train_trans_nash_original_image.py --train_save $SAVE_PATH --batchsize 8 --fuse_weight $FUSE_WEIGHT
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best.pth" --fuse_weight $FUSE_WEIGHT | tee log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best2.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best2.pth" --fuse_weight $FUSE_WEIGHT | tee -a log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-59.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-59.pth" --fuse_weight $FUSE_WEIGHT | tee -a log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-99.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-99.pth" --fuse_weight $FUSE_WEIGHT | tee -a log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# # echo `cat -A log_$SAVE_PATH.txt` | xargs -d '$' -IXXX python3 ../../tools/slack_bot.py --text XXX

# SAVE_PATH='Transfuse_nashv2_original_image_fuse_weight_01125'
# FUSE_WEIGHT=0.1125
# python3 ./train_trans_nash_original_image.py --train_save $SAVE_PATH --batchsize 8 --fuse_weight $FUSE_WEIGHT --analytic True --analytic_version v2
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best.pth" --fuse_weight $FUSE_WEIGHT | tee log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best2.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best2.pth" --fuse_weight $FUSE_WEIGHT | tee -a log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-59.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-59.pth" --fuse_weight $FUSE_WEIGHT | tee -a log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-99.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-99.pth" --fuse_weight $FUSE_WEIGHT | tee -a log_$SAVE_PATH.txt
# echo '---------------------------------------------------'
# echo `cat -A log_$SAVE_PATH.txt` | xargs -d '$' -IXXX python3 ../../tools/slack_bot.py --text XXX

SAVE_PATH='Transfuse_nashv2_original_image_fuse_weight_01125'
FUSE_WEIGHT=0.5
python3 ./train_trans_nash_original_image.py --train_save $SAVE_PATH --batchsize 8 --fuse_weight $FUSE_WEIGHT
python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best.pth" --fuse_weight $FUSE_WEIGHT | tee log_$SAVE_PATH.txt
echo '---------------------------------------------------'
python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best2.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best2.pth" --fuse_weight $FUSE_WEIGHT | tee -a log_$SAVE_PATH.txt
echo '---------------------------------------------------'
python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-59.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-59.pth" --fuse_weight $FUSE_WEIGHT | tee -a log_$SAVE_PATH.txt
echo '---------------------------------------------------'
python3 ./test_trans_nash_original_image.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-99.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-99.pth" --fuse_weight $FUSE_WEIGHT | tee -a log_$SAVE_PATH.txt
echo '---------------------------------------------------'
echo `cat -A log_$SAVE_PATH.txt` | xargs -d '$' -IXXX python3 ../../tools/slack_bot.py --text XXX

