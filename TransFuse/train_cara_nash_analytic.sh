SAVE_PATH='tmp'

# echo 'nash mtl'

# python3 ./train_trans_caranet_nash.py \
#     --epoch 5 \
#     --train_save $SAVE_PATH \
#     --batchsize 8

# python3 ./test_trans_caranet_nash.py \
#     --pth_path "./snapshots/$SAVE_PATH/Transfuse-best.pth" \
#     --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best.pth"

# echo '---------------------------------------------------'

# python3 ./test_trans_caranet_nash.py \
#     --pth_path "./snapshots/$SAVE_PATH/Transfuse-best2.pth" \
#     --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best2.pth"

# echo '---------------------------------------------------'

# python3 ./test_trans_caranet_nash.py \
#     --pth_path "./snapshots/$SAVE_PATH/Transfuse-59.pth" \
#     --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-59.pth"

# echo '---------------------------------------------------'

# python3 ./test_trans_caranet_nash.py \
#     --pth_path "./snapshots/$SAVE_PATH/Transfuse-99.pth" \
#     --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-99.pth"

# echo '---------------------------------------------------'


# SAVE_PATH='TransCaraNet_nash_analytic_v1'

# echo 'nash analytic v1'

# python3 ./train_trans_caranet_nash.py \
#     --epoch 6 \
#     --train_save $SAVE_PATH \
#     --analytic True \
#     --analytic_version v1 \
#     --batchsize 8

# python3 ./test_trans_caranet_nash.py \
#     --pth_path "./snapshots/$SAVE_PATH/Transfuse-best.pth" \
#     --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best.pth"

# echo '---------------------------------------------------'

# python3 ./test_trans_caranet_nash.py \
#     --pth_path "./snapshots/$SAVE_PATH/Transfuse-best2.pth" \
#     --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best2.pth"

# echo '---------------------------------------------------'

# python3 ./test_trans_caranet_nash.py \
#     --pth_path "./snapshots/$SAVE_PATH/Transfuse-59.pth" \
#     --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-59.pth"

# echo '---------------------------------------------------'

# python3 ./test_trans_caranet_nash.py \
#     --pth_path "./snapshots/$SAVE_PATH/Transfuse-99.pth" \
#     --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-99.pth"

# echo '---------------------------------------------------'


# SAVE_PATH='TransCaraNet_nash_analytic_v2'

echo 'nash analytic v2'

python3 ./train_trans_caranet_nash.py \
    --epoch 6 \
    --train_save $SAVE_PATH \
    --analytic True \
    --analytic_version v2 \
    --batchsize 8

# python3 ./test_trans_caranet_nash.py \
#     --pth_path "./snapshots/$SAVE_PATH/Transfuse-best.pth" \
#     --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best.pth"

# echo '---------------------------------------------------'

# python3 ./test_trans_caranet_nash.py \
#     --pth_path "./snapshots/$SAVE_PATH/Transfuse-best2.pth" \
#     --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best2.pth"

# echo '---------------------------------------------------'

# python3 ./test_trans_caranet_nash.py \
#     --pth_path "./snapshots/$SAVE_PATH/Transfuse-59.pth" \
#     --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-59.pth"

# echo '---------------------------------------------------'

# python3 ./test_trans_caranet_nash.py \
#     --pth_path "./snapshots/$SAVE_PATH/Transfuse-99.pth" \
#     --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-99.pth"

# echo '---------------------------------------------------'


# SAVE_PATH='TransCaraNet_nash_analytic_v3'

echo 'nash analytic v3'

python3 ./train_trans_caranet_nash.py \
    --epoch 6 \
    --analytic True \
    --analytic_version v3 \
    --train_save $SAVE_PATH \
    --batchsize 8

# python3 ./test_trans_caranet_nash.py \
#     --pth_path "./snapshots/$SAVE_PATH/Transfuse-best.pth" \
#     --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best.pth"

# echo '---------------------------------------------------'

# python3 ./test_trans_caranet_nash.py \
#     --pth_path "./snapshots/$SAVE_PATH/Transfuse-best2.pth" \
#     --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best2.pth"

# echo '---------------------------------------------------'

# python3 ./test_trans_caranet_nash.py \
#     --pth_path "./snapshots/$SAVE_PATH/Transfuse-59.pth" \
#     --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-59.pth"

# echo '---------------------------------------------------'

# python3 ./test_trans_caranet_nash.py \
#     --pth_path "./snapshots/$SAVE_PATH/Transfuse-99.pth" \
#     --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-99.pth"

# echo '---------------------------------------------------'


