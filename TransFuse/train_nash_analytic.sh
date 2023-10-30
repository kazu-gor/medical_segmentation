SAVE_PATH='TransFuse_nash_mtl_b8'
python3 ./train_trans_nash.py --train_save $SAVE_PATH --batchsize 8
python3 ./test_trans_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best.pth"
echo '---------------------------------------------------'
python3 ./test_trans_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best2.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best2.pth"
echo '---------------------------------------------------'
python3 ./test_trans_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-59.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-59.pth"
echo '---------------------------------------------------'
python3 ./test_trans_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-99.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-99.pth"
echo '---------------------------------------------------'


SAVE_PATH='TransFuse_nash_analytic_v1_b8'
python3 ./train_trans_nash.py --analytic True --analytic_version v1 --train_save $SAVE_PATH --batchsize 8
python3 ./test_trans_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best.pth"
echo '---------------------------------------------------'
python3 ./test_trans_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best2.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best2.pth"
echo '---------------------------------------------------'
python3 ./test_trans_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-59.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-59.pth"
echo '---------------------------------------------------'
python3 ./test_trans_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-99.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-99.pth"
echo '---------------------------------------------------'


SAVE_PATH='TransFuse_nash_analytic_v2_b8'
python3 ./train_trans_nash.py --analytic True --analytic_version v2 --train_save $SAVE_PATH --batchsize 8
python3 ./test_trans_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best.pth"
echo '---------------------------------------------------'
python3 ./test_trans_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best2.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best2.pth"
echo '---------------------------------------------------'
python3 ./test_trans_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-59.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-59.pth"
echo '---------------------------------------------------'
python3 ./test_trans_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-99.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-99.pth"
echo '---------------------------------------------------'


SAVE_PATH='TransFuse_nash_analytic_v3_b8'
python3 ./train_trans_nash.py --analytic True --analytic_version v3 --train_save $SAVE_PATH --batchsize 8
python3 ./test_trans_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best.pth"
echo '---------------------------------------------------'
python3 ./test_trans_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-best2.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-best2.pth"
echo '---------------------------------------------------'
python3 ./test_trans_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-59.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-59.pth"
echo '---------------------------------------------------'
python3 ./test_trans_nash.py --pth_path "./snapshots/$SAVE_PATH/Transfuse-99.pth" --pth_path2 "./snapshots/$SAVE_PATH/Discriminator-99.pth"
echo '---------------------------------------------------'

