SAVE_PATH='resnet_b16'
# python ./train_res2net.py --train_save $SAVE_PATH --batchsize 16
python ./test_res2net.py --pth_path2 ./snapshots/$SAVE_PATH/Discriminator-best2.pth --save_path ./results/$SAVE_PATH/
python ./test_res2net.py --pth_path2 ./snapshots/$SAVE_PATH/Discriminator-59.pth --save_path ./results/$SAVE_PATH/
python ./test_res2net.py --pth_path2 ./snapshots/$SAVE_PATH/Discriminator-99.pth --save_path ./results/$SAVE_PATH/
