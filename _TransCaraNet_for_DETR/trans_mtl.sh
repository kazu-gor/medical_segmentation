export PYTHONPATH="/home/student/git/laboratory/python/py/detr:$PYTHONPATH"
export PYTHONPATH="/home/student/git/laboratory/python/py/tools:$PYTHONPATH"

python3 ../../tools/slack_bot.py --text "`cat ./trans_mtl.sh`"

MTL='nashmtl'
TRAIN_SAVE_PATH="transfuse_$MTL"
mkdir -p ./logs/$TRAIN_SAVE_PATH
python ../../tools/slack_bot.py --text "Start training and evaluation $MTL"
# python ./train_trans_nash.py \
#     --tuning True \
#     --mtl $MTL \
#     --train_save $TRAIN_SAVE_PATH \
#     | tee ./logs/$TRAIN_SAVE_PATH/training_result.txt
python ./test_trans_nash.py \
    --save_path "./results/$TRAIN_SAVE_PATH/" \
    --pth_path "./snapshots/$TRAIN_SAVE_PATH/Transfuse-59.pth" \
    --pth_path2 "./snapshots/$TRAIN_SAVE_PATH/Discriminator-59.pth" \
    | tee ./logs/$TRAIN_SAVE_PATH/evaluation_results.txt

MTL='mgda'
TRAIN_SAVE_PATH="transfuse_$MTL"
mkdir -p ./logs/$TRAIN_SAVE_PATH
python ../../tools/slack_bot.py --text "Start training and evaluation $MTL"
# python ./train_trans_nash.py \
#     --tuning True \
#     --mtl $MTL \
#     --train_save $TRAIN_SAVE_PATH \
#     | tee ./logs/$TRAIN_SAVE_PATH/training_result.txt
python ./test_trans_nash.py \
    --save_path "./results/$TRAIN_SAVE_PATH/" \
    --pth_path "./snapshots/$TRAIN_SAVE_PATH/Transfuse-59.pth" \
    --pth_path2 "./snapshots/$TRAIN_SAVE_PATH/Discriminator-59.pth" \
    | tee ./logs/$TRAIN_SAVE_PATH/evaluation_results.txt

MTL='pcgrad'
TRAIN_SAVE_PATH="transfuse_$MTL"
mkdir -p ./logs/$TRAIN_SAVE_PATH
python ../../tools/slack_bot.py --text "Start training and evaluation $MTL"
# python ./train_trans_nash.py \
#     --tuning True \
#     --mtl $MTL \
#     --train_save $TRAIN_SAVE_PATH \
#     | tee ./logs/$TRAIN_SAVE_PATH/training_result.txt
python ./test_trans_nash.py \
    --save_path "./results/$TRAIN_SAVE_PATH/" \
    --pth_path "./snapshots/$TRAIN_SAVE_PATH/Transfuse-59.pth" \
    --pth_path2 "./snapshots/$TRAIN_SAVE_PATH/Discriminator-59.pth" \
    | tee ./logs/$TRAIN_SAVE_PATH/evaluation_results.txt

MTL='stl'
TRAIN_SAVE_PATH="transfuse_$MTL"
mkdir -p ./logs/$TRAIN_SAVE_PATH
python ../../tools/slack_bot.py --text "Start training and evaluation $MTL"
# python ./train_trans_nash.py \
#     --tuning True \
#     --mtl $MTL \
#     --train_save $TRAIN_SAVE_PATH \
#     | tee ./logs/$TRAIN_SAVE_PATH/training_result.txt
python ./test_trans_nash.py \
    --save_path "./results/$TRAIN_SAVE_PATH/" \
    --pth_path "./snapshots/$TRAIN_SAVE_PATH/Transfuse-59.pth" \
    --pth_path2 "./snapshots/$TRAIN_SAVE_PATH/Discriminator-59.pth" \
    | tee ./logs/$TRAIN_SAVE_PATH/evaluation_results.txt

