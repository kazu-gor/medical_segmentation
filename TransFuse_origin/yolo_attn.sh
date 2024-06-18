# =====================================================================
# Git Update part
# =====================================================================

echo ">>> git pull"
git pull

cd /home/student/git/laboratory/python/py/murano_program/TransFuse_origin/ultralytics/
echo ">>> git pull"
git pull

cd /home/student/git/laboratory/python/py/murano_program/TransFuse_origin

# =====================================================================
# Attention TransFuse
# =====================================================================

TRAIN_WEIGHT="polyp491_2"
echo ">>> TRAIN_WEIGHT=$TRAIN_WEIGHT"
SAVE_WEIGHT="${TRAIN_WEIGHT}_AttnTransFuse"
python3 ../../tools/slack_bot.py --text ">>> TRAIN_WEIGHT=$TRAIN_WEIGHT"

conf=(0.01 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50)
max_det=(10 7 5 3 1)

for co in ${conf[@]}
do
    for md in ${max_det[@]}
    do

        echo "conf: $co"
        echo "max_det: $md"
        python3 ../../tools/slack_bot.py --text ">>> conf=$co, max_det=$md"

        echo ">>> python3 ./test_yolo.py"
        # if attention dir is not exist, make it
        if [ ! -d "./dataset_attn/sekkai_TrainDataset/attention" ]; then
            mkdir -p ./dataset_attn/sekkai_TrainDataset/attention
        fi
        if [ ! -d "./dataset_attn/sekkai_ValDataset/attention" ]; then
            mkdir -p ./dataset_attn/sekkai_ValDataset/attention
        fi
        if [ ! -d "./dataset_attn/sekkai_TestDataset/attention" ]; then
            mkdir -p ./dataset_attn/sekkai_TestDataset/attention
        fi
        if [ -d "./dataset_attn/sekkai_TestDataset/attention" ]; then
            rm -rf ./dataset_attn/sekkai_TestDataset/attention_1/*
            rm -rf ./dataset_attn/sekkai_TrainDataset/attention_1/*
            rm -rf ./dataset_attn/sekkai_ValDataset/attention_1/*
        fi
        python3 ./test_yolo.py --mode attention --max_det $md --conf $co --weight $TRAIN_WEIGHT | tee ./logs/attention_test_yolo.log
        mv ./dataset_attn/sekkai_TrainDataset/attention/* ./dataset_attn/sekkai_TrainDataset/attention_1/
        mv ./dataset_attn/sekkai_ValDataset/attention/* ./dataset_attn/sekkai_ValDataset/attention_1/
        mv ./dataset_attn/sekkai_TestDataset/attention/* ./dataset_attn/sekkai_TestDataset/attention_1/

        # if [ -d "./dataset_attn/sekkai_TestDataset/attention" ]; then
        #     rm -rf ./dataset_attn/sekkai_TestDataset/attention_2/*
        #     rm -rf ./dataset_attn/sekkai_TrainDataset/attention_2/*
        #     rm -rf ./dataset_attn/sekkai_ValDataset/attention_2/*
        # fi
        # python3 ./test_yolo.py --mode attention --max_det 5 --conf $co --weight $TRAIN_WEIGHT | tee ./logs/attention_test_yolo.log
        # # if attention dir is not exist, make it
        # if [ ! -d "./dataset_attn/sekkai_TrainDataset/attention" ]; then
        #     mkdir -p ./dataset_attn/sekkai_TrainDataset/attention
        # fi
        # if [ ! -d "./dataset_attn/sekkai_ValDataset/attention" ]; then
        #     mkdir -p ./dataset_attn/sekkai_ValDataset/attention
        # fi
        # if [ ! -d "./dataset_attn/sekkai_TestDataset/attention" ]; then
        #     mkdir -p ./dataset_attn/sekkai_TestDataset/attention
        # fi
        # mv ./dataset_attn/sekkai_TrainDataset/attention/* ./dataset_attn/sekkai_TrainDataset/attention_2/
        # mv ./dataset_attn/sekkai_ValDataset/attention/* ./dataset_attn/sekkai_ValDataset/attention_2/
        # mv ./dataset_attn/sekkai_TestDataset/attention/* ./dataset_attn/sekkai_TestDataset/attention_2/
        python3 ../../tools/slack_bot.py --text "test_yolo.py is done."

        # echo ">>> python3 ./train_attn_trans.py"
        # python3 ./train_attn_trans.py --train_save "${SAVE_WEIGHT}_${co}" | tee ./logs/train_attn_trans.log
        # python3 ../../tools/slack_bot.py --text "Attention TransFuse training is done"

        echo ">>> python3 ./test_attn_trans.py"
        python3 ./test_attn_trans.py --pth_path "./snapshots/${SAVE_WEIGHT}_${co}/TransFuse-best.pth"  1> >(tee ./logs/train_attn_trans_stdout.log >&1 ) 2> >(tee ./logs/train_attn_trans_stderr.log >&2)
        python3 ./test_attn_trans.py --pth_path "./snapshots/${SAVE_WEIGHT}_${co}/Transfuse-59.pth"  1> >(tee ./logs/train_attn_trans_stdout.log >&1 ) 2> >(tee ./logs/train_attn_trans_stderr.log >&2)
        python3 ./test_attn_trans.py --pth_path "./snapshots/${SAVE_WEIGHT}_${co}/Transfuse-99.pth"  1> >(tee ./logs/train_attn_trans_stdout.log >&1 ) 2> >(tee ./logs/train_attn_trans_stderr.log >&2)

        if [ $? -ne 0 ]; then
            python3 ../../tools/slack_bot.py --text "`cat ./logs/train_attn_trans_stderr.log`"
        else
            python3 ../../tools/slack_bot.py --text "`cat ./logs/train_attn_trans_stdout.log`"
        fi
    done
done

# =====================================================================
# Original TransFuse
# =====================================================================

python3 ../../tools/slack_bot.py --text ">>> Train and Test Original TransFuse"

echo ">>> python3 ./train_trans.py"
python3 ./train_trans.py --train_save "TransFuse" | tee ./logs/train_trans.log
python3 ../../tools/slack_bot.py --text "TransFuse Training is done"

echo ">>> python3 ./test_trans.py"
python3 ./test_trans.py --pth_path "./snapshots/TransFuse/TransFuse-best.pth"| tee ./logs/test_trans.log
python3 ../../tools/slack_bot.py --text "`cat ./logs/test_yolo.log`"

# =====================================================================
# Original TransCaraNet
# =====================================================================

python3 ../../tools/slack_bot.py --text ">>> Train and Test Original TransCaraNet"

echo ">>> python3 ./train_trans_caranet_origin.py"
python3 ./train_trans_caranet_origin.py | tee ./logs/train_trans_caranet_origin.log
python3 ../../tools/slack_bot.py --text "TransCaraNet Training is done"

echo ">>> python3 ./test_trans_caranet_origin.py --epoch best"
python3 ./test_trans_caranet_origin.py  --epoch best 1> >(tee ./logs/test_trans_caranet_origin_best_stdout.log >&1 ) 2> >(tee ./logs/test_trans_caranet_origin_best_stderr.log >&2)

if [ $? -ne 0 ]; then
    python3 ../../tools/slack_bot.py --text "`cat ./logs/test_trans_caranet_origin_best_stderr.log`"
else
    python3 ../../tools/slack_bot.py --text "`cat ./logs/test_trans_caranet_origin_best_stdout.log`"
fi
