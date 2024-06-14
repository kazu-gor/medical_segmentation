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
# Setting env
# =====================================================================

TRAIN_WEIGHT="polyp491_20"
echo ">>> TRAIN_WEIGHT=$TRAIN_WEIGHT"
SAVE_WEIGHT="${TRAIN_WEIGHT}_AttnTransFuse"

python3 ../../tools/slack_bot.py --text ">>> TRAIN_WEIGHT=$TRAIN_WEIGHT"

# =====================================================================
# Attention TransFuse
# =====================================================================

# echo ">>> python3 ./train_yolo.py"
# python3 ./train_yolo.py | tee ./logs/train_yolo.log
# python3 ../../tools/slack_bot.py --text "YOLO Training is done"

conf=(0.01 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50)
max_det=(10)

# conf=(0.10)
# max_det=(10, 8, 6, 4)

for co in ${conf[@]}
do
    # for md in ${max_det[@]}
    # do
    echo "conf: $co"
    echo "conf: $md"
    python3 ../../tools/slack_bot.py --text ">>> conf=$co, max_det=$md"

    echo ">>> python3 ./test_yolo.py"
    python3 ./test_yolo.py --mode attention --max_det 10 --conf $co --weight $TRAIN_WEIGHT | tee ./logs/attention_test_yolo.log
    mv ./dataset_attn/sekkai_TrainDataset/attention/ ./dataset_attn/sekkai_TrainDataset/attention_1
    mv ./dataset_attn/sekkai_ValDataset/attention/ ./dataset_attn/sekkai_ValDataset/attention_1
    mv ./dataset_attn/sekkai_TestDataset/attention/ ./dataset_attn/sekkai_TestDataset/attention_1
    python3 ./test_yolo.py --mode attention --max_det 5 --conf $co --weight $TRAIN_WEIGHT | tee ./logs/attention_test_yolo.log
    mv ./dataset_attn/sekkai_TrainDataset/attention/ ./dataset_attn/sekkai_TrainDataset/attention_2
    mv ./dataset_attn/sekkai_ValDataset/attention/ ./dataset_attn/sekkai_ValDataset/attention_2
    mv ./dataset_attn/sekkai_TestDataset/attention/ ./dataset_attn/sekkai_TestDataset/attention_2
    python3 ../../tools/slack_bot.py --text "test_yolo.py is done."

    echo ">>> python3 ./train_attn_trans.py"
    python3 ./train_attn_trans.py --train_save $SAVE_WEIGHT | tee ./logs/train_attn_trans.log
    python3 ../../tools/slack_bot.py --text "Attention TransFuse training is done"

    echo ">>> python3 ./test_attn_trans.py"
    python3 ./test_attn_trans.py --pth_path "./snapshots/$SAVE_WEIGHT/TransFuse-best.pth" | tee ./logs/test_attn_trans.log
    python3 ../../tools/slack_bot.py --text "`cat ./logs/test_attn_trans.log`"
    # done
done
