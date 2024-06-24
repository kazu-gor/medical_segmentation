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

# for co in ${conf[@]}
# do
#     for md in ${max_det[@]}
#     do

#         echo "conf: $co"
#         echo "max_det: $md"
#         python3 ../../tools/slack_bot.py --text ">>> conf=$co, max_det=$md"

#         echo ">>> python3 ./test_yolo.py"
#         # if attention dir is not exist, make it
#         if [ ! -d "./dataset_attn/sekkai_TrainDataset/attention" ]; then
#             mkdir -p ./dataset_attn/sekkai_TrainDataset/attention
#         fi
#         if [ ! -d "./dataset_attn/sekkai_ValDataset/attention" ]; then
#             mkdir -p ./dataset_attn/sekkai_ValDataset/attention
#         fi
#         if [ ! -d "./dataset_attn/sekkai_TestDataset/attention" ]; then
#             mkdir -p ./dataset_attn/sekkai_TestDataset/attention
#         fi
#         if [ -d "./dataset_attn/sekkai_TestDataset/attention" ]; then
#             rm -rf ./dataset_attn/sekkai_TrainDataset/attention_${co}_${md}/*
#             rm -rf ./dataset_attn/sekkai_ValDataset/attention_${co}_${md}/*
#             rm -rf ./dataset_attn/sekkai_TestDataset/attention_${co}_${md}/*
#         fi
#         python3 ./test_yolo.py --mode attention --max_det $md --conf $co --weight $TRAIN_WEIGHT | tee ./logs/attention_test_yolo.log
#         mkdir ./dataset_attn/sekkai_TrainDataset/attention_${co}_${md}
#         mkdir ./dataset_attn/sekkai_ValDataset/attention_${co}_${md}
#         mkdir ./dataset_attn/sekkai_TestDataset/attention_${co}_${md}

#         mv ./dataset_attn/sekkai_TrainDataset/attention/* ./dataset_attn/sekkai_TrainDataset/attention_${co}_${md}
#         mv ./dataset_attn/sekkai_ValDataset/attention/* ./dataset_attn/sekkai_ValDataset/attention_${co}_${md}
#         mv ./dataset_attn/sekkai_TestDataset/attention/* ./dataset_attn/sekkai_TestDataset/attention_${co}_${md}
#     done
# done

for co in ${conf[@]}
do
    for md in ${max_det[@]}
    do
        echo "conf: $co"
        echo "max_det: $md"
        python3 ../../tools/slack_bot.py --text ">>> conf=$co, max_det=$md"

        echo ">>> python3 ./test_yolo.py"
        if [ ! -d "./dataset_attn/TrainDataset/attention" ]; then
            mkdir -p ./dataset_attn/TrainDataset/attention
        fi
        if [ ! -d "./dataset_attn/ValDataset/attention" ]; then
            mkdir -p ./dataset_attn/ValDataset/attention
        fi
        if [ ! -d "./dataset_attn/TestDataset/attention" ]; then
            mkdir -p ./dataset_attn/TestDataset/attention
        fi
        if [ -d "./dataset_attn/TestDataset/attention" ]; then
            rm -rf ./dataset_attn/TrainDataset/attention_${co}_${md}/*
            rm -rf ./dataset_attn/ValDataset/attention_${co}_${md}/*
            rm -rf ./dataset_attn/TestDataset/attention_${co}_${md}/*
        fi
        python3 ./test_yolo.py --mode attention --max_det $md --conf $co --weight $TRAIN_WEIGHT | tee ./logs/attention_test_yolo.log
        mkdir ./dataset_attn/TrainDataset/attention_${co}_${md}
        mkdir ./dataset_attn/ValDataset/attention_${co}_${md}
        mkdir ./dataset_attn/TestDataset/attention_${co}_${md}

        mv ./dataset_attn/TrainDataset/attention/* ./dataset_attn/TrainDataset/attention_${co}_${md}
        mv ./dataset_attn/ValDataset/attention/* ./dataset_attn/ValDataset/attention_${co}_${md}
        mv ./dataset_attn/TestDataset/attention/* ./dataset_attn/TestDataset/attention_${co}_${md}
    done
done
