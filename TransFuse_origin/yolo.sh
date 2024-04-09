TRAIN_WEIGHT="polyp491_88"

echo ">>> git pull"
git pull

cd /home/student/git/laboratory/python/py/murano_program/TransFuse_origin/ultralytics/
echo ">>> git pull"
git pull

cd /home/student/git/laboratory/python/py/murano_program/TransFuse_origin

# ---------------------- training part ----------------------

echo ">>> python3 ./train_yolo.py"
python3 ./train_yolo.py | tee ./logs/train_yolo.log
python3 ../../tools/slack_bot.py --text "`cat ./logs/train_yolo.log`"

echo ">>> python3 ./test_yolo.py"
python3 ./test_yolo.py --mode train | tee ./logs/test_yolo.log
python3 ../../tools/slack_bot.py --text "`cat ./logs/test_yolo.log`"

mv ./ultralytics/runs/detect/$TRAIN_WEIGHT/weights/last.pt ./ultralytics/runs/detect/$TRAIN_WEIGHT/weights/epoch100.pt

echo ">>> python3 ./train_trans_caranet.py"
python3 ./train_trans_caranet.py --train_save $TRAIN_WEIGHT | tee ./logs/train_trans_caranet.log
python3 ../../tools/slack_bot.py --text "`cat ./logs/train_trans_caranet.log`"

# ---------------------- test part ----------------------

echo ">>> python3 ./test_yolo.py"
python3 ./test_yolo.py --mode sekkai --weight $TRAIN_WEIGHT | tee ./logs/test_yolo.log
python3 ../../tools/slack_bot.py --text "`cat ./logs/test_yolo.log`"

echo ">>> python3 ./test_trans_caranet.py --epoch best"
python3 ./test_trans_caranet.py --train_weight $TRAIN_WEIGHT --epoch best | tee ./logs/test_trans_caranet_best.log
python3 ../../tools/slack_bot.py --text "`cat ./logs/test_trans_caranet_best.log`"

echo ">>> python3 ./test_trans_caranet.py --epoch 39"
python3 ./test_trans_caranet.py --train_weight $TRAIN_WEIGHT --epoch 39 | tee ./logs/test_trans_caranet_39.log
python3 ../../tools/slack_bot.py --text "`cat ./logs/test_trans_caranet_39.log`"

echo ">>> python3 ./test_trans_caranet.py --epoch 49"
python3 ./test_trans_caranet.py --train_weight $TRAIN_WEIGHT --epoch 49 | tee ./logs/test_trans_caranet_49.log
python3 ../../tools/slack_bot.py --text "`cat ./logs/test_trans_caranet_49.log`"

echo ">>> python3 ./test_trans_caranet.py --epoch 59"
python3 ./test_trans_caranet.py --train_weight $TRAIN_WEIGHT --epoch 59 | tee ./logs/test_trans_caranet_59.log
python3 ../../tools/slack_bot.py --text "`cat ./logs/test_trans_caranet_59.log`"

echo ">>> python3 ./test_trans_caranet.py --epoch 69"
python3 ./test_trans_caranet.py --train_weight $TRAIN_WEIGHT --epoch 69 | tee ./logs/test_trans_caranet_69.log
python3 ../../tools/slack_bot.py --text "`cat ./logs/test_trans_caranet_69.log`"

echo ">>> python3 ./test_trans_caranet.py --epoch 99"
python3 ./test_trans_caranet.py --train_weight $TRAIN_WEIGHT --epoch 99 | tee ./logs/test_trans_caranet_99.log
python3 ../../tools/slack_bot.py --text "`cat ./logs/test_trans_caranet_99.log`"
