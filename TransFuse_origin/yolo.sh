echo ">>> yolo.sh"

echo ">>> git pull"
git pull

cd ./ultralytics/
echo ">>> git pull"
git pull

cd ../

echo ">>> python3 ./train_yolo.py"
python3 ./train_yolo.py | tee ./logs/train_yolo.log
python3 ../../tools/slack_bot.py --text "`cat ./logs/train_yolo.log`"

echo ">>> python3 ./test_yolo.py"
python3 ./test_yolo.py | tee ./logs/test_yolo.log
python3 ../../tools/slack_bot.py --text "`cat ./logs/test_yolo.log`"

echo ">>> python3 ./train_trans_caranet.py"
python3 ./train_trans_caranet.py | tee ./logs/train_trans_caranet.log
python3 ../../tools/slack_bot.py --text "`cat ./logs/train_trans_caranet.log`"

echo ">>> python3 ./test_trans_caranet.py --epoch best"
python3 ./test_trans_caranet.py --epoch best | tee ./logs/test_trans_caranet_best.log
python3 ../../tools/slack_bot.py --text "`cat ./logs/test_trans_caranet_best.log`"

echo ">>> python3 ./test_trans_caranet.py --epoch 49"
python3 ./test_trans_caranet.py --epoch 49 | tee ./logs/test_trans_caranet_49.log
python3 ../../tools/slack_bot.py --text "`cat ./logs/test_trans_caranet_49.log`"

echo ">>> python3 ./test_trans_caranet.py --epoch 59"
python3 ./test_trans_caranet.py --epoch 59 | tee ./logs/test_trans_caranet_59.log
python3 ../../tools/slack_bot.py --text "`cat ./logs/test_trans_caranet_59.log`"

echo ">>> python3 ./test_trans_caranet.py --epoch 69"
python3 ./test_trans_caranet.py --epoch 69 | tee ./logs/test_trans_caranet_69.log
python3 ../../tools/slack_bot.py --text "`cat ./logs/test_trans_caranet_69.log`"

echo ">>> python3 ./test_trans_caranet.py --epoch 99"
python3 ./test_trans_caranet.py --epoch 99 | tee ./logs/test_trans_caranet_99.log
python3 ../../tools/slack_bot.py --text "`cat ./logs/test_trans_caranet_99.log`"