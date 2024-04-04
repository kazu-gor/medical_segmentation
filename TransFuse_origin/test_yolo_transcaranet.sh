git pull
cd ./ultrultralytics/
git pull 
cd ..

echo '--------------------- YOLOv8 Test ---------------------'
python3 ./test_yolo.py ; python3 ../../tools/slack_bot.py --text "[process] YOLOv8 test is done."

echo '--------------------- TransCaranet Test ---------------------'
python3 ./test_trans_caranet.py ; python3 ../../tools/slack_bot.py --text "[process] TransCaranet test is done."
