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
# Original TransFuse
# =====================================================================

# python3 ../../tools/slack_bot.py --text ">>> Train and Test Original TransFuse"

# echo ">>> python3 ./train_trans.py"
# python3 ./train_trans.py --train_save "TransFuse" | tee ./logs/train_trans.log
# python3 ../../tools/slack_bot.py --text "TransFuse Training is done"

# echo ">>> python3 ./test_trans.py"
# python3 ./test_trans.py --pth_path "./snapshots/TransFuse/TransFuse-best.pth" 1> >(tee ./logs/test_trans_stdout.log >&1 ) 2> >(tee ./logs/test_trans_stderr.log >&2)
# python3 ../../tools/slack_bot.py --text "`cat ./logs/test_trans_stdout.log`"
# python3 ../../tools/slack_bot.py --text "`cat ./logs/test_trans_stderr.log`"

# =====================================================================
# Attention TransFuse
# =====================================================================

TRAIN_WEIGHT="polyp491_2"
echo ">>> TRAIN_WEIGHT=$TRAIN_WEIGHT"
SAVE_WEIGHT="${TRAIN_WEIGHT}_AttnTransFuse"
python3 ../../tools/slack_bot.py --text ">>> TRAIN_WEIGHT=$TRAIN_WEIGHT"

conf=(0.01 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50)
max_det=(10 5 3)

for co in ${conf[@]}
do
    for md in ${max_det[@]}
    do
        python3 ../../tools/slack_bot.py --text "conf: ${co}, max_det: ${md}"

        # echo ">>> python3 ./train_attn_trans.py"
        # python3 ./train_attn_trans.py --train_save "${SAVE_WEIGHT}_${co}" --conf ${co} --max_det ${md} | tee ./logs/train_attn_trans.log
        # python3 ../../tools/slack_bot.py --text "Attention TransFuse training is done"

        # echo ">>> python3 ./test_attn_trans.py"
        # python3 ./test_attn_trans.py --pth_path "./snapshots/${SAVE_WEIGHT}_${co}/TransFuse-best.pth" --conf ${co} --max_det ${md} 1> >(tee ./logs/train_attn_trans_stdout_${co}_${md}.log >&1 ) 2> >(tee ./logs/train_attn_trans_stderr_${co}_${md}_best.log >&2)

        # python3 ../../tools/slack_bot.py --text "`cat ./logs/train_attn_trans_stdout_${co}_${md}.log`"
        # python3 ../../tools/slack_bot.py --text "`cat ./logs/train_attn_trans_stderr_${co}_${md}.log`"

        # python3 ./test_attn_trans.py --pth_path "./snapshots/${SAVE_WEIGHT}_${co}/Transfuse-59.pth" --conf ${co} --max_det ${md} 1> >(tee ./logs/train_attn_trans_stdout_${co}_${md}.log >&1 ) 2> >(tee ./logs/train_attn_trans_stderr_${co}_${md}_59.log >&2)
        # python3 ../../tools/slack_bot.py --text "`cat ./logs/train_attn_trans_stdout_${co}_${md}.log`"
        # python3 ../../tools/slack_bot.py --text "`cat ./logs/train_attn_trans_stderr_${co}_${md}.log`"

        # python3 ./test_attn_trans.py --pth_path "./snapshots/${SAVE_WEIGHT}_${co}/Transfuse-99.pth" --conf ${co} --max_det ${md} 1> >(tee ./logs/train_attn_trans_stdout_${co}_${md}.log >&1 ) 2> >(tee ./logs/train_attn_trans_stderr_${co}_${md}_99.log >&2)
        # python3 ../../tools/slack_bot.py --text "`cat ./logs/train_attn_trans_stdout_${co}_${md}.log`"
        # python3 ../../tools/slack_bot.py --text "`cat ./logs/train_attn_trans_stderr_${co}_${md}.log`"

        # =====================================================================
        # Training part mtl
        # =====================================================================

        TRAIN_SAVE="${TRAIN_WEIGHT}_${co}_${md}"

        python3 ./train_attn_trans_nash.py --train_save $TRAIN_SAVE --conf ${co} --max_det ${md} | tee ./logs/train_attn_trans_nash_${co}_${md}.log
        python3 ../../tools/slack_bot.py --text "`cat ./logs/train_attn_trans_nash_${co}_${md}.log`"

        # =====================================================================
        # Test part mtl
        # =====================================================================

        python3 ./test_attn_trans_nash.py --conf ${co} --max_det ${md} --pth_path "./snapshots/$TRAIN_SAVE/Transfuse-best.pth" --pth_path2 "./snapshots/$TRAIN_SAVE/Discriminator-best.pth" 1> >(tee ./logs/train_attn_trans_stdout_${co}_${md}.log >&1 ) 2> >(tee ./logs/train_attn_trans_stderr_${co}_${md}_99.log >&2)
        python3 ../../tools/slack_bot.py --text "`cat ./logs/test_yolo_caranet_nash.log`"
    done
done

# =====================================================================
# Original TransCaraNet
# =====================================================================

python3 ../../tools/slack_bot.py --text ">>> Train and Test Original TransCaraNet"

echo ">>> python3 ./train_trans_caranet_origin.py"
python3 ./train_trans_caranet_origin.py | tee ./logs/train_trans_caranet_origin.log
python3 ../../tools/slack_bot.py --text "TransCaraNet Training is done"

echo ">>> python3 ./test_trans_caranet_origin.py --epoch best"
python3 ./test_trans_caranet_origin.py  --epoch best 1> >(tee ./logs/test_trans_caranet_origin_best_stdout.log >&1 ) 2> >(tee ./logs/test_trans_caranet_origin_best_stderr.log >&2)
python3 ../../tools/slack_bot.py --text "`cat ./logs/test_trans_caranet_origin_best_stdout.log`"
python3 ../../tools/slack_bot.py --text "`cat ./logs/test_trans_caranet_origin_best_stderr.log`"
