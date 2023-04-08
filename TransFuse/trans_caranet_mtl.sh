python train_trans_caranet_nash.py --tuning True --mtl 'nashmtl' || \
    python ../../git/laboratory/python/py/tools/slack_bot.py --text 'Error: TransCaraNet training Failed.'

python test_trans_caranet_nash.py --pth_path './snapshots/Transfuse_S/Transfuse-best.pth' --pth_path2 './snapshots/Transfuse_S/Discriminator-best.pth' || \
    python ../../git/laboratory/python/py/tools/slack_bot.py --text 'Error: TransCaraNet evaluation Failed.'


