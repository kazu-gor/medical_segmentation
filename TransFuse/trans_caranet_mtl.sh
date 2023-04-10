python train_trans_caranet_nash.py --tuning True --mtl 'pcgrad' --train_save 'trans_caranet_pcgrad' || \
    python ../../git/laboratory/python/py/tools/slack_bot.py --text 'Error: TransCaraNet (pcgrad) training Failed.'

python test_trans_caranet_nash.py --pth_path './snapshots/trans_caranet_pcgrad/Transfuse-best.pth' --pth_path2 './snapshots/trans_caranet_pcgrad/Discriminator-best.pth' || \
    python ../../git/laboratory/python/py/tools/slack_bot.py --text 'Error: TransCaraNet (pcgrad) evaluation Failed.'

#####################################################################################################################

python train_trans_caranet_nash.py --tuning True --mtl 'mgda' --train_save 'trans_caranet_mgda' || \
    python ../../git/laboratory/python/py/tools/slack_bot.py --text 'Error: TransCaraNet (mgda) training Failed.'

python test_trans_caranet_nash.py --pth_path './snapshots/trans_caranet_mgda/Transfuse-best.pth' --pth_path2 './snapshots/trans_caranet_mgda/Discriminator-best.pth' || \
    python ../../git/laboratory/python/py/tools/slack_bot.py --text 'Error: TransCaraNet (mgda) evaluation Failed.'

#####################################################################################################################

python train_trans_caranet_nash.py --tuning True --mtl 'stl' --train_save 'trans_caranet_stl' || \
    python ../../git/laboratory/python/py/tools/slack_bot.py --text 'Error: TransCaraNet (stl) training Failed.'

python test_trans_caranet_nash.py --pth_path './snapshots/trans_caranet_stl/Transfuse-best.pth' --pth_path2 './snapshots/trans_caranet_stl/Discriminator-best.pth' || \
    python ../../git/laboratory/python/py/tools/slack_bot.py --text 'Error: TransCaraNet (stl) evaluation Failed.'

#####################################################################################################################

python train_trans_caranet_nash.py --tuning True --mtl 'ls' --train_save 'trans_caranet_ls' || \
    python ../../git/laboratory/python/py/tools/slack_bot.py --text 'Error: TransCaraNet (ls) training Failed.'

python test_trans_caranet_nash.py --pth_path './snapshots/trans_caranet_ls/Transfuse-best.pth' --pth_path2 './snapshots/trans_caranet_ls/Discriminator-best.pth' || \
    python ../../git/laboratory/python/py/tools/slack_bot.py --text 'Error: TransCaraNet (ls) evaluation Failed.'

#####################################################################################################################
