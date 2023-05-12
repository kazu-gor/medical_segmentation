python ../../git/laboratory/python/py/tools/slack_bot.py --text 'Evaluation: Nash-MTL'
python test_trans_caranet_nash.py --pth_path './snapshots/trans_caranet_nash/Transfuse-best.pth' --pth_path2 './snapshots/trans_caranet_nash/Discriminator-best.pth' || \
    python ../../git/laboratory/python/py/tools/slack_bot.py --text 'Error: TransCaraNet (nashmtl) evaluation Failed.'

#####################################################################################################################

# python ../../git/laboratory/python/py/tools/slack_bot.py --text 'Evaluation: PCGrad'
# python test_trans_caranet_nash.py --pth_path './snapshots/trans_caranet_pcgrad/Transfuse-best.pth' --pth_path2 './snapshots/trans_caranet_pcgrad/Discriminator-best.pth' || \
#     python ../../git/laboratory/python/py/tools/slack_bot.py --text 'Error: TransCaraNet (pcgrad) evaluation Failed.'

#####################################################################################################################

# python ../../git/laboratory/python/py/tools/slack_bot.py --text 'Evaluation: MGDA'
# python test_trans_caranet_nash.py --pth_path './snapshots/trans_caranet_mgda/Transfuse-best.pth' --pth_path2 './snapshots/trans_caranet_mgda/Discriminator-best.pth' || \
#     python ../../git/laboratory/python/py/tools/slack_bot.py --text 'Error: TransCaraNet (mgda) evaluation Failed.'

#####################################################################################################################

# python ../../git/laboratory/python/py/tools/slack_bot.py --text 'Evaluation: STL'
# python test_trans_caranet_nash.py --pth_path './snapshots/trans_caranet_stl/Transfuse-best.pth' --pth_path2 './snapshots/trans_caranet_stl/Discriminator-best.pth' || \
#     python ../../git/laboratory/python/py/tools/slack_bot.py --text 'Error: TransCaraNet (stl) evaluation Failed.'

#####################################################################################################################

# python ../../git/laboratory/python/py/tools/slack_bot.py --text 'Evaluation: ls'
# python test_trans_caranet_nash.py --pth_path './snapshots/trans_caranet_ls/Transfuse-best.pth' --pth_path2 './snapshots/trans_caranet_ls/Discriminator-best.pth' || \
#     python ../../git/laboratory/python/py/tools/slack_bot.py --text 'Error: TransCaraNet (ls) evaluation Failed.'

#####################################################################################################################
