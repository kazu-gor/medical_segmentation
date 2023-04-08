# Segmentationの実験
#　python train_trans_caranet.py
TransCaraNetのSegmentationの学習
# python train_trans.py
TransFuseのSegmentationの学習

オプション
非健常者データのみ
--train_path './dataset/sekkai_TrainDataset'
--val_path './dataset/sekkai_ValDataset'
健常者データを含む
--train_path './dataset/TrainDataset'
--val_path './dataset/ValDataset'

重みの保存場所
./snapshots/Transfuse_S
損失のグラフの場所
./fig

# python test_trans_caranet.py
TransFuseのSegmentationのテスト
# python test_trans.py
TransFuseのSegmentationのテスト

オプション
非健常者データのみ
テストデータ
--test_path './dataset/sekkai_TestDataset'
検証データ
--test_path './dataset/sekkai_ValDataset'

健常者データを含む
テストデータ
--test_path './dataset/TestDataset'
検証データ
--test_path './dataset/ValDataset'

出力画像の保存場所
./results/Transfuse_S





# MAEの実験
コードはsegmentationの実験と同じ
./libディレクトリにあるmodels_vit.pyの95行目の重みファイルを変えて実験する

健常者データを含めた1141枚で事前学習した重み
checkpoint = torch.load('checkpoint-399_vit_l_352.pth')
非健常者データのみの491枚で事前学習した重み
checkpoint = torch.load('checkpoint-399_vit_l_352_491.pth')
ImageNetのみで事前学習した重み
checkpoint = torch.load('mae_pretrain_vit_large.pth')





# nash-mtlの実験
# python test_trans_caranet_nash.py
TransCaraNetのnash-mtlの学習
# python test_trans_nash.py
TransFuseのnash-mtlの学習

segmentationモデルをチューニングするか否か
--tuning True
の場合「./weights/修論/segmentation/」に入っているsegmentationの実験で学習した重みを用いてチューニングする
--tuning False
の場合segmentationモデルのチューニングを行わない

mtl手法の選択
nashmtlの場合
--mtl 'nashmtl'
他の手法については'https://github.com/AvivNavon/nash-mtl'を確認

修論の実験(C-1)
segmentationモデルのチューニングあり
python train_trans_caranet_nash.py --tuning True --mtl 'nashmtl'
修論の実験(C-2)
segmentationモデルのチューニングなし
python train_trans_caranet_nash.py --tuning False --mtl 'nashmtl'

# python test_trans_caranet_nash.py
TransCaraNetのnash-mtlのテスト
# python test_trans_nash.py
TransFuseのnash-mtlのテスト

segmentationモデルの重み
--pth_path './snapshots/Transfuse_S/Transfuse-best.pth'
識別器の重み
--pth_path2 './snapshots/Transfuse_S/Discriminator-best.pth'

修論の実験(C-1)
segmentationモデルのチューニングあり
修論の実験(C-2)
segmentationモデルのチューニングなし
どちらの実験も同じ
python test_trans_caranet_nash.py --pth_path './snapshots/Transfuse_S/Transfuse-best.pth' --pth_path2 './snapshots/Transfuse_S/Discriminator-best.pth'





# STLの実験
# python train_trans_caranet_finetuning.py
TransCaraNetのSTLの学習
# python train_trans_finetuning.py
TransFuseのSTLの学習

segmentationモデルをチューニングするか否か
--tuning_calcification True
の場合「./weights/修論/segmentation/」に入っているsegmentationの実験で学習した重みを用いてチューニングする
--tuning_calcification False
の場合segmentationモデルのチューニングを行わない

--segmentation_grad True
の場合segmentationモデルのパラメータ更新を行う
--segmentation_grad False
の場合segmentationモデルのパラメータ更新を行わない

修論の実験(B-1)
segmentationモデルのチューニングあり、パラメータ更新は行わない
python train_trans_caranet_finetuning.py --tuning_calcification True --segmentation_grad False

修論の実験(B-2)
segmentationモデルのチューニングなし、パラメータ更新を行う
python train_trans_caranet_finetuning.py --tuning_calcification False --segmentation_grad True

# python test_trans_caranet_nash.py
TransCaraNetのSTLのテスト
# python test_trans_nash.py
TransFuseのSTLのテスト

segmentationモデルの重み
--pth_path './snapshots/Transfuse_S/Transfuse-best.pth'
識別器の重み
--pth_path2 './snapshots/Transfuse_S/Discriminator-best.pth'

修論の実験(B-1)
segmentationモデルのチューニングあり、パラメータ更新は行わない
python test_trans_caranet_nash.py --pth_path './weights/修論/segmentation/TransCaraNet+MAE_calsification/石灰化ありのみ/Transfuse-best.pth' --pth_path2 './snapshots/Transfuse_S/Discriminator-best.pth'
python test_trans_nash.py --pth_path './weights/修論/segmentation/TransFuse-L+MAE/vit-l_352/石灰化ありのみ/Transfuse-best.pth' --pth_path2 './snapshots/Transfuse_S/Discriminator-best.pth'

修論の実験(B-2)
segmentationモデルのチューニングなし、パラメータ更新を行う
TransCaraNet
python test_trans_caranet_nash.py --pth_path './snapshots/Transfuse_S/Transfuse-best.pth' --pth_path2 './snapshots/Transfuse_S/Discriminator-best.pth'
TransFuse
python test_trans_nash.py --pth_path './snapshots/Transfuse_S/Transfuse-best.pth' --pth_path2 './snapshots/Transfuse_S/Discriminator-best.pth'




# 従来手法の面積を特徴量として識別を行う実験
segmentationの実験で学習した重みを用いて
TransCaraNet
python test_trans_caranet.py
TransFuse
python test_trans.py
「検証データ（--test_path './dataset/ValDataset/' もしくは　--test_path './dataset/sekkai_ValDataset/'」の画像を生成（modelとdatasetの「非健常者データのみ」か「健常者データを含む」については学習とテストで矛盾しないように）
↓
python thresh_calculate.py
面積のしきい値を求める
↓
python f_measure.py
で評価
51行目の「lower_size」を先ほど求めたものに設定
↓
python auc.py
でroc_curveとAUCを求める
roc_curveはfigディレクトリに「roc_curve.png」として保存





# ディレクトリ
#bbox_result
f_measure.pyの出力画像の保存場所

# dataset
修論で使用した画像

#fig
学習時の損失のグラフとroc曲線の保存場所

# lib
TransCaraNetとTransFuseのコード

#results
testしたときの出力画像が保存される場所

# roc_curve
修論に載せたroc_curveを保存しているだけのディレクトリ

# snapshots
学習した重みが保存されれる場所

# utils
dataloaderとmtlの実験で使用するコードが入っている

# weights
卒論と修論の実験で使用した重み


