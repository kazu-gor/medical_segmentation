# Segmentationの実験
#　python MyTrainVal.py
PraNetとU_PraNet、CaraNetのSegmentationの学習
オプション
PraNetの学習
--model p
U_PraNetの学習
--model u
CaraNetの学習
--model c

非健常者データのみ
--train_path './dataset/sekkai_TrainDataset'
--val_path './dataset/sekkai_ValDataset'
健常者データを含む
--train_path './dataset/TrainDataset'
--val_path './dataset/ValDataset'

重みの保存場所
./snapshots/PraNet_Res2Net
損失のグラフの場所
./fig

# python MyTestVal.py
PraNetとU_PraNet、CaraNetのSegmentationのテスト
オプション
PraNetのテスト
--model p
U_PraNetのテスト
--model u
CaraNetのテスト
--model c

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
./results/PraNet





# nash-mtlの実験
# python train_nash.py
PraNetとU_PraNet、CaraNetのnash-mtlの学習
PraNetの学習
--model p
U_PraNetの学習
--model u
CaraNetの学習
--model c

segmentationモデルをチューニングするか否か
--tuning True
の場合「./weights/修論/segmentation/」に入っているpython MyTrainVal.pyで学習した重みを用いてチューニングする
--tuning False
の場合segmentationモデルのチューニングを行わない

mtl手法の選択
nashmtlの場合
--mtl 'nashmtl'
他の手法については'https://github.com/AvivNavon/nash-mtl'を確認

修論の実験(C-1)
segmentationモデルのチューニングあり
python train_nash.py --model p --tuning True --mtl 'nashmtl'
修論の実験(C-2)
segmentationモデルのチューニングなし
python train_nash.py --model p --tuning False --mtl 'nashmtl'

# python test_nash.py
PraNetとU_PraNet、CaraNetのnash-mtlのテスト
PraNetのテスト
--model p
U_PraNetのテスト
--model u
CaraNetのテスト
--model c

segmentationモデルの重み
--pth_path './snapshots/PraNet_Res2Net/PraNet-best.pth'
識別器の重み
--pth_path2 './snapshots/PraNet_Res2Net/Discriminator-best.pth'

修論の実験(C-1)
segmentationモデルのチューニングあり
修論の実験(C-2)
segmentationモデルのチューニングなし
どちらの実験も同じ
python test_nash.py --pth_path './snapshots/PraNet_Res2Net/PraNet-best.pth' --pth_path2 './snapshots/PraNet_Res2Net/Discriminator-best.pth'





# STLの実験
# python train_finetuning.py
PraNetとU_PraNet、CaraNetのSTLの学習
PraNetの学習
--model p
U_PraNetの学習
--model u
CaraNetの学習
--model c

segmentationモデルをチューニングするか否か
--tuning_calcification True
の場合「./weights/修論/segmentation/」に入っているpython MyTrainVal.pyで学習した重みを用いてチューニングする
--tuning_calcification False
の場合segmentationモデルのチューニングを行わない

--segmentation_grad True
の場合segmentationモデルのパラメータ更新を行う
--segmentation_grad False
の場合segmentationモデルのパラメータ更新を行わない

修論の実験(B-1)
segmentationモデルのチューニングあり、パラメータ更新は行わない
python train_finetuning.py --model p --tuning_calcification True --segmentation_grad False

修論の実験(B-2)
segmentationモデルのチューニングなし、パラメータ更新を行う
python train_finetuning.py --model p --tuning_calcification False --segmentation_grad True

# python test_nash.py
PraNetとU_PraNet、CaraNetのSTLのテスト
PraNetのテスト
--model p
U_PraNetのテスト
--model u
CaraNetのテスト
--model c

segmentationモデルの重み
--pth_path './snapshots/PraNet_Res2Net/PraNet-best.pth'
識別器の重み
--pth_path2 './snapshots/PraNet_Res2Net/Discriminator-best.pth'

修論の実験(B-1)
segmentationモデルのチューニングあり、パラメータ更新は行わない
python test_nash.py --pth_path './weights/修論/segmentation/PraNet/石灰化ありのみ/PraNet-best.pth' --pth_path2 './snapshots/PraNet_Res2Net/Discriminator-best.pth'

修論の実験(B-2)
segmentationモデルのチューニングなし、パラメータ更新を行う
python test_nash.py --pth_path './snapshots/PraNet_Res2Net/PraNet-best.pth' --pth_path2 './snapshots/PraNet_Res2Net/Discriminator-best.pth'





# 従来手法の面積を特徴量として識別を行う実験
python MyTrainVal.py
で学習した重みを用いて
python MyTestVal.py
で「検証データ（--test_path './dataset/ValDataset/' もしくは　--test_path './dataset/sekkai_ValDataset/'」の画像を生成（modelとdatasetの「非健常者データのみ」か「健常者データを含む」については学習とテストで矛盾しないように）
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





# python split.py
dataset作成用
# python split_train_sekkainomi.py
trainのみ石灰化を含む画像のみにして分ける
その他はsplit.pyと同じ
上2つのコードのrandom_stateの値を揃える必要がある





# ディレクトリ
#bbox_result
f_measure.pyの出力画像の保存場所

# dataset
修論で使用した画像

#fig
学習時の損失のグラフとroc曲線の保存場所

# lab
評価やデータセット作成のためのコードをまとめたもの
卒論のときのものも混ざっている

# lib
PraNetとU_PraNet、CaraNetのコード

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
TransFuseとTransCaraNetはTransFuseのweightsディレクトリに入っているものと同じ

#発表
ISCIT2021で使用した実験データ

# f_measure.py
F値を求める


