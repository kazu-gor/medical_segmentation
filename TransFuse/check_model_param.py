from lib.TransFuse_l import TransFuse_L

model = TransFuse_L()
transformer = model.transformer
print('Transformerのパラメータ')
for name, param in transformer.named_parameters():
    print(name)

print('ResNetのパラメータ')
resnet = model.Resnet
for name, param in resnet.named_parameters():
    print(name)
