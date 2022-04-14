## Pytorch视觉模型库--timm

⌚️:2021年12月17日

📚参考

- [GitHub](https://github.com/rwightman/pytorch-image-models)
- [Lean Tutorials](https://rwightman.github.io/pytorch-image-models/)
- [Comprehensive Tutorials](https://fastai.github.io/timmdocs/#)
- [paperswithcode 排名](https://paperswithcode.com/lib/timm)

---

## 1. timm是什么？

Py**t**orch **I**mage **M**odels (timm) 整合了常用的models, layers, utilities, optimizers, schedulers, data-loaders / augmentations, and reference training / validation scripts，它的目的是将各种SOTA模型整合在一起，并具有再现ImageNet训练结果的能力。

**注意**：timm库只针对ImageNet训练，也就是只有训练主干网络所涉及的内容。

## 2. 安装

```python
pip install timm
```

建议在python>=3.6, pytorch>=1.4环境下使用

## 3. 使用

```python
import timm
```

### 3.1  查询模型：timm.list_models()

查看存在的所有模型

```python
model_list = timm.list_models()
print(len(model_list), model_list[:3])
# Results 541 ['adv_inception_v3', 'botnet26t_256', 'botnet50ts_256']
```

查看具有预训练参数的模型

```python
model_pretrain_list = timm.list_models(pretrained=True)
print(len(model_pretrain_list), model_pretrain_list[:3])
# Results：396 ['adv_inception_v3', 'cait_m36_384', 'cait_m48_448']
```

检索特定模型，采用模糊查询，如resnet系列

```python
model_resnet = timm.list_models('*resnet*')
print(len(model_resnet), model_resnet[:3])
# Results: 117 ['cspresnet50', 'cspresnet50d', 'cspresnet50w']
```

可进一步查看想用的模型是否提供了预训练参数

```python
print('resnet50: ', 'resnet50' in model_pretrain_list,
      'resnet101: ', 'resnet101' in model_pretrain_list)
# Results:     resnet50:  True      resnet101:  False
```

### 3.2 创建模型：timm.create_model()

创建预定义的完整的分类模型，可通过pretrained选项选择是否加载预训练参数

```python
import torch
x = torch.randn([1, 3, 224, 224])
model_resnet50 = timm.create_model('resnet50', pretrained=True)
out = model_resnet50(x)
print(out.shape)
# Results: torch.Size([1, 1000])
```

改变输出类别数目，微调模型：num_classes

```python
model_resnet50_finetune = timm.create_model('resnet50', pretrained=True, num_classes=10)
out = model_resnet50_finetune (x)
print(out.shape)
# Results: torch.Size([1, 10])
```

改变输入通道数：in_chans

```python
# 通道数改变后，对应的权重参数会进行相应的处理，此处不作详细说明，
#可参照：https://fastai.github.io/timmdocs/models或直接查看源代码
x = torch.randn([1, 1, 224, 224])
feature_extractor = timm.create_model('resnet50', in_chans=1, features_only=True, out_indices=[1, 3, 4])
```

获取分类层前（倒数第二层）的特征

①直接调用forward_features()函数

```python
x = torch.randn([1, 3, 224, 224])
Backbone1 = timm.create_model('vit_base_patch16_224')
Backbone2 = timm.create_model('resnet50')
feature1 = Backbone1.forward_features(x)
feature2 = Backbone2.forward_features(x)
print('vit_feature:', feature1.shape, 'resnet_feature:', feature2.shape)
# Results: vit_feature: torch.Size([1, 768])    resnet_feature: torch.Size([1, 2048, 7, 7])
```

②直接创建没有池化和分类层的模型，对于基于CNN的模型可以这样做

```python
x = torch.randn([1, 3, 224, 224])
Backbone1 = timm.create_model('resnet50', num_classes=0, global_pool='')
Backbone2 = timm.create_model('resnet50', num_classes=0)
feature1 = Backbone1(x)
feature2 = Backbone2(x)
print('before pooling:', feature1.shape, 'after pooling:', feature2.shape)
# Results: before pooling: torch.Size([1, 2048, 7, 7])    after pooling: torch.Size([1, 2048])
```

③通过移除层来获得

```python
x = torch.randn([1, 3, 224, 224])
Backbone1 = timm.create_model('resnet50')
Backbone2 = timm.create_model('resnet50')
Backbone1.reset_classifier(0, '')
Backbone2.reset_classifier(0)
feature1 = Backbone1(x)
feature2 = Backbone2(x)
print('before pooling:', feature1.shape, 'after pooling:', feature2.shape)
# Results: before pooling: torch.Size([1, 2048, 7, 7]) after pooling: torch.Size([1, 2048])
```

获取中间层特征：features_only

```python
x = torch.randn([1, 3, 224, 224])
feature_extractor = timm.create_model('resnet50', features_only=True)  # 并非所有model都有此选项
feature_list = feature_extractor(x)
for a in feature_list:
    print(a.shape)
# Results:
# torch.Size([1, 64, 112, 112])
# torch.Size([1, 256, 56, 56])
# torch.Size([1, 512, 28, 28])
# torch.Size([1, 1024, 14, 14])
# torch.Size([1, 2048, 7, 7])
```

可通过out_indices参数指定从哪个level获取feature

```python
feature_extractor = timm.create_model('resnet50', features_only=True, out_indices=[1, 3, 4])
feature_list = feature_extractor(x)
for a in feature_list:
    print(a.shape)
# Results:
# torch.Size([1, 256, 56, 56])
# torch.Size([1, 1024, 14, 14])
# torch.Size([1, 2048, 7, 7])
```