# Change Detection Laboratory

致力于为基于深度学习的变化检测算法的开发和实验提供一个统一、轻量、便捷的项目模板（project template）。

[English](/README.md) | 简体中文

## 依赖库

> opencv-python==4.1.1  
  pytorch==1.3.1  
  torchvision==0.4.2  
  pyyaml==5.1.2  
  scikit-image==0.15.0  
  scikit-learn==0.21.3  
  scipy==1.3.1  
  tqdm==4.35.0

项目代码在 Python 3.7.4，Ubuntu 16.04 环境下测试通过。

## 快速入门

首先，从 GitHub 网站上拷贝此仓库。

```bash
git clone --recurse-submodules git@github.com:Bobholamovic/CDLab.git
cd CDLab
mkdir exp
cd src
```

在 `src/constants.py` 文件中将相应常量修改为数据集存放的位置。

### 模型训练

运行如下指令从头训练一个模型：

```bash
python train.py train --exp_config PATH_TO_CONFIG_FILE
```

在 `configs/` 文件夹中已经包含了一部分现成的配置文件，可供直接使用。 

训练脚本开始执行后，首先会输出配置信息到屏幕，然后会有出现一个提示符，指示您输入一些笔记。这些笔记将被记录到日志文件中。如果在一段时间后，您忘记了本次实验的具体内容，这些笔记可能有助于您回想起来。当然，您也可以选择按下回车键直接跳过。

如果需要从一个检查点（checkpoint）开始继续训练，运行如下指令：

```bash
python train.py train --exp_config PATH_TO_CONFIG_FILE --resume PATH_TO_CHECKPOINT
```

以下是对一些可选参数的介绍：

- `anew`: 如果您希望指定的检查点只是用于初始化模型参数，指定此选项。请注意，从一个不兼容的模型中获取部分层的参数对待训练的模型进行初始化也是允许的。
- `save_on`: 如果需要在进行模型评估的同时储存模型的输出结果，指定此选项。项目默认采用基于 epoch 的训练器。在每个 epoch 末尾，训练器也将在验证集上评估模型的性能。
- `log_off`: 指定此选项以禁用日志文件。
- `tb_on`: 指定此选项以启用 tensorboard 日志。
- `debug_on`: 指定此选项以在程序崩溃处自动设置断点，便于进行事后调试。

在训练过程中或训练完成后，您可以在 `exp/DATASET_NAME/weights/` 文件夹下查看模型权重文件，在 `exp/DATASET_NAME/logs` 文件夹下查看日志文件，在 `exp/DATASET_NAME/out` 文件夹下查看输出的变化图。

### 模型评估

使用如下指令评估训练好的模型：

```bash
python train.py eval --exp_config PATH_TO_CONFIG_FILE --resume PATH_TO_CHECKPOINT --save_on
```

## 预置模型列表

模型名称 | 对应名称 | 链接
:-:|:-:|:-:
CDNet | `CDNet` | [paper](https://doi.org/10.1007/s10514-018-9734-5)
FC-EF | `UNet` | [paper](https://ieeexplore.ieee.org/abstract/document/8451652)
FC-Siam-conc | `SiamUNet_conc` | [paper](https://ieeexplore.ieee.org/abstract/document/8451652)
FC-Siam-diff | `SiamUNet_diff` | [paper](https://ieeexplore.ieee.org/abstract/document/8451652)

## 预置数据集列表

数据集名称 | 对应名称 | 链接
:-:|:-:|:-:
SZTAKI AirChange Benchmark set: Szada set | `AC_Szada` | [source](http://web.eee.sztaki.hu/remotesensing/airchange_benchmark.html)
SZTAKI AirChange Benchmark set: Tiszadob set | `AC_Tiszadob` | [source](http://web.eee.sztaki.hu/remotesensing/airchange_benchmark.html)
Onera Satellite Change Detection dataset | `OSCD` | [source](https://rcdaudt.github.io/oscd/)
Synthetic images and real season-varying remote sensing images | `Lebedev` | [source](https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9)

## 开源许可证

该项目采用 [Unlicense 开源许可证](/LICENSE)。

## 二次开发

此部分文档仍待完善。如果您想了解更多关于此项目的设计思想和实现细节，可参考[此处](https://github.com/Bobholamovic/DuduLearnsToCode-Template/blob/main/take_a_look_if_you_want.md)。

## 贡献

欢迎大家为本项目贡献代码和意见。